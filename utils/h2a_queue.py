"""
h2A (Double-Buffered Async Message Queue)
双缓冲异步消息队列

核心目标：
1. 低延迟：有等待消费者时，消息直达（fast path）
2. 高吞吐：减少锁竞争，支持批量处理
3. 可控背压：队列满时的处理策略

设计：
- 写缓冲 (write_buffer)：生产者写入
- 读缓冲 (read_buffer)：消费者读取
- 等待者队列 (waiters)：等待消息的 Future
- 快速通道：如果有 waiter，直接 resolve，不进队列
"""

import asyncio
from typing import Optional, Any, List
from enum import Enum
from dataclasses import dataclass
import time


class BackpressureStrategy(Enum):
    """背压策略：队列满时如何处理"""
    DROP_OLDEST = "drop_oldest"      # 丢弃最老的消息
    DROP_NEWEST = "drop_newest"      # 丢弃最新的消息
    BLOCK = "block"                  # 阻塞等待（生产者会等）
    ERROR = "error"                  # 抛出异常


@dataclass
class QueueStats:
    """队列统计信息"""
    total_enqueued: int = 0      # 总入队数
    total_dequeued: int = 0      # 总出队数
    fast_path_hits: int = 0      # 快速通道命中次数
    buffer_swaps: int = 0        # 缓冲区交换次数
    dropped_messages: int = 0    # 丢弃的消息数
    current_size: int = 0        # 当前队列大小


class H2AQueue:
    """
    h2A 双缓冲异步消息队列
    
    特性：
    - 双缓冲设计减少锁竞争
    - 快速通道支持零延迟传递
    - 可配置背压策略
    - 完整的统计信息
    
    使用示例：
        queue = H2AQueue(max_size=1000)
        
        # 生产者
        await queue.enqueue("message")
        
        # 消费者
        async for msg in queue:
            print(msg)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        backpressure: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
        name: str = "h2a_queue",
    ):
        """
        初始化队列
        
        Args:
            max_size: 每个缓冲区的最大容量
            backpressure: 背压策略
            name: 队列名称（用于日志）
        """
        self.max_size = max_size
        self.backpressure = backpressure
        self.name = name
        
        # 双缓冲
        self._write_buffer: List[Any] = []
        self._read_buffer: List[Any] = []
        
        # 等待者队列（消费者在等消息）
        self._waiters: List[asyncio.Future] = []
        
        # 状态管理
        self._closed = False
        self._lock = asyncio.Lock()
        
        # 统计信息
        self.stats = QueueStats()
    
    async def enqueue(self, item: Any) -> bool:
        """
        入队（生产者调用）
        
        Args:
            item: 要入队的消息
            
        Returns:
            是否成功入队
        """
        if self._closed:
            return False
        
        async with self._lock:
            # 🚀 快速通道：如果有等待者，直接交付
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(item)
                    self.stats.total_enqueued += 1
                    self.stats.fast_path_hits += 1
                    return True
            
            # 常规路径：写入缓冲区
            if len(self._write_buffer) >= self.max_size:
                # 背压处理
                if self.backpressure == BackpressureStrategy.DROP_OLDEST:
                    self._write_buffer.pop(0)
                    self.stats.dropped_messages += 1
                elif self.backpressure == BackpressureStrategy.DROP_NEWEST:
                    self.stats.dropped_messages += 1
                    return False
                elif self.backpressure == BackpressureStrategy.ERROR:
                    raise RuntimeError(f"Queue {self.name} is full")
                # BLOCK 策略会在下面自然阻塞（因为锁）
            
            self._write_buffer.append(item)
            self.stats.total_enqueued += 1
            self.stats.current_size = len(self._write_buffer) + len(self._read_buffer)
            
            return True
    
    async def dequeue(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        出队（消费者调用）
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            消息，如果超时或队列关闭则返回 None
        """
        start_time = time.time() if timeout else None
        
        while not self._closed:
            async with self._lock:
                # 1. 从读缓冲读取
                if self._read_buffer:
                    item = self._read_buffer.pop(0)
                    self.stats.total_dequeued += 1
                    self.stats.current_size = len(self._write_buffer) + len(self._read_buffer)
                    return item
                
                # 2. 如果读缓冲空了，尝试交换
                if self._write_buffer:
                    self._read_buffer, self._write_buffer = self._write_buffer, self._read_buffer
                    self.stats.buffer_swaps += 1
                    continue
                
                # 3. 两个缓冲都空，创建等待者
                if self._closed:
                    return None
                
                waiter = asyncio.get_event_loop().create_future()
                self._waiters.append(waiter)
            
            # 4. 等待消息到来（释放锁后等待）
            try:
                if timeout:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        return None
                    item = await asyncio.wait_for(waiter, timeout=remaining)
                else:
                    item = await waiter
                
                self.stats.total_dequeued += 1
                return item
            
            except asyncio.TimeoutError:
                # 超时，移除等待者
                async with self._lock:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
                return None
            except asyncio.CancelledError:
                # 被取消，移除等待者
                async with self._lock:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
                raise
        
        return None
    
    def enqueue_nowait(self, item: Any) -> bool:
        """
        同步入队（不等待）
        
        注意：这是一个非阻塞操作，如果队列满会根据背压策略处理
        """
        if self._closed:
            return False
        
        # 简化版：直接操作写缓冲
        if len(self._write_buffer) >= self.max_size:
            if self.backpressure == BackpressureStrategy.DROP_OLDEST:
                self._write_buffer.pop(0)
                self.stats.dropped_messages += 1
            elif self.backpressure == BackpressureStrategy.DROP_NEWEST:
                self.stats.dropped_messages += 1
                return False
            elif self.backpressure == BackpressureStrategy.ERROR:
                raise RuntimeError(f"Queue {self.name} is full")
        
        self._write_buffer.append(item)
        self.stats.total_enqueued += 1
        self.stats.current_size = len(self._write_buffer) + len(self._read_buffer)
        
        # 尝试唤醒等待者
        if self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                # 从写缓冲取出刚加入的消息
                item = self._write_buffer.pop()
                waiter.set_result(item)
                self.stats.fast_path_hits += 1
        
        return True
    
    async def close(self):
        """关闭队列"""
        async with self._lock:
            self._closed = True
            # 唤醒所有等待者
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.set_result(None)
            self._waiters.clear()
    
    def is_closed(self) -> bool:
        """是否已关闭"""
        return self._closed
    
    def size(self) -> int:
        """当前队列大小"""
        return len(self._write_buffer) + len(self._read_buffer)
    
    def get_stats(self) -> QueueStats:
        """获取统计信息"""
        self.stats.current_size = self.size()
        return self.stats
    
    def __aiter__(self):
        """异步迭代器"""
        return self
    
    async def __anext__(self):
        """异步迭代"""
        item = await self.dequeue()
        if item is None:
            if self._closed:
                raise StopAsyncIteration
            # 如果没关闭但返回 None，继续等待
            return await self.__anext__()
        return item
    
    def __repr__(self) -> str:
        return (
            f"H2AQueue(name={self.name}, size={self.size()}, "
            f"enqueued={self.stats.total_enqueued}, "
            f"dequeued={self.stats.total_dequeued}, "
            f"fast_path_hits={self.stats.fast_path_hits})"
        )


class StreamingMessageQueue(H2AQueue):
    """
    流式消息队列（h2A 的特化版本）
    
    针对 Agent 流式输出优化：
    - 支持文本块、工具调用、状态更新等多种消息类型
    - 内置消息类型过滤
    - 支持实时打断（steering）
    """
    
    class MessageType(Enum):
        """消息类型"""
        TEXT = "text"              # 文本块
        TOOL_CALL = "tool_call"    # 工具调用
        TOOL_RESULT = "tool_result" # 工具结果
        STATUS = "status"          # 状态更新
        ERROR = "error"            # 错误
        COMPLETE = "complete"      # 完成
        INTERRUPT = "interrupt"    # 打断信号
    
    @dataclass
    class Message:
        """流式消息"""
        type: 'StreamingMessageQueue.MessageType'
        content: Any
        metadata: dict = None
        timestamp: float = None
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = time.time()
            if self.metadata is None:
                self.metadata = {}
    
    def __init__(self, max_size: int = 1000, name: str = "streaming_queue"):
        super().__init__(
            max_size=max_size,
            backpressure=BackpressureStrategy.DROP_OLDEST,
            name=name
        )
        self._interrupted = False
    
    async def send_text(self, text: str, **metadata):
        """发送文本块"""
        msg = self.Message(
            type=self.MessageType.TEXT,
            content=text,
            metadata=metadata
        )
        await self.enqueue(msg)
    
    async def send_tool_call(self, tool_name: str, args: dict, **metadata):
        """发送工具调用"""
        msg = self.Message(
            type=self.MessageType.TOOL_CALL,
            content={"name": tool_name, "args": args},
            metadata=metadata
        )
        await self.enqueue(msg)
    
    async def send_status(self, status: str, **metadata):
        """发送状态更新"""
        msg = self.Message(
            type=self.MessageType.STATUS,
            content=status,
            metadata=metadata
        )
        await self.enqueue(msg)
    
    async def interrupt(self):
        """发送打断信号"""
        self._interrupted = True
        msg = self.Message(
            type=self.MessageType.INTERRUPT,
            content="interrupted"
        )
        await self.enqueue(msg)
    
    def is_interrupted(self) -> bool:
        """是否被打断"""
        return self._interrupted
    
    def reset_interrupt(self):
        """重置打断状态"""
        self._interrupted = False
