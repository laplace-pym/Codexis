"""
Streaming Executor - 流式执行器

基于 h2A 队列实现的流式 Agent 执行器，支持：
1. 实时流式输出（边思考边输出）
2. 低延迟响应（快速通道）
3. 随时打断（steering）
"""

import asyncio
from typing import Optional, AsyncIterator
from dataclasses import dataclass

from llm.base import BaseLLM, Message, ToolDefinition
from tools.registry import ToolRegistry
from utils.h2a_queue import StreamingMessageQueue
from utils.logger import Logger, get_logger

from .base import AgentState
from .executor import AgentExecutor


@dataclass
class StreamEvent:
    """流式事件"""
    type: str  # "thinking", "tool_call", "tool_result", "complete", "error"
    content: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StreamingExecutor(AgentExecutor):
    """
    流式执行器
    
    继承自 AgentExecutor，增加流式输出能力。
    
    使用示例：
        executor = StreamingExecutor(llm, tools)
        
        async for event in executor.execute_stream(task):
            if event.type == "thinking":
                print(event.content, end="", flush=True)
            elif event.type == "tool_call":
                print(f"\\n🔧 {event.metadata['tool_name']}")
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        max_iterations: int = 10,
        verbose: bool = True,
        queue_size: int = 1000,
    ):
        super().__init__(llm, tool_registry, max_iterations, verbose)
        self.queue_size = queue_size
        self._current_queue: Optional[StreamingMessageQueue] = None
    
    async def execute_stream(
        self,
        task: str,
        state: Optional[AgentState] = None,
        context: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        流式执行任务
        
        Args:
            task: 任务描述
            state: 可选的已有状态
            context: 额外上下文
            
        Yields:
            StreamEvent: 流式事件
        """
        # 创建消息队列
        queue = StreamingMessageQueue(
            max_size=self.queue_size,
            name=f"stream_{task[:20]}"
        )
        self._current_queue = queue
        
        try:
            # 在后台任务中执行
            execution_task = asyncio.create_task(
                self._execute_with_streaming(task, state, context, queue)
            )
            
            # 从队列中流式读取事件
            async for msg in queue:
                if msg is None:
                    break
                
                # 转换为 StreamEvent
                event = StreamEvent(
                    type=msg.type.value,
                    content=str(msg.content),
                    metadata=msg.metadata
                )
                
                yield event
                
                # 检查是否完成
                if msg.type == StreamingMessageQueue.MessageType.COMPLETE:
                    break
                
                # 检查是否被打断
                if msg.type == StreamingMessageQueue.MessageType.INTERRUPT:
                    execution_task.cancel()
                    break
            
            # 等待执行任务完成
            try:
                await execution_task
            except asyncio.CancelledError:
                pass
        
        finally:
            await queue.close()
            self._current_queue = None
    
    async def _execute_with_streaming(
        self,
        task: str,
        state: Optional[AgentState],
        context: Optional[str],
        queue: StreamingMessageQueue,
    ):
        """带流式输出的执行逻辑"""
        try:
            # 发送开始状态
            await queue.send_status("started", task=task)
            
            # 初始化状态
            if state is None:
                state = AgentState(max_iterations=self.max_iterations)
            state.start_task(task)
            
            # 构建初始消息
            messages = self._build_initial_messages(task, context)
            tool_definitions = self.tools.get_definitions()
            
            # 主执行循环
            while not state.is_complete and state.increment_iteration():
                try:
                    # 发送迭代状态
                    await queue.send_status(
                        f"iteration_{state.iteration}",
                        iteration=state.iteration,
                        max_iterations=self.max_iterations
                    )
                    
                    # 调用 LLM（这里可以进一步优化为流式）
                    response = self.llm.chat_sync(
                        messages=messages,
                        tools=tool_definitions,
                        temperature=0.7,
                    )
                    
                    # 发送思考内容
                    if response.content:
                        await queue.send_text(
                            response.content,
                            type="thinking",
                            iteration=state.iteration
                        )
                        
                        state.history.add_think(response.content)
                        messages.append(Message.assistant(
                            response.content,
                            tool_calls=response.tool_calls if response.has_tool_calls else None
                        ))
                    
                    # 处理工具调用
                    if response.has_tool_calls:
                        for tc in response.tool_calls:
                            # 发送工具调用事件
                            await queue.send_tool_call(
                                tc.name,
                                tc.arguments,
                                tool_id=tc.id
                            )
                            
                            # 执行工具
                            result = self.tools.execute(tc.name, **tc.arguments)
                            
                            # 发送工具结果
                            await queue.enqueue(
                                StreamingMessageQueue.Message(
                                    type=StreamingMessageQueue.MessageType.TOOL_RESULT,
                                    content=str(result),
                                    metadata={
                                        "tool_name": tc.name,
                                        "success": result.success
                                    }
                                )
                            )
                            
                            # 更新消息历史
                            messages.append(Message.tool(
                                content=str(result),
                                tool_call_id=tc.id,
                                name=tc.name,
                            ))
                        
                        state.clear_errors()
                        
                        # 检查提前完成
                        self._check_early_completion(state)
                        if state.is_complete:
                            break
                    
                    else:
                        # 没有工具调用，检查是否完成
                        if self._is_task_complete(response.content, state):
                            state.complete(response.content)
                        else:
                            messages.append(Message.user(
                                "Please continue with the task. Use tools if needed, "
                                "or indicate completion if done."
                            ))
                
                except Exception as e:
                    await queue.enqueue(
                        StreamingMessageQueue.Message(
                            type=StreamingMessageQueue.MessageType.ERROR,
                            content=str(e),
                            metadata={"iteration": state.iteration}
                        )
                    )
                    
                    if not state.record_error():
                        state.complete(f"Task failed after too many errors: {str(e)}")
                        break
                    
                    messages.append(Message.user(
                        f"An error occurred: {str(e)}\n"
                        "Please try a different approach or fix the issue."
                    ))
            
            # 检查是否用完迭代次数
            if not state.is_complete:
                state.complete(
                    f"Task incomplete after {self.max_iterations} iterations."
                )
            
            # 发送完成事件
            await queue.enqueue(
                StreamingMessageQueue.Message(
                    type=StreamingMessageQueue.MessageType.COMPLETE,
                    content=state.final_result or "completed",
                    metadata={
                        "iterations": state.iteration,
                        "success": state.is_complete
                    }
                )
            )
        
        except asyncio.CancelledError:
            # 被打断
            await queue.send_status("cancelled")
            raise
        
        except Exception as e:
            await queue.enqueue(
                StreamingMessageQueue.Message(
                    type=StreamingMessageQueue.MessageType.ERROR,
                    content=f"Fatal error: {str(e)}",
                    metadata={"fatal": True}
                )
            )
            raise
        
        finally:
            await queue.close()
    
    async def interrupt(self):
        """打断当前执行"""
        if self._current_queue:
            await self._current_queue.interrupt()
