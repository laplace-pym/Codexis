#!/usr/bin/env python3
"""
流式执行演示

展示 h2A 双缓冲队列和流式 Agent 的能力：
1. 实时流式输出
2. 低延迟响应
3. 随时打断（steering）

运行：
    python examples/streaming_demo.py
"""

import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.h2a_queue import H2AQueue, StreamingMessageQueue
from agent.streaming_executor import StreamingExecutor
from llm.factory import LLMFactory
from tools.registry import create_default_registry
from utils.logger import get_logger


async def demo_basic_h2a():
    """演示基础 h2A 队列特性"""
    print("=" * 70)
    print("🧪 演示 1: 基础 h2A 队列")
    print("=" * 70)
    
    queue = H2AQueue(max_size=10, name="demo")
    
    # 生产者任务
    async def producer():
        for i in range(5):
            await queue.enqueue(f"Message {i}")
            print(f"✅ 生产: Message {i}")
            await asyncio.sleep(0.1)
        await queue.close()
    
    # 消费者任务
    async def consumer():
        async for msg in queue:
            if msg is None:
                break
            print(f"📨 消费: {msg}")
            await asyncio.sleep(0.2)  # 模拟处理时间
    
    # 并发运行
    await asyncio.gather(producer(), consumer())
    
    # 显示统计
    stats = queue.get_stats()
    print(f"\n📊 统计:")
    print(f"  总入队: {stats.total_enqueued}")
    print(f"  总出队: {stats.total_dequeued}")
    print(f"  快速通道命中: {stats.fast_path_hits}")
    print(f"  缓冲区交换: {stats.buffer_swaps}")


async def demo_fast_path():
    """演示快速通道（零延迟）"""
    print("\n" + "=" * 70)
    print("🧪 演示 2: 快速通道（消费者先等待）")
    print("=" * 70)
    
    queue = H2AQueue(max_size=10, name="fast_path")
    
    # 消费者先等待
    async def waiting_consumer():
        print("📨 消费者: 开始等待消息...")
        msg = await queue.dequeue()
        print(f"📨 消费者: 收到消息 '{msg}' （快速通道！）")
    
    # 生产者稍后发送
    async def delayed_producer():
        await asyncio.sleep(0.5)
        print("✅ 生产者: 发送消息")
        await queue.enqueue("快速通道消息")
    
    # 先启动消费者，再启动生产者
    consumer_task = asyncio.create_task(waiting_consumer())
    await asyncio.sleep(0.1)  # 确保消费者先等待
    producer_task = asyncio.create_task(delayed_producer())
    
    await asyncio.gather(consumer_task, producer_task)
    await queue.close()
    
    stats = queue.get_stats()
    print(f"\n📊 快速通道命中: {stats.fast_path_hits} / {stats.total_enqueued}")


async def demo_streaming_queue():
    """演示流式消息队列"""
    print("\n" + "=" * 70)
    print("🧪 演示 3: 流式消息队列")
    print("=" * 70)
    
    queue = StreamingMessageQueue(max_size=100, name="streaming")
    
    # 模拟 Agent 输出
    async def agent_simulator():
        await queue.send_status("开始执行任务")
        await asyncio.sleep(0.1)
        
        await queue.send_text("正在分析任务...")
        await asyncio.sleep(0.2)
        
        await queue.send_tool_call("write_file", {"path": "test.py", "content": "..."})
        await asyncio.sleep(0.1)
        
        await queue.send_text("代码已生成")
        await asyncio.sleep(0.1)
        
        await queue.enqueue(
            StreamingMessageQueue.Message(
                type=StreamingMessageQueue.MessageType.COMPLETE,
                content="任务完成"
            )
        )
        await queue.close()
    
    # 消费者
    async def ui_consumer():
        async for msg in queue:
            if msg is None:
                break
            
            if msg.type == StreamingMessageQueue.MessageType.TEXT:
                print(f"💭 思考: {msg.content}")
            elif msg.type == StreamingMessageQueue.MessageType.TOOL_CALL:
                print(f"🔧 工具: {msg.content['name']}")
            elif msg.type == StreamingMessageQueue.MessageType.STATUS:
                print(f"📍 状态: {msg.content}")
            elif msg.type == StreamingMessageQueue.MessageType.COMPLETE:
                print(f"✅ 完成: {msg.content}")
                break
    
    await asyncio.gather(agent_simulator(), ui_consumer())
    
    stats = queue.get_stats()
    print(f"\n📊 消息数: {stats.total_enqueued}")


async def demo_streaming_agent():
    """演示流式 Agent 执行"""
    print("\n" + "=" * 70)
    print("🧪 演示 4: 流式 Agent（需要 API Key）")
    print("=" * 70)
    
    try:
        # 创建流式执行器
        llm = LLMFactory.create("openai")
        tools = create_default_registry()
        executor = StreamingExecutor(llm, tools, max_iterations=3)
        
        task = "写一个简单的 Hello World Python 脚本"
        print(f"任务: {task}\n")
        print("-" * 70)
        
        # 流式执行
        async for event in executor.execute_stream(task):
            if event.type == "thinking":
                print(f"💭 {event.content[:80]}...")
            elif event.type == "tool_call":
                tool_name = event.metadata.get("tool_name", "unknown")
                print(f"🔧 调用工具: {tool_name}")
            elif event.type == "tool_result":
                success = event.metadata.get("success", False)
                status = "✅" if success else "❌"
                print(f"{status} 工具结果: {event.content[:50]}...")
            elif event.type == "status":
                print(f"📍 {event.content}")
            elif event.type == "complete":
                print(f"✅ 完成: {event.content}")
            elif event.type == "error":
                print(f"❌ 错误: {event.content}")
        
        print("-" * 70)
        print("流式执行完成！")
    
    except Exception as e:
        print(f"❌ 演示失败（可能缺少 API Key）: {e}")


async def main():
    """运行所有演示"""
    print("\n🚀 h2A 双缓冲异步队列演示\n")
    
    # 基础演示（不需要 API）
    await demo_basic_h2a()
    await demo_fast_path()
    await demo_streaming_queue()
    
    # Agent 演示（需要 API）
    print("\n是否要运行流式 Agent 演示？（需要有效的 API Key）")
    print("按 Enter 继续，Ctrl+C 跳过")
    
    try:
        input()
        await demo_streaming_agent()
    except KeyboardInterrupt:
        print("\n⏭️  跳过 Agent 演示")
    
    print("\n" + "=" * 70)
    print("🎉 所有演示完成")
    print("=" * 70)
    
    print("\n💡 关键特性:")
    print("  1. 快速通道: 消费者等待时，消息零延迟直达")
    print("  2. 双缓冲: 减少锁竞争，提升吞吐")
    print("  3. 流式输出: 实时反馈，提升用户体验")
    print("  4. 可打断: 支持 steering，随时改变方向")


if __name__ == "__main__":
    asyncio.run(main())
