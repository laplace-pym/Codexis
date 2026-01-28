#!/usr/bin/env python3
"""
测试优化效果的脚本

验证：
1. 简单任务能被快速识别并加速执行
2. 代码验证成功后能提前退出
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import CodingAgent, TaskAnalyzer
from utils.logger import get_logger

def test_task_analyzer():
    """测试任务复杂度分析器"""
    print("=" * 60)
    print("🧪 测试任务复杂度分析器")
    print("=" * 60)
    
    analyzer = TaskAnalyzer()
    
    test_cases = [
        # (任务描述, 预期是否简单)
        ("写一个九九乘法表", True),
        ("打印 Hello World", True),
        ("计算两个数的最大公约数", True),
        ("开发一个用户管理系统，包含前端和后端，支持数据库存储", False),
        ("创建一个 Web 爬虫框架，支持分布式爬取和数据清洗", False),
        ("实现斐波那契数列函数", True),
        ("搭建微服务架构，包含 API 网关、服务发现、负载均衡", False),
    ]
    
    for task, expected_simple in test_cases:
        result = analyzer.analyze(task)
        status = "✅" if result.is_simple == expected_simple else "❌"
        print(f"\n{status} 任务: {task}")
        print(f"   判断: {'简单' if result.is_simple else '复杂'} (置信度: {result.confidence:.0%})")
        print(f"   理由: {result.reason}")
        print(f"   建议迭代次数: {result.suggested_max_iterations}")


def test_simple_task():
    """测试简单任务的快速执行"""
    print("\n" + "=" * 60)
    print("🧪 测试简单任务快速执行")
    print("=" * 60)
    
    logger = get_logger("INFO")
    
    try:
        agent = CodingAgent(
            provider="openai",
            max_iterations=10,
            auto_fix=False,
            logger=logger,
        )
        
        # 简单任务：应该被识别并快速完成
        task = "写一个 Python 函数，计算 1 到 100 的和"
        
        print(f"\n任务: {task}")
        print("-" * 60)
        
        result = agent.run(task)
        
        print("\n" + "=" * 60)
        print("✅ 测试完成")
        print("=" * 60)
        
        if agent.state:
            print(f"实际迭代次数: {agent.state.iteration}")
            print(f"是否完成: {agent.state.is_complete}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n🚀 开始测试优化功能\n")
    
    # 测试 1: 任务分析器
    test_task_analyzer()
    
    # 测试 2: 简单任务快速执行（需要 API）
    print("\n\n是否要测试实际的 Agent 执行？（需要有效的 API Key）")
    response = input("输入 'y' 继续，其他键跳过: ").strip().lower()
    
    if response == 'y':
        test_simple_task()
    else:
        print("\n⏭️  跳过 Agent 执行测试")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
