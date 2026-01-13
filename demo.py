#!/usr/bin/env python3
"""
FakeClaude Code - Demo Script

This demo shows how to use the Coding Agent programmatically.
It demonstrates:
1. Basic code generation and execution
2. Tool usage (file operations)
3. Auto-fix loop for error recovery
4. Custom configuration
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config, LLMConfig
from utils.logger import get_logger
from llm.factory import LLMFactory
from tools.registry import create_default_registry
from executor.sandbox import Sandbox, SandboxConfig, Language
from agent import CodingAgent


def demo_sandbox_execution():
    """Demo: Execute code in sandbox."""
    print("\n" + "="*60)
    print("Demo 1: Sandbox Code Execution")
    print("="*60)
    
    config = SandboxConfig(timeout=10)
    
    with Sandbox(config) as sandbox:
        # Simple Python code
        code = """
import math

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate first 10 fibonacci numbers
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")

print(f"\\nPi = {math.pi}")
"""
        result = sandbox.execute_code(code, Language.PYTHON)
        
        print(f"\nCode:\n{code}")
        print(f"\nOutput:\n{result.stdout}")
        print(f"Exit code: {result.exit_code}")
        print(f"Success: {result.success}")


def demo_tools():
    """Demo: Use tools directly."""
    print("\n" + "="*60)
    print("Demo 2: Tool Usage")
    print("="*60)
    
    registry = create_default_registry()
    
    # List available tools
    print("\nAvailable tools:")
    for tool_name in registry.list_tools():
        tool = registry.get(tool_name)
        print(f"  - {tool_name}: {tool.description[:60]}...")
    
    # Demo: Execute Python code
    print("\n--- Execute Python Tool ---")
    result = registry.execute("execute_python", code="print('Hello from tool!')")
    print(f"Result: {result.output}")
    
    # Demo: List directory
    print("\n--- List Directory Tool ---")
    result = registry.execute("list_directory", path=".")
    print(f"Result:\n{result.output[:500]}...")


def demo_llm_adapter():
    """Demo: LLM Adapter (requires API key)."""
    print("\n" + "="*60)
    print("Demo 3: LLM Adapter")
    print("="*60)
    
    try:
        config = Config.load()
        llm = LLMFactory.create()
        
        print(f"\nUsing: {llm}")
        print("\nNote: This demo requires a valid API key in .env")
        print("Skipping actual API call to avoid charges.")
        
        # Uncomment to test actual API call:
        # from llm.base import Message
        # response = llm.chat_sync([
        #     Message.user("Say 'Hello, World!' and nothing else.")
        # ])
        # print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"LLM not configured: {e}")
        print("Create a .env file with your API key to test this demo.")


def demo_agent_simple():
    """Demo: Simple agent usage (no API call)."""
    print("\n" + "="*60)
    print("Demo 4: Agent Components")
    print("="*60)
    
    from agent.base import AgentState, AgentHistory, StepType
    
    # Create state
    state = AgentState(max_iterations=10)
    state.start_task("Create a hello world script")
    
    # Simulate steps
    state.history.add_think("Analyzing the task...")
    state.history.add_tool_call(
        tool_name="write_file",
        tool_args={"path": "hello.py", "content": "print('Hello!')"},
        description="Writing hello.py"
    )
    state.history.add_code_execution(
        code="print('Hello!')",
        stdout="Hello!\n",
        stderr="",
        exit_code=0,
    )
    
    print(f"\nState summary:")
    print(state.get_context_summary())
    
    print(f"\nHistory:")
    for step in state.history.steps:
        print(f"  - [{step.step_type.value}] {step.content}")


def demo_full_agent():
    """Demo: Full agent with actual LLM (requires API key)."""
    print("\n" + "="*60)
    print("Demo 5: Full Agent Demo")
    print("="*60)
    
    print("\nThis demo would run the full agent with LLM.")
    print("To test, ensure you have a valid API key in .env")
    print("\nExample usage:")
    print("""
    from agent import CodingAgent
    
    agent = CodingAgent(provider="deepseek")
    result = agent.run("Create a Python script that prints the current date")
    print(result)
    """)
    
    # Uncomment to run actual agent:
    # try:
    #     agent = CodingAgent()
    #     result = agent.run("Create a Python function that calculates factorial")
    #     print(f"\nResult: {result}")
    # except Exception as e:
    #     print(f"Error: {e}")


def demo_code_with_error_fix():
    """Demo: Show error detection and sandbox."""
    print("\n" + "="*60)
    print("Demo 6: Error Detection")
    print("="*60)
    
    config = SandboxConfig(timeout=5)
    
    with Sandbox(config) as sandbox:
        # Code with intentional error
        buggy_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)  # This will raise ZeroDivisionError
print(result)
"""
        print(f"Running buggy code:")
        print(buggy_code)
        
        result = sandbox.execute_code(buggy_code, Language.PYTHON)
        
        print(f"\nExecution result:")
        print(f"  Exit code: {result.exit_code}")
        print(f"  Success: {result.success}")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        
        # Now with fixed code
        fixed_code = """
def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b

result = divide(10, 0)
print(result)

result = divide(10, 2)
print(result)
"""
        print(f"\nRunning fixed code:")
        print(fixed_code)
        
        result = sandbox.execute_code(fixed_code, Language.PYTHON)
        
        print(f"\nExecution result:")
        print(f"  Exit code: {result.exit_code}")
        print(f"  Success: {result.success}")
        print(f"  Output: {result.stdout}")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# FakeClaude Code - Demo Suite")
    print("#"*60)
    
    # Run demos that don't require API keys
    demo_sandbox_execution()
    demo_tools()
    demo_agent_simple()
    demo_code_with_error_fix()
    
    # These require API keys
    demo_llm_adapter()
    demo_full_agent()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nTo use the full agent with LLM:")
    print("1. Copy env.example to .env")
    print("2. Add your API key (DeepSeek, OpenAI, or Anthropic)")
    print("3. Run: python main.py")
    print("="*60)


if __name__ == "__main__":
    main()
