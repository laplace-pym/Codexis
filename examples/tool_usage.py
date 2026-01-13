#!/usr/bin/env python3
"""
Example: Direct tool usage without LLM.

This example demonstrates how to use the tool system directly,
which is useful for:
- Testing tools before using them with the agent
- Building custom automation scripts
- Understanding tool capabilities

Usage:
    python examples/tool_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools import create_default_registry
from utils.logger import get_logger


def demo_file_tools(registry, logger):
    """Demonstrate file operation tools."""
    logger.separator("File Tools Demo")
    
    # List current directory
    logger.info("Listing current directory...")
    result = registry.execute("list_directory", path=".")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    
    # Read a file
    logger.info("\nReading README.md...")
    result = registry.execute("read_file", path="./README.md", start_line=1, end_line=20)
    if result.success:
        print(result.output)
    else:
        print(f"Error: {result.error}")
    
    # Write and read a test file
    logger.info("\nWriting test file...")
    test_content = """# Test File
This is a test file created by the tool demo.

def hello():
    print("Hello, World!")
"""
    result = registry.execute(
        "write_file", 
        path="./workspace/test_demo.py", 
        content=test_content
    )
    print(result.output)


def demo_search_tools(registry, logger):
    """Demonstrate search tools."""
    logger.separator("Search Tools Demo")
    
    # Grep search
    logger.info("Searching for 'def.*execute' in Python files...")
    result = registry.execute(
        "grep",
        pattern=r"def\s+execute",
        path="./tools",
        file_pattern="*.py",
        max_results=10
    )
    print(result.output)
    
    # Find symbol
    logger.info("\nFinding 'CodingAgent' symbol...")
    result = registry.execute(
        "find_symbol",
        symbol="CodingAgent",
        path="./agent",
        symbol_type="class"
    )
    print(result.output)


def demo_code_execution(registry, logger):
    """Demonstrate code execution tools."""
    logger.separator("Code Execution Demo")
    
    # Execute Python code
    logger.info("Executing Python code in sandbox...")
    code = """
import math

def calculate_primes(n):
    '''Find first n prime numbers.'''
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % p != 0 for p in primes):
            primes.append(num)
        num += 1
    return primes

primes = calculate_primes(10)
print(f"First 10 primes: {primes}")
print(f"Sum: {sum(primes)}")
"""
    
    result = registry.execute("execute_python", code=code)
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")


def demo_analysis_tools(registry, logger):
    """Demonstrate code analysis tools."""
    logger.separator("Analysis Tools Demo")
    
    # Analyze code structure
    logger.info("Analyzing agent/coding_agent.py...")
    result = registry.execute("analyze_code", path="./agent/coding_agent.py")
    print(result.output)
    
    # Summarize file
    logger.info("\nSummarizing README.md...")
    result = registry.execute("summarize", path="./README.md", max_length=300)
    print(result.output)


def demo_patch_tools(registry, logger):
    """Demonstrate code modification tools."""
    logger.separator("Patch Tools Demo")
    
    # Create a test file
    original_code = '''def greet(name):
    """Greet someone."""
    print(f"Hello, {name}!")

def main():
    greet("World")

if __name__ == "__main__":
    main()
'''
    
    registry.execute("write_file", path="./workspace/greet.py", content=original_code)
    logger.info("Created workspace/greet.py")
    
    # Edit a block
    logger.info("\nEditing code block...")
    result = registry.execute(
        "edit_block",
        path="./workspace/greet.py",
        old_content='    print(f"Hello, {name}!")',
        new_content='    message = f"Hello, {name}! Welcome!"\n    print(message)\n    return message'
    )
    print(result.output)
    
    # Show the updated file
    logger.info("\nUpdated file:")
    result = registry.execute("read_file", path="./workspace/greet.py")
    print(result.output)
    
    # Create diff
    logger.info("\nCreating diff between versions...")
    modified_code = '''def greet(name):
    """Greet someone with a nice message."""
    message = f"Hello, {name}! Welcome!"
    print(message)
    return message

def main():
    result = greet("World")
    print(f"Returned: {result}")

if __name__ == "__main__":
    main()
'''
    
    result = registry.execute(
        "create_diff",
        original=original_code,
        modified=modified_code,
        filename="greet.py"
    )
    print(result.output)


def demo_test_generation(registry, logger):
    """Demonstrate test generation."""
    logger.separator("Test Generation Demo")
    
    # Create a module to test
    module_code = '''"""Calculator module."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    registry.execute("write_file", path="./workspace/calculator.py", content=module_code)
    logger.info("Created workspace/calculator.py")
    
    # Generate tests
    logger.info("\nGenerating tests...")
    result = registry.execute(
        "generate_tests",
        path="./workspace/calculator.py",
        test_style="pytest"
    )
    print(result.output)


def main():
    """Run all tool demos."""
    logger = get_logger("INFO")
    logger.separator("🔧 Tool Usage Examples")
    
    # Create default registry with all tools
    registry = create_default_registry()
    
    logger.info(f"Registered tools: {', '.join(registry.list_tools())}")
    
    # Ensure workspace exists
    Path("./workspace").mkdir(exist_ok=True)
    
    # Run demos
    demos = [
        ("File Tools", demo_file_tools),
        ("Search Tools", demo_search_tools),
        ("Code Execution", demo_code_execution),
        ("Analysis Tools", demo_analysis_tools),
        ("Patch Tools", demo_patch_tools),
        ("Test Generation", demo_test_generation),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func(registry, logger)
        except Exception as e:
            logger.error(f"{name} failed: {e}")
        print("\n")
    
    logger.separator("Demo Complete")


if __name__ == "__main__":
    main()
