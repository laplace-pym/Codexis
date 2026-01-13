#!/usr/bin/env python3
"""
Example: Full agent workflow with planning and auto-fix.

This example demonstrates the complete agent capabilities:
1. Task planning and breakdown
2. Code generation
3. Tool usage
4. Code execution in sandbox
5. Error analysis and automatic fixing

Usage:
    python examples/full_workflow.py

Requirements:
    - Valid API key in .env file
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import CodingAgent, ErrorAnalyzer
from executor import Sandbox, SandboxConfig, Language
from utils.logger import get_logger
from utils.config import Config


def demo_error_analysis(logger):
    """Demonstrate error analysis capabilities."""
    logger.separator("Error Analysis Demo")
    
    analyzer = ErrorAnalyzer()
    
    # Example errors to analyze
    errors = [
        ("NameError: name 'undefined_var' is not defined", "x = undefined_var + 1"),
        ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", "result = 5 + 'hello'"),
        ("IndexError: list index out of range", "items = [1, 2, 3]\nprint(items[5])"),
        ("KeyError: 'missing_key'", "d = {'a': 1}\nprint(d['missing_key'])"),
        ("ModuleNotFoundError: No module named 'nonexistent'", "import nonexistent"),
    ]
    
    for error, code in errors:
        logger.info(f"\nAnalyzing: {error[:50]}...")
        analysis = analyzer.analyze(error, code)
        
        print(f"  Type: {analysis.error_type.value}")
        print(f"  Fixable: {analysis.fixable}")
        print(f"  Suggestion: {analysis.suggestion[:80]}...")


def demo_sandbox_execution(logger):
    """Demonstrate sandbox code execution."""
    logger.separator("Sandbox Execution Demo")
    
    config = SandboxConfig(timeout=10)
    
    # Test cases: (description, code, expected_success)
    test_cases = [
        (
            "Simple calculation",
            """
result = sum(range(1, 101))
print(f"Sum of 1-100: {result}")
""",
            True
        ),
        (
            "File operations in sandbox",
            """
# Create a file in sandbox
with open("test.txt", "w") as f:
    f.write("Hello from sandbox!")

# Read it back
with open("test.txt", "r") as f:
    content = f.read()
    
print(f"File content: {content}")
""",
            True
        ),
        (
            "Error handling",
            """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"

print(safe_divide(10, 2))
print(safe_divide(10, 0))
""",
            True
        ),
        (
            "Intentional error for testing",
            """
# This will fail
undefined_function()
""",
            False
        ),
    ]
    
    with Sandbox(config) as sandbox:
        for desc, code, expected in test_cases:
            logger.info(f"\nRunning: {desc}")
            result = sandbox.execute_code(code, Language.PYTHON)
            
            status = "✅" if result.success == expected else "❌"
            print(f"  {status} Success: {result.success} (expected: {expected})")
            
            if result.stdout:
                print(f"  Output: {result.stdout.strip()[:100]}")
            if result.stderr and not result.success:
                print(f"  Error: {result.stderr.strip()[:100]}")


def demo_full_agent(logger):
    """Demonstrate full agent capabilities."""
    logger.separator("Full Agent Demo")
    
    try:
        # Create agent with planning enabled
        agent = CodingAgent(
            max_iterations=10,
            max_fix_attempts=3,
            auto_fix=True,
        )
    except Exception as e:
        logger.error(f"Could not create agent: {e}")
        logger.info("This demo requires a valid API key in .env")
        return
    
    # Complex task that requires planning
    task = """
    Create a Python module for a simple task manager:
    
    1. Create a Task class with:
       - title (str)
       - description (str) 
       - priority (int, 1-5)
       - completed (bool)
       
    2. Create a TaskManager class with:
       - add_task(task): Add a new task
       - complete_task(title): Mark task as completed
       - get_pending_tasks(): Get all incomplete tasks
       - get_tasks_by_priority(priority): Filter by priority
       
    3. Add a main() function that:
       - Creates a TaskManager
       - Adds 3 sample tasks
       - Completes one task
       - Prints pending tasks
       
    4. Execute and verify the output
    """
    
    logger.info("Running complex task with planning...")
    logger.info(f"Task: {task[:100]}...")
    
    # Run with plan first
    result = agent.run(task, plan_first=True)
    
    logger.separator("Result")
    print(result)
    
    # Show execution history
    if agent.state and agent.state.history:
        logger.separator("Execution History")
        for i, step in enumerate(agent.state.history.steps, 1):
            status = "✓" if step.status.value == "success" else "✗"
            print(f"  {i}. [{status}] {step.step_type.value}: {step.content[:60]}...")


def demo_code_generation_and_fix(logger):
    """Demonstrate code generation with error fixing."""
    logger.separator("Code Generation + Auto-Fix Demo")
    
    try:
        agent = CodingAgent(
            max_iterations=5,
            max_fix_attempts=3,
            auto_fix=True,
        )
    except Exception as e:
        logger.error(f"Could not create agent: {e}")
        return
    
    # Task that might generate code with errors initially
    task = """
    Write a Python function called 'parse_csv_data' that:
    1. Takes a CSV string as input (not a file)
    2. Parses it and returns a list of dictionaries
    3. First row should be headers
    4. Handle edge cases (empty string, single row)
    5. Include type hints and docstring
    6. Test with sample data
    """
    
    logger.info("Running task with auto-fix enabled...")
    result = agent.run_with_auto_fix(task)
    
    logger.separator("Result")
    print(result)


def main():
    """Run all full workflow demos."""
    logger = get_logger("INFO")
    logger.separator("🚀 Full Workflow Examples")
    
    # Load config to check if API key is available
    try:
        config = Config.load()
        has_api_key = config.deepseek.is_valid() or config.openai.is_valid() or config.anthropic.is_valid()
    except:
        has_api_key = False
    
    # Demos that don't require API
    demo_error_analysis(logger)
    demo_sandbox_execution(logger)
    
    # Demos that require API
    if has_api_key:
        logger.info("\nAPI key found. Running full agent demos...")
        demo_full_agent(logger)
        demo_code_generation_and_fix(logger)
    else:
        logger.warning("\nNo API key found. Skipping full agent demos.")
        logger.info("To run full demos, add your API key to .env file.")
    
    logger.separator("Demo Complete")


if __name__ == "__main__":
    main()
