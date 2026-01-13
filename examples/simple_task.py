#!/usr/bin/env python3
"""
Example: Simple task execution with CodingAgent.

This example shows how to use the CodingAgent to execute a simple
coding task with automatic error fixing.

Usage:
    python examples/simple_task.py
    
Requirements:
    - Valid API key in .env file
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import CodingAgent
from utils.logger import get_logger


def main():
    """Run a simple task with the CodingAgent."""
    # Initialize logger
    logger = get_logger("INFO")
    logger.separator("Simple Task Example")
    
    # Create agent
    # Note: Requires valid API key in .env
    try:
        agent = CodingAgent(
            provider="deepseek",  # or "openai", "anthropic"
            max_iterations=10,
            auto_fix=True,
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        logger.info("Make sure you have a valid API key in .env file")
        return
    
    # Task to execute
    task = """
    Create a Python function called 'fibonacci' that:
    1. Takes a number n as input
    2. Returns a list of the first n Fibonacci numbers
    3. Include a docstring and type hints
    4. Test the function with n=10
    """
    
    logger.info(f"Task: {task.strip()}")
    logger.separator()
    
    # Run with automatic error fixing
    result = agent.run_with_auto_fix(task)
    
    logger.separator("Result")
    print(result)


if __name__ == "__main__":
    main()
