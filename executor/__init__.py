"""
Executor Module - Code execution and sandbox environment.

Provides safe, isolated code execution with:
- Timeout protection
- Output capture (stdout, stderr, exit code)
- Multi-language support (Python, JavaScript, Bash)
- Automatic cleanup
- Sandbox pooling for improved performance
"""

from .sandbox import Sandbox, SandboxConfig, ExecutionResult, Language, execute_python_safely
from .sandbox_pool import (
    SandboxPool,
    get_sandbox_pool,
    execute_with_pool,
    execute_with_pool_sync,
)

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "ExecutionResult",
    "Language",
    "execute_python_safely",
    # Sandbox pool
    "SandboxPool",
    "get_sandbox_pool",
    "execute_with_pool",
    "execute_with_pool_sync",
]
