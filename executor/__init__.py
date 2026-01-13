"""
Executor Module - Code execution and sandbox environment.

Provides safe, isolated code execution with:
- Timeout protection
- Output capture (stdout, stderr, exit code)
- Multi-language support (Python, JavaScript, Bash)
- Automatic cleanup
"""

from .sandbox import Sandbox, SandboxConfig, ExecutionResult, Language, execute_python_safely

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "ExecutionResult",
    "Language",
    "execute_python_safely",
]
