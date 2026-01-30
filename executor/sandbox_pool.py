"""
Sandbox Pool - Pre-warmed sandbox environments for faster code execution.
"""

import asyncio
import threading
from typing import Optional
from pathlib import Path

from .sandbox import Sandbox, SandboxConfig, ExecutionResult, Language


class SandboxPool:
    """
    Pool of pre-warmed sandbox environments for faster code execution.

    Features:
    - Pre-creates sandbox directories for immediate use
    - Thread-safe sandbox acquisition and release
    - Automatic cleanup and reset between uses
    - Configurable pool size

    Usage:
        pool = SandboxPool(pool_size=3)
        pool.warmup()

        # Acquire a sandbox
        sandbox = await pool.acquire()

        # Use the sandbox
        result = sandbox.execute_code("print('Hello')")

        # Release back to pool
        await pool.release(sandbox)

        # Or use async context manager
        async with pool.sandbox() as sandbox:
            result = sandbox.execute_code("print('Hello')")
    """

    def __init__(
        self,
        pool_size: int = 3,
        config: Optional[SandboxConfig] = None,
    ):
        """
        Initialize the sandbox pool.

        Args:
            pool_size: Number of sandboxes to pre-create
            config: Configuration for sandbox instances
        """
        self.pool_size = pool_size
        self.config = config or SandboxConfig()
        self._available: asyncio.Queue = asyncio.Queue()
        self._lock = threading.Lock()
        self._initialized = False
        self._total_created = 0

    def warmup(self) -> None:
        """Pre-create sandbox environments."""
        with self._lock:
            if self._initialized:
                return

            for _ in range(self.pool_size):
                sandbox = self._create_sandbox()
                self._available.put_nowait(sandbox)

            self._initialized = True

    def warmup_sync(self) -> None:
        """Synchronous warmup for non-async contexts."""
        self.warmup()

    def _create_sandbox(self) -> Sandbox:
        """Create and initialize a new sandbox."""
        sandbox = Sandbox(self.config)
        sandbox._create_sandbox()  # Pre-create the directory
        self._total_created += 1
        return sandbox

    async def acquire(self, timeout: float = 30.0) -> Sandbox:
        """
        Acquire a sandbox from the pool.

        Args:
            timeout: Maximum time to wait for an available sandbox

        Returns:
            A ready-to-use Sandbox instance

        Raises:
            asyncio.TimeoutError: If no sandbox available within timeout
        """
        if not self._initialized:
            self.warmup()

        try:
            sandbox = await asyncio.wait_for(
                self._available.get(),
                timeout=timeout
            )
            return sandbox
        except asyncio.TimeoutError:
            # Pool exhausted, create a new sandbox
            return self._create_sandbox()

    def acquire_sync(self, timeout: float = 30.0) -> Sandbox:
        """
        Synchronously acquire a sandbox from the pool.

        Args:
            timeout: Maximum time to wait

        Returns:
            A ready-to-use Sandbox instance
        """
        if not self._initialized:
            self.warmup()

        try:
            sandbox = self._available.get_nowait()
            return sandbox
        except asyncio.QueueEmpty:
            # Pool exhausted, create a new sandbox
            return self._create_sandbox()

    async def release(self, sandbox: Sandbox) -> None:
        """
        Release a sandbox back to the pool.

        The sandbox is reset (cleaned) before being returned to the pool.

        Args:
            sandbox: The sandbox to release
        """
        # Reset the sandbox for reuse
        self._reset_sandbox(sandbox)

        # Return to pool if not at capacity
        if self._available.qsize() < self.pool_size:
            await self._available.put(sandbox)
        else:
            # Pool at capacity, cleanup this sandbox
            sandbox._cleanup()

    def release_sync(self, sandbox: Sandbox) -> None:
        """
        Synchronously release a sandbox back to the pool.

        Args:
            sandbox: The sandbox to release
        """
        # Reset the sandbox for reuse
        self._reset_sandbox(sandbox)

        # Return to pool if not at capacity
        if self._available.qsize() < self.pool_size:
            self._available.put_nowait(sandbox)
        else:
            # Pool at capacity, cleanup this sandbox
            sandbox._cleanup()

    def _reset_sandbox(self, sandbox: Sandbox) -> None:
        """
        Reset a sandbox for reuse.

        Clears all files but keeps the directory.
        """
        if sandbox._sandbox_dir and sandbox._sandbox_dir.exists():
            # Clear all files in the sandbox directory
            for item in sandbox._sandbox_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                except Exception:
                    pass

    async def sandbox(self):
        """
        Async context manager for sandbox acquisition.

        Usage:
            async with pool.sandbox() as sb:
                result = sb.execute_code("print('Hello')")
        """
        return _SandboxContextManager(self)

    def shutdown(self) -> None:
        """Cleanup all sandboxes in the pool."""
        with self._lock:
            while not self._available.empty():
                try:
                    sandbox = self._available.get_nowait()
                    sandbox._cleanup()
                except asyncio.QueueEmpty:
                    break

            self._initialized = False

    @property
    def available_count(self) -> int:
        """Number of sandboxes currently available."""
        return self._available.qsize()

    @property
    def total_created(self) -> int:
        """Total number of sandboxes created (including overflow)."""
        return self._total_created


class _SandboxContextManager:
    """Async context manager for sandbox pool."""

    def __init__(self, pool: SandboxPool):
        self.pool = pool
        self.sandbox: Optional[Sandbox] = None

    async def __aenter__(self) -> Sandbox:
        self.sandbox = await self.pool.acquire()
        return self.sandbox

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.sandbox:
            await self.pool.release(self.sandbox)
            self.sandbox = None


# Global pool instance (lazy initialization)
_global_pool: Optional[SandboxPool] = None


def get_sandbox_pool(pool_size: int = 3) -> SandboxPool:
    """
    Get or create the global sandbox pool.

    Args:
        pool_size: Pool size (only used on first call)

    Returns:
        The global SandboxPool instance
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = SandboxPool(pool_size=pool_size)
        _global_pool.warmup()
    return _global_pool


async def execute_with_pool(
    code: str,
    language: Language = Language.PYTHON,
    timeout: Optional[int] = None,
) -> ExecutionResult:
    """
    Execute code using a sandbox from the global pool.

    Args:
        code: Code to execute
        language: Programming language
        timeout: Execution timeout

    Returns:
        ExecutionResult with output and status
    """
    pool = get_sandbox_pool()
    sandbox = await pool.acquire()

    try:
        return sandbox.execute_code(code, language=language, timeout=timeout)
    finally:
        await pool.release(sandbox)


def execute_with_pool_sync(
    code: str,
    language: Language = Language.PYTHON,
    timeout: Optional[int] = None,
) -> ExecutionResult:
    """
    Synchronously execute code using a sandbox from the global pool.

    Args:
        code: Code to execute
        language: Programming language
        timeout: Execution timeout

    Returns:
        ExecutionResult with output and status
    """
    pool = get_sandbox_pool()
    sandbox = pool.acquire_sync()

    try:
        return sandbox.execute_code(code, language=language, timeout=timeout)
    finally:
        pool.release_sync(sandbox)
