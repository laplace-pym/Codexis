"""
Sandbox - Isolated execution environment for running code safely.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    timeout: int = 30
    max_memory_mb: int = 512
    max_output_size: int = 1024 * 1024  # 1MB
    allow_network: bool = False
    working_dir: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    error: Optional[str] = None
    files_created: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.exit_code == 0 and not self.timed_out and self.error is None


class Sandbox:
    """
    Isolated sandbox environment for safe code execution.
    
    Features:
    - Isolated file system (temporary directory)
    - Configurable timeout
    - Output capture (stdout, stderr)
    - Support for multiple languages
    - Automatic cleanup
    
    Usage:
        sandbox = Sandbox(SandboxConfig(timeout=30))
        
        # Execute code
        result = sandbox.execute_code(
            code="print('Hello, World!')",
            language=Language.PYTHON
        )
        
        # Or run with files
        result = sandbox.execute_with_files(
            main_code="import helper; helper.run()",
            files={"helper.py": "def run(): print('Helper!')"},
            language=Language.PYTHON
        )
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._sandbox_dir: Optional[Path] = None
        
    def __enter__(self) -> "Sandbox":
        """Create sandbox directory on context enter."""
        self._create_sandbox()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup sandbox on context exit."""
        self._cleanup()
    
    def _create_sandbox(self) -> Path:
        """Create isolated sandbox directory."""
        if self._sandbox_dir is None:
            self._sandbox_dir = Path(tempfile.mkdtemp(prefix="fakeclaude_sandbox_"))
        return self._sandbox_dir
    
    def _cleanup(self):
        """Remove sandbox directory and all contents."""
        if self._sandbox_dir and self._sandbox_dir.exists():
            try:
                shutil.rmtree(self._sandbox_dir)
            except Exception:
                pass
            self._sandbox_dir = None
    
    def _get_interpreter(self, language: Language) -> list[str]:
        """Get the interpreter command for a language."""
        interpreters = {
            Language.PYTHON: [sys.executable],
            Language.JAVASCRIPT: ["node"],
            Language.BASH: ["bash"],
        }
        return interpreters.get(language, [])
    
    def _get_file_extension(self, language: Language) -> str:
        """Get file extension for a language."""
        extensions = {
            Language.PYTHON: ".py",
            Language.JAVASCRIPT: ".js",
            Language.BASH: ".sh",
        }
        return extensions.get(language, ".txt")
    
    def execute_code(
        self,
        code: str,
        language: Language = Language.PYTHON,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox.
        
        Args:
            code: Source code to execute
            language: Programming language
            timeout: Override default timeout
            
        Returns:
            ExecutionResult with output and status
        """
        return self.execute_with_files(
            main_code=code,
            files={},
            language=language,
            timeout=timeout,
        )
    
    def execute_with_files(
        self,
        main_code: str,
        files: dict[str, str],
        language: Language = Language.PYTHON,
        timeout: Optional[int] = None,
        main_filename: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code with additional supporting files.
        
        Args:
            main_code: Main source code to execute
            files: Additional files (filename -> content)
            language: Programming language
            timeout: Override default timeout
            main_filename: Custom filename for main code
            
        Returns:
            ExecutionResult with output and status
        """
        sandbox_dir = self._create_sandbox()
        timeout = timeout or self.config.timeout
        
        try:
            # Write additional files
            for filename, content in files.items():
                file_path = sandbox_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
            
            # Write main file
            ext = self._get_file_extension(language)
            main_file = main_filename or f"main{ext}"
            main_path = sandbox_dir / main_file
            main_path.write_text(main_code, encoding="utf-8")
            
            # Get interpreter
            interpreter = self._get_interpreter(language)
            if not interpreter:
                return ExecutionResult(
                    stdout="",
                    stderr="",
                    exit_code=1,
                    error=f"Unsupported language: {language.value}"
                )
            
            # Build command
            cmd = interpreter + [str(main_path)]
            
            # Set up environment
            env = os.environ.copy()
            env["HOME"] = str(sandbox_dir)
            env["TMPDIR"] = str(sandbox_dir)
            env["PYTHONPATH"] = str(sandbox_dir)
            
            # Execute
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=sandbox_dir,
                    env=env,
                )
                
                # List files created during execution
                files_created = []
                for item in sandbox_dir.rglob("*"):
                    if item.is_file() and item.name != main_file:
                        files_created.append(str(item.relative_to(sandbox_dir)))
                
                return ExecutionResult(
                    stdout=self._truncate_output(result.stdout),
                    stderr=self._truncate_output(result.stderr),
                    exit_code=result.returncode,
                    files_created=files_created,
                )
                
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    stdout="",
                    stderr="",
                    exit_code=124,
                    timed_out=True,
                    error=f"Execution timed out after {timeout} seconds"
                )
            except FileNotFoundError as e:
                return ExecutionResult(
                    stdout="",
                    stderr="",
                    exit_code=127,
                    error=f"Interpreter not found: {interpreter[0]}. Error: {str(e)}"
                )
                
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                error=f"Sandbox execution error: {str(e)}"
            )
    
    def _truncate_output(self, output: str) -> str:
        """Truncate output if it exceeds maximum size."""
        max_size = self.config.max_output_size
        if len(output) > max_size:
            return output[:max_size] + f"\n... (truncated, exceeded {max_size} bytes)"
        return output
    
    def read_file(self, filename: str) -> Optional[str]:
        """Read a file from the sandbox directory."""
        if self._sandbox_dir is None:
            return None
        
        file_path = self._sandbox_dir / filename
        if file_path.exists() and file_path.is_file():
            return file_path.read_text(encoding="utf-8")
        return None
    
    def list_files(self) -> list[str]:
        """List all files in the sandbox directory."""
        if self._sandbox_dir is None:
            return []
        
        files = []
        for item in self._sandbox_dir.rglob("*"):
            if item.is_file():
                files.append(str(item.relative_to(self._sandbox_dir)))
        return files
    
    def write_file(self, filename: str, content: str) -> bool:
        """Write a file to the sandbox directory."""
        if self._sandbox_dir is None:
            self._create_sandbox()
        
        try:
            file_path = self._sandbox_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False
    
    def get_sandbox_path(self) -> Optional[Path]:
        """Get the sandbox directory path."""
        return self._sandbox_dir


def execute_python_safely(
    code: str,
    timeout: int = 30,
    files: Optional[dict[str, str]] = None,
) -> ExecutionResult:
    """
    Convenience function to execute Python code safely.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        files: Additional files to create
        
    Returns:
        ExecutionResult with output and status
    """
    config = SandboxConfig(timeout=timeout)
    
    with Sandbox(config) as sandbox:
        return sandbox.execute_with_files(
            main_code=code,
            files=files or {},
            language=Language.PYTHON,
        )
