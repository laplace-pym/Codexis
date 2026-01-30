"""
Code Executor Tools - Tools for executing code safely.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import asyncio
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


def estimate_timeout(code: str, base_timeout: int = 10) -> int:
    """
    Estimate appropriate timeout based on code complexity.

    Args:
        code: The code to analyze
        base_timeout: Base timeout in seconds

    Returns:
        Estimated timeout in seconds
    """
    timeout = base_timeout

    # Has network operations -> increase timeout
    network_patterns = [
        r'\brequests\b', r'\burllib\b', r'\bhttpx\b', r'\baiohttp\b',
        r'\bsocket\b', r'\.get\s*\(', r'\.post\s*\(',
        r'\bfetch\b', r'\baxios\b',
    ]
    if any(re.search(pattern, code) for pattern in network_patterns):
        timeout += 20

    # Has file I/O with large data -> increase timeout
    file_patterns = [
        r'\.read\s*\(', r'\.write\s*\(',
        r'open\s*\(', r'with\s+open',
        r'\.to_csv\b', r'\.read_csv\b',
        r'\.to_parquet\b', r'\.read_parquet\b',
    ]
    if any(re.search(pattern, code) for pattern in file_patterns):
        timeout += 10

    # Has loops with potentially large iterations
    loop_patterns = [
        r'for\s+\w+\s+in\s+range\s*\(\s*\d{4,}',  # range(1000+)
        r'while\s+True',
        r'for\s+.*\s+in\s+',
    ]
    if any(re.search(pattern, code) for pattern in loop_patterns):
        timeout += 15

    # Has ML/DL operations -> increase timeout significantly
    ml_patterns = [
        r'\btorch\b', r'\btensorflow\b', r'\bkeras\b',
        r'\.fit\s*\(', r'\.train\s*\(',
        r'\bmodel\b', r'\boptimizer\b',
        r'\.forward\s*\(',
    ]
    if any(re.search(pattern, code) for pattern in ml_patterns):
        timeout += 30

    # Has subprocess/shell execution
    subprocess_patterns = [
        r'\bsubprocess\b', r'\bos\.system\b',
        r'\bPopen\b', r'\.run\s*\(',
    ]
    if any(re.search(pattern, code) for pattern in subprocess_patterns):
        timeout += 10

    # Cap at 120 seconds maximum
    return min(timeout, 120)


class ExecuteCodeTool(BaseTool):
    """Tool for executing shell commands."""
    
    def __init__(self, timeout: int = 30, working_dir: Optional[str] = None):
        self.timeout = timeout
        self.working_dir = working_dir
    
    @property
    def name(self) -> str:
        return "execute_command"
    
    @property
    def description(self) -> str:
        return "Execute a shell command and return its output."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional, default: 30)"
                }
            },
            "required": ["command"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        command = kwargs.get("command")
        working_dir = kwargs.get("working_dir", self.working_dir)
        timeout = kwargs.get("timeout", self.timeout)
        
        try:
            # Resolve working directory
            if working_dir:
                cwd = Path(working_dir).expanduser().resolve()
                if not cwd.exists():
                    return ToolResult.error_result(f"Working directory not found: {working_dir}")
            else:
                cwd = None
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            
            exec_result = ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
            
            # Format output
            output_parts = []
            if exec_result.stdout:
                output_parts.append(f"stdout:\n{exec_result.stdout}")
            if exec_result.stderr:
                output_parts.append(f"stderr:\n{exec_result.stderr}")
            output_parts.append(f"exit_code: {exec_result.exit_code}")
            
            output = "\n".join(output_parts)
            
            return ToolResult(
                success=exec_result.exit_code == 0,
                output=output,
                error=exec_result.stderr if exec_result.exit_code != 0 else None,
                data=exec_result,
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult.error_result(
                f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult.error_result(f"Error executing command: {str(e)}")


class ExecutePythonTool(BaseTool):
    """Tool for executing Python code."""
    
    def __init__(self, timeout: int = 120, working_dir: Optional[str] = None):
        self.timeout = timeout
        self.working_dir = working_dir
    
    @property
    def name(self) -> str:
        return "execute_python"
    
    @property
    def description(self) -> str:
        return "Execute Python code and return its output. The code is written to a temporary file and executed."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for execution (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional, default: 30)"
                },
                "save_as": {
                    "type": "string",
                    "description": "Save the code to this file path (optional)"
                }
            },
            "required": ["code"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code")
        working_dir = kwargs.get("working_dir", self.working_dir)
        timeout = kwargs.get("timeout", self.timeout)
        save_as = kwargs.get("save_as")
        
        temp_file = None
        
        try:
            # Determine file path
            if save_as:
                file_path = Path(save_as).expanduser().resolve()
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                    encoding="utf-8"
                )
                temp_file.write(code)
                temp_file.close()
                file_path = Path(temp_file.name)
            
            # Resolve working directory
            if working_dir:
                cwd = Path(working_dir).expanduser().resolve()
                if not cwd.exists():
                    cwd.mkdir(parents=True, exist_ok=True)
            else:
                cwd = file_path.parent
            
            # Execute Python code
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            
            exec_result = ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
            
            # Format output
            output_parts = []
            if save_as:
                output_parts.append(f"Saved to: {file_path}")
            if exec_result.stdout:
                output_parts.append(f"stdout:\n{exec_result.stdout}")
            if exec_result.stderr:
                output_parts.append(f"stderr:\n{exec_result.stderr}")
            output_parts.append(f"exit_code: {exec_result.exit_code}")
            
            output = "\n".join(output_parts)
            
            return ToolResult(
                success=exec_result.exit_code == 0,
                output=output,
                error=exec_result.stderr if exec_result.exit_code != 0 else None,
                data=exec_result,
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult.error_result(
                f"Code execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult.error_result(f"Error executing Python code: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file and not save_as:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    async def execute_async(self, **kwargs) -> ToolResult:
        """
        Asynchronously execute Python code.

        Non-blocking execution that allows other async tasks to run.

        Args:
            Same as execute()

        Returns:
            ToolResult with execution output
        """
        code = kwargs.get("code")
        working_dir = kwargs.get("working_dir", self.working_dir)
        timeout = kwargs.get("timeout")
        save_as = kwargs.get("save_as")
        smart_timeout = kwargs.get("smart_timeout", True)

        # Use smart timeout estimation if not explicitly provided
        if timeout is None:
            if smart_timeout:
                timeout = estimate_timeout(code, base_timeout=15)
            else:
                timeout = self.timeout

        temp_file = None

        try:
            # Determine file path
            if save_as:
                file_path = Path(save_as).expanduser().resolve()
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                    encoding="utf-8"
                )
                temp_file.write(code)
                temp_file.close()
                file_path = Path(temp_file.name)

            # Resolve working directory
            if working_dir:
                cwd = Path(working_dir).expanduser().resolve()
                if not cwd.exists():
                    cwd.mkdir(parents=True, exist_ok=True)
            else:
                cwd = file_path.parent

            # Execute Python code asynchronously
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = process.returncode

                exec_result = ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                )

                # Format output
                output_parts = []
                if save_as:
                    output_parts.append(f"Saved to: {file_path}")
                if exec_result.stdout:
                    output_parts.append(f"stdout:\n{exec_result.stdout}")
                if exec_result.stderr:
                    output_parts.append(f"stderr:\n{exec_result.stderr}")
                output_parts.append(f"exit_code: {exec_result.exit_code}")

                output = "\n".join(output_parts)

                return ToolResult(
                    success=exec_result.exit_code == 0,
                    output=output,
                    error=exec_result.stderr if exec_result.exit_code != 0 else None,
                    data=exec_result,
                )

            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                return ToolResult.error_result(
                    f"Code execution timed out after {timeout} seconds"
                )

        except Exception as e:
            return ToolResult.error_result(f"Error executing Python code: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file and not save_as:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass


class ExecuteInSandboxTool(BaseTool):
    """Tool for executing code in an isolated sandbox environment."""
    
    def __init__(self, timeout: int = 120, sandbox_dir: Optional[str] = None):
        self.timeout = timeout
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="sandbox_")
    
    @property
    def name(self) -> str:
        return "execute_in_sandbox"
    
    @property
    def description(self) -> str:
        return "Execute code in an isolated sandbox environment for safety."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, javascript, bash)",
                    "enum": ["python", "javascript", "bash"]
                },
                "files": {
                    "type": "object",
                    "description": "Additional files to create in sandbox (filename -> content)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds"
                }
            },
            "required": ["code", "language"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code")
        language = kwargs.get("language", "python")
        files = kwargs.get("files", {})
        timeout = kwargs.get("timeout", self.timeout)
        
        # Create isolated sandbox directory
        sandbox_path = Path(tempfile.mkdtemp(prefix="sandbox_"))
        
        try:
            # Create additional files
            for filename, content in files.items():
                file_path = sandbox_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            
            # Determine execution command based on language
            if language == "python":
                main_file = sandbox_path / "main.py"
                main_file.write_text(code, encoding="utf-8")
                cmd = [sys.executable, str(main_file)]
            elif language == "javascript":
                main_file = sandbox_path / "main.js"
                main_file.write_text(code, encoding="utf-8")
                cmd = ["node", str(main_file)]
            elif language == "bash":
                main_file = sandbox_path / "main.sh"
                main_file.write_text(code, encoding="utf-8")
                cmd = ["bash", str(main_file)]
            else:
                return ToolResult.error_result(f"Unsupported language: {language}")
            
            # Execute in sandbox
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox_path,
                env={
                    **os.environ,
                    "HOME": str(sandbox_path),
                    "TMPDIR": str(sandbox_path),
                }
            )
            
            exec_result = ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
            
            # Format output
            output_parts = [f"Sandbox: {sandbox_path}"]
            if exec_result.stdout:
                output_parts.append(f"stdout:\n{exec_result.stdout}")
            if exec_result.stderr:
                output_parts.append(f"stderr:\n{exec_result.stderr}")
            output_parts.append(f"exit_code: {exec_result.exit_code}")
            
            output = "\n".join(output_parts)
            
            return ToolResult(
                success=exec_result.exit_code == 0,
                output=output,
                error=exec_result.stderr if exec_result.exit_code != 0 else None,
                data=exec_result,
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult.error_result(
                f"Sandbox execution timed out after {timeout} seconds"
            )
        except FileNotFoundError as e:
            return ToolResult.error_result(
                f"Interpreter not found: {str(e)}. "
                f"Make sure {language} is installed."
            )
        except Exception as e:
            return ToolResult.error_result(f"Sandbox execution error: {str(e)}")
        finally:
            # Clean up sandbox
            try:
                shutil.rmtree(sandbox_path)
            except:
                pass

    async def execute_async(self, **kwargs) -> ToolResult:
        """
        Asynchronously execute code in sandbox.

        Non-blocking execution that allows other async tasks to run.

        Args:
            Same as execute()

        Returns:
            ToolResult with execution output
        """
        code = kwargs.get("code")
        language = kwargs.get("language", "python")
        files = kwargs.get("files", {})
        timeout = kwargs.get("timeout")
        smart_timeout = kwargs.get("smart_timeout", True)

        # Use smart timeout estimation if not explicitly provided
        if timeout is None:
            if smart_timeout:
                timeout = estimate_timeout(code, base_timeout=15)
            else:
                timeout = self.timeout

        # Create isolated sandbox directory
        sandbox_path = Path(tempfile.mkdtemp(prefix="sandbox_"))

        try:
            # Create additional files
            for filename, content in files.items():
                file_path = sandbox_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Determine execution command based on language
            if language == "python":
                main_file = sandbox_path / "main.py"
                main_file.write_text(code, encoding="utf-8")
                cmd = [sys.executable, str(main_file)]
            elif language == "javascript":
                main_file = sandbox_path / "main.js"
                main_file.write_text(code, encoding="utf-8")
                cmd = ["node", str(main_file)]
            elif language == "bash":
                main_file = sandbox_path / "main.sh"
                main_file.write_text(code, encoding="utf-8")
                cmd = ["bash", str(main_file)]
            else:
                return ToolResult.error_result(f"Unsupported language: {language}")

            # Set up environment
            env = {
                **os.environ,
                "HOME": str(sandbox_path),
                "TMPDIR": str(sandbox_path),
            }

            # Execute in sandbox asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_path,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = process.returncode

                exec_result = ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                )

                # Format output
                output_parts = [f"Sandbox: {sandbox_path}"]
                if exec_result.stdout:
                    output_parts.append(f"stdout:\n{exec_result.stdout}")
                if exec_result.stderr:
                    output_parts.append(f"stderr:\n{exec_result.stderr}")
                output_parts.append(f"exit_code: {exec_result.exit_code}")

                output = "\n".join(output_parts)

                return ToolResult(
                    success=exec_result.exit_code == 0,
                    output=output,
                    error=exec_result.stderr if exec_result.exit_code != 0 else None,
                    data=exec_result,
                )

            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                return ToolResult.error_result(
                    f"Sandbox execution timed out after {timeout} seconds"
                )

        except FileNotFoundError as e:
            return ToolResult.error_result(
                f"Interpreter not found: {str(e)}. "
                f"Make sure {language} is installed."
            )
        except Exception as e:
            return ToolResult.error_result(f"Sandbox execution error: {str(e)}")
        finally:
            # Clean up sandbox
            try:
                shutil.rmtree(sandbox_path)
            except:
                pass
