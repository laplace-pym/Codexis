"""
File Tools - Tools for file system operations.
"""

import os
import fnmatch
from pathlib import Path
from typing import Optional

from .base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file at the specified path."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start reading from this line (1-indexed, optional)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Stop reading at this line (inclusive, optional)"
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        encoding = kwargs.get("encoding", "utf-8")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            if not file_path.is_file():
                return ToolResult.error_result(f"Path is not a file: {path}")
            
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            
            # Handle line range
            if start_line is not None or end_line is not None:
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]
                # Add line numbers
                numbered_lines = [
                    f"{i}| {line}" 
                    for i, line in enumerate(lines, start=start_idx + 1)
                ]
                content = "".join(numbered_lines)
            else:
                content = "".join(lines)
            
            return ToolResult.success_result(
                content,
                data={"path": str(file_path), "lines": len(lines)}
            )
            
        except UnicodeDecodeError:
            return ToolResult.error_result(
                f"Cannot decode file with encoding '{encoding}'. "
                "Try a different encoding."
            )
        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error_result(f"Error reading file: {str(e)}")


class WriteFileTool(BaseTool):
    """Tool for writing content to files."""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file if it doesn't exist, or overwrites if it does."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to file instead of overwriting",
                    "default": False
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist",
                    "default": True
                }
            },
            "required": ["path", "content"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content")
        encoding = kwargs.get("encoding", "utf-8")
        append = kwargs.get("append", False)
        create_dirs = kwargs.get("create_dirs", True)
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            # Create parent directories if needed
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = "a" if append else "w"
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            action = "Appended to" if append else "Wrote"
            return ToolResult.success_result(
                f"{action} {len(content)} characters to {file_path}",
                data={"path": str(file_path), "bytes": len(content.encode(encoding))}
            )
            
        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error_result(f"Error writing file: {str(e)}")


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List the contents of a directory."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list recursively",
                    "default": False
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py')"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with .)",
                    "default": False
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        recursive = kwargs.get("recursive", False)
        pattern = kwargs.get("pattern")
        include_hidden = kwargs.get("include_hidden", False)
        
        try:
            dir_path = Path(path).expanduser().resolve()
            
            if not dir_path.exists():
                return ToolResult.error_result(f"Directory not found: {path}")
            
            if not dir_path.is_dir():
                return ToolResult.error_result(f"Path is not a directory: {path}")
            
            entries = []
            
            if recursive:
                for item in dir_path.rglob("*"):
                    rel_path = item.relative_to(dir_path)
                    if not include_hidden and any(p.startswith(".") for p in rel_path.parts):
                        continue
                    if pattern and not fnmatch.fnmatch(item.name, pattern):
                        continue
                    
                    prefix = "📁" if item.is_dir() else "📄"
                    entries.append(f"{prefix} {rel_path}")
            else:
                for item in sorted(dir_path.iterdir()):
                    if not include_hidden and item.name.startswith("."):
                        continue
                    if pattern and not fnmatch.fnmatch(item.name, pattern):
                        continue
                    
                    prefix = "📁" if item.is_dir() else "📄"
                    size = ""
                    if item.is_file():
                        size = f" ({item.stat().st_size:,} bytes)"
                    entries.append(f"{prefix} {item.name}{size}")
            
            if not entries:
                return ToolResult.success_result(
                    f"Directory is empty: {path}",
                    data={"path": str(dir_path), "count": 0}
                )
            
            output = f"Contents of {dir_path}:\n" + "\n".join(entries)
            return ToolResult.success_result(
                output,
                data={"path": str(dir_path), "count": len(entries)}
            )
            
        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error_result(f"Error listing directory: {str(e)}")


class SearchFilesTool(BaseTool):
    """Tool for searching files by name or content."""
    
    @property
    def name(self) -> str:
        return "search_files"
    
    @property
    def description(self) -> str:
        return "Search for files by name pattern or content."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory to search in"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g., '*.py')"
                },
                "content": {
                    "type": "string",
                    "description": "Text to search for within files"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 50
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        pattern = kwargs.get("pattern", "*")
        content = kwargs.get("content")
        max_results = kwargs.get("max_results", 50)
        
        try:
            dir_path = Path(path).expanduser().resolve()
            
            if not dir_path.exists():
                return ToolResult.error_result(f"Directory not found: {path}")
            
            results = []
            
            for file_path in dir_path.rglob(pattern):
                if len(results) >= max_results:
                    break
                    
                if not file_path.is_file():
                    continue
                
                if content:
                    # Search file content
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        
                        if content.lower() in file_content.lower():
                            # Find line numbers with matches
                            lines = file_content.split("\n")
                            match_lines = [
                                i + 1 for i, line in enumerate(lines)
                                if content.lower() in line.lower()
                            ]
                            rel_path = file_path.relative_to(dir_path)
                            results.append(f"📄 {rel_path} (lines: {match_lines[:5]})")
                    except (UnicodeDecodeError, PermissionError):
                        continue
                else:
                    rel_path = file_path.relative_to(dir_path)
                    results.append(f"📄 {rel_path}")
            
            if not results:
                return ToolResult.success_result(
                    "No matching files found.",
                    data={"count": 0}
                )
            
            header = f"Found {len(results)} matches"
            if len(results) >= max_results:
                header += f" (limited to {max_results})"
            
            output = f"{header}:\n" + "\n".join(results)
            return ToolResult.success_result(
                output,
                data={"count": len(results), "files": results}
            )
            
        except Exception as e:
            return ToolResult.error_result(f"Error searching files: {str(e)}")


class DeleteFileTool(BaseTool):
    """Tool for deleting files."""
    
    @property
    def name(self) -> str:
        return "delete_file"
    
    @property
    def description(self) -> str:
        return "Delete a file at the specified path."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to delete"
                }
            },
            "required": ["path"]
        }
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {path}")
            
            if not file_path.is_file():
                return ToolResult.error_result(
                    f"Path is not a file: {path}. Use a different tool for directories."
                )
            
            file_path.unlink()
            return ToolResult.success_result(f"Deleted file: {file_path}")
            
        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.error_result(f"Error deleting file: {str(e)}")
