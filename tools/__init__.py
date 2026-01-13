"""
Tools Module - Extensible tool system for the Coding Agent.

This module provides a comprehensive set of tools for:
- File operations (read, write, list, search, delete)
- Code execution (Python, shell commands, sandboxed execution)
- Document parsing (PDF, Word, images with OCR)
- Code search (grep-like, symbol finding, search & replace)
- Code modification (patches, diffs, block editing)
- Code analysis (structure analysis, test generation, summarization)
"""

from .base import BaseTool, ToolResult
from .registry import ToolRegistry, tool, create_default_registry
from .file_tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    SearchFilesTool,
    DeleteFileTool,
)
from .code_executor import ExecuteCodeTool, ExecutePythonTool, ExecuteInSandboxTool
from .doc_tools import ReadPDFTool, ReadDocxTool, ReadImageTool
from .search_tools import GrepTool, FindSymbolTool, ReplaceInFilesTool
from .patch_tools import ApplyPatchTool, EditBlockTool, InsertCodeTool, CreateDiffTool
from .analysis_tools import AnalyzeCodeTool, GenerateTestsTool, SummarizeTool

__all__ = [
    # Base
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "create_default_registry",
    
    # File tools
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "SearchFilesTool",
    "DeleteFileTool",
    
    # Code execution
    "ExecuteCodeTool",
    "ExecutePythonTool",
    "ExecuteInSandboxTool",
    
    # Document tools
    "ReadPDFTool",
    "ReadDocxTool",
    "ReadImageTool",
    
    # Search tools
    "GrepTool",
    "FindSymbolTool",
    "ReplaceInFilesTool",
    
    # Patch/Edit tools
    "ApplyPatchTool",
    "EditBlockTool",
    "InsertCodeTool",
    "CreateDiffTool",
    
    # Analysis tools
    "AnalyzeCodeTool",
    "GenerateTestsTool",
    "SummarizeTool",
]
