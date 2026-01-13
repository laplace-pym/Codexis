"""
Tool Registry - Central registry for all available tools.
"""

from typing import Optional, Callable, Type
from functools import wraps

from .base import BaseTool, ToolResult
from llm.base import ToolDefinition


class ToolRegistry:
    """
    Central registry for managing tools.
    
    Features:
    - Register tools by class or instance
    - Auto-discovery of tools
    - Get tool definitions for LLM
    - Execute tools by name
    
    Usage:
        # Create registry
        registry = ToolRegistry()
        
        # Register tools
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        
        # Or register class (will instantiate)
        registry.register_class(ReadFileTool)
        
        # Get all tool definitions
        tools = registry.get_definitions()
        
        # Execute a tool
        result = registry.execute("read_file", path="/some/file.txt")
    """
    
    _global_instance: Optional["ToolRegistry"] = None
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    @classmethod
    def get_global(cls) -> "ToolRegistry":
        """Get the global registry instance."""
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.
        
        Args:
            tool: Tool instance to register
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, got {type(tool)}")
        
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
    
    def register_class(self, tool_class: Type[BaseTool], **init_kwargs) -> None:
        """
        Register a tool class (will be instantiated).
        
        Args:
            tool_class: Tool class to register
            **init_kwargs: Arguments to pass to tool constructor
        """
        if not issubclass(tool_class, BaseTool):
            raise TypeError(f"Expected BaseTool subclass, got {tool_class}")
        
        tool = tool_class(**init_kwargs)
        self.register(tool)
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinition list for all registered tools."""
        return [tool.to_definition() for tool in self._tools.values()]
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            ToolResult from the tool execution
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.error_result(f"Unknown tool: {name}")
        
        # Check if arguments had parsing errors
        if "_parse_error" in kwargs:
            raw_preview = kwargs.pop("_raw_preview", "N/A")
            return ToolResult.error_result(
                f"Failed to parse tool arguments. This usually happens when the LLM returns "
                f"malformed JSON. Raw arguments preview: {raw_preview[:200]}"
            )
        
        # Validate arguments
        validation_error = tool.validate_args(**kwargs)
        if validation_error:
            return ToolResult.error_result(validation_error)
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult.error_result(str(e))
    
    async def execute_async(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool asynchronously.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.error_result(f"Unknown tool: {name}")
        
        validation_error = tool.validate_args(**kwargs)
        if validation_error:
            return ToolResult.error_result(validation_error)
        
        try:
            return await tool.execute_async(**kwargs)
        except Exception as e:
            return ToolResult.error_result(str(e))
    
    def list_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __iter__(self):
        return iter(self._tools.values())


def tool(registry: Optional[ToolRegistry] = None):
    """
    Decorator to register a tool class.
    
    Usage:
        @tool()
        class MyTool(BaseTool):
            ...
        
        # Or with specific registry
        @tool(my_registry)
        class MyTool(BaseTool):
            ...
    """
    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        target_registry = registry or ToolRegistry.get_global()
        target_registry.register_class(cls)
        return cls
    return decorator


def create_default_registry() -> ToolRegistry:
    """
    Create a registry with all default tools registered.
    
    Returns:
        ToolRegistry with standard tools
    """
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
    
    registry = ToolRegistry()
    
    # File tools
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchFilesTool())
    registry.register(DeleteFileTool())
    
    # Code execution
    registry.register(ExecuteCodeTool())
    registry.register(ExecutePythonTool())
    registry.register(ExecuteInSandboxTool())
    
    # Document tools
    registry.register(ReadPDFTool())
    registry.register(ReadDocxTool())
    registry.register(ReadImageTool())
    
    # Search tools (grep-like)
    registry.register(GrepTool())
    registry.register(FindSymbolTool())
    registry.register(ReplaceInFilesTool())
    
    # Patch/Edit tools
    registry.register(ApplyPatchTool())
    registry.register(EditBlockTool())
    registry.register(InsertCodeTool())
    registry.register(CreateDiffTool())
    
    # Analysis tools
    registry.register(AnalyzeCodeTool())
    registry.register(GenerateTestsTool())
    registry.register(SummarizeTool())
    
    return registry
