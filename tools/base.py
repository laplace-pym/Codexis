"""
Base Tool - Abstract interface for all tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from llm.base import ToolDefinition


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None  # Structured data if applicable
    
    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}\n{self.output}" if self.output else f"Error: {self.error}"
    
    @classmethod
    def success_result(cls, output: str, data: Any = None) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, output=output, data=data)
    
    @classmethod
    def error_result(cls, error: str, output: str = "") -> "ToolResult":
        """Create an error result."""
        return cls(success=False, output=output, error=error)


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    To create a new tool:
    1. Extend this class
    2. Implement the required properties and methods
    3. Register with ToolRegistry
    
    Example:
        class MyTool(BaseTool):
            @property
            def name(self) -> str:
                return "my_tool"
            
            @property
            def description(self) -> str:
                return "Does something useful"
            
            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input value"}
                    },
                    "required": ["input"]
                }
            
            def execute(self, **kwargs) -> ToolResult:
                input_val = kwargs.get("input")
                # Do something
                return ToolResult.success_result(f"Processed: {input_val}")
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        JSON Schema for the tool parameters.
        
        Example:
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    }
                },
                "required": ["path"]
            }
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult indicating success/failure and output
        """
        pass
    
    async def execute_async(self, **kwargs) -> ToolResult:
        """
        Async version of execute. Override if the tool needs async operations.
        Default implementation calls the sync version.
        """
        return self.execute(**kwargs)
    
    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM API."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
    
    def validate_args(self, **kwargs) -> Optional[str]:
        """
        Validate tool arguments.
        
        Returns:
            None if valid, error message string if invalid
        """
        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})
        
        # Check required arguments
        for req in required:
            if req not in kwargs or kwargs[req] is None:
                return f"Missing required argument: {req}"
        
        # Type checking could be added here
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
