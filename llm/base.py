"""
Base LLM Adapter - Abstract interface for all LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class MessageRole(Enum):
    """Role in a conversation message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in the conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None  # For tool response messages
    tool_calls: Optional[list["ToolCall"]] = None  # For assistant messages with tool calls
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for API calls."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[list["ToolCall"]] = None) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        """Create a tool response message."""
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolCall:
    """Represents a tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    
    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r}, arguments={self.arguments!r})"


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional[dict] = None
    raw_response: Optional[Any] = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema format
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class BaseLLM(ABC):
    """
    Abstract base class for LLM adapters.
    
    All LLM providers must implement this interface.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initialize the LLM adapter.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            model: Model name/identifier
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of available tools
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse with content and/or tool calls
        """
        pass
    
    @abstractmethod
    def chat_sync(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Synchronous version of chat.
        """
        pass
    
    def get_system_prompt(self) -> str:
        """
        Get the default system prompt for the coding agent.
        Override in subclasses for provider-specific prompts.
        """
        return """You are an expert coding assistant. Your role is to help users with programming tasks by:

1. Understanding their requirements clearly
2. Breaking down complex tasks into manageable steps
3. Writing clean, well-documented code
4. Using tools when necessary to read files, execute code, etc.
5. Automatically fixing errors when code execution fails

When you need to perform actions, use the available tools. Always explain your reasoning.

When generating code:
- Write complete, runnable code
- Include necessary imports
- Add helpful comments
- Follow best practices for the language

When errors occur:
- Analyze the error message carefully
- Identify the root cause
- Propose and implement a fix
- Re-execute to verify the fix works"""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, base_url={self.base_url!r})"
