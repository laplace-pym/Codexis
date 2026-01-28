"""
Chat Mode - Simple conversation without tool calling.

This module provides a lightweight chat mode for simple Q&A interactions
that don't require the full agent tool-calling capabilities.
"""

from typing import Optional, AsyncIterator, List
from dataclasses import dataclass, field

from llm.base import BaseLLM, Message, MessageRole
from llm.factory import LLMFactory
from utils.config import Config


@dataclass
class ChatHistory:
    """Maintains conversation history for chat mode."""
    messages: List[Message] = field(default_factory=list)
    max_history: int = 50

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(Message.user(content))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append(Message.assistant(content))
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep only the most recent messages."""
        if len(self.messages) > self.max_history:
            # Keep system message if present, then trim from the beginning
            if self.messages and self.messages[0].role == MessageRole.SYSTEM:
                self.messages = [self.messages[0]] + self.messages[-(self.max_history - 1):]
            else:
                self.messages = self.messages[-self.max_history:]

    def clear(self) -> None:
        """Clear all history except system message."""
        if self.messages and self.messages[0].role == MessageRole.SYSTEM:
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def get_messages(self) -> List[Message]:
        """Get all messages for API call."""
        return self.messages.copy()


class ChatMode:
    """
    Simple chat mode without tool calling.

    This mode is optimized for:
    - Quick Q&A responses
    - General conversation
    - Lower latency (no tool overhead)

    Usage:
        chat = ChatMode()
        response = chat.chat("What is Python?")

        # With context
        response = chat.chat("Explain this code", context="def foo(): pass")

        # Streaming
        async for chunk in chat.chat_stream("Tell me a story"):
            print(chunk, end="")
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. You provide clear, accurate, and helpful responses.

When asked about code or technical topics:
- Provide clear explanations
- Include code examples when helpful
- Be concise but thorough

Keep your responses focused and relevant to the user's question."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize chat mode.

        Args:
            llm: LLM instance to use
            provider: LLM provider name if llm not provided
            system_prompt: Custom system prompt
            temperature: Response temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        if llm:
            self.llm = llm
        else:
            config = Config.load()
            self.llm = LLMFactory.create(provider or config.default_provider)

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = ChatHistory()

        # Initialize with system message
        self.history.messages.append(Message.system(self.system_prompt))

    def chat(
        self,
        message: str,
        context: Optional[str] = None,
        include_history: bool = True,
    ) -> str:
        """
        Send a chat message and get a response.

        Args:
            message: User message
            context: Optional context to include (e.g., document content)
            include_history: Whether to include conversation history

        Returns:
            Assistant response string
        """
        # Build the user message
        user_content = message
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {message}"

        # Add to history
        self.history.add_user_message(user_content)

        # Get messages for API call
        if include_history:
            messages = self.history.get_messages()
        else:
            messages = [
                Message.system(self.system_prompt),
                Message.user(user_content),
            ]

        # Call LLM (no tools)
        response = self.llm.chat_sync(
            messages=messages,
            tools=None,  # No tools in chat mode
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Add response to history
        self.history.add_assistant_message(response.content)

        return response.content

    async def chat_async(
        self,
        message: str,
        context: Optional[str] = None,
        include_history: bool = True,
    ) -> str:
        """
        Async version of chat.

        Args:
            message: User message
            context: Optional context
            include_history: Whether to include history

        Returns:
            Assistant response string
        """
        # Build the user message
        user_content = message
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {message}"

        # Add to history
        self.history.add_user_message(user_content)

        # Get messages for API call
        if include_history:
            messages = self.history.get_messages()
        else:
            messages = [
                Message.system(self.system_prompt),
                Message.user(user_content),
            ]

        # Call LLM async
        response = await self.llm.chat(
            messages=messages,
            tools=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Add response to history
        self.history.add_assistant_message(response.content)

        return response.content

    async def chat_stream(
        self,
        message: str,
        context: Optional[str] = None,
        include_history: bool = True,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response.

        Args:
            message: User message
            context: Optional context
            include_history: Whether to include history

        Yields:
            Response chunks as they arrive
        """
        # Build the user message
        user_content = message
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {message}"

        # Add to history
        self.history.add_user_message(user_content)

        # Get messages for API call
        if include_history:
            messages = self.history.get_messages()
        else:
            messages = [
                Message.system(self.system_prompt),
                Message.user(user_content),
            ]

        # Check if LLM supports streaming
        if hasattr(self.llm, 'chat_stream'):
            full_response = ""
            async for chunk in self.llm.chat_stream(
                messages=messages,
                tools=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                full_response += chunk
                yield chunk

            # Add complete response to history
            self.history.add_assistant_message(full_response)
        else:
            # Fallback to non-streaming
            response = await self.llm.chat(
                messages=messages,
                tools=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self.history.add_assistant_message(response.content)
            yield response.content

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = prompt
        # Update the first message if it's a system message
        if self.history.messages and self.history.messages[0].role == MessageRole.SYSTEM:
            self.history.messages[0] = Message.system(prompt)
        else:
            self.history.messages.insert(0, Message.system(prompt))
