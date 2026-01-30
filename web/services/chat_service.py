"""
Chat Service - Handles chat and agent message processing.
"""

import uuid
from typing import Optional, Dict, AsyncIterator, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

from agent import CodingAgent, ChatMode


@dataclass
class Session:
    """Represents a chat session."""
    id: str
    mode: Literal["chat", "agent"]
    agent: CodingAgent
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    document_ids: list[str] = field(default_factory=list)
    # Interactive mode support
    action_queue: Optional[asyncio.Queue] = field(default=None)
    execution_mode: Literal["auto", "interactive"] = field(default="auto")

    def __post_init__(self):
        """Initialize the action queue."""
        if self.action_queue is None:
            self.action_queue = asyncio.Queue()

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() - self.last_activity > timedelta(minutes=timeout_minutes)

    def set_execution_mode(self, mode: Literal["auto", "interactive"]) -> None:
        """Set the execution mode for this session."""
        self.execution_mode = mode

    async def send_user_action(self, action: dict) -> None:
        """Send a user action to the action queue."""
        if self.action_queue:
            await self.action_queue.put(action)


class SessionManager:
    """Manages chat sessions."""

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize session manager.

        Args:
            cleanup_interval: Seconds between cleanup runs
        """
        self._sessions: Dict[str, Session] = {}
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None

    def create_session(
        self,
        mode: Literal["chat", "agent"] = "agent",
        provider: Optional[str] = None,
    ) -> Session:
        """
        Create a new session.

        Args:
            mode: Operating mode (chat or agent)
            provider: LLM provider name

        Returns:
            New Session instance
        """
        session_id = str(uuid.uuid4())
        agent = CodingAgent(provider=provider)
        agent.set_mode(mode)

        session = Session(
            id=session_id,
            mode=mode,
            agent=agent,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        mode: Literal["chat", "agent"] = "agent",
    ) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                # Update mode if different
                if session.mode != mode:
                    session.mode = mode
                    session.agent.set_mode(mode)
                return session

        return self.create_session(mode=mode)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired(self, timeout_minutes: int = 60) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired(timeout_minutes)
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    async def start_cleanup_loop(self) -> None:
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(self._cleanup_interval)
                self.cleanup_expired()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    def stop_cleanup_loop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


class ChatService:
    """
    Service for handling chat/agent messages.
    """

    def __init__(self, session_manager: Optional[SessionManager] = None):
        """
        Initialize chat service.

        Args:
            session_manager: Session manager instance
        """
        self.sessions = session_manager or SessionManager()

    async def process_message(
        self,
        message: str,
        mode: Literal["chat", "agent"] = "agent",
        session_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict:
        """
        Process a chat message.

        Args:
            message: User message
            mode: Operating mode
            session_id: Optional session ID
            context: Optional context (document contents, etc.)

        Returns:
            Response dict with session_id, content, and optional tool_calls
        """
        # Get or create session
        session = self.sessions.get_or_create_session(session_id, mode)

        # Process message
        response = session.agent.process_message(
            message=message,
            context=context,
            mode=mode,
        )

        result = {
            "session_id": session.id,
            "content": response,
            "mode": mode,
        }

        # Include tool calls for agent mode
        if mode == "agent" and session.agent.state and session.agent.state.history:
            tool_calls = []
            for step in session.agent.state.history.steps:
                if step.step_type.value == "tool_call":
                    tool_calls.append({
                        "tool": step.tool_name,
                        "args": step.tool_args,
                        "result": step.tool_result[:200] if step.tool_result else None,
                        "success": step.status.value == "success",
                    })
            if tool_calls:
                result["tool_calls"] = tool_calls

        return result

    async def process_message_stream(
        self,
        message: str,
        mode: Literal["chat", "agent"] = "agent",
        session_id: Optional[str] = None,
        context: Optional[str] = None,
        execution_mode: Literal["auto", "interactive"] = "auto",
    ) -> AsyncIterator[dict]:
        """
        Stream a response.

        Args:
            message: User message
            mode: Operating mode
            session_id: Session ID
            context: Optional context
            execution_mode: "auto" for automatic execution, "interactive" for user confirmation

        Yields:
            Event dictionaries
        """
        session = self.sessions.get_or_create_session(session_id, mode)
        session.set_execution_mode(execution_mode)

        # Yield session info first
        yield {"type": "session", "session_id": session.id}

        # Stream the response with appropriate mode
        if execution_mode == "interactive" and mode == "agent":
            # Use interactive streaming with code preview
            async for event in session.agent.process_message_stream_interactive(
                message=message,
                context=context,
                action_queue=session.action_queue,
            ):
                yield event
        else:
            # Standard streaming
            async for event in session.agent.process_message_stream(
                message=message,
                context=context,
                mode=mode,
            ):
                yield event
