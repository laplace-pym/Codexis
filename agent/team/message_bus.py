"""
MessageBus - Thread-safe inter-agent communication system.

Provides pub/sub style messaging between team members with support for
direct messages, broadcasts, and global listeners for monitoring.
"""

import queue
import threading
import logging
from typing import Callable, Optional

from .models import TeamMessage, TeamMessageType

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Thread-safe message bus for inter-agent communication.

    Each registered member gets a dedicated inbox (queue). Messages can be
    sent to specific members or broadcast to all. Global listeners receive
    copies of every message for monitoring/streaming purposes.
    """

    def __init__(self):
        self._inboxes: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._global_listeners: list[Callable[[TeamMessage], None]] = []

    def register(self, name: str) -> None:
        """
        Create an inbox for a team member.

        Args:
            name: Unique member name.
        """
        with self._lock:
            if name in self._inboxes:
                logger.warning(f"Member '{name}' already registered, resetting inbox")
            self._inboxes[name] = queue.Queue()
            logger.debug(f"Registered inbox for '{name}'")

    def unregister(self, name: str) -> None:
        """
        Remove a member's inbox.

        Args:
            name: Member name to remove.
        """
        with self._lock:
            if name in self._inboxes:
                del self._inboxes[name]
                logger.debug(f"Unregistered inbox for '{name}'")

    def send(self, message: TeamMessage) -> None:
        """
        Send a message. If recipient is "all", broadcast to all members
        except the sender. Otherwise send to the specific recipient's inbox.

        Always notifies global listeners regardless of recipient.

        Args:
            message: The message to send.
        """
        with self._lock:
            if message.recipient == "all":
                for name, inbox in self._inboxes.items():
                    if name != message.sender:
                        inbox.put(message)
                logger.debug(
                    f"Broadcast from '{message.sender}': {message.type.value}"
                )
            else:
                inbox = self._inboxes.get(message.recipient)
                if inbox is not None:
                    inbox.put(message)
                    logger.debug(
                        f"Message from '{message.sender}' to "
                        f"'{message.recipient}': {message.type.value}"
                    )
                else:
                    logger.warning(
                        f"No inbox for recipient '{message.recipient}', "
                        f"message dropped"
                    )

            # Notify all global listeners
            for listener in self._global_listeners:
                try:
                    listener(message)
                except Exception as e:
                    logger.error(f"Global listener error: {e}")

    def receive(self, name: str, timeout: Optional[float] = None) -> Optional[TeamMessage]:
        """
        Blocking receive from a member's inbox.

        Args:
            name: Member name whose inbox to read.
            timeout: Max seconds to wait. None means block forever.

        Returns:
            The next message, or None if timeout expired.
        """
        inbox = self._get_inbox(name)
        if inbox is None:
            return None
        try:
            return inbox.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def receive_nowait(self, name: str) -> Optional[TeamMessage]:
        """
        Non-blocking receive from a member's inbox.

        Args:
            name: Member name whose inbox to read.

        Returns:
            The next message, or None if inbox is empty.
        """
        inbox = self._get_inbox(name)
        if inbox is None:
            return None
        try:
            return inbox.get_nowait()
        except queue.Empty:
            return None

    def has_messages(self, name: str) -> bool:
        """
        Check if a member has pending messages.

        Args:
            name: Member name to check.

        Returns:
            True if there are messages in the inbox.
        """
        inbox = self._get_inbox(name)
        if inbox is None:
            return False
        return not inbox.empty()

    def add_listener(self, callback: Callable[[TeamMessage], None]) -> None:
        """
        Add a global listener that receives copies of all messages.

        Useful for monitoring, logging, or web streaming.

        Args:
            callback: Function called with each message.
        """
        with self._lock:
            self._global_listeners.append(callback)

    def broadcast_shutdown(self, sender: str = "leader") -> None:
        """
        Send a SHUTDOWN message to all registered members.

        Args:
            sender: Who is initiating the shutdown (default: "leader").
        """
        message = TeamMessage(
            type=TeamMessageType.SHUTDOWN,
            sender=sender,
            recipient="all",
            content="Shutdown requested",
        )
        self.send(message)

    def _get_inbox(self, name: str) -> Optional[queue.Queue]:
        """Get a member's inbox, or None if not registered."""
        with self._lock:
            inbox = self._inboxes.get(name)
        if inbox is None:
            logger.warning(f"No inbox for '{name}'")
        return inbox
