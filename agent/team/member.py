"""
TeamMember - Wraps a CodingAgent for participation in a team.

Each team member runs its assigned task in a background thread,
communicates results via the MessageBus, and reports status through
event callbacks.
"""

import threading
from typing import Optional, Callable

from ..coding_agent import CodingAgent
from .models import (
    TeamTask,
    TeamMessage,
    TeamMessageType,
    TeamMemberInfo,
)
from .message_bus import MessageBus


class TeamMember:
    """
    A team member that wraps a CodingAgent for collaborative execution.

    Usage:
        member = TeamMember(name="backend", role="backend developer")
        member.start_task(task)
        member.wait()
    """

    def __init__(
        self,
        name: str,
        role: str = "general",
        agent: Optional[CodingAgent] = None,
        provider: Optional[str] = None,
        message_bus: Optional[MessageBus] = None,
        on_event: Optional[Callable] = None,
    ):
        """
        Initialize a TeamMember.

        Args:
            name: Unique name for this member.
            role: Role description (e.g. "backend developer").
            agent: Pre-configured CodingAgent. If None, one is created.
            provider: LLM provider name, used when creating a new agent.
            message_bus: Shared MessageBus for inter-member communication.
            on_event: Callback invoked on lifecycle events.
        """
        self.name = name
        self.role = role
        self.agent = agent or CodingAgent(provider=provider)
        self.message_bus = message_bus
        self.on_event = on_event

        self.info = TeamMemberInfo(
            name=name,
            role=role,
            provider=provider,
        )

        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[TeamTask] = None
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_task(self, task: TeamTask) -> None:
        """
        Begin executing a task in a background daemon thread.

        Args:
            task: The TeamTask to execute.
        """
        self._current_task = task
        task.start(self.name)

        self.info.status = "working"
        self.info.current_task_id = task.id

        self._emit_event({
            "type": "task_started",
            "member": self.name,
            "task_id": task.id,
            "task_title": task.title,
        })

        self._thread = threading.Thread(
            target=self._execute_task,
            args=(task,),
            daemon=True,
        )
        self._thread.start()

    def is_busy(self) -> bool:
        """Return True if the member's execution thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until the current task finishes."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def shutdown(self) -> None:
        """Signal the member to stop and mark status as done."""
        self._shutdown.set()
        self.info.status = "done"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_task(self, task: TeamTask) -> None:
        """Run the task via the wrapped CodingAgent (runs in a thread)."""
        try:
            context = self._build_task_context(task)
            result = self.agent.run(
                task=task.description,
                context=context,
                auto_detect_complexity=True,
            )

            task.complete(result)
            self.info.status = "idle"
            self.info.current_task_id = None
            self.info.tasks_completed += 1

            self._emit_event({
                "type": "task_completed",
                "member": self.name,
                "task_id": task.id,
                "result": result,
            })

            if self.message_bus:
                self.message_bus.send(TeamMessage(
                    type=TeamMessageType.TASK_COMPLETED,
                    sender=self.name,
                    recipient="leader",
                    content=f"Task '{task.title}' completed.",
                    data={"task_id": task.id, "result": result},
                ))

        except Exception as exc:
            error_msg = str(exc)
            task.fail(error_msg)
            self.info.status = "idle"
            self.info.current_task_id = None
            self.info.tasks_failed += 1

            self._emit_event({
                "type": "task_failed",
                "member": self.name,
                "task_id": task.id,
                "error": error_msg,
            })

            if self.message_bus:
                self.message_bus.send(TeamMessage(
                    type=TeamMessageType.TASK_FAILED,
                    sender=self.name,
                    recipient="leader",
                    content=f"Task '{task.title}' failed: {error_msg}",
                    data={"task_id": task.id, "error": error_msg},
                ))

    def _build_task_context(self, task: TeamTask) -> str:
        """Build a context string describing the member's role and task."""
        lines = [
            f"You are team member '{self.name}' with role: {self.role}.",
            f"Task: {task.title}",
        ]
        if task.dependencies:
            lines.append(f"This task depends on: {', '.join(task.dependencies)}")
        if task.metadata:
            parent_info = task.metadata.get("parent_task")
            if parent_info:
                lines.append(f"Parent task: {parent_info}")
        return "\n".join(lines)

    def _emit_event(self, event: dict) -> None:
        """Call the on_event callback if one is registered."""
        if self.on_event is not None:
            self.on_event(event)
