"""
Team - Main team orchestration.

Coordinates multiple TeamMembers to execute a complex task by decomposing
it via the TeamLeader, assigning sub-tasks, and polling for completion
with dependency-aware scheduling.
"""

import time
import uuid
import threading
from typing import Optional, Callable
from datetime import datetime

from .models import (
    TeamTask,
    TeamTaskStatus,
    TeamMessage,
    TeamMessageType,
    TeamMemberInfo,
    TeamStatus,
    TeamProgress,
)
from .member import TeamMember
from .leader import TeamLeader
from .message_bus import MessageBus
from utils.logger import get_logger


class Team:
    """
    A group of agents collaborating on a complex task.

    Lifecycle:
        1. Create team and add members
        2. Call execute(task) with a high-level task description
        3. Leader decomposes into sub-tasks
        4. Members execute sub-tasks in parallel (respecting dependencies)
        5. Returns TeamProgress with final status

    Usage:
        team = Team(provider="deepseek")
        team.add_member("architect", role="architect")
        team.add_member("developer", role="developer")
        team.add_member("tester", role="tester")
        progress = team.execute("Build a REST API with tests")
    """

    def __init__(
        self,
        team_id: Optional[str] = None,
        provider: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        self.team_id = team_id or uuid.uuid4().hex[:8]
        self.provider = provider
        self.on_event = on_event

        self.message_bus = MessageBus()
        self.leader = TeamLeader(provider=provider)
        self.members: dict[str, TeamMember] = {}
        self.tasks: list[TeamTask] = []
        self.status = TeamStatus.IDLE
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        self.logger = get_logger()

        # Register leader on message bus
        self.message_bus.register("leader")

        # Wire up global message listener for web streaming
        if on_event:
            self.message_bus.add_listener(
                lambda msg: on_event({
                    "type": "team_message",
                    "team_id": self.team_id,
                    "message": msg.to_dict(),
                })
            )

    # ------------------------------------------------------------------
    # Member management
    # ------------------------------------------------------------------

    def add_member(
        self,
        name: str,
        role: str = "general",
        provider: Optional[str] = None,
    ) -> TeamMember:
        """Add a member to the team."""
        member = TeamMember(
            name=name,
            role=role,
            provider=provider or self.provider,
            message_bus=self.message_bus,
            on_event=self.on_event,
        )
        self.members[name] = member
        self.message_bus.register(name)
        return member

    def remove_member(self, name: str) -> None:
        """Remove a member from the team."""
        member = self.members.pop(name, None)
        if member:
            member.shutdown()
            self.message_bus.unregister(name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> TeamProgress:
        """
        Execute a high-level task with the team.

        Steps:
            1. Leader decomposes task into sub-tasks
            2. Sub-tasks are executed by members (parallel, dependency-aware)
            3. Returns final progress

        Args:
            task: High-level task description
            context: Additional context

        Returns:
            TeamProgress with final status
        """
        self.status = TeamStatus.PLANNING
        self.start_time = datetime.now()

        self._emit_event({
            "type": "team_status",
            "status": "planning",
            "task": task,
        })

        # Step 1: Leader decomposes the task
        member_names_roles = [
            (name, m.role) for name, m in self.members.items()
        ]

        self.logger.info(
            f"Team leader decomposing task for {len(self.members)} members..."
        )
        self.tasks = self.leader.decompose_task(
            task=task,
            member_names_roles=member_names_roles,
            context=context,
        )

        # Inject parent task info into metadata
        for t in self.tasks:
            t.metadata["parent_task"] = task

        self._emit_event({
            "type": "team_tasks_created",
            "tasks": [t.to_dict() for t in self.tasks],
        })

        self.logger.info(f"Created {len(self.tasks)} sub-tasks:")
        for t in self.tasks:
            deps = f" (depends on: {', '.join(t.dependencies)})" if t.dependencies else ""
            self.logger.info(f"  - [{t.assigned_to}] {t.title}{deps}")

        # Step 2: Execute tasks
        self.status = TeamStatus.EXECUTING

        self._emit_event({
            "type": "team_status",
            "status": "executing",
        })

        self._execute_tasks()

        # Step 3: Finalize
        self.end_time = datetime.now()
        failed_tasks = [
            t for t in self.tasks if t.status == TeamTaskStatus.FAILED
        ]

        if failed_tasks and len(failed_tasks) == len(self.tasks):
            self.status = TeamStatus.FAILED
        else:
            self.status = TeamStatus.COMPLETED

        progress = self.get_progress()

        self._emit_event({
            "type": "team_complete",
            "progress": progress.to_dict(),
        })

        self.logger.info(
            f"Team finished: {progress.completed_tasks}/{progress.total_tasks} "
            f"completed, {progress.failed_tasks} failed, "
            f"{progress.elapsed_seconds:.1f}s elapsed"
        )

        # Cleanup
        self.message_bus.broadcast_shutdown()

        return progress

    def _execute_tasks(self) -> None:
        """
        Execute all tasks, respecting dependencies.

        Strategy:
        - Tasks with no unresolved dependencies are eligible to start
        - Assign eligible tasks to their designated members when idle
        - Wait for completions, then check for newly eligible tasks
        - Deadlock detection: if nothing is in progress and nothing eligible
        """
        completed_ids: set[str] = set()
        started_ids: set[str] = set()

        while True:
            # Update completed set
            for task in self.tasks:
                if task.status == TeamTaskStatus.COMPLETED:
                    completed_ids.add(task.id)

            # Find eligible tasks (dependencies met, not started yet)
            eligible = []
            for task in self.tasks:
                if task.id in started_ids:
                    continue
                if task.status in (
                    TeamTaskStatus.COMPLETED,
                    TeamTaskStatus.FAILED,
                ):
                    continue
                deps_met = all(
                    dep_id in completed_ids for dep_id in task.dependencies
                )
                if deps_met:
                    eligible.append(task)

            # Start eligible tasks on their assigned members
            for task in eligible:
                member = self.members.get(task.assigned_to)
                if member and not member.is_busy():
                    self.logger.info(
                        f"Assigning '{task.title}' to {task.assigned_to}"
                    )
                    member.start_task(task)
                    started_ids.add(task.id)

            # Check if all tasks are done
            all_done = all(
                t.status
                in (TeamTaskStatus.COMPLETED, TeamTaskStatus.FAILED)
                for t in self.tasks
            )
            if all_done:
                break

            # Deadlock detection
            any_in_progress = any(
                t.status == TeamTaskStatus.IN_PROGRESS for t in self.tasks
            )
            if not any_in_progress and not eligible:
                self.logger.warning(
                    "Deadlock detected: no tasks can proceed. "
                    "Failing remaining pending tasks."
                )
                for task in self.tasks:
                    if task.status == TeamTaskStatus.PENDING:
                        task.fail("Deadlock: unresolved dependencies")
                break

            # Brief sleep to avoid busy-waiting
            time.sleep(0.5)

        # Wait for all active threads to finish
        for member in self.members.values():
            member.wait(timeout=300)

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def get_progress(self) -> TeamProgress:
        """Get current progress snapshot."""
        elapsed = 0.0
        if self.start_time:
            end = self.end_time or datetime.now()
            elapsed = (end - self.start_time).total_seconds()

        return TeamProgress(
            team_id=self.team_id,
            status=self.status,
            total_tasks=len(self.tasks),
            completed_tasks=sum(
                1
                for t in self.tasks
                if t.status == TeamTaskStatus.COMPLETED
            ),
            failed_tasks=sum(
                1
                for t in self.tasks
                if t.status == TeamTaskStatus.FAILED
            ),
            in_progress_tasks=sum(
                1
                for t in self.tasks
                if t.status == TeamTaskStatus.IN_PROGRESS
            ),
            pending_tasks=sum(
                1
                for t in self.tasks
                if t.status == TeamTaskStatus.PENDING
            ),
            members=[m.info for m in self.members.values()],
            elapsed_seconds=elapsed,
        )

    def get_task(self, task_id: str) -> Optional[TeamTask]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_event(self, event: dict) -> None:
        """Emit a team-level event."""
        event["team_id"] = self.team_id
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass
