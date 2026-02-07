"""
Team Models - Data structures for the agent team system.

Defines enums, dataclasses, and types used across team components
for task management, messaging, and progress tracking.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
from enum import Enum


class TeamTaskStatus(Enum):
    """Status of a task within a team."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TeamMessageType(Enum):
    """Types of messages exchanged between team members."""
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    CONTEXT_SHARE = "context_share"
    HELP_REQUEST = "help_request"
    BROADCAST = "broadcast"
    SHUTDOWN = "shutdown"


class TeamStatus(Enum):
    """Overall status of a team."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TeamTask:
    """
    A task assigned within a team.

    Tasks have lifecycle methods (start, complete, fail) and
    can declare dependencies on other tasks by ID.
    """
    title: str
    description: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TeamTaskStatus = TeamTaskStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def start(self, member_name: str) -> None:
        """Mark this task as started by a team member."""
        self.status = TeamTaskStatus.IN_PROGRESS
        self.assigned_to = member_name
        self.started_at = datetime.now()

    def complete(self, result: str) -> None:
        """Mark this task as successfully completed."""
        self.status = TeamTaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark this task as failed."""
        self.status = TeamTaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds from start to completion, or None if not finished."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "duration": self.duration,
        }


@dataclass
class TeamMessage:
    """
    A message exchanged between team members.
    """
    type: TeamMessageType
    sender: str
    recipient: str
    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    data: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TeamMemberInfo:
    """
    Summary information about a team member's current state.
    """
    name: str
    role: str
    provider: Optional[str] = None
    status: str = "idle"  # idle, working, done, error
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "role": self.role,
            "provider": self.provider,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        }


@dataclass
class TeamProgress:
    """
    Snapshot of overall team progress.
    """
    team_id: str
    status: TeamStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    in_progress_tasks: int
    pending_tasks: int
    members: list[TeamMemberInfo] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def completion_ratio(self) -> float:
        """Fraction of tasks completed (0.0 to 1.0)."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "team_id": self.team_id,
            "status": self.status.value,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "in_progress_tasks": self.in_progress_tasks,
            "pending_tasks": self.pending_tasks,
            "completion_ratio": self.completion_ratio,
            "members": [m.to_dict() for m in self.members],
            "elapsed_seconds": self.elapsed_seconds,
        }
