"""
Agent Team Module - Multi-agent team collaboration system.

This module provides components for orchestrating teams of AI agents
that collaborate on complex tasks through task decomposition,
assignment, and message-based coordination.
"""

from .models import (
    TeamTaskStatus,
    TeamMessageType,
    TeamStatus,
    TeamTask,
    TeamMessage,
    TeamMemberInfo,
    TeamProgress,
)
from .message_bus import MessageBus
from .member import TeamMember
from .leader import TeamLeader
from .team import Team
from .manager import TeamManager

__all__ = [
    # Enums
    "TeamTaskStatus",
    "TeamMessageType",
    "TeamStatus",
    # Data structures
    "TeamTask",
    "TeamMessage",
    "TeamMemberInfo",
    "TeamProgress",
    # Components
    "MessageBus",
    "TeamMember",
    "TeamLeader",
    "Team",
    "TeamManager",
]
