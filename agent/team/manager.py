"""
TeamManager - Factory for creating and managing team instances.

Follows the same pattern as SessionManager in the web layer.
"""

from typing import Optional, Callable

from .team import Team
from .models import TeamProgress


class TeamManager:
    """
    Creates and manages Team instances.

    Usage:
        manager = TeamManager()
        team = manager.create_default_team(provider="deepseek")
        progress = team.execute("Build a REST API with tests")
    """

    def __init__(self):
        self._teams: dict[str, Team] = {}

    def create_team(
        self,
        members_config: list[dict],
        provider: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> Team:
        """
        Create a team with the given member configuration.

        Args:
            members_config: List of dicts with "name", optional "role", optional "provider"
            provider: Default LLM provider for all members
            on_event: Callback for streaming events

        Returns:
            Configured Team instance
        """
        team = Team(provider=provider, on_event=on_event)

        for mc in members_config:
            team.add_member(
                name=mc["name"],
                role=mc.get("role", "general"),
                provider=mc.get("provider", provider),
            )

        self._teams[team.team_id] = team
        return team

    def create_default_team(
        self,
        provider: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> Team:
        """
        Create a default team with standard roles.

        Default members:
        - architect: designs the solution structure
        - developer: implements the code
        - tester: writes and runs tests
        """
        return self.create_team(
            members_config=[
                {"name": "architect", "role": "architect"},
                {"name": "developer", "role": "developer"},
                {"name": "tester", "role": "tester"},
            ],
            provider=provider,
            on_event=on_event,
        )

    def get_team(self, team_id: str) -> Optional[Team]:
        """Get a team by ID."""
        return self._teams.get(team_id)

    def delete_team(self, team_id: str) -> bool:
        """Delete a team and shut down its members."""
        team = self._teams.pop(team_id, None)
        if team:
            team.message_bus.broadcast_shutdown()
            return True
        return False

    def list_teams(self) -> list[TeamProgress]:
        """List all teams with their current progress."""
        return [team.get_progress() for team in self._teams.values()]
