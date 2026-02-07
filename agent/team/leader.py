"""
TeamLeader - Decomposes tasks into sub-tasks and assigns them to team members.
"""

import json
import re
from typing import Optional

from llm.base import BaseLLM, Message
from llm.factory import LLMFactory
from .models import TeamTask, TeamMessage, TeamMessageType
from utils.logger import get_logger


DECOMPOSITION_PROMPT = """You are a team leader responsible for decomposing complex tasks into independent sub-tasks and assigning them to team members.

Your goal is to:
1. Analyze the given task thoroughly
2. Break it down into independent, parallelizable sub-tasks where possible
3. Assign each sub-task to the most appropriate team member based on their role
4. Specify dependencies between tasks (which tasks must complete before others can start)

Rules:
- Each sub-task should be self-contained and clearly scoped
- Minimize dependencies between tasks to maximize parallelism
- Assign tasks based on member roles and capabilities
- Use exact member names from the provided list
- Task titles should be concise but descriptive

Output your decomposition in this exact JSON format:
```json
{
    "reasoning": "Your analysis of why you broke the task down this way...",
    "tasks": [
        {
            "title": "Short descriptive title",
            "description": "Detailed description of what needs to be done",
            "assigned_to": "member_name",
            "dependencies": ["title of task this depends on"]
        }
    ]
}
```

Important:
- The "dependencies" array should contain the TITLES of other tasks that must complete first
- If a task has no dependencies, use an empty array []
- Every task must be assigned to exactly one team member using their exact name"""


class TeamLeader:
    """
    Decomposes complex tasks into sub-tasks and assigns them to team members.

    Uses an LLM to analyze tasks and create optimal work distribution
    across available team members based on their roles.
    """

    def __init__(self, llm: Optional[BaseLLM] = None, provider: Optional[str] = None):
        """
        Initialize the TeamLeader.

        Args:
            llm: Optional pre-configured LLM instance
            provider: Optional provider name (used if llm is not provided)
        """
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMFactory.create(provider)
        self.logger = get_logger()

    def decompose_task(
        self,
        task: str,
        member_names_roles: list[tuple[str, str]],
        context: Optional[str] = None,
    ) -> list[TeamTask]:
        """
        Decompose a task into sub-tasks assigned to team members.

        Args:
            task: The main task description to decompose
            member_names_roles: List of (name, role) tuples for available members
            context: Optional additional context for decomposition

        Returns:
            List of TeamTask objects with assignments and dependencies
        """
        self.logger.info(f"Decomposing task into sub-tasks for {len(member_names_roles)} members")

        members_desc = "\n".join(
            f"- {name} (Role: {role})" for name, role in member_names_roles
        )

        messages = [
            Message.system(DECOMPOSITION_PROMPT),
        ]

        user_content = f"Task: {task}\n\nAvailable team members:\n{members_desc}"
        if context:
            user_content += f"\n\nAdditional context:\n{context}"

        messages.append(Message.user(user_content))

        response = self.llm.chat_sync(
            messages=messages,
            temperature=0.3,
        )

        return self._parse_decomposition(response.content, member_names_roles)

    def _parse_decomposition(
        self,
        response: str,
        member_names_roles: list[tuple[str, str]],
    ) -> list[TeamTask]:
        """
        Parse LLM response into a list of TeamTask objects.

        Args:
            response: Raw LLM response text
            member_names_roles: List of (name, role) tuples for validation

        Returns:
            List of TeamTask objects
        """
        valid_names = {name for name, _ in member_names_roles}
        fallback_name = member_names_roles[0][0] if member_names_roles else "default"

        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                raw_tasks = data.get("tasks", [])

                tasks = []
                title_to_id = {}

                # First pass: create tasks and build title->id mapping
                for raw_task in raw_tasks:
                    title = raw_task.get("title", "Untitled task")
                    description = raw_task.get("description", "")
                    assigned_to = raw_task.get("assigned_to", fallback_name)

                    if assigned_to not in valid_names:
                        self.logger.warning(
                            f"Invalid member '{assigned_to}' for task '{title}', "
                            f"falling back to '{fallback_name}'"
                        )
                        assigned_to = fallback_name

                    team_task = TeamTask(
                        title=title,
                        description=description,
                        assigned_to=assigned_to,
                    )
                    tasks.append(team_task)
                    title_to_id[title] = team_task.id

                # Second pass: resolve dependency titles to task IDs
                for i, raw_task in enumerate(raw_tasks):
                    dep_titles = raw_task.get("dependencies", [])
                    for dep_title in dep_titles:
                        dep_id = title_to_id.get(dep_title)
                        if dep_id:
                            tasks[i].dependencies.append(dep_id)
                        else:
                            self.logger.warning(
                                f"Unknown dependency '{dep_title}' for task '{tasks[i].title}'"
                            )

                if tasks:
                    self.logger.info(f"Decomposed into {len(tasks)} sub-tasks")
                    return tasks

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self.logger.warning(f"Failed to parse decomposition JSON: {e}")

        # Fallback: return a single task with the full description
        self.logger.warning("Using fallback single-task decomposition")
        return [
            TeamTask(
                title="Complete task",
                description=response,
                assigned_to=fallback_name,
            )
        ]
