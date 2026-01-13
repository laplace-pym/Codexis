"""
Planner - Analyzes tasks and creates execution plans.
"""

from dataclasses import dataclass
from typing import Optional
from llm.base import BaseLLM, Message, ToolDefinition


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    description: str
    action_type: str  # "tool", "code", "think", "verify"
    details: Optional[dict] = None
    completed: bool = False


@dataclass
class ExecutionPlan:
    """An execution plan for a task."""
    task: str
    steps: list[PlanStep]
    reasoning: str
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next uncompleted step."""
        for step in self.steps:
            if not step.completed:
                return step
        return None
    
    def mark_complete(self, step_index: int) -> None:
        """Mark a step as completed."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].completed = True
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.completed for step in self.steps)
    
    @property
    def progress(self) -> float:
        """Get completion progress (0-1)."""
        if not self.steps:
            return 1.0
        return sum(1 for s in self.steps if s.completed) / len(self.steps)


PLANNER_SYSTEM_PROMPT = """You are an expert task planner for a coding assistant.

Your job is to analyze the user's task and create a clear, step-by-step execution plan.

For each task, you should:
1. Understand what the user wants to achieve
2. Break it down into concrete, actionable steps
3. Identify what tools or actions are needed for each step

Available action types:
- "tool": Use a specific tool (file operations, etc.)
- "code": Write or generate code
- "think": Analyze or reason about something
- "verify": Test or verify the result

Output your plan in this exact JSON format:
```json
{
    "reasoning": "Your analysis of the task...",
    "steps": [
        {
            "description": "Step description",
            "action_type": "code|tool|think|verify",
            "details": {"key": "value"}
        }
    ]
}
```

Be specific and practical. Each step should be achievable in a single action."""


class Planner:
    """
    Planner component that analyzes tasks and creates execution plans.
    
    The planner uses an LLM to:
    1. Understand the user's intent
    2. Break down complex tasks
    3. Create structured execution plans
    """
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
    def create_plan(
        self,
        task: str,
        context: Optional[str] = None,
        available_tools: Optional[list[ToolDefinition]] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for a task.
        
        Args:
            task: The user's task description
            context: Additional context (files, previous actions, etc.)
            available_tools: List of available tools
            
        Returns:
            ExecutionPlan with steps to execute
        """
        # Build the prompt
        messages = [
            Message.system(PLANNER_SYSTEM_PROMPT),
        ]
        
        # Add context if provided
        user_content = f"Task: {task}"
        if context:
            user_content += f"\n\nContext:\n{context}"
        
        if available_tools:
            tools_desc = "\n".join(
                f"- {t.name}: {t.description}"
                for t in available_tools
            )
            user_content += f"\n\nAvailable tools:\n{tools_desc}"
        
        messages.append(Message.user(user_content))
        
        # Get plan from LLM
        response = self.llm.chat_sync(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent planning
        )
        
        # Parse the response
        return self._parse_plan(task, response.content)
    
    def _parse_plan(self, task: str, response: str) -> ExecutionPlan:
        """Parse LLM response into an ExecutionPlan."""
        import json
        import re
        
        # Try to extract JSON from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        
        if json_match:
            try:
                plan_data = json.loads(json_match.group(1))
                
                steps = []
                for step_data in plan_data.get("steps", []):
                    steps.append(PlanStep(
                        description=step_data.get("description", ""),
                        action_type=step_data.get("action_type", "think"),
                        details=step_data.get("details"),
                    ))
                
                return ExecutionPlan(
                    task=task,
                    steps=steps,
                    reasoning=plan_data.get("reasoning", ""),
                )
            except json.JSONDecodeError:
                pass
        
        # Fallback: create a simple plan
        return ExecutionPlan(
            task=task,
            steps=[
                PlanStep(
                    description="Analyze the task",
                    action_type="think",
                ),
                PlanStep(
                    description="Implement the solution",
                    action_type="code",
                ),
                PlanStep(
                    description="Verify the result",
                    action_type="verify",
                ),
            ],
            reasoning=response,
        )
    
    def refine_plan(
        self,
        plan: ExecutionPlan,
        feedback: str,
    ) -> ExecutionPlan:
        """
        Refine a plan based on execution feedback.
        
        Args:
            plan: Current execution plan
            feedback: Feedback from execution (errors, results, etc.)
            
        Returns:
            Updated ExecutionPlan
        """
        # Build refinement prompt
        current_plan = "\n".join(
            f"{i+1}. [{s.action_type}] {s.description} {'✓' if s.completed else ''}"
            for i, s in enumerate(plan.steps)
        )
        
        messages = [
            Message.system(PLANNER_SYSTEM_PROMPT),
            Message.user(f"""Original task: {plan.task}

Current plan:
{current_plan}

Feedback from execution:
{feedback}

Please update the plan based on this feedback. If steps need to be added, modified, or removed, provide the updated plan."""),
        ]
        
        response = self.llm.chat_sync(
            messages=messages,
            temperature=0.3,
        )
        
        # Parse and return refined plan
        refined = self._parse_plan(plan.task, response.content)
        
        # Preserve completed steps status
        for i, step in enumerate(refined.steps):
            if i < len(plan.steps) and plan.steps[i].completed:
                step.completed = True
        
        return refined
