"""
Agent Base - Core data structures for agent state management.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
from enum import Enum


class StepType(Enum):
    """Type of agent step."""
    THINK = "think"           # Agent reasoning
    TOOL_CALL = "tool_call"   # Tool invocation
    TOOL_RESULT = "tool_result"  # Tool result
    CODE_GEN = "code_gen"     # Code generation
    CODE_EXEC = "code_exec"   # Code execution
    FIX = "fix"               # Error fix attempt
    COMPLETE = "complete"     # Task completion


class StepStatus(Enum):
    """Status of a step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentStep:
    """
    Represents a single step in agent execution.
    """
    step_type: StepType
    content: str
    status: StepStatus = StepStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For tool calls
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    
    # For code execution
    code: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    
    # Additional data
    metadata: dict = field(default_factory=dict)
    
    def mark_success(self, result: Optional[str] = None):
        """Mark step as successful."""
        self.status = StepStatus.SUCCESS
        if result:
            self.tool_result = result
    
    def mark_failed(self, error: str):
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.metadata["error"] = error
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.step_type.value,
            "content": self.content,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "metadata": self.metadata,
        }


@dataclass
class AgentHistory:
    """
    Maintains history of agent steps for a task.
    """
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add_step(self, step: AgentStep) -> None:
        """Add a step to history."""
        self.steps.append(step)
    
    def add_think(self, thought: str) -> AgentStep:
        """Add a thinking step."""
        step = AgentStep(
            step_type=StepType.THINK,
            content=thought,
            status=StepStatus.SUCCESS,
        )
        self.add_step(step)
        return step
    
    def add_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        description: str = "",
    ) -> AgentStep:
        """Add a tool call step."""
        step = AgentStep(
            step_type=StepType.TOOL_CALL,
            content=description or f"Calling {tool_name}",
            status=StepStatus.IN_PROGRESS,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        self.add_step(step)
        return step
    
    def add_code_execution(
        self,
        code: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ) -> AgentStep:
        """Add a code execution step."""
        step = AgentStep(
            step_type=StepType.CODE_EXEC,
            content=f"Executing code ({len(code)} chars)",
            status=StepStatus.SUCCESS if exit_code == 0 else StepStatus.FAILED,
            code=code,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )
        self.add_step(step)
        return step
    
    def get_last_step(self) -> Optional[AgentStep]:
        """Get the most recent step."""
        return self.steps[-1] if self.steps else None
    
    def get_failed_steps(self) -> list[AgentStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]
    
    def finish(self) -> None:
        """Mark history as complete."""
        self.end_time = datetime.now()
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
        }


@dataclass
class AgentState:
    """
    Current state of the agent.
    
    Tracks:
    - Current task
    - Conversation history
    - Execution history
    - Working files
    """
    task: Optional[str] = None
    history: Optional[AgentHistory] = None
    
    # Context
    working_directory: str = "."
    files_context: dict[str, str] = field(default_factory=dict)  # filename -> content
    
    # Execution state
    iteration: int = 0
    max_iterations: int = 10
    is_complete: bool = False
    final_result: Optional[str] = None
    
    # Error tracking
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
    
    def start_task(self, task: str) -> None:
        """Start a new task."""
        self.task = task
        self.history = AgentHistory(task=task)
        self.iteration = 0
        self.is_complete = False
        self.final_result = None
        self.consecutive_errors = 0
    
    def increment_iteration(self) -> bool:
        """
        Increment iteration counter.
        
        Returns:
            True if we can continue, False if max iterations reached
        """
        self.iteration += 1
        return self.iteration <= self.max_iterations
    
    def record_error(self) -> bool:
        """
        Record an error.
        
        Returns:
            True if we can continue, False if max errors reached
        """
        self.consecutive_errors += 1
        return self.consecutive_errors <= self.max_consecutive_errors
    
    def clear_errors(self) -> None:
        """Clear error counter after successful operation."""
        self.consecutive_errors = 0
    
    def complete(self, result: str) -> None:
        """Mark task as complete."""
        self.is_complete = True
        self.final_result = result
        if self.history:
            self.history.finish()
    
    def add_file_context(self, filename: str, content: str) -> None:
        """Add a file to the context."""
        self.files_context[filename] = content
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        parts = []
        
        if self.task:
            parts.append(f"Task: {self.task}")
        
        parts.append(f"Iteration: {self.iteration}/{self.max_iterations}")
        
        if self.files_context:
            files_list = ", ".join(self.files_context.keys())
            parts.append(f"Files in context: {files_list}")
        
        if self.history:
            parts.append(f"Steps completed: {len(self.history.steps)}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task": self.task,
            "working_directory": self.working_directory,
            "files_context": list(self.files_context.keys()),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "is_complete": self.is_complete,
            "final_result": self.final_result,
            "history": self.history.to_dict() if self.history else None,
        }
