"""
Coding Agent - Main agent class that orchestrates everything.
"""

from typing import Optional, Callable
from pathlib import Path

from llm.base import BaseLLM, Message
from llm.factory import LLMFactory
from tools.registry import ToolRegistry, create_default_registry
from executor.sandbox import Sandbox, SandboxConfig, Language, ExecutionResult
from utils.config import Config
from utils.logger import Logger, get_logger

from .base import AgentState, AgentHistory, StepType
from .planner import Planner, ExecutionPlan
from .executor import AgentExecutor
from .error_analyzer import ErrorAnalyzer, AutoFixer


class CodingAgent:
    """
    Main Coding Agent - Claude Code style AI assistant.
    
    This agent combines:
    - LLM for reasoning and code generation
    - Tool system for file operations
    - Sandbox for safe code execution
    - Auto-repair loop for fixing errors
    
    Usage:
        # Simple usage
        agent = CodingAgent()
        result = agent.run("Create a Python script that calculates fibonacci numbers")
        
        # With custom configuration
        agent = CodingAgent(
            provider="deepseek",
            max_iterations=15,
            auto_fix=True,
        )
        result = agent.run(task)
        
        # Interactive mode
        agent.interactive()
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        provider: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Config] = None,
        max_iterations: int = 10,
        max_fix_attempts: int = 3,
        auto_fix: bool = True,
        workspace_dir: Optional[str] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the Coding Agent.
        
        Args:
            llm: LLM instance to use. If not provided, creates one based on provider.
            provider: LLM provider name (deepseek, openai, anthropic)
            tool_registry: Tool registry. If not provided, creates default registry.
            config: Configuration instance. If not provided, loads from environment.
            max_iterations: Maximum iterations for task execution
            max_fix_attempts: Maximum attempts to fix errors
            auto_fix: Whether to automatically fix errors
            workspace_dir: Working directory for file operations
            logger: Logger instance
        """
        # Load configuration
        self.config = config or Config.load()
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            self.llm = LLMFactory.create(provider or self.config.default_provider)
        
        # Initialize tools
        self.tools = tool_registry or create_default_registry()
        
        # Initialize components
        self.planner = Planner(self.llm)
        self.executor = AgentExecutor(
            llm=self.llm,
            tool_registry=self.tools,
            max_iterations=max_iterations,
        )
        
        # Settings
        self.max_iterations = max_iterations
        self.max_fix_attempts = max_fix_attempts
        self.auto_fix = auto_fix
        self.workspace_dir = Path(workspace_dir or self.config.agent.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.state: Optional[AgentState] = None
        self.logger = logger or get_logger(self.config.agent.log_level)
        
        # Error handling
        self.error_analyzer = ErrorAnalyzer()
        self.auto_fixer = AutoFixer(max_attempts=max_fix_attempts)
        
        # Callbacks
        self._on_step: Optional[Callable] = None
        self._on_tool_call: Optional[Callable] = None
    
    def run(
        self,
        task: str,
        context: Optional[str] = None,
        plan_first: bool = False,
    ) -> str:
        """
        Execute a task.
        
        Args:
            task: Natural language task description
            context: Additional context (file contents, etc.)
            plan_first: Whether to create a plan before executing
            
        Returns:
            Final result string
        """
        # 显示完整任务，长度超过100时才截断
        task_display = task if len(task) <= 100 else task[:100] + "..."
        self.logger.separator(f"Task: {task_display}")
        
        # Optionally create a plan first
        if plan_first:
            self.logger.agent_thinking("Creating execution plan...")
            plan = self.planner.create_plan(
                task=task,
                context=context,
                available_tools=self.tools.get_definitions(),
            )
            self._log_plan(plan)
            # Add plan to context
            plan_context = self._plan_to_context(plan)
            context = f"{context}\n\n{plan_context}" if context else plan_context
        
        # Execute the task
        self.state = self.executor.execute(
            task=task,
            context=context,
        )
        
        # Log result
        if self.state.is_complete:
            self.logger.info(f"Task completed in {self.state.iteration} iterations")
            if self.state.history:
                self.logger.info(f"Total steps: {len(self.state.history.steps)}")
        
        return self.state.final_result or "Task execution completed."
    
    def run_with_auto_fix(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Execute a task with automatic error fixing.
        
        This method will:
        1. Execute the task
        2. If code execution fails, analyze the error
        3. Attempt to fix and re-execute
        4. Repeat until success or max attempts reached
        
        Args:
            task: Natural language task description
            context: Additional context
            
        Returns:
            Final result string
        """
        # 显示完整任务，长度超过100时才截断
        task_display = task if len(task) <= 100 else task[:100] + "..."
        self.logger.separator(f"Task (auto-fix): {task_display}")
        
        # Reset auto-fixer for new task
        self.auto_fixer.reset()
        
        fix_attempts = 0
        last_error = None
        last_code = None
        
        while fix_attempts < self.max_fix_attempts:
            # Add error context if this is a retry
            run_context = context
            if last_error:
                # Use enhanced error analysis
                analysis = self.error_analyzer.analyze(last_error, last_code)
                fix_context = self.auto_fixer.get_fix_context(analysis)
                
                run_context = f"{context}\n\n{fix_context}" if context else fix_context
                
                self.logger.agent_thinking(
                    f"Attempting fix #{fix_attempts + 1}...\n"
                    f"Error type: {analysis.error_type.value}\n"
                    f"Suggestion: {analysis.suggestion}"
                )
            
            # Run the task
            result = self.run(task=task, context=run_context)
            
            # Check for execution errors
            if self.state and self.state.history:
                failed_steps = self.state.history.get_failed_steps()
                code_failures = [
                    s for s in failed_steps 
                    if s.step_type in (StepType.CODE_EXEC, StepType.TOOL_CALL)
                    and s.exit_code is not None and s.exit_code != 0
                ]
                
                if code_failures and self.auto_fix:
                    failure = code_failures[-1]
                    last_error = self._format_error(failure)
                    last_code = failure.code
                    
                    # Check if we should attempt fix
                    should_fix, analysis = self.auto_fixer.should_attempt_fix(last_error, last_code or "")
                    
                    if should_fix:
                        fix_attempts += 1
                        self.logger.warning(f"Code execution failed. Error: {last_error[:100]}...")
                        
                        # Record the attempt
                        if analysis:
                            self.auto_fixer.record_attempt(analysis, "", False)
                        continue
                    else:
                        # Cannot fix - return with explanation
                        if analysis and not analysis.fixable:
                            return (
                                f"Task failed with unfixable error: {analysis.error_type.value}\n"
                                f"{analysis.message}\n"
                                f"Suggestion: {analysis.suggestion}"
                            )
            
            # Success!
            return result
        
        # Max attempts reached
        if last_error:
            analysis = self.error_analyzer.analyze(last_error, last_code)
            return (
                f"Task failed after {fix_attempts} fix attempts.\n"
                f"Error type: {analysis.error_type.value}\n"
                f"Last error: {analysis.message}\n"
                f"Suggestion: {analysis.suggestion}"
            )
        
        return f"Task failed after {fix_attempts} fix attempts."
    
    def execute_code(
        self,
        code: str,
        language: str = "python",
        save_as: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code directly in sandbox.
        
        Args:
            code: Code to execute
            language: Programming language
            save_as: Optional file path to save the code
            
        Returns:
            ExecutionResult with stdout, stderr, exit_code
        """
        self.logger.code_block(code, language, "Executing Code")
        
        lang_map = {
            "python": Language.PYTHON,
            "javascript": Language.JAVASCRIPT,
            "bash": Language.BASH,
        }
        lang = lang_map.get(language, Language.PYTHON)
        
        config = SandboxConfig(
            timeout=self.config.agent.execution_timeout,
        )
        
        with Sandbox(config) as sandbox:
            result = sandbox.execute_code(code, language=lang)
        
        # Log result
        self.logger.execution_result(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )
        
        # Save file if requested
        if save_as:
            file_path = self.workspace_dir / save_as
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code, encoding="utf-8")
            self.logger.info(f"Saved to: {file_path}")
        
        return result
    
    def interactive(self):
        """
        Run the agent in interactive mode.
        
        This provides a REPL-like interface for continuous interaction.
        """
        self.logger.separator("FakeClaude Code - Interactive Mode")
        self.logger.info("Type 'exit' or 'quit' to exit.")
        self.logger.info("Type 'help' for available commands.")
        self.logger.separator()
        
        while True:
            try:
                user_input = self.logger.user_input("You")
                
                if not user_input.strip():
                    continue
                
                # Handle special commands
                cmd = user_input.strip().lower()
                if cmd in ("exit", "quit", "q"):
                    self.logger.info("Goodbye!")
                    break
                elif cmd == "help":
                    self._print_help()
                    continue
                elif cmd == "status":
                    self._print_status()
                    continue
                elif cmd == "history":
                    self._print_history()
                    continue
                elif cmd == "clear":
                    self.state = None
                    self.logger.info("State cleared.")
                    continue
                
                # Execute as task
                self.logger.separator()
                result = self.run_with_auto_fix(user_input)
                self.logger.separator()
                self.logger.info(result)
                
            except KeyboardInterrupt:
                self.logger.info("\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                self.logger.error(f"Error: {str(e)}")
    
    def _log_plan(self, plan: ExecutionPlan):
        """Log an execution plan."""
        self.logger.plan([s.description for s in plan.steps])
    
    def _plan_to_context(self, plan: ExecutionPlan) -> str:
        """Convert plan to context string."""
        steps = "\n".join(
            f"{i+1}. [{s.action_type}] {s.description}"
            for i, s in enumerate(plan.steps)
        )
        return f"Execution Plan:\n{steps}\n\nReasoning: {plan.reasoning}"
    
    def _format_error(self, step) -> str:
        """Format an error from a failed step."""
        parts = []
        if step.stderr:
            parts.append(f"stderr: {step.stderr}")
        if step.tool_result:
            parts.append(f"result: {step.tool_result}")
        if step.metadata.get("error"):
            parts.append(f"error: {step.metadata['error']}")
        return "\n".join(parts) or "Unknown error"
    
    def _print_help(self):
        """Print help information."""
        self.logger.info("""
Available commands:
  help     - Show this help
  status   - Show current agent status
  history  - Show execution history
  clear    - Clear current state
  exit     - Exit interactive mode

Just type your task in natural language to execute it.
""")
    
    def _print_status(self):
        """Print current agent status."""
        if self.state:
            self.logger.info(self.state.get_context_summary())
        else:
            self.logger.info("No active task.")
    
    def _print_history(self):
        """Print execution history."""
        if self.state and self.state.history:
            for i, step in enumerate(self.state.history.steps):
                status = "✓" if step.status.value == "success" else "✗"
                self.logger.info(f"  {i+1}. [{status}] {step.step_type.value}: {step.content[:50]}...")
        else:
            self.logger.info("No history available.")
    
    # Callback setters
    def on_step(self, callback: Callable):
        """Register a callback for each step."""
        self._on_step = callback
        return self
    
    def on_tool_call(self, callback: Callable):
        """Register a callback for tool calls."""
        self._on_tool_call = callback
        return self
