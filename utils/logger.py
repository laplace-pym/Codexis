"""
Logging utilities with rich console output.
"""

import sys

# 导入 readline 启用方向键编辑和命令历史功能
# macOS 和 Linux 自带 readline，Windows 需要 pyreadline3
try:
    import readline
except ImportError:
    # 如果没有 readline，方向键可能无法使用，但不影响基本功能
    pass
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.markup import escape


class Logger:
    """
    Logger with rich formatting for the Coding Agent.
    
    Provides beautiful console output for:
    - Agent actions and thoughts
    - Tool calls and results
    - Code execution output
    - Errors and warnings
    """
    
    def __init__(self, level: str = "INFO", console: Optional[Console] = None):
        self.console = console or Console()
        self.level = level.upper()
        self._level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
        }
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on current level."""
        return self._level_priority.get(level, 0) >= self._level_priority.get(self.level, 0)
    
    def _timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S")
    
    def debug(self, message: str):
        """Log debug message."""
        if self._should_log("DEBUG"):
            self.console.print(f"[dim]{self._timestamp()}[/dim] [blue]DEBUG[/blue] {escape(message)}")
    
    def info(self, message: str):
        """Log info message."""
        if self._should_log("INFO"):
            self.console.print(f"[dim]{self._timestamp()}[/dim] [green]INFO[/green]  {escape(message)}")
    
    def warning(self, message: str):
        """Log warning message."""
        if self._should_log("WARNING"):
            self.console.print(f"[dim]{self._timestamp()}[/dim] [yellow]WARN[/yellow]  {escape(message)}")
    
    def error(self, message: str):
        """Log error message."""
        if self._should_log("ERROR"):
            self.console.print(f"[dim]{self._timestamp()}[/dim] [red]ERROR[/red] {escape(message)}")
    
    def agent_thinking(self, thought: str):
        """Display agent's thinking process."""
        panel = Panel(
            Text(thought, style="italic"),
            title="🧠 Agent Thinking",
            border_style="blue",
        )
        self.console.print(panel)
    
    def agent_action(self, action: str, details: Optional[str] = None):
        """Display agent action."""
        text = Text()
        text.append("⚡ ", style="yellow")
        text.append(action, style="bold")
        if details:
            text.append(f"\n   {details}", style="dim")
        self.console.print(text)
    
    def tool_call(self, tool_name: str, args: dict):
        """Display tool call information."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        
        for key, value in args.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."
            table.add_row(key, str_value)
        
        panel = Panel(
            table,
            title=f"🔧 Tool: {tool_name}",
            border_style="cyan",
        )
        self.console.print(panel)
    
    def tool_result(self, result: str, success: bool = True):
        """Display tool execution result."""
        style = "green" if success else "red"
        icon = "✅" if success else "❌"
        
        # Truncate very long results
        if len(result) > 500:
            result = result[:500] + "\n... (truncated)"
        
        panel = Panel(
            result,
            title=f"{icon} Result",
            border_style=style,
        )
        self.console.print(panel)
    
    def code_block(self, code: str, language: str = "python", title: str = "Generated Code"):
        """Display formatted code block."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        panel = Panel(
            syntax,
            title=f"📝 {title}",
            border_style="green",
        )
        self.console.print(panel)
    
    def execution_result(self, stdout: str, stderr: str, exit_code: int):
        """Display code execution result."""
        # Status
        if exit_code == 0:
            status = Text("✅ Success", style="bold green")
        else:
            status = Text(f"❌ Failed (exit code: {exit_code})", style="bold red")
        
        self.console.print(status)
        
        if stdout.strip():
            self.console.print(Panel(
                stdout,
                title="stdout",
                border_style="green",
            ))
        
        if stderr.strip():
            self.console.print(Panel(
                stderr,
                title="stderr",
                border_style="red",
            ))
    
    def plan(self, steps: list[str]):
        """Display execution plan."""
        table = Table(title="📋 Execution Plan", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Step", style="white")
        table.add_column("Status", width=10)
        
        for i, step in enumerate(steps, 1):
            table.add_row(str(i), step, "⏳ Pending")
        
        self.console.print(table)
    
    def step_complete(self, step_num: int, step: str, success: bool = True):
        """Mark a plan step as complete."""
        icon = "✅" if success else "❌"
        style = "green" if success else "red"
        self.console.print(f"  [{style}]{icon} Step {step_num}: {step}[/{style}]")
    
    def separator(self, title: str = ""):
        """Print a separator line."""
        if title:
            # 不使用 rule 避免中文被截断，改用简单的分隔线+标题
            self.console.print("─" * 60)
            self.console.print(f"[bold cyan]📋 {title}[/bold cyan]")
            self.console.print("─" * 60)
        else:
            self.console.rule()
    
    def user_input(self, prompt: str = "You") -> str:
        """
        Get input from user with nice formatting.
        
        Uses readline for arrow key support and proper line editing.
        Uses standard input() with prompt to avoid buffer issues with Rich.
        """
        # 不使用 Rich console，直接用 input() 带提示符
        # 这样可以避免 Rich 输出缓冲和 input 缓冲冲突导致的字符丢失
        print()  # 换行
        try:
            # 直接使用 input 的 prompt 参数，避免缓冲区问题
            return input(f"{prompt}: ")
        except EOFError:
            return "exit"


# Global logger instance
_logger: Optional[Logger] = None


def get_logger(level: str = "INFO") -> Logger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = Logger(level=level)
    return _logger
