"""
Agent Module - Core agent architecture.

This module provides the main agent components:
- CodingAgent: Main agent class that orchestrates everything
- Planner: Creates execution plans from user tasks
- AgentExecutor: Executes plans using LLM and tools
- ErrorAnalyzer: Analyzes errors and suggests fixes
- AutoFixer: Automatic code fixing coordination
"""

from .base import AgentState, AgentStep, AgentHistory, StepType, StepStatus
from .planner import Planner, ExecutionPlan, PlanStep
from .executor import AgentExecutor
from .coding_agent import CodingAgent
from .error_analyzer import ErrorAnalyzer, ErrorAnalysis, ErrorType, AutoFixer

__all__ = [
    # State management
    "AgentState",
    "AgentStep", 
    "AgentHistory",
    "StepType",
    "StepStatus",
    # Planning
    "Planner",
    "ExecutionPlan",
    "PlanStep",
    # Execution
    "AgentExecutor",
    # Main agent
    "CodingAgent",
    # Error handling
    "ErrorAnalyzer",
    "ErrorAnalysis",
    "ErrorType",
    "AutoFixer",
]
