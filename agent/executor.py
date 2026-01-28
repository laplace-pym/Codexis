"""
Agent Executor - Executes planned steps using tools and LLM.
"""

import json
import asyncio
from typing import Optional, Callable, Any
from dataclasses import dataclass

from llm.base import BaseLLM, Message, ToolCall, LLMResponse, ToolDefinition
from tools.registry import ToolRegistry
from tools.base import ToolResult
from utils.logger import get_logger
from .base import AgentState, AgentStep, StepType, StepStatus


@dataclass
class ExecutionContext:
    """Context passed during execution."""
    state: AgentState
    messages: list[Message]
    tool_results: dict[str, ToolResult]


EXECUTOR_SYSTEM_PROMPT = """You are an expert coding assistant that helps users write and execute code.

Your capabilities:
1. Generate clean, working code based on requirements
2. Use tools to read/write files, execute code, and more
3. Analyze errors and fix them automatically
4. Verify your work by testing

IMPORTANT RULES:
- For EACH new task, create a dedicated subfolder under "./code/" directory
- Folder name should be short and descriptive in English (e.g., "./code/fibonacci/", "./code/attention_mechanism/", "./code/web_crawler/")
- Save all files for the task in that folder
- Example: write_file with path "./code/my_task/main.py"
- After writing code, use execute_python tool to run and verify it
- Keep responses concise and action-oriented
- Work efficiently - complete the task in as few steps as possible

When you need to perform an action, use the appropriate tool.
When generating code, make it complete and runnable.
When you encounter an error, analyze it carefully and fix it.

CRITICAL: After completing a task, you MUST explicitly say "任务完成" or "Task complete" and briefly summarize what you did. This is required to signal that the task is finished."""


class AgentExecutor:
    """
    Executes agent actions using LLM and tools.
    
    The executor handles the main loop:
    1. Get LLM response (may include tool calls)
    2. Execute any tool calls
    3. Feed results back to LLM
    4. Repeat until task is complete
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = get_logger()
    
    def execute(
        self,
        task: str,
        state: Optional[AgentState] = None,
        context: Optional[str] = None,
    ) -> AgentState:
        """
        Execute a task to completion.
        
        Args:
            task: The task to execute
            state: Optional existing state to continue from
            context: Additional context for the LLM
            
        Returns:
            Final AgentState with results
        """
        # Initialize state
        if state is None:
            state = AgentState(max_iterations=self.max_iterations)
        state.start_task(task)
        
        # Build initial messages
        messages = self._build_initial_messages(task, context)
        
        # Get tool definitions
        tool_definitions = self.tools.get_definitions()
        
        # Main execution loop
        while not state.is_complete and state.increment_iteration():
            try:
                if self.verbose:
                    self.logger.info(f"Iteration {state.iteration}/{self.max_iterations}...")
                
                # Get LLM response
                response = self.llm.chat_sync(
                    messages=messages,
                    tools=tool_definitions,
                    temperature=0.7,
                )
                
                # Record thinking
                if response.content:
                    state.history.add_think(response.content)
                    if self.verbose:
                        self.logger.agent_thinking(response.content[:500] + "..." if len(response.content) > 500 else response.content)
                    messages.append(Message.assistant(
                        response.content,
                        tool_calls=response.tool_calls if response.has_tool_calls else None
                    ))
                
                # Handle tool calls
                if response.has_tool_calls:
                    tool_messages = self._execute_tools(response.tool_calls, state)
                    messages.extend(tool_messages)
                    state.clear_errors()
                    
                    # 🚀 优化：工具调用后检查是否提前完成
                    if state.is_complete:
                        if self.verbose:
                            self.logger.info("✅ Task auto-completed after successful execution")
                        break
                else:
                    # No tool calls - check if task is complete
                    if self._is_task_complete(response.content, state):
                        state.complete(response.content)
                    else:
                        # Ask LLM to continue
                        messages.append(Message.user(
                            "Please continue with the task. Use tools if needed, "
                            "or indicate completion if done."
                        ))
                        
            except Exception as e:
                if not state.record_error():
                    state.complete(f"Task failed after too many errors: {str(e)}")
                    break
                
                error_msg = f"Error during execution: {str(e)}"
                state.history.add_think(error_msg)
                messages.append(Message.user(
                    f"An error occurred: {str(e)}\n"
                    "Please try a different approach or fix the issue."
                ))
        
        # Check if we ran out of iterations
        if not state.is_complete:
            state.complete(
                f"Task incomplete after {self.max_iterations} iterations. "
                "Consider breaking down the task into smaller parts."
            )
        
        return state
    
    def _build_initial_messages(
        self,
        task: str,
        context: Optional[str],
    ) -> list[Message]:
        """Build the initial message list."""
        messages = [
            Message.system(EXECUTOR_SYSTEM_PROMPT),
        ]
        
        user_content = f"Task: {task}"
        if context:
            user_content += f"\n\nContext:\n{context}"
        
        messages.append(Message.user(user_content))
        return messages
    
    def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        state: AgentState,
    ) -> list[Message]:
        """
        Execute a list of tool calls.
        
        Returns:
            List of tool result messages to add to conversation
        """
        messages = []
        
        for tc in tool_calls:
            # Record tool call
            step = state.history.add_tool_call(
                tool_name=tc.name,
                tool_args=tc.arguments,
                description=f"Calling {tc.name}",
            )
            
            # Log tool call
            if self.verbose:
                self.logger.tool_call(tc.name, tc.arguments)
            
            # Execute tool
            result = self.tools.execute(tc.name, **tc.arguments)
            
            # Log result
            if self.verbose:
                self.logger.tool_result(str(result)[:300], result.success)
            
            # Update step
            if result.success:
                step.mark_success(result.output)
            else:
                step.mark_failed(result.error or "Unknown error")
            
            # Add file context if it's a read operation
            if tc.name == "read_file" and result.success:
                path = tc.arguments.get("path", "unknown")
                state.add_file_context(path, result.output)
            
            # Add code execution result to history
            if tc.name in ("execute_python", "execute_command"):
                data = result.data
                if data:
                    state.history.add_code_execution(
                        code=tc.arguments.get("code", tc.arguments.get("command", "")),
                        stdout=data.stdout if hasattr(data, "stdout") else "",
                        stderr=data.stderr if hasattr(data, "stderr") else "",
                        exit_code=data.exit_code if hasattr(data, "exit_code") else 0,
                    )
            
            # Create tool result message
            messages.append(Message.tool(
                content=str(result),
                tool_call_id=tc.id,
                name=tc.name,
            ))
        
        # 🚀 优化：检测"写文件 + 执行成功"组合，提前标记完成
        self._check_early_completion(state)
        
        return messages
    
    def _check_early_completion(self, state: AgentState) -> None:
        """
        🚀 优化：检测是否满足提前完成条件
        
        如果同时满足：
        1. 写入了文件（write_file 成功）
        2. 执行了代码且成功（execute_python/execute_in_sandbox exit_code=0）
        
        则自动标记任务完成，避免多余的后续迭代。
        """
        if state.is_complete:
            return
        
        has_write = False
        has_successful_exec = False
        
        for step in state.history.steps:
            # 检查是否有成功的写文件操作
            if step.tool_name == "write_file" and step.status.value == "success":
                has_write = True
            
            # 检查是否有成功的代码执行（沙箱验证通过）
            if step.tool_name in ("execute_python", "execute_in_sandbox", "execute_command"):
                if step.exit_code is not None and step.exit_code == 0:
                    has_successful_exec = True
        
        # 如果同时满足两个条件，提前标记完成
        if has_write and has_successful_exec:
            state.complete(
                "任务自动完成：代码已写入文件并在沙箱中验证成功。\n"
                f"最后执行输出：{state.history.steps[-1].stdout[:200] if state.history.steps else ''}"
            )
    
    def _is_task_complete(self, response: str, state: AgentState) -> bool:
        """
        Check if the LLM indicates task completion.
        
        Uses multiple strategies:
        1. Keyword matching (English and Chinese)
        2. Heuristic: code written + executed successfully = complete
        """
        # Must have done at least something
        if len(state.history.steps) < 2:
            return False
        
        # English completion indicators
        english_indicators = [
            "task is complete",
            "task complete",
            "completed successfully",
            "finished successfully",
            "done",
            "here is the result",
            "the solution is",
            "successfully created",
            "successfully written",
            "code has been saved",
        ]
        
        # Chinese completion indicators (中文完成关键词)
        chinese_indicators = [
            "任务完成",
            "已完成",
            "完成了",
            "成功创建",
            "成功生成",
            "代码已保存",
            "已经写好",
            "已经完成",
            "运行成功",
            "测试通过",
        ]
        
        response_lower = response.lower()
        
        # Check English indicators
        for indicator in english_indicators:
            if indicator in response_lower:
                return True
        
        # Check Chinese indicators (case-sensitive for Chinese)
        for indicator in chinese_indicators:
            if indicator in response:
                return True
        
        # Heuristic: If we've written a file AND executed code successfully, consider complete
        has_write = False
        has_successful_exec = False
        
        for step in state.history.steps:
            if step.tool_name == "write_file" and step.status.value == "success":
                has_write = True
            if step.tool_name in ("execute_python", "execute_command") and step.exit_code == 0:
                has_successful_exec = True
        
        if has_write and has_successful_exec:
            return True
        
        return False
    
    def execute_single_step(
        self,
        messages: list[Message],
        state: AgentState,
    ) -> tuple[LLMResponse, list[Message]]:
        """
        Execute a single step (for more fine-grained control).
        
        Returns:
            Tuple of (LLM response, new messages to add)
        """
        tool_definitions = self.tools.get_definitions()
        
        response = self.llm.chat_sync(
            messages=messages,
            tools=tool_definitions,
        )
        
        new_messages = []
        
        if response.content:
            new_messages.append(Message.assistant(
                response.content,
                tool_calls=response.tool_calls if response.has_tool_calls else None
            ))
        
        if response.has_tool_calls:
            tool_messages = self._execute_tools(response.tool_calls, state)
            new_messages.extend(tool_messages)
        
        return response, new_messages


    def execute_with_callback(
        self,
        task: str,
        callback: Callable[[dict], None],
        state: Optional[AgentState] = None,
        context: Optional[str] = None,
    ) -> AgentState:
        """
        Execute a task with real-time callbacks for progress updates.
        
        Args:
            task: The task to execute
            callback: Function called with event dicts for each progress update
            state: Optional existing state to continue from
            context: Additional context for the LLM
            
        Returns:
            Final AgentState with results
        """
        # Initialize state
        if state is None:
            state = AgentState(max_iterations=self.max_iterations)
        state.start_task(task)
        
        # Build initial messages
        messages = self._build_initial_messages(task, context)
        
        # Get tool definitions
        tool_definitions = self.tools.get_definitions()
        
        # Emit iteration event
        callback({
            "type": "iteration",
            "content": f"Starting task execution...",
            "metadata": {"iteration": 0, "max_iterations": self.max_iterations}
        })
        
        # Main execution loop
        while not state.is_complete and state.increment_iteration():
            try:
                # Emit iteration event
                callback({
                    "type": "iteration",
                    "content": f"Iteration {state.iteration}/{self.max_iterations}",
                    "metadata": {"iteration": state.iteration, "max_iterations": self.max_iterations}
                })
                
                if self.verbose:
                    self.logger.info(f"Iteration {state.iteration}/{self.max_iterations}...")
                
                # Get LLM response
                response = self.llm.chat_sync(
                    messages=messages,
                    tools=tool_definitions,
                    temperature=0.7,
                )
                
                # Record and emit thinking
                if response.content:
                    state.history.add_think(response.content)
                    thinking_content = response.content[:500] + "..." if len(response.content) > 500 else response.content
                    if self.verbose:
                        self.logger.agent_thinking(thinking_content)
                    
                    callback({
                        "type": "thinking",
                        "content": thinking_content,
                    })
                    
                    messages.append(Message.assistant(
                        response.content,
                        tool_calls=response.tool_calls if response.has_tool_calls else None
                    ))
                
                # Handle tool calls
                if response.has_tool_calls:
                    tool_messages = self._execute_tools_with_callback(
                        response.tool_calls, state, callback
                    )
                    messages.extend(tool_messages)
                    state.clear_errors()
                    
                    if state.is_complete:
                        if self.verbose:
                            self.logger.info("✅ Task auto-completed after successful execution")
                        callback({
                            "type": "status",
                            "content": "Task auto-completed after successful execution",
                        })
                        break
                else:
                    # No tool calls - check if task is complete
                    if self._is_task_complete(response.content, state):
                        state.complete(response.content)
                    else:
                        messages.append(Message.user(
                            "Please continue with the task. Use tools if needed, "
                            "or indicate completion if done."
                        ))
                        
            except Exception as e:
                error_msg = str(e)
                callback({
                    "type": "error",
                    "content": f"Error: {error_msg}",
                })
                
                if not state.record_error():
                    state.complete(f"Task failed after too many errors: {error_msg}")
                    break
                
                state.history.add_think(f"Error during execution: {error_msg}")
                messages.append(Message.user(
                    f"An error occurred: {error_msg}\n"
                    "Please try a different approach or fix the issue."
                ))
        
        # Check if we ran out of iterations
        if not state.is_complete:
            state.complete(
                f"Task incomplete after {self.max_iterations} iterations. "
                "Consider breaking down the task into smaller parts."
            )
        
        return state

    def _execute_tools_with_callback(
        self,
        tool_calls: list[ToolCall],
        state: AgentState,
        callback: Callable[[dict], None],
    ) -> list[Message]:
        """
        Execute tools with callback for progress updates.
        """
        messages = []
        
        for tc in tool_calls:
            # Record tool call
            step = state.history.add_tool_call(
                tool_name=tc.name,
                tool_args=tc.arguments,
                description=f"Calling {tc.name}",
            )
            
            # Emit tool_call event
            callback({
                "type": "tool_call",
                "content": f"Calling {tc.name}",
                "metadata": {
                    "tool": tc.name,
                    "args": tc.arguments,
                }
            })
            
            # Log tool call
            if self.verbose:
                self.logger.tool_call(tc.name, tc.arguments)
            
            # Execute tool
            result = self.tools.execute(tc.name, **tc.arguments)
            
            # Log result
            if self.verbose:
                self.logger.tool_result(str(result)[:300], result.success)
            
            # Update step
            if result.success:
                step.mark_success(result.output)
            else:
                step.mark_failed(result.error or "Unknown error")
            
            # Emit tool_result event
            result_preview = str(result)[:500] if len(str(result)) > 500 else str(result)
            callback({
                "type": "tool_result",
                "content": result_preview,
                "metadata": {
                    "success": result.success,
                    "tool": tc.name,
                }
            })
            
            # Add file context if it's a read operation
            if tc.name == "read_file" and result.success:
                path = tc.arguments.get("path", "unknown")
                state.add_file_context(path, result.output)
            
            # Add code execution result to history
            if tc.name in ("execute_python", "execute_command"):
                data = result.data
                if data:
                    state.history.add_code_execution(
                        code=tc.arguments.get("code", tc.arguments.get("command", "")),
                        stdout=data.stdout if hasattr(data, "stdout") else "",
                        stderr=data.stderr if hasattr(data, "stderr") else "",
                        exit_code=data.exit_code if hasattr(data, "exit_code") else 0,
                    )
            
            # Create tool result message
            messages.append(Message.tool(
                content=str(result),
                tool_call_id=tc.id,
                name=tc.name,
            ))
        
        # Check early completion
        self._check_early_completion(state)
        
        return messages
