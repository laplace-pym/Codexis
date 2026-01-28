# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Codexis is an autonomous AI coding agent (similar to Claude Code itself) that generates, modifies, and executes code based on natural language instructions. It features a complete tool-calling system, sandbox execution, and automatic error repair.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive mode
python main.py

# Single task execution
python main.py --task "Create a fibonacci function"

# With specific LLM provider
python main.py --provider openai --task "..."
python main.py --provider anthropic

# With execution plan first
python main.py --plan --task "..."

# Verbose output
python main.py -v --task "..."

# Run demo
python demo.py
```

## Architecture

### Core Components

**CodingAgent** (`agent/coding_agent.py`): Main orchestrator that combines:
- `Planner`: Breaks tasks into execution steps
- `AgentExecutor`: Runs the LLM+tool loop until task completion
- `ErrorAnalyzer` + `AutoFixer`: Automatic error analysis and repair
- `TaskAnalyzer`: Detects task complexity to optimize execution strategy

**Execution Flow**: User task → Planner (optional) → Executor loop (LLM → Tool calls → Results) → Auto-fix on failure → Final result

### LLM Layer (`llm/`)

Factory pattern with pluggable adapters:
- `LLMFactory.create(provider)` returns configured LLM
- Supported: `deepseek` (default), `openai`, `anthropic`
- Extend by subclassing `BaseLLM` and calling `LLMFactory.register_provider()`

### Tool System (`tools/`)

Registry-based tool system:
- `ToolRegistry` holds all tools, `create_default_registry()` gives standard set
- Tool categories: file ops, code execution, document parsing, search, patch/edit, analysis
- Extend by subclassing `BaseTool` with `name`, `description`, `parameters` properties and `execute()` method

### Sandbox (`executor/sandbox.py`)

Safe code execution environment supporting Python, JavaScript, and Bash with configurable timeout.

## Configuration

Copy `env.example` to `.env` and set API keys:
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
- `DEFAULT_LLM_PROVIDER`: which provider to use by default
- `MAX_ITERATIONS`: agent loop limit (default 10)
- `EXECUTION_TIMEOUT`: code execution timeout in seconds

## Key Behaviors

The executor (`agent/executor.py`) has specific completion detection:
- Checks for completion keywords in both English and Chinese
- Auto-completes when file write + successful execution are detected together
- Simple tasks use "fast mode" with reduced iteration limits

Generated code is saved under `./code/<task_name>/` subdirectories.
