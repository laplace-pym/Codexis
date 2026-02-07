#!/usr/bin/env python3
"""
FakeClaude Code - AI Coding Agent

A Claude Code-style AI coding assistant that can:
- Generate and modify code based on natural language
- Execute code and automatically fix errors
- Use tools to interact with the file system
- Work in an interactive REPL mode

Usage:
    # Interactive mode
    python main.py
    
    # Single task
    python main.py --task "Create a fibonacci function"
    
    # With specific provider
    python main.py --provider openai --task "..."
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from utils.logger import get_logger
from agent import CodingAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FakeClaude Code - AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # Interactive mode
    python main.py --task "Create hello world"        # Single task
    python main.py --provider openai --task "..."     # Use OpenAI
    python main.py --provider anthropic               # Use Claude
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task to execute (if not provided, starts interactive mode)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["deepseek", "openai", "anthropic"],
        help="LLM provider to use (default: from .env or deepseek)"
    )
    
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=10,
        help="Maximum iterations for task execution (default: 10)"
    )
    
    parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="Disable automatic error fixing"
    )
    
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        help="Working directory for file operations"
    )
    
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Create execution plan before running task"
    )

    parser.add_argument(
        "--team",
        action="store_true",
        help="Use team mode with multiple agents working in parallel"
    )

    parser.add_argument(
        "--team-size",
        type=int,
        default=3,
        help="Number of team members (default: 3)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="FakeClaude Code v0.1.0"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = Config.load()
    
    # Set log level
    log_level = "DEBUG" if args.verbose else config.agent.log_level
    logger = get_logger(log_level)
    
    # Print banner
    logger.separator("🤖 Codexis!为您服务！pym真帅")
    logger.info(f"Provider: {args.provider or config.default_provider}")
    
    try:
        # Create agent
        agent = CodingAgent(
            provider=args.provider,
            config=config,
            max_iterations=args.max_iterations,
            auto_fix=not args.no_auto_fix,
            workspace_dir=args.workspace,
            logger=logger,
        )
        
        if args.task:
            # Single task mode
            logger.info(f"Task: {args.task}")
            logger.separator()

            if args.team:
                from agent.team import TeamManager
                logger.separator("Team Mode")
                logger.info(f"Creating team with default members...")
                manager = TeamManager()
                team = manager.create_default_team(provider=args.provider or config.default_provider)
                progress = team.execute(task=args.task)
                logger.separator("Team Result")
                logger.info(f"Status: {progress.status.value}")
                logger.info(f"Tasks: {progress.completed_tasks}/{progress.total_tasks} completed")
                logger.info(f"Failed: {progress.failed_tasks}")
                logger.info(f"Duration: {progress.elapsed_seconds:.1f}s")
                for member in progress.members:
                    logger.info(f"  {member.name} ({member.role}): {member.tasks_completed} done, {member.tasks_failed} failed")
                for task in team.tasks:
                    icon = "+" if task.status.value == "completed" else "x"
                    logger.info(f"  [{icon}] {task.title}: {(task.result or task.error or '')[:100]}")
            elif args.plan:
                result = agent.run(args.task, plan_first=True)
                logger.separator("Result")
                print(result)
            else:
                result = agent.run_with_auto_fix(args.task)
                logger.separator("Result")
                print(result)
        else:
            # Interactive mode
            agent.interactive()
            
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
