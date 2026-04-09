"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools

logger = logging.getLogger(__name__)

# Default instruction template for the meta agent
DEFAULT_INSTRUCTION_TEMPLATE = """Modify any part of the codebase at `{repo_path}`.

You have access to the following tools: {available_tools}

- Use `search` to find files and content before making changes
- Use `bash` to explore the directory structure and run commands
- Use `editor` to view, create, and modify files

Guidelines for making improvements:
1. Start by exploring the codebase structure with `bash` and `editor`
2. Identify areas that could benefit from improvements (error handling, validation, robustness)
3. Make targeted, focused changes that improve the code quality
4. Test your changes by viewing the modified files
5. Ensure all changes maintain backward compatibility
{eval_info}{iteration_info}

Begin by exploring the codebase to understand its structure, then make targeted improvements."""


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses the agentic loop with bash, editor, and search tools
    to explore and modify the codebase. It can incorporate evaluation
    feedback to guide improvements.
    
    Attributes:
        model: The LLM model identifier to use.
        temperature: Sampling temperature for the LLM.
        log_fn: Logging function for agent activity.
        stats: Statistics about agent runs.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model identifier to use.
            temperature: Sampling temperature for the LLM (0.0 = deterministic).
        """
        self.model = model
        self.temperature = temperature
        self.log_fn: Callable = logger.info
        self.stats = {
            "total_runs": 0,
            "total_tool_calls": 0,
            "files_modified": set(),
            "files_created": set(),
        }
    
    def get_stats(self) -> dict:
        """Get statistics about meta agent activity.
        
        Returns:
            Dictionary with statistics (files counts converted to lists).
        """
        return {
            "total_runs": self.stats["total_runs"],
            "total_tool_calls": self.stats["total_tool_calls"],
            "files_modified": list(self.stats["files_modified"]),
            "files_created": list(self.stats["files_created"]),
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "total_runs": 0,
            "total_tool_calls": 0,
            "files_modified": set(),
            "files_created": set(),
        }

    def _read_eval_results(self, eval_path: str) -> str:
        """Read and format evaluation results from a file.
        
        Args:
            eval_path: Path to the evaluation results file.
            
        Returns:
            Formatted evaluation info string, or error message if reading fails.
        """
        try:
            import json
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
                if isinstance(eval_data, dict):
                    score = eval_data.get('score', 'N/A')
                    errors = eval_data.get('errors', [])
                    eval_info = f"\n\nPrevious evaluation results:\n- Score: {score}\n"
                    if errors:
                        eval_info += f"- Errors: {errors[:3]}\n"  # Show first 3 errors
                    return eval_info
                return "\n\nPrevious evaluation results: (invalid format)\n"
        except Exception as e:
            return f"\n\nCould not read evaluation results: {e}\n"

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
        max_time_seconds: float | None = 300.0,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results (can be empty string).
            iterations_left: Remaining iterations (budget info).
            max_time_seconds: Maximum time for the agentic loop (default 5 minutes).

        Returns:
            Message history from the agentic loop.
        """
        self.stats["total_runs"] += 1
        available_tools = list_available_tools()
        
        # Read evaluation results if available
        eval_info = ""
        if eval_path:
            eval_info = self._read_eval_results(eval_path)
        
        iteration_info = f"\nIterations remaining: {iterations_left}" if iterations_left is not None else ""
        
        instruction = DEFAULT_INSTRUCTION_TEMPLATE.format(
            repo_path=repo_path,
            available_tools=', '.join(available_tools),
            eval_info=eval_info,
            iteration_info=iteration_info,
        )

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            max_time_seconds=max_time_seconds,
        )
        
        # Update statistics from message history
        self._update_stats_from_history(msg_history, repo_path)

        return msg_history
    
    def _update_stats_from_history(self, msg_history: list[dict], repo_path: str) -> None:
        """Update statistics based on message history.
        
        Args:
            msg_history: The message history from the agentic loop.
            repo_path: The repository path for identifying file operations.
        """
        for msg in msg_history:
            # Count tool calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                self.stats["total_tool_calls"] += len(msg.get("tool_calls", []))
            
            # Track file modifications from tool results
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if "edited successfully" in content or "File" in content and "edited" in content:
                    # Extract filename from content
                    if "File " in content:
                        try:
                            filename = content.split("File ")[1].split(" ")[0]
                            self.stats["files_modified"].add(filename)
                        except IndexError:
                            pass
                elif "created successfully" in content:
                    # Extract filename from content
                    if "File " in content:
                        try:
                            filename = content.split("File ")[1].split(" ")[0]
                            self.stats["files_created"].add(filename)
                        except IndexError:
                            pass
