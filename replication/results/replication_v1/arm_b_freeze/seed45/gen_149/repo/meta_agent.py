"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)

        Returns:
            Message history from the agentic loop
        """
        available_tools = list_available_tools()
        
        # Read evaluation results if available
        eval_info = ""
        if eval_path:
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
            except Exception as e:
                eval_info = f"\n\nCould not read evaluation results: {e}\n"
        
        iteration_info = f"\nIterations remaining: {iterations_left}" if iterations_left is not None else ""
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools: {', '.join(available_tools)}

- Use `search` to find files and content before making changes
- Use `bash` to explore the directory structure and run commands
- Use `editor` to view, create, and modify files
- Use `file_info` to get detailed metadata about files (size, modification time, permissions)
- Use `test_runner` to run Python tests and validate your changes
- Use `docstring_analyzer` to analyze and improve code documentation

Guidelines for making improvements:
1. Start by exploring the codebase structure with `bash`, `file_info`, and `editor`
2. Use `code_analyzer` to identify code quality issues (unused imports, undefined variables, complexity)
3. Use `docstring_analyzer` to check documentation quality and find missing/incomplete docstrings
4. Identify areas that could benefit from improvements (error handling, validation, robustness, documentation)
5. Make targeted, focused changes that improve the code quality
6. Test your changes by running tests with `test_runner` (pytest, unittest, or discover)
7. View modified files with `editor` to verify changes
8. Ensure all changes maintain backward compatibility
{eval_info}{iteration_info}

Begin by exploring the codebase to understand its structure, then make targeted improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
