"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build a detailed instruction for the meta agent."""
        
        # Check if eval file exists and read first part
        eval_context = ""
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    content = f.read(2000)  # First 2000 chars
                    eval_context = f"\n\nPrevious evaluation results (first 2000 chars):\n```\n{content}\n```"
            except Exception as e:
                eval_context = f"\n\nCould not read evaluation results: {e}"
        
        iterations_info = f"\nIterations remaining: {iterations_left}" if iterations_left is not None else ""
        
        return f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify the codebase at `{repo_path}` to improve its performance on mathematical grading tasks.

## Available Tools

You have access to:
- `bash`: Run shell commands (cd, ls, find, etc.)
- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- `search`: Search for files and content (find_files, grep, find_in_files)

## Guidelines

1. First, explore the codebase to understand its structure
2. Identify areas for improvement based on the evaluation results
3. Make targeted, focused changes that improve the agent's capabilities
4. Test your changes if possible
5. Document what you changed and why

## Key Files

- `task_agent.py`: Main task-solving agent (the one being evaluated)
- `agent/agentic_loop.py`: Agentic loop with tool calling
- `agent/llm_client.py`: LLM client wrapper
- `agent/tools/`: Tool implementations (bash, editor, search)

{eval_context}
{iterations_info}

## Task

Modify any part of the codebase at `{repo_path}` to improve performance. Focus on:
- Better prompting strategies
- Improved error handling
- More robust JSON extraction
- Better tool usage patterns

Start by exploring the current state of the codebase."""

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
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
