"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
"""

from __future__ import annotations

import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.bash_tool import set_allowed_root as set_bash_root
from agent.tools.editor_tool import set_allowed_root as set_editor_root
from agent.tools.search_tool import set_allowed_root as set_search_root

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _setup_tool_roots(self, repo_path: str) -> None:
        """Configure tool access restrictions to the repo path."""
        abs_path = os.path.abspath(repo_path)
        set_bash_root(abs_path)
        set_editor_root(abs_path)
        set_search_root(abs_path)
        self.log_fn(f"Tool access restricted to: {abs_path}")

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build the meta-agent instruction with context."""
        parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository to modify: `{repo_path}`",
            f"",
        ]
        
        if iterations_left is not None:
            parts.append(f"Iterations remaining: {iterations_left}")
            parts.append("")
        
        parts.extend([
            f"Available tools:",
            f"- `bash`: Run shell commands (cd, ls, grep, etc.)",
            f"- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            f"- `search`: Search for patterns in files using grep",
            f"",
            f"Your task:",
            f"1. First, explore the codebase to understand its structure",
            f"2. Identify areas for improvement (bugs, missing features, better error handling)",
            f"3. Make targeted modifications to improve the code",
            f"4. Focus on the task_agent.py which handles grading student solutions",
            f"",
            f"Key files to examine:",
            f"- task_agent.py: Main grading logic",
            f"- agent/llm_client.py: LLM communication",
            f"- agent/agentic_loop.py: Tool execution loop",
            f"- agent/tools/: Tool implementations",
            f"",
            f"Best practices:",
            f"- Use `search` to find specific code patterns quickly",
            f"- Use `editor view` to examine files before editing",
            f"- Make small, focused changes with `str_replace`",
            f"- Test your changes with `bash` if possible",
            f"",
            f"Begin by exploring the repository structure.",
        ])
        
        return "\n".join(parts)

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
        # Setup tool access restrictions
        self._setup_tool_roots(repo_path)
        
        # Build comprehensive instruction
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
