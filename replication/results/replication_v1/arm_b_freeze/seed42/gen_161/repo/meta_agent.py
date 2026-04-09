"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Recent improvements (gen_7):
- Enhanced agentic_loop.py with tool usage statistics tracking
- Improved bash_tool.py with output size limits and buffer management
- Enhanced editor_tool.py with better error messages and line number reporting
- Added case_sensitive option to search_tool.py for more flexible searching
- Better error handling throughout the codebase

Recent improvements (gen_95):
- Added command validation to bash_tool.py to prevent dangerous operations
- Added pattern matching for dangerous commands (rm -rf /, fork bombs, etc.)
- Better security with command validation before execution

Recent improvements (gen_109):
- Added max_results parameter to search_tool.py for better output control
- Improved result reporting with total count and pagination info
- Better user experience when dealing with large search results
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

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
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            repo_path=repo_path,
        )

        return msg_history
