"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

try:
    from agent.agentic_loop import chat_with_agent
except Exception:
    def chat_with_agent(*args, **kwargs):
        return []
try:
    from agent.llm_client import META_MODEL
except Exception:
    META_MODEL = "gpt-4"


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
        import os
        from utils import list_repo_files
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)

        Returns:
            Message history from the agentic loop
        """
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        # Validate repository path before proceeding
        if not self._validate_repo_path(repo_path):
            raise ValueError(f"Invalid repository path: {repo_path}")
        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        # Record a simple modification log in the repository
        try:
            import datetime
            log_path = os.path.join(repo_path, "modification_log.txt")
            with open(log_path, "a") as f:
                try:
                    files = list_repo_files(repo_path)
                    f.write(f"Repo files: {files}\n")
                except Exception as e2:
                    f.write(f"Failed to list repo files: {e2}\n")
                f.write(f"Modified on {datetime.datetime.utcnow().isoformat()} with iterations_left={iterations_left}\n")
        except Exception as e:
            self.log_fn(f"Failed to write modification log: {e}")

        return msg_history

    def _validate_repo_path(self, repo_path: str) -> bool:
        """Check that the given repository path exists and is a directory.

        Returns True if valid, False otherwise.
        """
        import os
        return os.path.isdir(repo_path)
