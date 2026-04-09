"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
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

        self.log_repo_files(repo_path)
        # Example usage of utils
        from utils import hello_world
        self.log_fn(f"Utility says: {hello_world()}")
        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history


    def apply_patch(self, file_path: str, patch: str) -> None:
        """Apply a simple text patch to a file.

        Args:
            file_path: Absolute path to the file to modify.
            patch: Text to append to the file. For simplicity, this method
                appends the patch at the end of the file.
        """
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n" + patch)
            self.log_fn(f"Applied patch to {file_path}")
        except Exception as e:
            self.log_fn(f"Failed to apply patch to {file_path}: {e}")

    def read_file(self, file_path: str) -> str:
        """Read and return the contents of a file.

        Args:
            file_path: Absolute path to the file.
        Returns:
            The file contents as a string, or an empty string on error.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.log_fn(f"Failed to read {file_path}: {e}")
            return ""


    def list_repo_files(self, repo_path: str) -> list[str]:
        """Return a list of all file paths under the repository directory.

        This helper can be used for debugging or inspection of the codebase.
        """
        import os
        file_paths = []
        for root, _, files in os.walk(repo_path):
            for f in files:
                file_paths.append(os.path.join(root, f))
        return file_paths


    def log_repo_files(self, repo_path: str) -> None:
        """Log all files in the repository for debugging purposes."""
        files = self.list_repo_files(repo_path)
        self.log_fn(f"Repository files ({len(files)}):")
        for f in files:
            self.log_fn(f" - {f}")


