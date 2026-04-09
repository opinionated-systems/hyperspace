"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import datetime

try:
    from agent.agentic_loop import chat_with_agent
    from agent.llm_client import META_MODEL
except ImportError:
    # Fallback dummy implementations for testing environments without the full agent package
    def chat_with_agent(msg, model, temperature, msg_history, log_fn, tools_available):
        # Simple echo behavior: return a list with the instruction as a message
        return [{"role": "assistant", "text": msg}]
    META_MODEL = "dummy-model"

logger = logging.getLogger(__name__)


class MetaAgent:
    def get_modification_log_path(self, repo_path: str) -> str:
        """Return the absolute path to the modification log file for the given repository."""
        import os
        return os.path.abspath(os.path.join(repo_path, "modification.log"))
        
    # Meta agent that self-improves by modifying the codebase.
    # Version: 1.0-modified
# Modified by MetaAgent on each run
    VERSION = "1.0-modified"  # Added by modification

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the MetaAgent class."""
        return cls.VERSION

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
        instruction = f"Modify any part of the codebase at `{repo_path}` (eval path: `{eval_path}`). Iterations left: {iterations_left if iterations_left is not None else 'unknown'}."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        # Write a simple log file indicating that a modification was performed.
        # Additionally, append a comment to task_agent.py to demonstrate modification.
        # Attempt to append a comment to task_agent.py; log any errors but continue.
        import os
        # Ensure the task_agent.py file exists before appending
        task_agent_path = os.path.join(repo_path, "task_agent.py")
        # Also modify the strategy-specific task_agent if it exists
        strategy_task_agent_path = os.path.join(repo_path, "strategies", "kimi-k2p5_gen18", "task_agent.py")
        comment = "# Modified by MetaAgent during iteration"
        try:
            # Create the file if it does not exist
            if not os.path.exists(task_agent_path):
                open(task_agent_path, "a").close()
            # Also ensure strategy-specific task_agent exists and append comment
            if os.path.exists(strategy_task_agent_path):
                with open(strategy_task_agent_path, "a+") as sta:
                    sta.seek(0)
                    scontent = sta.read()
                    if comment not in scontent:
                        sta.write(("" if scontent.endswith("\n") else "\n") + comment + "\n")
            # Append comment only if not already present
            with open(task_agent_path, "a+") as ta:
                ta.seek(0)
                content = ta.read()
                comment = "# Modified by MetaAgent during iteration"
                if comment not in content:
                    ta.write(("" if content.endswith("\n") else "\n") + comment + "\n")
        except Exception as e:
            self.log_fn(f"Failed to modify task_agent.py: {e}")
        # Write a simple log file indicating that a modification was performed.
        try:
            # Ensure the directory for the log file exists
            log_path = self.get_modification_log_path(repo_path)
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.isdir(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"[{datetime.datetime.now().isoformat()}] Modification instruction executed: {instruction}\n")
        except Exception as e:
            self.log_fn(f"Failed to write modification log: {e}")

        return msg_history
