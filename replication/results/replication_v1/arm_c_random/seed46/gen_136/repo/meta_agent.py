"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.utils import format_duration

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_count = 0
        self._start_time: float | None = None

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
        self._start_time = time.time()
        
        # Build a more detailed instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Available tools:",
            "- bash: Run shell commands (state is persistent)",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- search: Find files and search for patterns (grep, find)",
        ]
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])
        
        if eval_path:
            instruction_parts.extend([
                "",
                f"Previous evaluation results available at: {eval_path}",
                "Consider reviewing these results to identify areas for improvement.",
            ])
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"Starting meta-agent with model: {self.model}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Count modifications made
        self._modification_count = self._count_modifications(msg_history)
        
        elapsed = time.time() - self._start_time
        self.log_fn(f"Meta-agent completed in {format_duration(elapsed)}")
        self.log_fn(f"Modifications made: {self._modification_count}")

        return msg_history
    
    def _count_modifications(self, msg_history: list[dict]) -> int:
        """Count the number of file modifications in the message history."""
        count = 0
        for msg in msg_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Check for tool calls that modify files
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "")
                    if name == "editor":
                        try:
                            import json
                            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                            cmd = args.get("command", "")
                            if cmd in ("create", "str_replace", "insert"):
                                count += 1
                        except Exception:
                            pass
        return count
    
    def get_stats(self) -> dict:
        """Get statistics about the meta-agent run."""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
        return {
            "modifications": self._modification_count,
            "elapsed_seconds": elapsed,
            "model": self.model,
            "temperature": self.temperature,
        }
