"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._call_count = 0
        self._total_tool_calls = 0

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build the instruction for the meta agent."""
        parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Focus on improvements that will enhance the agent's performance.",
        ]
        
        if iterations_left is not None:
            parts.extend([
                "",
                f"Budget: {iterations_left} iterations remaining.",
            ])
        
        # Check if eval results exist and add context
        if eval_path and Path(eval_path).exists():
            parts.extend([
                "",
                f"Previous evaluation results available at: {eval_path}",
                "Consider reviewing these results to identify areas for improvement.",
            ])
        
        parts.extend([
            "",
            "Key files to consider modifying:",
            "- task_agent.py: Core grading logic and prompt engineering",
            "- agent/agentic_loop.py: Tool execution and agent loop",
            "- agent/llm_client.py: LLM interaction and caching",
            "- agent/tools/: Tool implementations",
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
        self._call_count += 1
        start_time = time.time()
        
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"MetaAgent call #{self._call_count} starting (temp={self.temperature})")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        elapsed = time.time() - start_time
        tool_calls = sum(1 for m in msg_history if m.get("role") == "assistant" and m.get("tool_calls"))
        self._total_tool_calls += tool_calls
        
        self.log_fn(
            f"MetaAgent call #{self._call_count} completed in {elapsed:.1f}s "
            f"({tool_calls} tool calls, {self._total_tool_calls} total)"
        )

        return msg_history
    
    def get_stats(self) -> dict:
        """Get meta agent statistics."""
        return {
            "calls": self._call_count,
            "total_tool_calls": self._total_tool_calls,
        }
