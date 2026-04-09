"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Enhanced with evaluation feedback integration and progress tracking.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools, get_tool_info
from agent.utils import ProgressTracker, format_duration

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_history: list[dict] = []

    def _load_eval_feedback(self, eval_path: str) -> dict | None:
        """Load evaluation feedback if available."""
        try:
            path = Path(eval_path)
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load eval feedback from {eval_path}: {e}")
        return None

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
            "You have access to bash, editor, and code_analysis tools to make changes.",
            "Focus on improving the agent's capabilities, robustness, or efficiency.",
        ]

        # Add available tools info
        available_tools = list_available_tools()
        parts.extend([
            "",
            "Available tools:",
        ])
        for tool_name in available_tools:
            tool_info = get_tool_info(tool_name)
            if tool_info:
                parts.append(f"  - {tool_name}: {tool_info.get('description', 'No description')[:100]}")

        # Add evaluation feedback if available
        eval_feedback = self._load_eval_feedback(eval_path)
        if eval_feedback:
            parts.extend([
                "",
                "Previous evaluation feedback:",
                json.dumps(eval_feedback, indent=2)[:2000],  # Limit length
            ])

        # Add budget info
        if iterations_left is not None:
            parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])

        # Add modification history context
        if self._modification_history:
            parts.extend([
                "",
                f"Previous modifications made: {len(self._modification_history)}",
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
        start_time = time.time()
        
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"MetaAgent starting with model={self.model}, temp={self.temperature}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        duration = time.time() - start_time
        self.log_fn(f"MetaAgent completed in {format_duration(duration)}")

        # Record this modification
        self._modification_history.append({
            "timestamp": time.time(),
            "duration": duration,
            "iterations_left": iterations_left,
            "message_count": len(msg_history),
        })

        return msg_history

    def get_history(self) -> list[dict]:
        """Get the modification history."""
        return list(self._modification_history)

    def reset_history(self) -> None:
        """Clear the modification history."""
        self._modification_history.clear()
