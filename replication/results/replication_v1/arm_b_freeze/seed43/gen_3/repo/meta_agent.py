"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Enhanced with structured self-improvement guidance and evaluation feedback integration.
"""

from __future__ import annotations

import json
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

    def _load_eval_results(self, eval_path: str) -> dict:
        """Load evaluation results if available."""
        if not eval_path or not os.path.exists(eval_path):
            return {}
        try:
            with open(eval_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_fn(f"Could not load eval results: {e}")
            return {}

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
        # Load previous evaluation results for context
        eval_results = self._load_eval_results(eval_path)
        
        # Build structured instruction
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository path: `{repo_path}`",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
        if eval_results:
            instruction_parts.extend([
                f"",
                f"Previous evaluation results:",
                f"```json",
                f"{json.dumps(eval_results, indent=2)[:2000]}",
                f"```",
            ])
        
        instruction_parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.",
            f"",
            f"Guidelines for improvement:",
            f"1. First, explore the codebase structure using the editor tool",
            f"2. Identify weaknesses in the current implementation",
            f"3. Make targeted improvements to task_agent.py (the main agent logic)",
            f"4. Consider improving:",
            f"   - Prompt engineering and instructions",
            f"   - Error handling and edge cases",
            f"   - JSON extraction logic",
            f"   - Reasoning capabilities",
            f"5. Test your changes by viewing the modified files",
            f"6. Make incremental, focused changes rather than large rewrites",
            f"",
            f"Available tools: bash (for exploring/running commands) and editor (for viewing/modifying files)",
        ])
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
