"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import json
import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


def _load_eval_summary(eval_path: str) -> str:
    """Load and summarize evaluation results if available."""
    if not eval_path or not os.path.exists(eval_path):
        return "No previous evaluation results available."

    try:
        with open(eval_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            accuracy = data.get("accuracy", "N/A")
            total = data.get("total", "N/A")
            correct = data.get("correct", "N/A")
            return f"Previous evaluation: {correct}/{total} correct ({accuracy:.2%} accuracy)"
        elif isinstance(data, list):
            total = len(data)
            correct = sum(1 for r in data if r.get("correct", False))
            accuracy = correct / total if total > 0 else 0
            return f"Previous evaluation: {correct}/{total} correct ({accuracy:.2%} accuracy)"
    except Exception as e:
        return f"Could not load evaluation results: {e}"


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
        eval_summary = _load_eval_summary(eval_path)
        budget_info = f"Iterations remaining: {iterations_left}" if iterations_left is not None else "Budget: unlimited"

        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify the codebase at `{repo_path}` to improve its performance on mathematical problem grading tasks.

Context:
- {eval_summary}
- {budget_info}

The codebase structure:
- task_agent.py: Main task agent that grades student answers (IMO problems)
- agent/llm_client.py: LLM client for API calls
- agent/agentic_loop.py: Agentic loop with tool calling
- agent/tools/: Editor and bash tools for file operations

Guidelines for improvement:
1. Focus on task_agent.py - this is where grading logic resides
2. Improve the prompt to be more specific about mathematical reasoning
3. Add better error handling and response normalization
4. Consider adding few-shot examples or chain-of-thought prompting
5. Ensure the response extraction is robust

Use the editor and bash tools to explore and modify files. Start by viewing the current task_agent.py to understand the existing implementation.

Modify any part of the codebase at `{repo_path}`."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
