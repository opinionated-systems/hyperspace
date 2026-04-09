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
    """Load and summarize evaluation results for context."""
    if not eval_path or not os.path.exists(eval_path):
        return "No previous evaluation results available."
    
    try:
        with open(eval_path, 'r') as f:
            data = json.load(f)
        
        # Extract key metrics
        total = len(data.get('results', []))
        correct = sum(1 for r in data.get('results', []) if r.get('correct', False))
        accuracy = correct / total * 100 if total > 0 else 0
        
        summary = f"Previous evaluation: {correct}/{total} correct ({accuracy:.1f}% accuracy)"
        
        # Include some error examples if available
        errors = [r for r in data.get('results', []) if not r.get('correct', False)]
        if errors:
            summary += f"\n\nSample errors ({min(3, len(errors))} of {len(errors)}):"
            for i, err in enumerate(errors[:3]):
                summary += f"\n  {i+1}. Problem: {err.get('problem_id', 'unknown')}"
                summary += f"\n     Expected: {str(err.get('expected', 'N/A'))[:100]}"
                summary += f"\n     Got: {str(err.get('prediction', 'N/A'))[:100]}"
        
        return summary
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
        # Load evaluation context
        eval_summary = _load_eval_summary(eval_path)
        
        # Build comprehensive instruction
        instruction = f"""Modify any part of the codebase at `{repo_path}` to improve the agent's performance.

{eval_summary}

ITERATIONS LEFT: {iterations_left if iterations_left is not None else 'unknown'}

YOUR TASK:
1. First, explore the codebase to understand its structure
2. Identify areas for improvement based on the evaluation results
3. Make targeted modifications to fix issues or enhance capabilities
4. Focus on the task_agent.py as this is what gets evaluated

GUIDELINES:
- Make incremental, focused changes
- Test your changes by viewing the modified files
- Consider error handling, prompt engineering, and output parsing improvements
- The agent grades mathematical olympiad problems, so accuracy is critical

Use the available tools (bash, editor) to explore and modify the codebase."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
