"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced with evaluation context for better self-improvement.
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

    def _load_eval_results(self, eval_path: str) -> dict | None:
        """Load and parse evaluation results if available."""
        if not eval_path or not os.path.exists(eval_path):
            return None
        
        try:
            with open(eval_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.log_fn(f"Could not load eval results from {eval_path}: {e}")
            return None

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None,
    ) -> str:
        """Build comprehensive instruction for the meta agent."""
        
        # Load evaluation context
        eval_results = self._load_eval_results(eval_path)
        
        parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository path: `{repo_path}`",
        ]
        
        # Add iteration context
        if iterations_left is not None:
            parts.append(f"Iterations remaining: {iterations_left}")
        
        # Add evaluation context if available
        if eval_results:
            parts.extend([
                f"",
                f"Previous evaluation results:",
            ])
            
            # Extract key metrics
            if isinstance(eval_results, dict):
                if 'accuracy' in eval_results:
                    parts.append(f"- Accuracy: {eval_results['accuracy']:.2%}")
                if 'total_samples' in eval_results:
                    parts.append(f"- Total samples: {eval_results['total_samples']}")
                if 'correct' in eval_results and 'incorrect' in eval_results:
                    parts.append(f"- Correct: {eval_results['correct']}, Incorrect: {eval_results['incorrect']}")
                
                # Add error patterns if available
                if 'error_patterns' in eval_results and eval_results['error_patterns']:
                    parts.append(f"- Common error patterns: {', '.join(eval_results['error_patterns'][:3])}")
        
        # Add guidance
        parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve performance.",
            f"",
            f"Guidelines:",
            f"1. First, explore the codebase to understand its structure",
            f"2. Identify areas for improvement based on the evaluation results",
            f"3. Make targeted, focused changes that address specific issues",
            f"4. Ensure changes maintain code quality and follow best practices",
            f"5. Test your changes if possible using the bash tool",
            f"",
            f"Focus on improving:",
            f"- Error handling and robustness",
            f"- JSON parsing and validation",
            f"- Input formatting and processing",
            f"- Overall reliability and accuracy",
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
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"MetaAgent starting with instruction length: {len(instruction)}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
