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


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _load_eval_results(self, eval_path: str) -> dict:
        """Load evaluation results from the given path."""
        if not eval_path or not os.path.exists(eval_path):
            return {}
        try:
            with open(eval_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_fn(f"Error loading eval results: {e}")
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
        # Load evaluation results for context
        eval_results = self._load_eval_results(eval_path)
        
        # Build a comprehensive instruction with context
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"## Repository Location",
            f"The codebase is located at: `{repo_path}`",
            f"",
            f"## Task",
            f"Modify any part of the codebase to improve the agent's performance on IMO (International Mathematical Olympiad) grading tasks.",
            f"",
        ]
        
        # Add evaluation context if available
        if eval_results:
            # Extract key metrics for better context
            metrics_summary = {}
            if isinstance(eval_results, dict):
                if "accuracy" in eval_results:
                    metrics_summary["accuracy"] = eval_results["accuracy"]
                if "total" in eval_results:
                    metrics_summary["total_samples"] = eval_results["total"]
                if "correct" in eval_results:
                    metrics_summary["correct"] = eval_results["correct"]
                if "errors" in eval_results and eval_results["errors"]:
                    metrics_summary["error_count"] = len(eval_results["errors"])
            
            instruction_parts.extend([
                f"## Previous Evaluation Results",
                f"```json",
                f"{json.dumps(eval_results, indent=2)[:2000]}",  # Limit size
                f"```",
                f"",
            ])
            
            if metrics_summary:
                instruction_parts.extend([
                    f"## Key Metrics",
                    f"```json",
                    f"{json.dumps(metrics_summary, indent=2)}",
                    f"```",
                    f"",
                ])
        
        if iterations_left is not None:
            instruction_parts.append(f"## Budget")
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append(f"")
        
        instruction_parts.extend([
            f"## Guidelines",
            f"1. First, explore the codebase structure using the `view` command",
            f"2. Identify areas for improvement based on the evaluation results",
            f"3. Make targeted modifications to improve performance",
            f"4. Focus on the task_agent.py which handles IMO grading",
            f"5. Ensure all changes are syntactically correct",
            f"6. Test your changes by viewing the modified files",
            f"7. Make incremental improvements - don't change too much at once",
            f"",
            f"## Common Improvement Areas",
            f"- Score extraction logic (handling edge cases, malformed JSON)",
            f"- Prompt engineering (clearer instructions, better structure)",
            f"- Error handling (graceful degradation, validation)",
            f"- Partial credit assessment (more accurate grading)",
            f"",
            f"## Available Tools",
            f"- `bash`: Run shell commands (persistent session)",
            f"- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            f"",
            f"Begin by exploring the repository structure and then make improvements.",
        ])
        
        instruction = "\n".join(instruction_parts)

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
        except Exception as e:
            self.log_fn(f"Error in meta-agent chat: {e}")
            # Return minimal history on error
            return [{"role": "user", "text": instruction}, {"role": "assistant", "text": f"Error: {e}"}]

        return msg_history
