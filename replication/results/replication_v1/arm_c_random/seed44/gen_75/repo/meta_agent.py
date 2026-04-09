"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Enhanced with evaluation context and structured improvement guidance.
"""

from __future__ import annotations

import json
import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


def _load_evaluation_results(eval_path: str) -> dict | None:
    """Load and parse evaluation results if available."""
    if not eval_path or not os.path.exists(eval_path):
        return None
    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load evaluation results: {e}")
        return None


def _format_eval_summary(eval_results: dict | None) -> str:
    """Format evaluation results into a readable summary."""
    if not eval_results:
        return "No previous evaluation results available."
    
    parts = ["## Previous Evaluation Results"]
    
    if "accuracy" in eval_results:
        parts.append(f"- Overall Accuracy: {eval_results['accuracy']:.2%}")
    if "total" in eval_results:
        parts.append(f"- Total Samples: {eval_results['total']}")
    if "correct" in eval_results:
        parts.append(f"- Correct: {eval_results['correct']}")
    if "incorrect" in eval_results:
        parts.append(f"- Incorrect: {eval_results['incorrect']}")
    
    # Include error patterns if available
    if "errors" in eval_results and eval_results["errors"]:
        parts.append("\n### Common Error Patterns:")
        for error in eval_results["errors"][:5]:  # Top 5 errors
            parts.append(f"- {error}")
    
    return "\n".join(parts)


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
        eval_results = _load_evaluation_results(eval_path)
        eval_summary = _format_eval_summary(eval_results)
        
        # Build comprehensive instruction
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify the codebase at `{repo_path}` to improve its performance on grading tasks.

{eval_summary}

## Available Tools
- `bash`: Run shell commands (state is persistent across calls)
- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)

## Key Files to Consider
- `task_agent.py`: The main agent that solves grading problems
- `agent/llm_client.py`: LLM client wrapper
- `agent/agentic_loop.py`: Agentic loop with tool calling
- `agent/tools/`: Tool implementations

## Improvement Strategies
1. **Analyze the current implementation** - View files to understand the code
2. **Identify weaknesses** - Look for error patterns in evaluation results
3. **Make targeted improvements** - Use str_replace to modify specific parts
4. **Test your changes** - Use bash to run tests if available

## Guidelines
- Make focused, incremental improvements
- Preserve the existing interface and function signatures
- Add comments explaining your changes
- Consider adding error handling or validation

Budget: {iterations_left if iterations_left is not None else 'unlimited'} iterations remaining.

Start by exploring the codebase structure and understanding the current implementation."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
