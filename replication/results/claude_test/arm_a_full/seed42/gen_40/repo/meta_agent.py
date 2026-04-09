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

# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

_SYSTEM_CONTEXT = """\
You are a self-improving AI agent. Your goal is to improve the task agent
codebase so that it scores higher on the IMO grading benchmark.

The benchmark evaluates how accurately the task agent assigns integer scores
to student answers for International Mathematical Olympiad problems. Each
problem is scored 0–7 and the metric is exact-match accuracy.

Key files you should focus on:
  • {repo_path}/task_agent.py  — the agent that grades each problem (most
    impactful; this is what the evaluator calls directly)
  • {repo_path}/agent/llm_client.py  — LLM wrapper (model aliases, retries)
  • {repo_path}/agent/agentic_loop.py  — agentic loop used by the meta agent
  • {repo_path}/agent/tools/  — bash and editor tools

Suggested improvement directions (in rough priority order):
  1. Improve the task_agent.py prompt: add chain-of-thought, clearer grading
     rubric instructions, few-shot examples, or self-consistency voting.
  2. Add a verification step: after producing a score, ask the model to
     double-check it against the grading guidelines.
  3. Improve JSON extraction robustness so scores are never lost.
  4. Consider switching to a stronger EVAL_MODEL if available.
  5. Fix any bugs or inefficiencies you find anywhere in the codebase.

Always read the relevant files before editing them. Test your changes with
the bash tool where possible. Make sure task_agent.py remains importable and
that TaskAgent.forward() still returns (str, list[dict]).
"""

_EVAL_SUMMARY_TEMPLATE = """\
Previous evaluation results (from {eval_path}):
{summary}
"""


def _load_eval_summary(eval_path: str, max_chars: int = 3000) -> str:
    """Load and summarise the previous evaluation results file."""
    if not eval_path or not os.path.exists(eval_path):
        return "(no previous evaluation results available)"
    try:
        with open(eval_path) as f:
            content = f.read()
        # Truncate to avoid blowing the context window
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"(could not read eval results: {e})"


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
        # Build a rich, context-aware instruction.
        system_ctx = _SYSTEM_CONTEXT.format(repo_path=repo_path)

        eval_summary = _load_eval_summary(eval_path)
        eval_section = _EVAL_SUMMARY_TEMPLATE.format(
            eval_path=eval_path,
            summary=eval_summary,
        )

        budget_note = ""
        if iterations_left is not None:
            budget_note = (
                f"\nRemaining self-improvement iterations: {iterations_left}. "
                + ("Focus on the highest-impact changes." if iterations_left <= 2
                   else "You have time for thorough improvements.")
            )

        instruction = (
            f"{system_ctx}\n"
            f"{eval_section}"
            f"{budget_note}\n\n"
            f"Modify any part of the codebase at `{repo_path}`."
        )

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
