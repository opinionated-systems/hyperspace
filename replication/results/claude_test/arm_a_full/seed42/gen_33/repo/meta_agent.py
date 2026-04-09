"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Improvements over the initial version:
- The instruction now includes the eval_path so the meta-agent can read
  previous evaluation results and make informed, targeted improvements.
- The remaining iteration budget (iterations_left) is surfaced in the
  instruction so the meta-agent can calibrate how ambitious its changes
  should be (e.g. conservative fixes late in the run vs. bold rewrites
  early on).
- A brief strategy hint is appended to guide the meta-agent toward the
  most impactful files (task_agent.py first, then supporting modules).
"""

from __future__ import annotations

import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)

_STRATEGY_HINT = """\

Strategy hints (in rough priority order):
1. `task_agent.py` — the TaskAgent.forward() prompt and JSON extraction
   logic have the most direct impact on evaluation scores.  Improve the
   system prompt, add chain-of-thought, or tighten the grading rubric.
2. `agent/agentic_loop.py` — controls the meta-agent's own tool-use loop.
   Increasing max_tool_calls or adding retry logic can help.
3. `agent/tools/` — bash_tool.py and editor_tool.py are the meta-agent's
   hands.  Fixing edge cases here improves reliability.
4. `agent/llm_client.py` — retry logic, timeout, and audit logging.

Always read the evaluation results at `{eval_path}` before making changes
so you understand which problems are being missed and why.
"""


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
        budget_note = (
            f"You have {iterations_left} iteration(s) remaining after this one."
            if iterations_left is not None
            else "This is the final iteration."
        )

        eval_note = (
            f"Previous evaluation results are at `{eval_path}`. "
            "Read them to understand current performance before making changes."
            if eval_path and os.path.exists(eval_path)
            else "No previous evaluation results are available yet."
        )

        strategy = _STRATEGY_HINT.format(eval_path=eval_path)

        instruction = (
            f"Modify any part of the codebase at `{repo_path}`.\n\n"
            f"{eval_note}\n\n"
            f"{budget_note}\n"
            f"{strategy}"
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
