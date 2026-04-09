"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced instruction with self-improvement guidance.
"""

from __future__ import annotations

import logging
import json

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


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
        # Load evaluation results if available
        eval_summary = ""
        if eval_path:
            try:
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                    total = len(eval_data.get('results', []))
                    correct = sum(1 for r in eval_data.get('results', []) if r.get('correct', False))
                    accuracy = correct / total * 100 if total > 0 else 0
                    eval_summary = f"\nPrevious evaluation: {correct}/{total} correct ({accuracy:.1f}%)"
                    # Include some error examples
                    errors = [r for r in eval_data.get('results', []) if not r.get('correct', False)][:3]
                    if errors:
                        eval_summary += "\nSample errors:"
                        for e in errors:
                            eval_summary += f"\n  - Expected: {e.get('expected', '?')}, Got: {e.get('predicted', '?')}"
            except Exception as e:
                eval_summary = f"\nCould not load evaluation results: {e}"

        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify the agent code at `{repo_path}` to improve its performance on mathematical grading tasks.{eval_summary}

ITERATIONS LEFT: {iterations_left if iterations_left is not None else 'unknown'}

AVAILABLE TOOLS:
- bash: Run shell commands (state is persistent across calls)
- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)

RECOMMENDED WORKFLOW:
1. First, explore the codebase structure using the editor tool
2. Read the key files: task_agent.py, meta_agent.py, and files in agent/
3. Identify potential improvements based on the evaluation results
4. Make targeted modifications to improve performance
5. Verify your changes are syntactically correct

FOCUS AREAS FOR IMPROVEMENT:
- Better prompting strategies in task_agent.py
- More robust JSON extraction logic
- Improved error handling and edge cases
- Better handling of different grading formats
- Enhanced reasoning capabilities

Start by exploring the codebase, then make improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
