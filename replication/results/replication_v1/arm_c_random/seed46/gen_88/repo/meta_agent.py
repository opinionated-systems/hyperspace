"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

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
        # Read evaluation results if available
        eval_context = ""
        if eval_path:
            try:
                import json
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                    if isinstance(eval_data, dict):
                        accuracy = eval_data.get('accuracy', 'N/A')
                        total = eval_data.get('total', 'N/A')
                        correct = eval_data.get('correct', 'N/A')
                        eval_context = f"""
Previous Evaluation Results:
- Accuracy: {accuracy}
- Correct: {correct}/{total}

Focus on improving the task_agent.py to achieve better grading accuracy."""
            except Exception as e:
                eval_context = f"Could not read evaluation results: {e}"

        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Repository path: `{repo_path}`
{eval_context}

Your goal is to modify the codebase to improve performance on mathematical grading tasks.

Key files to consider:
- task_agent.py: The main agent that grades student answers
- agent/llm_client.py: LLM client configuration
- agent/agentic_loop.py: Agent execution loop

Instructions:
1. First, explore the codebase to understand the current implementation
2. Identify areas for improvement (prompt engineering, error handling, extraction logic)
3. Make targeted modifications to improve grading accuracy
4. Focus on the task_agent.py forward() method and JSON extraction functions

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
