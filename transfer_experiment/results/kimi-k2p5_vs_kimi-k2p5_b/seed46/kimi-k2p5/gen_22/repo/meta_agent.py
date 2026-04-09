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
        # Build context about the current state
        context = f"""Repository path: {repo_path}
Evaluation path: {eval_path}
Iterations remaining: {iterations_left}

Your task is to improve the task_agent.py to achieve better performance on the IMO grading task.

Key improvement areas to consider:
1. Better distinction between "almost" and "partial" categories
2. Improved prompt engineering for the LLM
3. Better extraction and normalization of predictions
4. More effective use of grading guidelines hints

Use the editor and bash tools to:
1. View the current task_agent.py implementation
2. Analyze the evaluation results to understand error patterns
3. Make targeted improvements to the code
4. Verify your changes are syntactically correct

Focus on making concrete, testable improvements that address the most common error patterns."""

        instruction = f"Modify any part of the codebase at `{repo_path}`.\n\n{context}"

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
