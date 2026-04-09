"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced instruction with context about the codebase structure.
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
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify the codebase at `{repo_path}` to improve the agent's performance on grading tasks.

## Repository Structure:
- `task_agent.py`: The main task agent that evaluates student answers. This is the primary file to improve.
- `meta_agent.py`: The meta agent (this file) that modifies the codebase.
- `agent/`: Directory containing agent infrastructure:
  - `agentic_loop.py`: The agentic loop with tool calling
  - `llm_client.py`: LLM client wrapper
  - `tools/`: Tool implementations (bash, editor)

## Key Files to Focus On:
1. **task_agent.py**: Contains the TaskAgent class with the `forward()` method. This is where grading logic lives.
   - The `forward()` method receives inputs with: domain, problem, solution, grading_guidelines, student_answer
   - It should return a tuple of (prediction, msg_history)
   - Currently uses chain-of-thought prompting with JSON output

## Available Tools:
- `bash`: Run shell commands (persistent session)
- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)

## Instructions:
1. First, explore the codebase to understand the current implementation
2. Look at the evaluation results if available at `{eval_path}`
3. Identify areas for improvement in the task agent's reasoning or prompt engineering
4. Make targeted modifications to improve grading accuracy
5. You have {iterations_left if iterations_left is not None else 'unlimited'} iterations remaining

Start by exploring the repository structure and understanding the current implementation."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
