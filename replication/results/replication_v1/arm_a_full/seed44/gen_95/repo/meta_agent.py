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
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Repository path: `{repo_path}`
Evaluation results path: `{eval_path}`
Iterations remaining: {iterations_left if iterations_left is not None else 'unknown'}

Your goal is to modify the codebase to improve the agent's performance on IMO grading tasks.

Key files to consider:
- task_agent.py: The main task agent that grades IMO problems. Contains:
  - _extract_jsons(): JSON extraction from LLM responses
  - _FEW_SHOT_EXAMPLES: Training examples for grading
  - TaskAgent class with _build_prompt(), _validate_score(), _normalize_score(), _extract_prediction()
  - forward(): Main entry point for grading

- agent/llm_client.py: LLM client with get_response_from_llm() and get_response_from_llm_with_tools()
- agent/agentic_loop.py: Agentic loop with tool calling
- agent/tools/: Editor and bash tools for file operations

Available tools:
- editor: view, create, str_replace, insert, undo_edit
- bash: run shell commands

Strategy suggestions:
1. First, view the evaluation results to understand current performance
2. View the task_agent.py to understand the current implementation
3. Identify weaknesses (e.g., JSON extraction, prompt quality, score validation)
4. Make targeted improvements using str_replace
5. Test changes with bash commands if needed

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
