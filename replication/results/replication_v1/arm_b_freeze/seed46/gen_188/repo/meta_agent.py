"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.modification_count = 0
        self.session_start_time = None

    def _build_instruction(self, repo_path: str, eval_path: str, iterations_left: int | None) -> str:
        """Build a comprehensive instruction for the meta agent."""
        base_instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to bash and editor tools to make changes to the code.

Guidelines for modifications:
1. First, explore the codebase to understand its structure
2. Identify areas that need improvement based on the evaluation results
3. Make targeted, focused changes that improve the agent's performance
4. Test your changes if possible using bash commands
5. Ensure all changes are syntactically correct and maintain code quality

Remember: You are improving an AI agent that grades mathematical solutions. Focus on:
- Improving grading accuracy
- Better handling of edge cases
- More robust extraction of grades from responses
- Clearer prompting to the LLM
"""
        
        if iterations_left is not None:
            base_instruction += f"\n\nIterations remaining: {iterations_left}"
        
        if eval_path:
            base_instruction += f"\n\nEvaluation results available at: {eval_path}"
        
        return base_instruction

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
        self.session_start_time = time.time()
        self.log_fn(f"MetaAgent session started at {datetime.now().isoformat()}")
        self.log_fn(f"Repository path: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Log session summary
        duration = time.time() - self.session_start_time
        self.log_fn(f"MetaAgent session completed in {duration:.2f}s")
        self.log_fn(f"Total messages in history: {len(msg_history)}")

        return msg_history
