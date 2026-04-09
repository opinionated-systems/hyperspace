"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses an LLM to analyze and modify the codebase at a given
    repository path. It can be used for automated code improvement,
    refactoring, and bug fixing.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use. Defaults to META_MODEL.
            temperature: The temperature for LLM sampling. Defaults to 0.0.
        """
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
            
        Raises:
            FileNotFoundError: If the repo_path does not exist.
        """
        # Validate that the repository path exists
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        
        self.log_fn(f"Starting meta-agent on repository: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

Your goal is to improve the task agent's performance on IMO grading problems.

Key files to consider modifying:
- `{repo_path}/task_agent.py`: The main task agent that grades student solutions
- `{repo_path}/meta_agent.py`: This meta agent (if needed)

When modifying code:
1. First explore the repository structure to understand the codebase
2. Look at the evaluation results in `{eval_path}` to understand what needs improvement
3. Make targeted changes to improve accuracy
4. Ensure the code maintains the same interface and output format

Focus on:
- Improving the grading prompt for better accuracy
- Enhancing grade extraction logic
- Adding better error handling
- Improving the decision-making process"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"Meta-agent completed with {len(msg_history)} messages")

        return msg_history
