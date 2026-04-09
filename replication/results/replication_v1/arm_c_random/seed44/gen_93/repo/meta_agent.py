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
        # Build comprehensive instruction with context
        parts = []
        parts.append(f"You are a meta-agent tasked with improving an AI grading system.")
        parts.append(f"")
        parts.append(f"## TASK")
        parts.append(f"Modify any part of the codebase at `{repo_path}` to improve the grading agent's performance.")
        parts.append(f"")
        parts.append(f"## CODEBASE STRUCTURE")
        parts.append(f"- `task_agent.py`: The main grading agent that evaluates student answers")
        parts.append(f"- `meta_agent.py`: This meta-agent that modifies the codebase")
        parts.append(f"- `agent/llm_client.py`: LLM client for making API calls")
        parts.append(f"- `agent/agentic_loop.py`: Agentic loop with tool calling")
        parts.append(f"- `agent/tools/`: Tools for file editing and bash commands")
        parts.append(f"")
        parts.append(f"## EVALUATION")
        parts.append(f"Previous evaluation results are available at: `{eval_path}`")
        parts.append(f"Use the editor tool to view these results and understand what needs improvement.")
        parts.append(f"")
        
        if iterations_left is not None:
            parts.append(f"## BUDGET")
            parts.append(f"Iterations remaining: {iterations_left}")
            parts.append(f"")
        
        parts.append(f"## GUIDELINES")
        parts.append(f"1. First, explore the codebase to understand the current implementation")
        parts.append(f"2. Review the evaluation results to identify failure patterns")
        parts.append(f"3. Make targeted improvements to fix identified issues")
        parts.append(f"4. Focus on improving:")
        parts.append(f"   - JSON extraction reliability")
        parts.append(f"   - Grading accuracy and consistency")
        parts.append(f"   - Error handling and recovery")
        parts.append(f"   - Prompt quality and clarity")
        parts.append(f"")
        parts.append(f"5. Test your changes by viewing the modified files")
        parts.append(f"6. Ensure all changes are syntactically correct")
        parts.append(f"")
        parts.append(f"Start by exploring the repository structure and evaluation results.")
        
        instruction = "\n".join(parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
