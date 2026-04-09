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
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "The codebase structure is:",
            "- task_agent.py: Main task agent that grades student solutions",
            "- agent/llm_client.py: LLM client wrapper",
            "- agent/agentic_loop.py: Agentic loop with tool calling",
            "- agent/tools/: Tool implementations (bash, editor)",
            "",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append("")
        
        # Add evaluation context if available
        if eval_path:
            instruction_parts.append(f"Previous evaluation results are at: {eval_path}")
            instruction_parts.append("You can use the bash tool to view these results and understand what went wrong.")
            instruction_parts.append("")
        
        instruction_parts.append("Focus on improving the task_agent.py grading logic:")
        instruction_parts.append("1. Better JSON extraction from LLM responses")
        instruction_parts.append("2. More robust error handling and retries")
        instruction_parts.append("3. Improved prompting for grading accuracy")
        instruction_parts.append("4. Better handling of edge cases in student answers")
        instruction_parts.append("")
        instruction_parts.append("Use the editor and bash tools to explore and modify the code.")
        instruction_parts.append("Start by viewing the current task_agent.py to understand its structure.")
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
