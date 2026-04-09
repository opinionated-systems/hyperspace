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
        # Build a more informative instruction that includes context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path:
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            instruction_parts.append("Review these results to understand what improvements are needed.")
            instruction_parts.append("Look for patterns in errors - are they related to:")
            instruction_parts.append("  - JSON parsing failures?")
            instruction_parts.append("  - Incorrect grading decisions?")
            instruction_parts.append("  - Edge cases not handled?")
        
        # Add budget context
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                instruction_parts.append("This is the final iteration - make your most important improvements now.")
        
        instruction_parts.append("\nFocus on:")
        instruction_parts.append("1. Improving the task agent's accuracy on grading problems")
        instruction_parts.append("2. Better JSON extraction and validation (handle edge cases)")
        instruction_parts.append("3. More robust error handling and fallback strategies")
        instruction_parts.append("4. Clearer prompting for the LLM with explicit formatting rules")
        instruction_parts.append("5. Early detection of empty/invalid student answers")
        
        instruction_parts.append("\nKey files to consider modifying:")
        instruction_parts.append(f"- {repo_path}/task_agent.py: Main grading logic, JSON extraction, validation")
        instruction_parts.append(f"- {repo_path}/agent/llm_client.py: LLM interaction, retry logic")
        instruction_parts.append(f"- {repo_path}/agent/agentic_loop.py: Tool calling loop")
        
        instruction_parts.append("\nWhen making changes:")
        instruction_parts.append("1. First explore the codebase to understand current implementation")
        instruction_parts.append("2. Identify specific weaknesses from evaluation results")
        instruction_parts.append("3. Make targeted, focused improvements")
        instruction_parts.append("4. Test your changes if possible")
        instruction_parts.append("5. Ensure backward compatibility with existing interfaces")
        
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
