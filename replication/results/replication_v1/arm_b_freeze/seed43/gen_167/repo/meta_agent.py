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
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
        
        context_parts.append("\nAvailable tools:")
        context_parts.append("- bash: Run shell commands (persistent session)")
        context_parts.append("- editor: View, create, and edit files")
        context_parts.append("- search: Find patterns in files using grep")
        
        context_parts.append("\nKey files in the agent:")
        context_parts.append("- task_agent.py: The task-solving agent (main target for improvements)")
        context_parts.append("- agent/llm_client.py: LLM API client with retry logic")
        context_parts.append("- agent/agentic_loop.py: Tool execution loop")
        context_parts.append("- agent/tools/: Tool implementations (bash, editor, search)")
        
        context_parts.append("\nGuidelines:")
        context_parts.append("1. Start by exploring the codebase to understand current state")
        context_parts.append("2. Make focused, incremental improvements")
        context_parts.append("3. Test your changes with bash commands when possible")
        context_parts.append("4. Use search tool to find relevant code patterns")
        
        instruction = "\n".join(context_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
