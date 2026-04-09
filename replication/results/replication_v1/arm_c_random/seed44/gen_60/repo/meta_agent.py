"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os

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
        # Build a more informative instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "## Available Tools",
            "- `bash`: Run commands in a bash shell (state is persistent across calls)",
            "- `editor`: File editor with commands: view, create, str_replace, insert, undo_edit",
            "",
            "## Key Files to Consider",
            "- `task_agent.py`: The main task-solving agent. This is what gets evaluated.",
            "- `meta_agent.py`: The meta agent (this file) that modifies the codebase.",
            "- `agent/llm_client.py`: LLM client with get_response_from_llm() and get_response_from_llm_with_tools()",
            "- `agent/agentic_loop.py`: Agentic loop with chat_with_agent() for tool use",
            "- `agent/tools/`: Tool implementations (bash_tool.py, editor_tool.py, registry.py)",
            "",
            "## Suggested Workflow",
            "1. Use `editor view` to explore the current codebase structure",
            "2. Identify areas for improvement (e.g., better prompting, error handling, new features)",
            "3. Use `editor str_replace` to make targeted changes",
            "4. Verify changes with `bash` commands if needed",
            "",
            "Make meaningful improvements to help the task agent perform better on its evaluation tasks.",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"")
            instruction_parts.append(f"## Budget Info")
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"")
            instruction_parts.append(f"## Previous Evaluation Results")
            instruction_parts.append(f"Available at: {eval_path}")
            instruction_parts.append(f"Consider reviewing these results to identify failure patterns.")
        
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
