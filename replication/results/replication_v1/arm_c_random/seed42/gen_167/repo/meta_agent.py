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
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools:
- bash: Execute shell commands (ls, cat, grep, etc.)
- editor: View, create, and edit files (view, str_replace, create)
- file: Check file existence and metadata
- search: Search for patterns in files and directories (regex and text search)

Key files in the codebase:
- task_agent.py: Main task agent implementation with grading logic
- agent/agentic_loop.py: Agentic loop with tool calling
- agent/llm_client.py: LLM client wrapper
- agent/tools/: Tool implementations (bash_tool, editor_tool, file_tool, search_tool)

When modifying code:
1. First explore the codebase to understand the current implementation
2. Use the search tool to find relevant code patterns
3. Make targeted improvements to fix issues or enhance functionality
4. Ensure changes maintain backward compatibility
5. Add appropriate error handling and logging

Best practices:
- Use 'search' to find all occurrences of a function or pattern before modifying
- Use 'file' to check file sizes before reading large files
- Use 'editor' with view_range to examine specific sections of code
- Test your changes by examining the modified code after edits

Iterations remaining: {iterations_left if iterations_left is not None else 'unknown'}"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
