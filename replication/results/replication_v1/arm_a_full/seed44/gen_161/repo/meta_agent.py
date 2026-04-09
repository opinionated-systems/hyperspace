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
        # Build context-aware instruction with iteration info
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
        
        if eval_path:
            context_parts.append(f"Evaluation results available at: {eval_path}")
        
        instruction = f"""{chr(10).join(context_parts)}

You have access to the following tools:
- bash: Run shell commands (state is persistent across calls)
- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- search: Search for files and content (grep, find)
- stats: Get statistics about files and directories

RECOMMENDED WORKFLOW:
1. First, use `editor view` or `stats` to understand the codebase structure
2. Use `search grep` to find relevant code patterns
3. Use `editor view` with line ranges to examine specific code sections
4. Use `editor str_replace` to make targeted changes
5. Verify changes with `editor view` after editing

IMPORTANT GUIDELINES:
- Always use absolute paths with the editor tool
- Use `search` to find relevant code before making changes
- Make focused, incremental improvements
- Test your changes by viewing the modified files
- When using str_replace, ensure old_str matches exactly and is unique

The codebase is an IMO grading agent. Key files to consider:
- task_agent.py: The main grading logic (most important to improve)
- agent/llm_client.py: LLM interaction layer
- agent/agentic_loop.py: Agent execution loop
- agent/tools/: Various tool implementations
"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            tools_root=repo_path,
        )

        return msg_history
