"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools

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
        available_tools = list_available_tools()
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools: {', '.join(available_tools)}

## Tool Usage Guidelines:
- Use `search` to find files and content before making changes (search for patterns, function names, etc.)
- Use `bash` to explore the directory structure and run commands (e.g., `find`, `ls`, `wc -l`)
- Use `editor` to view, create, and modify files (view ranges with view_range, make targeted edits with str_replace)

## Workflow:
1. First, explore the codebase structure using `bash` and `editor view`
2. Identify areas for improvement (error handling, edge cases, prompt engineering, etc.)
3. Make targeted, focused changes that improve functionality
4. Verify your changes are syntactically correct

## Key Files to Consider:
- `task_agent.py`: The main grading agent (prompts, JSON extraction, grade normalization)
- `agent/agentic_loop.py`: The agentic execution loop (tool execution, error handling)
- `agent/llm_client.py`: LLM interaction layer
- `agent/tools/`: Tool implementations (editor, bash, search)

Start by exploring the codebase to understand its structure, then make targeted improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
