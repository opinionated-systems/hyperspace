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
        
        instruction = f"""You are a meta-agent tasked with improving an AI agent codebase at `{repo_path}`.

You have access to the following tools: {', '.join(available_tools)}

## Your Goal
Improve the task_agent.py and related files to make the agent more effective at grading student solutions. Focus on:
1. Better JSON extraction and error handling
2. More robust prompt engineering
3. Improved reasoning capabilities
4. Better handling of edge cases

## Tool Usage Guidelines

### search tool
- `find_files`: Find files matching a glob pattern (e.g., "*.py")
- `grep`: Search for text patterns in files with context lines
- `find_in_files`: Find files containing a specific pattern
- `view_file`: View a specific file with line numbers (use view_range for large files)

### bash tool
- Use for exploring directory structure
- Run tests or validation commands
- Check file existence and permissions

### editor tool
- `view`: View file or directory contents
- `create`: Create new files
- `str_replace`: Replace text (requires exact match)
- `insert`: Insert text after a specific line
- `undo_edit`: Undo the last edit to a file

## Workflow
1. First, explore the codebase structure using `search` and `bash`
2. Read the relevant files using `editor view` or `search view_file`
3. Identify areas for improvement
4. Make targeted changes using `editor str_replace` or `editor create`
5. Verify your changes work correctly

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
