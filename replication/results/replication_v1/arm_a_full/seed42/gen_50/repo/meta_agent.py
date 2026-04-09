"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.stats = {"modifications": 0, "errors": 0}

    def _get_repo_structure(self, repo_path: str) -> str:
        """Get a summary of the repository structure."""
        try:
            result = []
            for root, dirs, files in os.walk(repo_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                level = root.replace(repo_path, "").count(os.sep)
                indent = " " * 2 * level
                result.append(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files:
                    if not file.endswith(".pyc"):
                        result.append(f"{subindent}{file}")
            return "\n".join(result)
        except Exception as e:
            self.log_fn(f"Error getting repo structure: {e}")
            return "Unable to get repository structure"

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
        # Get repository structure for context
        repo_structure = self._get_repo_structure(repo_path)
        
        # Build iteration context
        iteration_context = ""
        if iterations_left is not None:
            iteration_context = f"\nIterations remaining: {iterations_left}"
        
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Repository path: `{repo_path}`
Repository structure:
```
{repo_structure}
```{iteration_context}

Your goal: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.

## Suggested Workflow:
1. **Explore**: Use `search` and `editor` (view) to understand the codebase structure
2. **Analyze**: Use `code_analysis` tool to identify potential issues in Python files
3. **Identify**: Look for areas to improve:
   - Error handling gaps
   - Missing input validation
   - Performance bottlenecks
   - Code clarity and documentation
   - Edge cases not handled
4. **Plan**: Decide on specific, targeted improvements
5. **Implement**: Use `editor` (str_replace, insert) to make changes
6. **Verify**: Use `bash` to run tests or validate changes

## Guidelines:
- Make focused, incremental improvements
- Maintain backward compatibility
- Add appropriate error handling and logging
- Follow existing code style and patterns
- Test your changes if possible

## Available Tools:
- `bash`: Run shell commands (tests, git, etc.)
- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- `search`: Find files by name or content
- `code_analysis`: Analyze Python code for issues and improvements

Begin by exploring the repository structure and understanding the current implementation."""

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.stats["modifications"] += 1
        except Exception as e:
            self.log_fn(f"Meta agent failed: {e}")
            self.stats["errors"] += 1
            raise

        return msg_history
    
    def get_stats(self) -> dict:
        """Return meta agent statistics."""
        return self.stats.copy()
