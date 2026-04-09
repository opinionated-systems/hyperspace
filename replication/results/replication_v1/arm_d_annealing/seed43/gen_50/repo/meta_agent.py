"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced with better context and structured improvement guidance.
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

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build a comprehensive instruction for the meta agent."""
        
        # Get basic repo info
        try:
            py_files = []
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden and cache directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for f in files:
                    if f.endswith('.py'):
                        py_files.append(os.path.join(root, f))
            file_list = "\n".join([f"  - {os.path.relpath(f, repo_path)}" for f in py_files[:20]])
            if len(py_files) > 20:
                file_list += f"\n  ... and {len(py_files) - 20} more files"
        except Exception:
            file_list = "  (unable to list files)"
        
        budget_info = ""
        if iterations_left is not None:
            budget_info = f"\n## Budget\nYou have {iterations_left} iterations remaining. Focus on high-impact changes."
        
        eval_info = ""
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()[:2000]
                eval_info = f"\n## Previous Evaluation Results\n```\n{eval_content}\n```\nUse these results to identify areas for improvement."
            except Exception:
                eval_info = "\n## Previous Evaluation\nPrevious evaluation results are available but could not be loaded."
        
        return f"""You are a meta-agent tasked with improving an AI agent's codebase. Your goal is to make targeted improvements that enhance the agent's performance, robustness, or efficiency.

## Repository Location
`{repo_path}`

## Python Files in Repository
{file_list}

## Your Task
1. First, explore the codebase to understand its structure and current implementation
2. Identify areas for improvement based on:
   - Code quality and best practices
   - Error handling and robustness
   - Performance optimizations
   - Missing functionality or edge cases
   - Issues from previous evaluations (if available)
3. Make targeted, focused improvements
4. Verify your changes work correctly

## Guidelines for Improvements
- Focus on ONE or TWO significant improvements rather than many small changes
- Ensure changes are backward compatible where possible
- Add better error handling and logging
- Improve documentation and code clarity
- Fix any obvious bugs or issues
- Consider edge cases and input validation

## Tools Available
- `bash`: Run shell commands (persistent session)
- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)

## Best Practices
1. Always view files before editing them
2. Use `str_replace` for precise edits (requires exact old_str match)
3. Test your changes with bash commands when possible
4. Make incremental changes and verify each one
5. Keep backups of important changes using undo_edit history
{eval_info}
{budget_info}

## Response Format
When you have completed your improvements, summarize:
1. What files you modified
2. What changes you made
3. Why these changes improve the agent
4. Any testing you performed

Start by exploring the repository structure."""

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
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"[MetaAgent] Starting improvement for repo: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"[MetaAgent] Iterations remaining: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            max_tool_calls=50,
            max_iterations=150,
        )
        
        self.log_fn(f"[MetaAgent] Completed. Total messages: {len(msg_history)}")
        
        return msg_history
