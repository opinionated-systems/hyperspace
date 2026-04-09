"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced with code analysis capabilities for better context.
"""

from __future__ import annotations

import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools import code_search_tool

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
        # Analyze the codebase structure to provide context
        try:
            structure_info = code_search_tool.analyze_code_structure(repo_path)
        except Exception as e:
            structure_info = f"Could not analyze codebase structure: {e}"
        
        # Check if evaluation results exist and provide context
        eval_info = ""
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()
                # Truncate if too long
                if len(eval_content) > 2000:
                    eval_info = f"\n\nPrevious evaluation results (truncated):\n{eval_content[:2000]}..."
                else:
                    eval_info = f"\n\nPrevious evaluation results:\n{eval_content}"
            except Exception as e:
                eval_info = f"\n\nCould not read evaluation results: {e}"
        
        # Build comprehensive instruction with context
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

## Codebase Structure
{structure_info}
{eval_info}

## Instructions
1. First, use the code_search tools to understand the codebase structure and find relevant code
2. Analyze the evaluation results to identify what needs improvement
3. Make targeted modifications to improve the agent's performance
4. Use the editor tool to make changes and bash tool to verify changes

Available tools:
- search_code: Search for patterns in code files
- find_function_definitions: Find function or class definitions
- analyze_code_structure: Analyze the overall codebase structure
- editor: View, create, and edit files
- bash: Run shell commands

Remember: You have {iterations_left if iterations_left else 'unlimited'} iterations remaining."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
