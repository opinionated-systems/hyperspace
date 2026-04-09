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


def _get_codebase_summary(repo_path: str) -> str:
    """Generate a summary of the codebase structure.
    
    This helps the meta agent understand what files are available
    and their purposes before making modifications.
    """
    summary_parts = ["\n=== Codebase Structure ===\n"]
    
    try:
        # List all Python files
        for root, dirs, files in os.walk(repo_path):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            level = root.replace(repo_path, '').count(os.sep)
            indent = '  ' * level
            rel_path = os.path.relpath(root, repo_path)
            if rel_path == '.':
                rel_path = 'root'
            summary_parts.append(f"{indent}{rel_path}/")
            
            subindent = '  ' * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            # Get docstring if available
                            docstring = ""
                            for line in lines[:10]:
                                if '"""' in line or "'''" in line:
                                    docstring = line.strip().strip('"').strip("'")
                                    break
                            summary_parts.append(f"{subindent}- {file} ({len(lines)} lines) {docstring[:50]}")
                    except Exception:
                        summary_parts.append(f"{subindent}- {file}")
    except Exception as e:
        summary_parts.append(f"Error scanning codebase: {e}")
    
    summary_parts.append("\n=== End Codebase Structure ===\n")
    return "\n".join(summary_parts)


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
        # Generate codebase summary for better context
        codebase_summary = _get_codebase_summary(repo_path)
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

{codebase_summary}

You have access to bash and editor tools to make modifications. Focus on improving:
1. The task_agent.py - which handles IMO grading tasks
2. The agent/ directory - which contains the agentic loop and tools

Use the editor tool to view files before modifying them, and bash to explore the structure."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
