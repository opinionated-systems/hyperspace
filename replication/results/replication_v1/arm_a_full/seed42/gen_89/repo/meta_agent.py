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
        """Get a summary of the repository structure with file sizes.
        
        Includes file sizes for better context and skips common non-source directories.
        """
        try:
            result = []
            total_size = 0
            file_count = 0
            
            # Directories to skip
            skip_dirs = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".tox", "node_modules", ".venv", "venv"}
            
            for root, dirs, files in os.walk(repo_path):
                # Skip unwanted directories
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                
                level = root.replace(repo_path, "").count(os.sep)
                indent = " " * 2 * level
                result.append(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                
                for file in files:
                    if file.endswith(".pyc") or file.startswith("."):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(file_path)
                        total_size += size
                        file_count += 1
                        # Format size nicely
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f}KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f}MB"
                        result.append(f"{subindent}{file} ({size_str})")
                    except OSError:
                        result.append(f"{subindent}{file}")
            
            # Add summary at the top
            if total_size < 1024 * 1024:
                size_summary = f"{total_size / 1024:.1f}KB"
            else:
                size_summary = f"{total_size / (1024 * 1024):.1f}MB"
            
            summary = f"Repository: {file_count} files, {size_summary} total\n"
            return summary + "\n".join(result)
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
        
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Repository path: `{repo_path}`
Repository structure:
```
{repo_structure}
```

Your goal: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.

Guidelines:
1. First explore the codebase to understand its structure
2. Identify areas for improvement (error handling, logic, efficiency)
3. Make targeted modifications using the available tools
4. Ensure changes maintain backward compatibility
5. Add appropriate error handling and logging

Available tools: bash, editor (view, create, str_replace, insert), search

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
