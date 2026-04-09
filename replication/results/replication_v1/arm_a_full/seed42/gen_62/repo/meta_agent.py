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
        self.stats = {
            "modifications": 0, 
            "errors": 0,
            "files_viewed": 0,
            "files_edited": 0,
            "searches_performed": 0,
            "commands_executed": 0,
        }

    def _get_repo_structure(self, repo_path: str) -> str:
        """Get a summary of the repository structure."""
        try:
            result = []
            file_count = 0
            dir_count = 0
            for root, dirs, files in os.walk(repo_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                level = root.replace(repo_path, "").count(os.sep)
                indent = " " * 2 * level
                result.append(f"{indent}{os.path.basename(root)}/")
                dir_count += 1
                subindent = " " * 2 * (level + 1)
                for file in files:
                    if not file.endswith(".pyc"):
                        result.append(f"{subindent}{file}")
                        file_count += 1
            
            # Add summary at the top
            summary = f"Repository: {os.path.basename(repo_path)}\n"
            summary += f"Total directories: {dir_count}, Total files: {file_count}\n\n"
            return summary + "\n".join(result)
        except Exception as e:
            self.log_fn(f"Error getting repo structure: {e}")
            return "Unable to get repository structure"
    
    def _analyze_tool_usage(self, msg_history: list[dict]) -> dict:
        """Analyze tool usage from message history."""
        stats = {
            "files_viewed": 0,
            "files_edited": 0,
            "searches_performed": 0,
            "commands_executed": 0,
        }
        
        for msg in msg_history:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "Here's the result of running `cat -n`" in content:
                    stats["files_viewed"] += 1
                if "File" in content and "edited" in content:
                    stats["files_edited"] += 1
                if "Found" in content and "result(s) for pattern" in content:
                    stats["searches_performed"] += 1
                if "Tool bash" in content:
                    stats["commands_executed"] += 1
        
        return stats

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
        
        # Add iteration context if available
        iteration_context = ""
        if iterations_left is not None:
            iteration_context = f"\nIterations remaining: {iterations_left}\n"
            if iterations_left <= 3:
                iteration_context += "WARNING: Running low on iterations. Focus on high-impact changes.\n"
        
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Repository path: `{repo_path}`
Repository structure:
```
{repo_structure}
```
{iteration_context}
Your goal: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.

Guidelines:
1. First explore the codebase to understand its structure
2. Identify areas for improvement (error handling, logic, efficiency)
3. Make targeted modifications using the available tools
4. Ensure changes maintain backward compatibility
5. Add appropriate error handling and logging
6. Test your changes by verifying files compile correctly

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
            
            # Update stats from tool usage analysis
            tool_stats = self._analyze_tool_usage(msg_history)
            self.stats["files_viewed"] += tool_stats["files_viewed"]
            self.stats["files_edited"] += tool_stats["files_edited"]
            self.stats["searches_performed"] += tool_stats["searches_performed"]
            self.stats["commands_executed"] += tool_stats["commands_executed"]
            
            self.log_fn(f"Meta agent completed: {tool_stats['files_viewed']} files viewed, "
                       f"{tool_stats['files_edited']} files edited, "
                       f"{tool_stats['searches_performed']} searches, "
                       f"{tool_stats['commands_executed']} commands")
            
        except Exception as e:
            self.log_fn(f"Meta agent failed: {e}")
            self.stats["errors"] += 1
            raise

        return msg_history
    
    def get_stats(self) -> dict:
        """Return meta agent statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self.stats = {
            "modifications": 0, 
            "errors": 0,
            "files_viewed": 0,
            "files_edited": 0,
            "searches_performed": 0,
            "commands_executed": 0,
        }
