"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze and modify its own codebase,
    enabling iterative self-improvement through automated code changes.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _validate_paths(self, repo_path: str, eval_path: str) -> None:
        """Validate that the provided paths exist and are accessible.
        
        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            
        Raises:
            FileNotFoundError: If either path does not exist
            NotADirectoryError: If repo_path is not a directory
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not os.path.isdir(repo_path):
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation path does not exist: {eval_path}")

    def _extract_tool_calls(self, msg_history: list[dict]) -> list[dict]:
        """Extract tool calls from message history for summary.
        
        Args:
            msg_history: The message history from the agentic loop
            
        Returns:
            List of tool call summaries
        """
        tool_calls = []
        for msg in msg_history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_calls.append({
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    })
        return tool_calls

    def _generate_summary(
        self,
        msg_history: list[dict],
        repo_path: str,
        iterations_left: int | None,
    ) -> dict[str, Any]:
        """Generate a summary of the meta-agent run.
        
        Args:
            msg_history: The message history from the agentic loop
            repo_path: Path to the modified repository
            iterations_left: Remaining iterations
            
        Returns:
            Summary dictionary with run statistics
        """
        tool_calls = self._extract_tool_calls(msg_history)
        
        # Count tool usage
        tool_counts = {}
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1
        
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repo_path": repo_path,
            "iterations_left": iterations_left,
            "total_messages": len(msg_history),
            "total_tool_calls": len(tool_calls),
            "tool_usage": tool_counts,
            "tools_used": list(tool_counts.keys()),
        }
        
        return summary

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
            
        Raises:
            FileNotFoundError: If repository path does not exist
        """
        # Validate paths before proceeding
        self._validate_paths(repo_path, eval_path)
        
        # Build instruction with optional budget information
        instruction = f"Modify any part of the codebase at `{repo_path}`."
        if iterations_left is not None:
            instruction += f"\nRemaining iterations: {iterations_left}"

        self.log_fn(f"Starting meta-agent with model={self.model}, temp={self.temperature}")
        
        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Generate and log summary
        summary = self._generate_summary(msg_history, repo_path, iterations_left)
        self.log_fn(f"Meta-agent completed: {summary['total_messages']} messages, {summary['total_tool_calls']} tool calls")
        self.log_fn(f"Tools used: {', '.join(summary['tools_used']) if summary['tools_used'] else 'none'}")

        return msg_history
