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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_log: list[dict] = []
        self._start_time: float | None = None

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str | None,
        iterations_left: int | None,
    ) -> str:
        """Build a comprehensive instruction for the meta agent."""
        parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository path: `{repo_path}`",
        ]
        
        if eval_path and os.path.exists(eval_path):
            parts.append(f"Evaluation results: `{eval_path}`")
            # Try to load and summarize evaluation results
            try:
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                if isinstance(eval_data, dict):
                    if 'accuracy' in eval_data:
                        parts.append(f"Current accuracy: {eval_data['accuracy']:.2%}")
                    if 'total' in eval_data and 'correct' in eval_data:
                        parts.append(f"Score: {eval_data['correct']}/{eval_data['total']}")
                    if 'errors' in eval_data and eval_data['errors']:
                        parts.append(f"Errors encountered: {len(eval_data['errors'])}")
            except Exception as e:
                logger.debug(f"Could not parse eval_path: {e}")
        
        if iterations_left is not None:
            parts.append(f"Iterations remaining: {iterations_left}")
            if iterations_left <= 3:
                parts.append("WARNING: Low iteration budget remaining. Focus on high-impact changes.")
        
        parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.",
            f"",
            f"Guidelines:",
            f"1. First, explore the codebase to understand its structure",
            f"2. Identify weaknesses or areas for improvement",
            f"3. Make targeted, focused changes",
            f"4. Prefer small, testable modifications over large rewrites",
            f"5. After making changes, verify the code is syntactically correct",
            f"",
            f"Available tools: bash (for exploration), editor (for modifications)",
        ])
        
        return "\n".join(parts)

    def _log_modification(self, msg_history: list[dict]) -> None:
        """Log details about the modifications made."""
        if not msg_history:
            return
        
        # Count tool calls
        tool_calls = 0
        editor_calls = 0
        bash_calls = 0
        
        for msg in msg_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Check for tool usage indicators in the content
                if "editor" in content.lower() or "str_replace" in content.lower():
                    editor_calls += 1
                if "bash" in content.lower() or "command" in content.lower():
                    bash_calls += 1
        
        modification_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "temperature": self.temperature,
            "total_messages": len(msg_history),
            "tool_calls": tool_calls,
            "editor_calls": editor_calls,
            "bash_calls": bash_calls,
            "duration_seconds": time.time() - self._start_time if self._start_time else None,
        }
        
        self._modification_log.append(modification_entry)
        self.log_fn(f"Meta-agent session complete: {modification_entry}")

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
        self._start_time = time.time()
        
        # Build comprehensive instruction
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"Starting meta-agent with model={self.model}, temp={self.temperature}")
        self.log_fn(f"Repository: {repo_path}")
        
        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            # Log the modifications
            self._log_modification(msg_history)
            
            return msg_history
            
        except Exception as e:
            logger.error(f"Meta-agent failed: {e}")
            raise

    def get_modification_summary(self) -> dict:
        """Get a summary of all modifications made by this agent."""
        if not self._modification_log:
            return {"total_sessions": 0}
        
        total_duration = sum(
            m.get("duration_seconds", 0) or 0 
            for m in self._modification_log
        )
        
        return {
            "total_sessions": len(self._modification_log),
            "total_editor_calls": sum(m.get("editor_calls", 0) for m in self._modification_log),
            "total_bash_calls": sum(m.get("bash_calls", 0) for m in self._modification_log),
            "total_duration_seconds": total_duration,
            "sessions": self._modification_log,
        }
