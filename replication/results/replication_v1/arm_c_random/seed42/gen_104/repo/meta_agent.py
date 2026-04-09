"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

This agent is responsible for self-improvement by analyzing and modifying
the codebase to enhance performance and capabilities.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The MetaAgent uses LLM-powered tools to analyze, debug, and enhance
the agent codebase. It operates in an iterative improvement loop
guided by evaluation feedback.
    """

    def __init__(
        self,
        model: str = META_MODEL,
        temperature: float = 0.0,
        log_fn: Callable | None = None
    ) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: Model name to use for LLM calls
            temperature: Sampling temperature for LLM
            log_fn: Optional custom logging function
        """
        self.model = model
        self.temperature = temperature
        self.log_fn = log_fn or logger.info
        self._start_time: float | None = None
        self._msg_history: list[dict] = []

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
        self._msg_history = []
        
        self.log_fn(f"Starting meta-agent improvement for: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        self.log_fn(f"Using model: {self.model}")

        # Build a comprehensive instruction that includes context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to the following tools:",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- bash: Run shell commands",
            "- file: Check file existence, size, and list directories",
            "",
            "Your goal is to improve the codebase. Consider:",
            "1. Analyzing the current code structure",
            "2. Identifying areas for improvement",
            "3. Making targeted modifications",
            "4. Ensuring code quality and best practices",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nNote: You have {iterations_left} iterations remaining in the budget.")
        
        instruction = "\n".join(instruction_parts)

        try:
            self._msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=self._msg_history,
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=60,
            )
            
            elapsed = time.time() - self._start_time
            self.log_fn(f"Meta-agent completed in {elapsed:.2f} seconds")
            self.log_fn(f"Generated {len(self._msg_history)} messages in conversation")
            
        except Exception as e:
            elapsed = time.time() - self._start_time
            self.log_fn(f"Meta-agent failed after {elapsed:.2f} seconds: {e}")
            raise

        return self._msg_history

    def get_last_response(self) -> str | None:
        """Get the last assistant response from the message history.
        
        Returns:
            The last assistant message content, or None if no history
        """
        for msg in reversed(self._msg_history):
            if msg.get("role") == "assistant":
                return msg.get("content") or msg.get("text")
        return None

    def get_modification_summary(self) -> dict:
        """Get a summary of modifications made during the session.
        
        Returns:
            Dictionary with modification statistics
        """
        modifications = {
            "editor_view": 0,
            "editor_create": 0,
            "editor_str_replace": 0,
            "editor_insert": 0,
            "editor_undo_edit": 0,
            "bash": 0,
            "file": 0,
            "total_messages": len(self._msg_history),
        }
        
        for msg in self._msg_history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tool_name = tc.get("function", {}).get("name", "")
                    if tool_name == "editor":
                        try:
                            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                            command = args.get("command", "unknown")
                            key = f"editor_{command}"
                            if key in modifications:
                                modifications[key] += 1
                        except:
                            pass
                    elif tool_name in modifications:
                        modifications[tool_name] += 1
        
        return modifications
