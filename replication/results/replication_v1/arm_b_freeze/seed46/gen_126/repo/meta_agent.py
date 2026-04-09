"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

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
        # Build comprehensive instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "## Context:",
        ]
        
        # Add evaluation results context if available
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()
                instruction_parts.extend([
                    f"Previous evaluation results from `{eval_path}`:",
                    "```",
                    eval_content[:2000] if len(eval_content) > 2000 else eval_content,
                    "```" if len(eval_content) > 2000 else "",
                ])
                if len(eval_content) > 2000:
                    instruction_parts.append("(truncated for brevity)")
            except Exception as e:
                self.log_fn(f"Warning: Could not read eval_path: {e}")
        
        # Add budget information
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"## Budget Information:",
                f"- Remaining iterations: {iterations_left}",
            ])
        
        # Add guidance for the meta agent
        instruction_parts.extend([
            "",
            "## Guidelines:",
            "1. First, explore the codebase to understand its structure",
            "2. Identify areas for improvement based on the evaluation results",
            "3. Make targeted, focused changes that address specific issues",
            "4. Verify your changes are syntactically correct",
            "5. Prefer small, incremental improvements over large rewrites",
            "",
            "Available tools: bash (for exploring and running commands) and editor (for viewing and modifying files).",
        ])
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with model: {self.model}")
        self.log_fn(f"Target repository: {repo_path}")
        start_time = time.time()

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            elapsed = time.time() - start_time
            self.log_fn(f"MetaAgent completed in {elapsed:.2f}s")
            
            # Log summary of changes made
            tool_calls_made = self._count_tool_calls(msg_history)
            self.log_fn(f"Tool calls made: {tool_calls_made}")
            
            return msg_history
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_fn(f"MetaAgent failed after {elapsed:.2f}s: {e}")
            raise

    def _count_tool_calls(self, msg_history: list[dict]) -> int:
        """Count the number of tool calls in the message history."""
        count = 0
        for msg in msg_history:
            if msg.get("role") == "assistant":
                # Check for tool_calls in the message
                tool_calls = msg.get("tool_calls", [])
                count += len(tool_calls)
        return count
