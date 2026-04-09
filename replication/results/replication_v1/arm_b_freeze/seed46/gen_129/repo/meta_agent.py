"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._execution_stats: dict[str, Any] = {}

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
        start_time = time.time()
        self.log_fn(f"MetaAgent starting at {datetime.now().isoformat()}")
        self.log_fn(f"Repository: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")

        # Build a more informative instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nYou have {iterations_left} iterations remaining to improve the codebase.")
        
        instruction_parts.append("\nFocus on:")
        instruction_parts.append("1. Improving error handling and robustness")
        instruction_parts.append("2. Adding input validation where missing")
        instruction_parts.append("3. Optimizing performance where possible")
        instruction_parts.append("4. Adding helpful logging and monitoring")
        instruction_parts.append("\nUse the bash and editor tools to explore and modify the code.")
        
        instruction = "\n".join(instruction_parts)

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=40,
                track_progress=True,
            )
            
            # Record execution stats with enhanced metrics
            duration = time.time() - start_time
            
            # Count tool calls from message history
            tool_calls_count = 0
            for msg in msg_history:
                if msg.get("role") == "assistant":
                    # Check for tool calls in the message
                    content = msg.get("text", "")
                    if "Tool" in content or "bash" in content or "editor" in content:
                        tool_calls_count += 1
            
            self._execution_stats = {
                "duration_seconds": duration,
                "num_messages": len(msg_history),
                "estimated_tool_calls": tool_calls_count,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "repo_path": repo_path,
                "iterations_left": iterations_left,
            }
            self.log_fn(f"MetaAgent completed in {duration:.2f}s with {len(msg_history)} messages")
            
            return msg_history
            
        except Exception as e:
            duration = time.time() - start_time
            self._execution_stats = {
                "duration_seconds": duration,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "repo_path": repo_path,
                "iterations_left": iterations_left,
            }
            self.log_fn(f"MetaAgent failed after {duration:.2f}s: {e}")
            logger.exception("MetaAgent execution failed")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics from the last run."""
        return self._execution_stats.copy()
