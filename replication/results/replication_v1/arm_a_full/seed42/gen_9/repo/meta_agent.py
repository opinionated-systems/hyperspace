"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from agent.agentic_loop import chat_with_agent, AgentMetrics
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


@dataclass
class MetaAgentResult:
    """Result from a meta agent run."""
    msg_history: list[dict] = field(default_factory=list)
    success: bool = False
    duration: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "duration": self.duration,
            "error": self.error,
            "num_messages": len(self.msg_history),
        }


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(
        self, 
        model: str = META_MODEL, 
        temperature: float = 0.0,
        max_tool_calls: int = 40,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.max_tool_calls = max_tool_calls

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> MetaAgentResult:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)

        Returns:
            MetaAgentResult with message history and status
        """
        start_time = time.time()
        
        # Build instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_path:
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with model={self.model}, temp={self.temperature}")
        
        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=self.max_tool_calls,
                track_metrics=True,
            )
            
            duration = time.time() - start_time
            self.log_fn(f"MetaAgent completed in {duration:.2f}s")
            
            return MetaAgentResult(
                msg_history=msg_history,
                success=True,
                duration=duration,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"MetaAgent failed after {duration:.2f}s: {e}")
            
            return MetaAgentResult(
                success=False,
                duration=duration,
                error=str(e),
            )

    def run_with_retry(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
        max_retries: int = 2,
    ) -> MetaAgentResult:
        """Run meta agent with automatic retry on failure.
        
        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)
            max_retries: maximum number of retry attempts
            
        Returns:
            MetaAgentResult from the successful run or last attempt
        """
        last_result = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.log_fn(f"Retry attempt {attempt}/{max_retries}")
                time.sleep(1.0 * attempt)  # Exponential backoff
            
            result = self.forward(repo_path, eval_path, iterations_left)
            last_result = result
            
            if result.success:
                if attempt > 0:
                    self.log_fn(f"Retry succeeded on attempt {attempt}")
                return result
        
        self.log_fn(f"All {max_retries + 1} attempts failed")
        return last_result
