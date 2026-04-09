"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Enhanced with session tracking, detailed logging, and iteration awareness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


@dataclass
class MetaAgentSession:
    """Track meta-agent session metrics."""
    repo_path: str
    eval_path: str
    start_time: float = field(default_factory=time.time)
    iterations_left: int | None = None
    end_time: float = 0.0
    message_history: list[dict] = field(default_factory=list)
    success: bool = False
    
    def complete(self, msg_history: list[dict], success: bool = True) -> None:
        """Mark session as complete."""
        self.end_time = time.time()
        self.message_history = msg_history
        self.success = success
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def get_summary(self) -> dict:
        """Get session summary."""
        return {
            "repo_path": self.repo_path,
            "eval_path": self.eval_path,
            "duration_seconds": self.duration,
            "iterations_left": self.iterations_left,
            "message_count": len(self.message_history),
            "success": self.success,
        }


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._session_history: list[MetaAgentSession] = []

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
        # Create session tracking
        session = MetaAgentSession(
            repo_path=repo_path,
            eval_path=eval_path,
            iterations_left=iterations_left,
        )
        
        self.log_fn(f"Starting meta-agent session for {repo_path}")
        self.log_fn(f"Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        # Build instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"You have {iterations_left} iterations remaining.")
        
        instruction = " ".join(instruction_parts)
        self.log_fn(f"Instruction: {instruction}")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            session.complete(msg_history, success=True)
            self.log_fn(f"Meta-agent session completed successfully in {session.duration:.2f}s")
            
        except Exception as e:
            session.complete([], success=False)
            self.log_fn(f"Meta-agent session failed after {session.duration:.2f}s: {e}")
            raise
        
        finally:
            self._session_history.append(session)
            summary = session.get_summary()
            self.log_fn(f"Session summary: {summary}")

        return session.message_history
    
    def get_session_history(self) -> list[MetaAgentSession]:
        """Get history of all sessions."""
        return list(self._session_history)
    
    def get_total_sessions(self) -> int:
        """Get total number of sessions run."""
        return len(self._session_history)
    
    def get_success_rate(self) -> float:
        """Get success rate of sessions."""
        if not self._session_history:
            return 0.0
        successful = sum(1 for s in self._session_history if s.success)
        return successful / len(self._session_history)
