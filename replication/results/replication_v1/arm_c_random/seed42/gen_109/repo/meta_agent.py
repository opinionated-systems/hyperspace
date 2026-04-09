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
from typing import Any

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The MetaAgent uses LLM-powered tools to analyze, debug, and enhance
    the agent codebase. It operates in an iterative improvement loop
    guided by evaluation feedback.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._start_time: float | None = None
        self._run_count = 0
        self._modification_history: list[dict] = []

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
        self._run_count += 1
        self._start_time = time.time()
        
        logger.info(f"=" * 60)
        logger.info(f"Meta-Agent Run #{self._run_count}")
        logger.info(f"=" * 60)
        logger.info(f"Repository: {repo_path}")
        logger.info(f"Evaluation: {eval_path}")
        if iterations_left is not None:
            logger.info(f"Budget: {iterations_left} iterations remaining")
        
        # Build a more detailed instruction with context
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Record this run
        elapsed = time.time() - self._start_time
        run_record = {
            "run_id": self._run_count,
            "timestamp": time.time(),
            "repo_path": repo_path,
            "eval_path": eval_path,
            "iterations_left": iterations_left,
            "duration_seconds": elapsed,
            "messages_count": len(msg_history),
            "model": self.model,
        }
        self._modification_history.append(run_record)
        
        logger.info(f"Meta-agent run #{self._run_count} completed in {elapsed:.2f}s")
        logger.info(f"Conversation: {len(msg_history)} messages")
        logger.info(f"=" * 60)

        return msg_history
    
    def _build_instruction(
        self, 
        repo_path: str, 
        eval_path: str, 
        iterations_left: int | None
    ) -> str:
        """Build a detailed instruction for the meta-agent."""
        budget_info = ""
        if iterations_left is not None:
            budget_info = f"\nYou have {iterations_left} iterations remaining in your budget."
        
        return f"""You are a meta-agent responsible for improving an AI agent codebase.

Your task: Modify any part of the codebase at `{repo_path}` to improve its performance.

Context:
- Repository path: {repo_path}
- Evaluation results path: {eval_path}{budget_info}
- Previous runs: {self._run_count}

Guidelines for improvement:
1. First, explore the codebase structure to understand the current implementation
2. Read the evaluation results to understand what needs improvement
3. Make targeted, meaningful changes that address specific issues
4. Ensure your changes maintain backward compatibility
5. Add appropriate error handling and logging
6. Test your changes conceptually before applying them

Available tools:
- bash: Execute shell commands to explore and test
- editor: View, create, and modify files
- file: Check file existence, get sizes, list directories

Start by exploring the repository structure and understanding the current state."""
    
    def get_history(self) -> list[dict]:
        """Get the modification history."""
        return list(self._modification_history)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the meta-agent's operation."""
        if not self._modification_history:
            return {"total_runs": 0}
        
        total_duration = sum(r["duration_seconds"] for r in self._modification_history)
        total_messages = sum(r["messages_count"] for r in self._modification_history)
        
        return {
            "total_runs": self._run_count,
            "total_duration_seconds": total_duration,
            "total_messages": total_messages,
            "avg_duration_seconds": total_duration / self._run_count,
            "avg_messages_per_run": total_messages / self._run_count,
        }
