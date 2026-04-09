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
        self._modifications_made: list[dict[str, Any]] = []

    def _build_contextual_instruction(
        self,
        repo_path: str,
        eval_path: str | None,
        iterations_left: int | None,
    ) -> str:
        """Build a comprehensive instruction with context for the meta agent.
        
        Includes information about available files, evaluation results,
        and remaining iterations to guide the agent's modifications.
        """
        parts = [
            f"You are the meta agent for a self-improving AI system.",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve performance.",
            "",
        ]
        
        # Add repository structure context
        parts.extend([
            "## Repository Structure",
            "Key files you can modify:",
            "- task_agent.py: The main task-solving agent (primary target for improvements)",
            "- agent/llm_client.py: LLM client wrapper and API handling",
            "- agent/agentic_loop.py: Tool execution and agentic loop logic",
            "- agent/tools/: Tool implementations (bash, editor)",
            "",
        ])
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            parts.extend([
                "## Previous Evaluation Results",
                f"Review the evaluation results at: {eval_path}",
                "Analyze errors and identify patterns to fix.",
                "",
            ])
        
        # Add iteration context
        if iterations_left is not None:
            parts.extend([
                "## Budget Information",
                f"Remaining iterations: {iterations_left}",
                "Focus on high-impact improvements." if iterations_left <= 3 else "You have sufficient iterations for thorough improvements.",
                "",
            ])
        
        # Add modification guidelines
        parts.extend([
            "## Guidelines",
            "1. Use the editor tool to view files before modifying them",
            "2. Make targeted, focused changes that address specific issues",
            "3. Test your changes by viewing the modified files",
            "4. Prefer improving task_agent.py as it directly affects task performance",
            "5. Keep changes minimal and well-justified",
            "",
            "Begin by exploring the codebase to understand the current implementation.",
        ])
        
        return "\n".join(parts)

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
        self._modifications_made = []
        
        # Build comprehensive instruction with context
        instruction = self._build_contextual_instruction(
            repo_path=repo_path,
            eval_path=eval_path if eval_path and os.path.exists(eval_path) else None,
            iterations_left=iterations_left,
        )
        
        self.log_fn(f"Starting meta agent with model: {self.model}")
        self.log_fn(f"Repository path: {repo_path}")
        if eval_path:
            self.log_fn(f"Evaluation path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
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
            self.log_fn(f"Meta agent completed in {elapsed:.2f}s")
            self.log_fn(f"Total messages in history: {len(msg_history)}")
            
            # Log summary of modifications
            if self._modifications_made:
                self.log_fn(f"Modifications made: {len(self._modifications_made)}")
                for mod in self._modifications_made:
                    self.log_fn(f"  - {mod.get('file', 'unknown')}: {mod.get('description', 'no description')}")
            
            return msg_history
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_fn(f"Meta agent failed after {elapsed:.2f}s: {e}")
            logger.exception("Meta agent execution failed")
            raise

    def record_modification(self, file: str, description: str, **kwargs) -> None:
        """Record a modification made by the agent for tracking purposes.
        
        This can be called by tools to track what changes were made.
        """
        mod = {
            "file": file,
            "description": description,
            "timestamp": time.time(),
            **kwargs,
        }
        self._modifications_made.append(mod)
        self.log_fn(f"Recorded modification: {file} - {description}")
