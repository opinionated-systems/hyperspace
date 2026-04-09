"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Enhanced with evaluation feedback integration and progress tracking.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL, get_cache_stats
from agent.utils import ProgressTracker, format_duration

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_history: list[dict] = []

    def _load_eval_feedback(self, eval_path: str) -> dict | None:
        """Load evaluation feedback if available."""
        try:
            path = Path(eval_path)
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load eval feedback from {eval_path}: {e}")
        return None

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build the instruction for the meta agent."""
        parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Focus on improving the agent's capabilities, robustness, or efficiency.",
        ]

        # Add evaluation feedback if available
        eval_feedback = self._load_eval_feedback(eval_path)
        if eval_feedback:
            parts.extend([
                "",
                "Previous evaluation feedback:",
                json.dumps(eval_feedback, indent=2)[:2000],  # Limit length
            ])

        # Add budget info
        if iterations_left is not None:
            parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])

        # Add modification history context
        if self._modification_history:
            parts.extend([
                "",
                f"Previous modifications made: {len(self._modification_history)}",
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
        
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"MetaAgent starting with model={self.model}, temp={self.temperature}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            # Check if any modifications were actually made
            modifications_made = self._detect_modifications(msg_history)
            if modifications_made:
                self.log_fn(f"Modifications detected: {modifications_made}")
            else:
                self.log_fn("No modifications were made to the codebase")
                
        except Exception as e:
            logger.error(f"MetaAgent failed: {e}", exc_info=True)
            # Return minimal history on failure
            msg_history = [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error: {e}"},
            ]

        duration = time.time() - start_time
        self.log_fn(f"MetaAgent completed in {format_duration(duration)}")

        # Record this modification
        cache_stats = get_cache_stats()
        self._modification_history.append({
            "timestamp": time.time(),
            "duration": duration,
            "iterations_left": iterations_left,
            "message_count": len(msg_history),
            "cache_stats": cache_stats,
            "modifications": modifications_made if 'modifications_made' in dir() else [],
        })

        return msg_history
    
    def _detect_modifications(self, msg_history: list[dict]) -> list[str]:
        """Detect what modifications were made by analyzing the message history.
        
        Returns a list of detected modification types.
        """
        modifications = []
        
        for msg in msg_history:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
                
            # Check for common modification patterns
            if "str_replace" in content or "create" in content or "insert" in content:
                if "file" not in modifications:
                    modifications.append("file")
            if "bash" in content and any(cmd in content for cmd in ["rm", "mv", "cp", "mkdir"]):
                if "filesystem" not in modifications:
                    modifications.append("filesystem")
            if "git" in content.lower():
                if "git" not in modifications:
                    modifications.append("git")
        
        return modifications

    def get_history(self) -> list[dict]:
        """Get the modification history."""
        return list(self._modification_history)

    def reset_history(self) -> None:
        """Clear the modification history."""
        self._modification_history.clear()
