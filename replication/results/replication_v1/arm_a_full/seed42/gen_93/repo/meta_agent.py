"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.stats = {"modifications": 0, "errors": 0}

    def _get_repo_structure(self, repo_path: str) -> str:
        """Get a summary of the repository structure."""
        try:
            result = []
            for root, dirs, files in os.walk(repo_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                level = root.replace(repo_path, "").count(os.sep)
                indent = " " * 2 * level
                result.append(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files:
                    if not file.endswith(".pyc"):
                        result.append(f"{subindent}{file}")
            return "\n".join(result)
        except Exception as e:
            self.log_fn(f"Error getting repo structure: {e}")
            return "Unable to get repository structure"

    def _load_eval_history(self, eval_path: str) -> str:
        """Load previous evaluation results for context."""
        if not eval_path or not os.path.exists(eval_path):
            return "No previous evaluation results available."
        
        try:
            import json
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            
            # Extract key metrics
            summary = []
            if isinstance(eval_data, dict):
                if 'score' in eval_data:
                    summary.append(f"Previous Score: {eval_data['score']}")
                if 'errors' in eval_data:
                    summary.append(f"Previous Errors: {eval_data['errors']}")
                if 'feedback' in eval_data:
                    summary.append(f"Feedback: {eval_data['feedback']}")
                if 'improvements_needed' in eval_data:
                    summary.append(f"Improvements Needed: {eval_data['improvements_needed']}")
            
            if summary:
                return "Previous Evaluation:\n" + "\n".join(summary)
            return "Previous evaluation data available but no summary metrics found."
        except Exception as e:
            self.log_fn(f"Error loading eval history: {e}")
            return f"Unable to load evaluation history: {e}"

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
        # Get repository structure for context
        repo_structure = self._get_repo_structure(repo_path)
        
        # Load evaluation history for context
        eval_history = self._load_eval_history(eval_path)
        
        # Build instruction with all context
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Repository path: `{repo_path}`
Repository structure:
```
{repo_structure}
```

{eval_history}

Your goal: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.

Guidelines:
1. First explore the codebase to understand its structure
2. Identify areas for improvement (error handling, logic, efficiency)
3. Make targeted modifications using the available tools
4. Ensure changes maintain backward compatibility
5. Add appropriate error handling and logging
6. Focus on fixing issues identified in previous evaluations

Available tools: bash, editor (view, create, str_replace, insert), search

Begin by exploring the repository structure and understanding the current implementation."""

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.stats["modifications"] += 1
        except Exception as e:
            self.log_fn(f"Meta agent failed: {e}")
            self.stats["errors"] += 1
            raise

        return msg_history
    
    def get_stats(self) -> dict:
        """Return meta agent statistics."""
        return self.stats.copy()
