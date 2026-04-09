"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced with better context about the codebase and evaluation results.
"""

from __future__ import annotations

import json
import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _load_eval_results(self, eval_path: str) -> dict:
        """Load evaluation results if available."""
        if not eval_path or not os.path.exists(eval_path):
            return {}
        try:
            with open(eval_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_fn(f"Error loading eval results: {e}")
            return {}

    def _get_codebase_summary(self, repo_path: str) -> str:
        """Get a summary of the codebase structure."""
        try:
            import subprocess
            result = subprocess.run(
                ['find', repo_path, '-type', 'f', '-name', '*.py', '-not', '-path', '*/\.*', '-not', '-path', '*/__pycache__/*'],
                capture_output=True, text=True, timeout=10
            )
            files = result.stdout.strip().split('\n') if result.stdout else []
            return '\n'.join([f'  - {f}' for f in files if f])[:500]
        except Exception:
            return "  (Unable to list files)"

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
        # Load evaluation results for context
        eval_results = self._load_eval_results(eval_path)
        codebase_summary = self._get_codebase_summary(repo_path)
        
        # Build comprehensive instruction
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository path: `{repo_path}`",
            f"",
            f"Python files in the codebase:",
            codebase_summary,
        ]
        
        # Add evaluation context if available
        if eval_results:
            instruction_parts.extend([
                f"",
                f"Previous evaluation results:",
                f"```json",
                json.dumps(eval_results, indent=2)[:1000],
                f"```",
            ])
        
        if iterations_left is not None:
            instruction_parts.extend([
                f"",
                f"Iterations remaining: {iterations_left}",
            ])
        
        instruction_parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.",
            f"",
            f"Guidelines:",
            f"1. Use the `editor` tool to view and modify files",
            f"2. Use the `bash` tool to run tests or check syntax",
            f"3. Focus on improving task_agent.py which handles the main grading task",
            f"4. Consider improving:",
            f"   - Prompt engineering for better reasoning",
            f"   - Error handling and robustness",
            f"   - JSON extraction and parsing",
            f"   - Response validation",
            f"5. Make incremental, focused changes",
            f"6. Test your changes if possible",
            f"",
            f"Start by exploring the codebase to understand the current implementation.",
        ])
        
        instruction = '\n'.join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
