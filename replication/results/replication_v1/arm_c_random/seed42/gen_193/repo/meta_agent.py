"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import json

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.file_tool import file_command
from agent.tools.bash_tool import bash_command

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _load_evaluation_results(self, eval_path: str) -> dict | None:
        """Load evaluation results from the eval_path if available."""
        try:
            # Try to find report.json in the eval_path
            import os
            report_path = os.path.join(eval_path, "eval_val", "staged", "report.json")
            if not os.path.exists(report_path):
                report_path = os.path.join(eval_path, "eval_train", "staged", "report.json")
            if not os.path.exists(report_path):
                return None
            
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_fn(f"Could not load evaluation results: {e}")
            return None

    def _get_repo_structure(self, repo_path: str) -> str:
        """Get the structure of the repository."""
        try:
            result = bash_command(f"find {repo_path} -type f -name '*.py' | head -20")
            return result
        except Exception as e:
            return f"Could not get repo structure: {e}"

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
        # Build instruction with context about remaining iterations
        instruction_parts = []
        
        # Add evaluation context if available
        eval_results = None
        if eval_path:
            eval_results = self._load_evaluation_results(eval_path)
        
        if eval_results:
            accuracy = eval_results.get('overall_accuracy', 'N/A')
            total_correct = eval_results.get('total_correct', 'N/A')
            total = eval_results.get('total', 'N/A')
            instruction_parts.append(
                f"Current evaluation results: {total_correct}/{total} correct ({accuracy*100:.1f}% accuracy). "
                f"Your goal is to improve these results by modifying the codebase."
            )
        
        instruction_parts.append(f"Modify any part of the codebase at `{repo_path}`.")
        
        if iterations_left is not None:
            instruction_parts.append(f"You have {iterations_left} iteration(s) remaining to improve the codebase.")
        
        if eval_path:
            instruction_parts.append(f"Previous evaluation results are available at `{eval_path}`.")
        
        # Add guidance on what to focus on
        instruction_parts.append(
            "\n\nFocus on improving the task_agent.py which is responsible for grading student solutions. "
            "Common issues include:\n"
            "1. JSON extraction from LLM responses\n"
            "2. Prompt clarity for the grading task\n"
            "3. Handling different response formats (points vs categories)\n"
            "\nUse the file and bash tools to explore the codebase, then use the editor tool to make changes."
        )
        
        instruction = " ".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:100]}...")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"MetaAgent completed with {len(msg_history)} messages in history.")
        
        return msg_history
