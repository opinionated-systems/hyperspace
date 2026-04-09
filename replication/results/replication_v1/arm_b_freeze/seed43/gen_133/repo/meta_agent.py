"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.bash_tool import set_allowed_root as bash_set_root
from agent.tools.editor_tool import set_allowed_root as editor_set_root

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
        # Set up tool scoping to restrict operations to the repo path
        abs_repo_path = os.path.abspath(repo_path)
        bash_set_root(abs_repo_path)
        editor_set_root(abs_repo_path)
        
        self.log_fn(f"MetaAgent starting with repo_path: {abs_repo_path}")
        self.log_fn(f"Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")
        
        # Build comprehensive instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{abs_repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Use the editor tool to view, create, and modify files.",
            "Use the bash tool to run commands and explore the repository.",
            "",
            "CODEBASE STRUCTURE:",
            "- task_agent.py: Main task agent for IMO grading. Contains the TaskAgent class and JSON extraction logic.",
            "- meta_agent.py: Meta agent that modifies the codebase (this file).",
            "- agent/agentic_loop.py: Agentic loop with native tool calling.",
            "- agent/llm_client.py: LLM client wrapper for API calls.",
            "- agent/tools/: Tool implementations (bash, editor).",
            "",
            "IMPROVEMENT GUIDELINES:",
            "- Focus on task_agent.py for grading accuracy improvements",
            "- The agent grades IMO (International Mathematical Olympiad) problems",
            "- Key areas to improve: JSON extraction, grading logic, prompt engineering",
            "- Maintain backward compatibility with existing interfaces",
        ]
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.extend([
                "",
                f"Previous evaluation results are available at: {eval_path}",
                "Consider these results when deciding what improvements to make.",
            ])
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"Budget: {iterations_left} iterations remaining.",
                "Focus on high-impact improvements that will improve task performance.",
            ])
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"Instruction: {instruction[:200]}...")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            self.log_fn(f"MetaAgent completed with {len(msg_history)} messages")
            return msg_history
            
        except Exception as e:
            self.log_fn(f"MetaAgent failed with error: {e}")
            # Return minimal history with error info
            return [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error: {e}"},
            ]
