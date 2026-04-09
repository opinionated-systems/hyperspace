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
from agent.tools.bash_tool import set_allowed_root as set_bash_root
from agent.tools.editor_tool import set_allowed_root as set_editor_root
from agent.tools.search_tool import set_allowed_root as set_search_root

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
        import time
        start_time = time.time()
        
        # Validate paths
        if not repo_path:
            raise ValueError("repo_path cannot be empty")
        if not os.path.isdir(repo_path):
            raise ValueError(f"Invalid repo_path: {repo_path} (not a directory)")
        
        # Set up scoped access for tools
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        set_search_root(abs_repo_path)
        
        # Build instruction with context
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI agent codebase.",
            f"",
            f"Repository location: `{repo_path}`",
            f"",
            f"Your goal: Modify any part of the codebase to improve its performance, robustness, or functionality.",
            f"",
            f"Available tools:",
            f"- bash: Run shell commands (cd, ls, grep, etc.)",
            f"- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            f"- search: Search for patterns in the codebase (regex supported, with file filtering)",
            f"",
            f"Guidelines:",
            f"1. First explore the codebase to understand its structure",
            f"2. Use the search tool to find relevant code patterns",
            f"3. Identify areas for improvement (error handling, logic, performance)",
            f"4. Make targeted changes using the editor tool",
            f"5. Verify your changes work correctly",
        ]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()[:3000]  # Increased context size
                instruction_parts.append(f"\nPrevious evaluation results:\n```\n{eval_content}\n```")
            except Exception as e:
                self.log_fn(f"Warning: Could not read eval_path: {e}")
        
        # Add budget info
        if iterations_left is not None:
            instruction_parts.append(f"\nBudget: {iterations_left} iterations remaining")
            if iterations_left < 10:
                instruction_parts.append("⚠️ Low budget - focus on high-impact changes only")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"[MetaAgent] Starting with model={self.model}")
        self.log_fn(f"[MetaAgent] Repository: {abs_repo_path}")
        self.log_fn(f"[MetaAgent] Evaluation: {eval_path if eval_path and os.path.exists(eval_path) else 'none'}")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=50,  # Increased from default 40
            )
            elapsed = time.time() - start_time
            self.log_fn(f"[MetaAgent] Completed in {elapsed:.1f}s with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_fn(f"[MetaAgent] Failed after {elapsed:.1f}s: {e}")
            import traceback
            self.log_fn(f"[MetaAgent] Traceback: {traceback.format_exc()}")
            raise
