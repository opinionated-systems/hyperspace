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
        # Set up scoped access for tools
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        
        self.log_fn(f"MetaAgent starting with repo_path: {abs_repo_path}")
        self.log_fn(f"Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")
        
        # Build a more detailed instruction
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify the codebase at: `{abs_repo_path}`

The codebase contains:
- task_agent.py: The main task agent that grades IMO (International Mathematical Olympiad) problems
- agent/: Directory containing supporting modules
  - agentic_loop.py: The agentic loop with tool calling
  - llm_client.py: LLM client wrapper
  - tools/: Tool implementations (bash, editor)

Instructions:
1. First, explore the codebase to understand its structure
2. Identify areas for improvement (error handling, prompt quality, extraction logic, etc.)
3. Make targeted improvements to enhance the agent's performance
4. Use the bash tool to explore and the editor tool to make changes

Start by viewing the current files to understand what needs improvement."""

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=40,
            )
            
            self.log_fn(f"MetaAgent completed with {len(msg_history)} messages")
            return msg_history
            
        except Exception as e:
            import traceback
            error_msg = f"MetaAgent failed: {e}\n{traceback.format_exc()}"
            self.log_fn(error_msg)
            logger.error(error_msg)
            return [{"role": "system", "text": error_msg}]
