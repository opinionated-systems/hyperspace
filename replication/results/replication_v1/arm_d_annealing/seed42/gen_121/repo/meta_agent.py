"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.bash_tool import set_allowed_root

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze and modify its own codebase,
    enabling autonomous self-improvement through iterative refinement.
    """

    def __init__(
        self, 
        model: str = META_MODEL, 
        temperature: float = 0.0,
        log_fn: Callable[[str], None] | None = None
    ) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use for modifications
            temperature: Sampling temperature for the model (0.0 = deterministic)
            log_fn: Optional custom logging function
        """
        self.model = model
        self.temperature = temperature
        self.log_fn = log_fn or logger.info

    def _validate_paths(self, repo_path: str, eval_path: str) -> None:
        """Validate that required paths exist.
        
        Args:
            repo_path: Path to the agent's repository
            eval_path: Path to previous evaluation results
            
        Raises:
            FileNotFoundError: If repo_path does not exist
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation path not found: {eval_path}")

    def _read_evaluation_results(self, eval_path: str) -> str:
        """Read and summarize evaluation results for context.
        
        Args:
            eval_path: Path to evaluation results
            
        Returns:
            Summary of evaluation results or empty string if not available
        """
        try:
            with open(eval_path, 'r') as f:
                content = f.read()
                # Return first 2000 chars to avoid overwhelming the context
                if len(content) > 2000:
                    return content[:2000] + "\n... [truncated]"
                return content
        except Exception as e:
            logger.warning(f"Could not read evaluation results: {e}")
            return ""

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
            
        Raises:
            FileNotFoundError: If repo_path does not exist
            RuntimeError: If the agentic loop fails
        """
        self._validate_paths(repo_path, eval_path)
        
        # Set the allowed root for bash commands to the repo path
        set_allowed_root(repo_path)
        
        if iterations_left is not None:
            self.log_fn(f"MetaAgent starting with {iterations_left} iterations remaining")
        
        # Read evaluation results for context
        eval_summary = self._read_evaluation_results(eval_path)
        
        # Build a more informative instruction
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Available tools:",
            "- `bash`: Run commands in a bash shell (state is persistent across calls)",
            "- `editor`: View, create, and edit files (commands: view, create, str_replace, insert, undo_edit)",
        ]
        
        if eval_summary:
            instruction_parts.extend([
                "",
                "Previous evaluation results (for context):",
                "```",
                eval_summary,
                "```",
            ])
        
        instruction_parts.extend([
            "",
            "Guidelines for modifications:",
            "1. First, explore the codebase to understand its structure",
            "2. Identify areas for improvement based on the evaluation results",
            "3. Make targeted, focused changes that address specific issues",
            "4. Test your changes if possible using bash commands",
            "5. Ensure the code remains syntactically correct and functional",
            "6. After making changes, verify syntax with: python3 -m py_compile <file.py>",
        ])
        
        instruction = "\n".join(instruction_parts)

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"MetaAgent completed successfully with {len(msg_history)} messages")
            
            # Log summary of changes made
            tool_calls_made = 0
            for msg in msg_history:
                if isinstance(msg, dict) and msg.get("tool_calls"):
                    tool_calls_made += len(msg.get("tool_calls", []))
            self.log_fn(f"Total tool calls made: {tool_calls_made}")
            
            return msg_history
        except Exception as e:
            logger.error(f"MetaAgent failed: {e}")
            raise RuntimeError(f"MetaAgent execution failed: {e}") from e
