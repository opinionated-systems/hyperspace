"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from typing import Any

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The meta agent uses an LLM with tool calling capabilities to analyze,
    debug, and improve the task agent's codebase. It has access to bash
    commands and file editing tools to make modifications.
    
    Attributes:
        model: The LLM model to use for meta-agent reasoning.
        temperature: Sampling temperature for the LLM (0.0 = deterministic).
        log_fn: Logging function for agent activity.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _build_meta_prompt(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build a comprehensive instruction prompt for the meta agent.
        
        This method constructs a detailed prompt that guides the meta agent
        in analyzing and improving the codebase effectively.
        
        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results.
            iterations_left: Remaining iterations (budget info).
            
        Returns:
            A formatted instruction string for the meta agent.
        """
        base_instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal is to modify any part of the codebase at `{repo_path}` to improve its performance.

You have access to the following tools:
- `bash`: Run shell commands to explore the repository
- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)

Recommended workflow:
1. First, explore the repository structure using `bash` and `editor view`
2. Read the evaluation results at `{eval_path}` to understand what needs improvement
3. Identify specific issues or areas for enhancement
4. Make targeted modifications to fix problems or improve performance
5. Verify your changes are syntactically correct

Guidelines for modifications:
- Make focused, incremental changes rather than large rewrites
- Preserve existing working functionality
- Add comments to explain complex logic
- Test your changes if possible

Repository path: {repo_path}
Evaluation results: {eval_path}
"""
        
        if iterations_left is not None:
            base_instruction += f"\nIterations remaining: {iterations_left}\n"
            if iterations_left <= 3:
                base_instruction += "WARNING: Low on iterations. Prioritize critical fixes only.\n"
        
        base_instruction += "\nBegin by exploring the repository and evaluation results."
        
        return base_instruction

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
        self.log_fn(f"Starting meta agent iteration. Repo: {repo_path}, Eval: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        # Use the enhanced prompt builder for better guidance
        instruction = self._build_meta_prompt(repo_path, eval_path, iterations_left)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"Meta agent completed. Total messages: {len(msg_history)}")
        
        # Log summary of changes made
        self._log_modification_summary(msg_history)

        return msg_history
    
    def _log_modification_summary(self, msg_history: list[dict]) -> None:
        """Log a summary of modifications made during the agentic loop.
        
        Analyzes the message history to identify and log key actions taken
        by the meta agent, such as file edits and command executions.
        
        Args:
            msg_history: The message history from the agentic loop.
        """
        if not msg_history:
            self.log_fn("No message history to summarize.")
            return
        
        # Count tool usage by looking for tool-related content
        tool_calls = 0
        for msg in msg_history:
            if isinstance(msg, dict):
                content = msg.get("text", "") or msg.get("content", "")
                if content and isinstance(content, str):
                    # Count editor and bash tool usage
                    if "editor" in content.lower() or "bash" in content.lower():
                        tool_calls += 1
        
        self.log_fn(f"Modification summary: {len(msg_history)} messages, ~{tool_calls} tool interactions")
