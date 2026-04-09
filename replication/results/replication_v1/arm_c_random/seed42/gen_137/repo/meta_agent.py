"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tool calling to analyze and modify its own
    codebase, enabling autonomous self-improvement through iterative refinement.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.start_time: float | None = None

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
        self.start_time = time.time()
        
        # Build comprehensive instruction with context
        instruction_parts = [
            f"You are an expert software engineer tasked with improving a codebase.",
            f"",
            f"Repository location: `{repo_path}`",
            f"",
            f"Available tools:",
            f"- `editor`: View, create, and edit files. Commands: view, create, str_replace, insert, undo_edit",
            f"- `bash`: Run shell commands",
            f"- `search`: Search for files and content (grep for content, find for files)",
            f"",
            f"Your task: Modify any part of the codebase to improve its functionality,",
            f"performance, reliability, or maintainability.",
            f"",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Budget: You have {iterations_left} iterations remaining.")
            instruction_parts.append(f"")
        
        instruction_parts.extend([
            f"Suggested workflow:",
            f"1. Use `editor view` to explore the codebase structure",
            f"2. Use `search grep` to find relevant code patterns",
            f"3. Identify areas for improvement",
            f"4. Make targeted modifications using `editor str_replace` or `editor create`",
            f"5. Verify changes with `bash` commands if needed",
            f"",
            f"Begin by exploring the repository structure.",
        ])
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"Starting meta-agent run on {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        elapsed = time.time() - self.start_time
        self.log_fn(f"Meta-agent run completed in {elapsed:.2f}s")

        return msg_history
