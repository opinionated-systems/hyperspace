"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The meta agent uses the agentic loop with tool calling to explore,
    analyze, and modify the codebase. It can make multiple modifications
    in a single forward pass.
    
    Attributes:
        model: The LLM model to use for meta-level reasoning
        temperature: Sampling temperature for the LLM
        log_fn: Logging function for debug output
        modifications_count: Counter for tracking code modifications
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.modifications_count = 0

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
        self.log_fn(f"Starting meta agent forward pass on: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Count modifications made in this forward pass
        # by looking for editor tool calls in the message history
        for msg in msg_history:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    if tc.get("function", {}).get("name") == "editor":
                        self.modifications_count += 1
        
        self.log_fn(f"Meta agent completed. Total modifications so far: {self.modifications_count}")

        return msg_history
