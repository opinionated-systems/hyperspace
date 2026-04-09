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
        # Build instruction with context about evaluation results if available
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add context about remaining iterations
        if iterations_left is not None:
            instruction_parts.append(f"\nRemaining iterations: {iterations_left}")
        
        # Try to load and summarize evaluation results
        if eval_path:
            try:
                import json
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                
                if isinstance(eval_data, dict):
                    accuracy = eval_data.get('accuracy', 'N/A')
                    total = eval_data.get('total_samples', 'N/A')
                    instruction_parts.append(f"\nPrevious evaluation accuracy: {accuracy} ({total} samples)")
                    
                    # Add error examples if available
                    errors = eval_data.get('errors', [])
                    if errors and len(errors) > 0:
                        instruction_parts.append(f"\nRecent errors ({min(3, len(errors))} shown):")
                        for i, err in enumerate(errors[:3], 1):
                            instruction_parts.append(f"  {i}. {err}")
            except Exception as e:
                self.log_fn(f"Could not load evaluation results: {e}")
        
        instruction = "\n".join(instruction_parts)
        self.log_fn(f"Meta agent instruction:\n{instruction}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
