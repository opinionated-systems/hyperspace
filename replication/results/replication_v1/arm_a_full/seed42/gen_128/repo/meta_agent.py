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
        # Build instruction with context about iterations left
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\n\nIterations remaining: {iterations_left}")
            if iterations_left <= 3:
                instruction_parts.append(" - Focus on high-impact changes only.")
            elif iterations_left <= 10:
                instruction_parts.append(" - Consider targeted improvements to core functionality.")
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\n\nPrevious evaluation results available at: {eval_path}")
            # Try to read and include summary of previous results
            try:
                import json
                for subdir in ["eval_train", "eval_val"]:
                    subpath = os.path.join(eval_path, subdir)
                    if os.path.isdir(subpath):
                        for fname in os.listdir(subpath):
                            if fname.endswith(".json"):
                                with open(os.path.join(subpath, fname)) as f:
                                    data = json.load(f)
                                    if isinstance(data, dict) and ("score" in data or "accuracy" in data):
                                        score = data.get("score") or data.get("accuracy")
                                        instruction_parts.append(f"\nPrevious {subdir} score: {score}")
                                        break
            except Exception:
                pass  # Ignore errors reading eval files
        
        instruction_parts.append("\n\nGuidelines for modifications:")
        instruction_parts.append("\n1. Use the editor tool to view files before modifying them")
        instruction_parts.append("\n2. Use the search tool to find relevant code patterns")
        instruction_parts.append("\n3. Make focused, incremental improvements")
        instruction_parts.append("\n4. Ensure changes are syntactically correct")
        instruction_parts.append("\n5. Test your understanding by viewing the modified code")
        
        instruction = "".join(instruction_parts)
        
        self.log_fn(f"[MetaAgent] Starting with instruction length: {len(instruction)} chars")
        self.log_fn(f"[MetaAgent] Target repo: {repo_path}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"[MetaAgent] Completed with {len(msg_history)} messages")

        return msg_history
