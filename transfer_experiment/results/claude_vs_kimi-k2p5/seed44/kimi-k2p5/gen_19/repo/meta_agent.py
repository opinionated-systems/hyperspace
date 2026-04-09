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


def _load_eval_summary(eval_path: str) -> str:
    """Load evaluation results if available."""
    if not eval_path or not os.path.exists(eval_path):
        return "No previous evaluation results available."
    
    try:
        import json
        report_path = os.path.join(eval_path, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                data = json.load(f)
            return f"Previous evaluation: {data.get('overall_accuracy', 'N/A')} accuracy, {data.get('total_correct', 'N/A')}/{data.get('total', 'N/A')} correct"
    except Exception:
        pass
    return "Evaluation results available but could not be parsed."


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
        eval_summary = _load_eval_summary(eval_path)
        
        instruction = f"""You are a meta-agent tasked with improving a task agent that grades student solutions to competition math problems.

## Context
- Repository path: `{repo_path}`
- {eval_summary}
- Iterations remaining: {iterations_left if iterations_left is not None else 'unknown'}

## Your task
Modify any part of the codebase at `{repo_path}` to improve the task agent's grading accuracy.

## Key files to consider
- `task_agent.py`: The main grading logic and prompts
- `meta_agent.py`: This meta-agent's own implementation

## Improvement strategies
1. **Prompt engineering**: Improve the grading instructions, add more examples, clarify edge cases
2. **Extraction logic**: Improve how the agent parses model responses and extracts labels
3. **Error handling**: Add better handling for malformed responses
4. **Structured reasoning**: Add steps that force the model to think more carefully

## Guidelines
- Use the `editor` and `bash` tools to make changes
- View files before modifying them
- Make focused, incremental improvements
- Test your changes by examining the code structure
- Preserve the existing interface (class names, method signatures)

Start by exploring the codebase to understand the current implementation, then make targeted improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
