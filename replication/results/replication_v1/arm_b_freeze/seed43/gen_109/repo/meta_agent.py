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
from agent.tools.registry import get_tool_descriptions

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
        # Build comprehensive instruction with context
        tool_help = get_tool_descriptions()
        
        instruction = f"""You are a meta-agent tasked with improving an AI agent's codebase.

Your goal: Modify any part of the codebase at `{repo_path}` to improve the agent's performance on IMO (International Mathematical Olympiad) grading tasks.

AVAILABLE TOOLS:
{tool_help}

WORKFLOW:
1. First, explore the codebase structure to understand what exists
2. Review the current implementation of task_agent.py and related files
3. Identify areas for improvement (better prompts, error handling, extraction logic, etc.)
4. Make targeted modifications to improve grading accuracy and robustness
5. Verify your changes are syntactically correct

IMPORTANT GUIDELINES:
- Focus on improving the task_agent.py which handles IMO problem grading
- Consider improvements to: prompt engineering, JSON extraction, error handling, retry logic
- Make incremental, testable changes rather than large rewrites
- Always verify files after editing with the view command
- Use absolute paths for all file operations

CONTEXT:
- Repository path: {repo_path}
- Evaluation results path: {eval_path}
- Iterations remaining: {iterations_left if iterations_left is not None else 'unlimited'}

Begin by exploring the codebase structure."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
