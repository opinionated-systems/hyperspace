"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools

logger = logging.getLogger(__name__)

# Default instruction template for the meta agent
DEFAULT_INSTRUCTION_TEMPLATE = """Modify any part of the codebase at `{repo_path}`.

You have access to the following tools: {available_tools}

- Use `search` to find files and content before making changes
- Use `bash` to explore the directory structure and run commands
- Use `editor` to view, create, and modify files

Guidelines for making improvements:
1. Start by exploring the codebase structure with `bash` and `editor`
2. Identify areas that could benefit from improvements (error handling, validation, robustness)
3. Make targeted, focused changes that improve the code quality
4. Test your changes by viewing the modified files
5. Ensure all changes maintain backward compatibility
{eval_info}{iteration_info}

Begin by exploring the codebase to understand its structure, then make targeted improvements."""


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses the agentic loop with bash, editor, and search tools
    to explore and modify the codebase. It can incorporate evaluation
    feedback to guide improvements.
    
    Attributes:
        model: The LLM model identifier to use.
        temperature: Sampling temperature for the LLM.
        log_fn: Logging function for agent activity.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model identifier to use.
            temperature: Sampling temperature for the LLM (0.0 = deterministic).
            
        Raises:
            ValueError: If model is not a valid string or temperature is out of range.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        
        if not isinstance(temperature, (int, float)):
            raise ValueError(f"temperature must be a number, got {type(temperature).__name__}")
        if temperature < 0 or temperature > 2:
            raise ValueError(f"temperature must be between 0 and 2, got {temperature}")
        
        self.model = model
        self.temperature = temperature
        self.log_fn: Callable = logger.info
        
        # Validate that the model is available
        try:
            from agent.llm_client import _get_client
            _get_client(model)
        except Exception as e:
            logger.warning(f"Could not validate model '{model}' at initialization: {e}")

    def _read_eval_results(self, eval_path: str) -> str:
        """Read and format evaluation results from a file.
        
        Args:
            eval_path: Path to the evaluation results file.
            
        Returns:
            Formatted evaluation info string, or error message if reading fails.
        """
        try:
            import json
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
                if isinstance(eval_data, dict):
                    score = eval_data.get('score', 'N/A')
                    errors = eval_data.get('errors', [])
                    eval_info = f"\n\nPrevious evaluation results:\n- Score: {score}\n"
                    if errors:
                        eval_info += f"- Errors: {errors[:3]}\n"  # Show first 3 errors
                    return eval_info
                return "\n\nPrevious evaluation results: (invalid format)\n"
        except Exception as e:
            return f"\n\nCould not read evaluation results: {e}\n"

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results (can be empty string).
            iterations_left: Remaining iterations (budget info).

        Returns:
            Message history from the agentic loop.
            
        Raises:
            ValueError: If repo_path is invalid.
            FileNotFoundError: If repo_path does not exist.
        """
        # Validate repo_path
        if not isinstance(repo_path, str) or not repo_path.strip():
            raise ValueError("repo_path must be a non-empty string")
        
        import os
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not os.path.isdir(repo_path):
            raise ValueError(f"Repository path is not a directory: {repo_path}")
        
        # Validate eval_path if provided
        if eval_path and not isinstance(eval_path, str):
            raise ValueError("eval_path must be a string or empty")
        
        # Validate iterations_left if provided
        if iterations_left is not None:
            if not isinstance(iterations_left, int):
                raise ValueError("iterations_left must be an integer or None")
            if iterations_left < 0:
                raise ValueError("iterations_left must be non-negative")
        
        available_tools = list_available_tools()
        
        # Read evaluation results if available
        eval_info = ""
        if eval_path:
            eval_info = self._read_eval_results(eval_path)
        
        iteration_info = f"\nIterations remaining: {iterations_left}" if iterations_left is not None else ""
        
        instruction = DEFAULT_INSTRUCTION_TEMPLATE.format(
            repo_path=repo_path,
            available_tools=', '.join(available_tools),
            eval_info=eval_info,
            iteration_info=iteration_info,
        )

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
