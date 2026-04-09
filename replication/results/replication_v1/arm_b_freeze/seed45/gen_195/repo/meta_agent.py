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
        """
        self.model = model
        self.temperature = temperature
        self.log_fn: Callable = logger.info

    def _read_eval_results(self, eval_path: str) -> str:
        """Read and format evaluation results from a file.
        
        Args:
            eval_path: Path to the evaluation results file.
            
        Returns:
            Formatted evaluation info string, or error message if reading fails.
        """
        import os
        
        # Validate eval_path
        if not eval_path or not isinstance(eval_path, str):
            return "\n\nNo evaluation results available (invalid path).\n"
        
        # Check if file exists
        if not os.path.exists(eval_path):
            return f"\n\nNo evaluation results available (file not found: {eval_path}).\n"
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(eval_path):
            return f"\n\nNo evaluation results available (not a file: {eval_path}).\n"
        
        try:
            import json
            with open(eval_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return "\n\nPrevious evaluation results: (empty file)\n"
                
                eval_data = json.loads(content)
                if isinstance(eval_data, dict):
                    score = eval_data.get('score', 'N/A')
                    errors = eval_data.get('errors', [])
                    total = eval_data.get('total', 'N/A')
                    passed = eval_data.get('passed', 'N/A')
                    
                    eval_info = f"\n\nPrevious evaluation results:\n- Score: {score}\n"
                    if total != 'N/A':
                        eval_info += f"- Total: {total}\n"
                    if passed != 'N/A':
                        eval_info += f"- Passed: {passed}\n"
                    if errors:
                        eval_info += f"- Errors ({len(errors)} total, showing first 3):\n"
                        for i, error in enumerate(errors[:3], 1):
                            error_str = str(error)[:200]  # Truncate long errors
                            eval_info += f"  {i}. {error_str}\n"
                    return eval_info
                elif isinstance(eval_data, list):
                    return f"\n\nPrevious evaluation results:\n- Results count: {len(eval_data)}\n"
                else:
                    return f"\n\nPrevious evaluation results: (unexpected format: {type(eval_data).__name__})\n"
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evaluation results as JSON: {e}")
            return f"\n\nPrevious evaluation results: (invalid JSON: {e})\n"
        except PermissionError as e:
            logger.warning(f"Permission denied reading evaluation results: {e}")
            return f"\n\nCould not read evaluation results: Permission denied\n"
        except Exception as e:
            logger.warning(f"Error reading evaluation results: {e}")
            return f"\n\nCould not read evaluation results: {type(e).__name__}\n"

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
        """
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
