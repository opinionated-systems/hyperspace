"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class MetaAgentConfig:
    """Configuration for the MetaAgent.
    
    Attributes:
        model: The LLM model identifier to use.
        temperature: Sampling temperature for the LLM.
        max_tool_calls: Maximum number of tool calls per iteration.
    """
    model: str = META_MODEL
    temperature: float = 0.0
    max_tool_calls: int = 40


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses the agentic loop with bash, editor, and search tools
    to explore and modify the codebase. It can incorporate evaluation
    feedback to guide improvements.
    
    Attributes:
        config: Configuration for the meta agent.
        log_fn: Logging function for agent activity.
    """

    def __init__(
        self,
        model: str = META_MODEL,
        temperature: float = 0.0,
        max_tool_calls: int = 40,
    ) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model identifier to use.
            temperature: Sampling temperature for the LLM (0.0 = deterministic).
            max_tool_calls: Maximum number of tool calls per iteration.
        """
        self.config = MetaAgentConfig(
            model=model,
            temperature=temperature,
            max_tool_calls=max_tool_calls,
        )
        self.log_fn: Callable = logger.info

    def _read_eval_results(self, eval_path: str) -> str:
        """Read and format evaluation results from a file.
        
        Args:
            eval_path: Path to the evaluation results file.
            
        Returns:
            Formatted evaluation info string, or error message if reading fails.
        """
        if not eval_path:
            return ""
            
        try:
            import json
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
                if isinstance(eval_data, dict):
                    score = eval_data.get('score', 'N/A')
                    errors = eval_data.get('errors', [])
                    eval_info = f"\n\nPrevious evaluation results:\n- Score: {score}\n"
                    if errors:
                        # Show first 3 errors with truncation
                        error_preview = errors[:3]
                        eval_info += f"- Errors: {error_preview}\n"
                    return eval_info
                return "\n\nPrevious evaluation results: (invalid format)\n"
        except FileNotFoundError:
            return f"\n\nEvaluation results file not found: {eval_path}\n"
        except json.JSONDecodeError as e:
            return f"\n\nInvalid JSON in evaluation results: {e}\n"
        except Exception as e:
            return f"\n\nCould not read evaluation results: {e}\n"

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None,
    ) -> str:
        """Build the instruction for the meta agent.
        
        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results.
            iterations_left: Remaining iterations.
            
        Returns:
            Formatted instruction string.
        """
        available_tools = list_available_tools()
        eval_info = self._read_eval_results(eval_path)
        iteration_info = f"\nIterations remaining: {iterations_left}" if iterations_left is not None else ""
        
        return DEFAULT_INSTRUCTION_TEMPLATE.format(
            repo_path=repo_path,
            available_tools=', '.join(available_tools),
            eval_info=eval_info,
            iteration_info=iteration_info,
        )

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
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.config.model,
            temperature=self.config.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            max_tool_calls=self.config.max_tool_calls,
        )

        return msg_history
