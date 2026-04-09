"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
    
    def _parse_evaluation_results(self, eval_path: str | None) -> str:
        """Parse evaluation results from a JSON file.
        
        Args:
            eval_path: Path to the evaluation results JSON file.
            
        Returns:
            A formatted string with evaluation information, or an empty string
            if no evaluation results are available.
        """
        if not eval_path:
            return ""
        
        try:
            import json
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
                if not isinstance(eval_data, dict):
                    return f"\n\nInvalid evaluation data format: expected dict, got {type(eval_data).__name__}\n"
                
                score = eval_data.get('score', 'N/A')
                errors = eval_data.get('errors', [])
                total = eval_data.get('total', 'N/A')
                
                eval_info = f"\n\nPrevious evaluation results:\n- Score: {score}/{total}\n"
                
                if errors:
                    eval_info += f"- Errors: {errors[:5]}\n"  # Show first 5 errors
                    # Add suggestions based on error patterns
                    eval_info += "\nFocus on fixing these error patterns in the codebase.\n"
                
                # Add additional metrics if available
                if 'accuracy' in eval_data:
                    eval_info += f"- Accuracy: {eval_data['accuracy']:.2%}\n"
                if 'avg_latency' in eval_data:
                    eval_info += f"- Avg Latency: {eval_data['avg_latency']:.2f}s\n"
                
                return eval_info
                
        except FileNotFoundError:
            return f"\n\nNo previous evaluation results found at {eval_path}\n"
        except json.JSONDecodeError as e:
            return f"\n\nInvalid JSON in evaluation results: {e}\n"
        except PermissionError:
            return f"\n\nPermission denied reading evaluation results at {eval_path}\n"
        except Exception as e:
            return f"\n\nCould not read evaluation results: {type(e).__name__}: {e}\n"

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
        # Validate inputs
        if not repo_path or not isinstance(repo_path, str):
            logger.error(f"Invalid repo_path: {repo_path}")
            return []
        
        # Validate repo_path exists
        import os
        if not os.path.exists(repo_path):
            logger.error(f"Repository path does not exist: {repo_path}")
            return []
        
        if not os.path.isdir(repo_path):
            logger.error(f"Repository path is not a directory: {repo_path}")
            return []
        
        available_tools = list_available_tools()
        logger.info(f"Available tools: {available_tools}")
        
        # Read evaluation results if available
        eval_info = self._parse_evaluation_results(eval_path)
        
        iteration_info = f"\nIterations remaining: {iterations_left}" if iterations_left is not None else ""
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools: {', '.join(available_tools)}

- Use `search` to find files and content before making changes
- Use `bash` to explore the directory structure and run commands
- Use `editor` to view, create, and modify files

Guidelines for making improvements:
1. Start by exploring the codebase structure with `bash` and `editor`
2. Identify areas that could benefit from improvements (error handling, validation, robustness)
3. Make targeted, focused changes that improve the code quality
4. Test your changes by viewing the modified files
5. Ensure all changes maintain backward compatibility
6. Add logging for better debugging capabilities
7. Improve input validation where missing
{eval_info}{iteration_info}

Begin by exploring the codebase to understand its structure, then make targeted improvements."""

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            return msg_history
        except Exception as e:
            logger.error(f"Error in meta agent chat: {e}", exc_info=True)
            return []
