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
        available_tools = list_available_tools()
        
        # Read evaluation results if available
        eval_info = ""
        if eval_path:
            try:
                import json
                import os
                
                # Validate path exists and is a file
                if not os.path.exists(eval_path):
                    eval_info = f"\n\nEvaluation file not found: {eval_path}\n"
                elif not os.path.isfile(eval_path):
                    eval_info = f"\n\nEvaluation path is not a file: {eval_path}\n"
                else:
                    with open(eval_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            eval_info = "\n\nEvaluation file is empty.\n"
                        else:
                            eval_data = json.loads(content)
                            if isinstance(eval_data, dict):
                                score = eval_data.get('score', 'N/A')
                                errors = eval_data.get('errors', [])
                                total = eval_data.get('total', None)
                                passed = eval_data.get('passed', None)
                                
                                eval_info = f"\n\nPrevious evaluation results:\n- Score: {score}\n"
                                if total is not None:
                                    eval_info += f"- Total: {total}\n"
                                if passed is not None:
                                    eval_info += f"- Passed: {passed}\n"
                                if errors:
                                    eval_info += f"- Errors ({len(errors)} total, showing first 3): {errors[:3]}\n"
                            elif isinstance(eval_data, list):
                                eval_info = f"\n\nPrevious evaluation results:\n- Results count: {len(eval_data)}\n"
                            else:
                                eval_info = f"\n\nPrevious evaluation results: {str(eval_data)[:200]}\n"
            except json.JSONDecodeError as e:
                eval_info = f"\n\nInvalid JSON in evaluation file: {e}\n"
            except UnicodeDecodeError as e:
                eval_info = f"\n\nCould not decode evaluation file (encoding issue): {e}\n"
            except PermissionError as e:
                eval_info = f"\n\nPermission denied reading evaluation file: {e}\n"
            except Exception as e:
                eval_info = f"\n\nCould not read evaluation results: {type(e).__name__}: {e}\n"
        
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
{eval_info}{iteration_info}

Begin by exploring the codebase to understand its structure, then make targeted improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
