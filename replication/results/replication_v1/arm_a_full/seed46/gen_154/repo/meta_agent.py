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
from agent.tools.bash_tool import set_allowed_root as set_bash_root
from agent.tools.editor_tool import set_allowed_root as set_editor_root

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
        # Validate inputs
        if not repo_path:
            self.log_fn("Error: repo_path is required")
            return []
        
        # Ensure repo_path is absolute
        repo_path = os.path.abspath(repo_path)
        
        # Check if repo_path exists
        if not os.path.exists(repo_path):
            self.log_fn(f"Error: repo_path does not exist: {repo_path}")
            return []
        
        # Set up tool roots for security
        try:
            set_bash_root(repo_path)
            set_editor_root(repo_path)
            self.log_fn(f"Set tool roots to: {repo_path}")
        except Exception as e:
            self.log_fn(f"Warning: Failed to set tool roots: {e}")
        
        # Build instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            try:
                # Try to read report.json first for structured data
                import json
                report_path = os.path.join(eval_path, "report.json")
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    # Extract key metrics
                    metrics = []
                    if "accuracy" in report:
                        metrics.append(f"Accuracy: {report['accuracy']:.4f}")
                    if "f1_macro" in report:
                        metrics.append(f"F1 Macro: {report['f1_macro']:.4f}")
                    if "total_samples" in report:
                        metrics.append(f"Samples: {report['total_samples']}")
                    if metrics:
                        instruction_parts.append(f"\nKey metrics: {', '.join(metrics)}")
                    
                    # Add error analysis if available
                    if "error_analysis" in report and report["error_analysis"]:
                        error_summary = str(report["error_analysis"])[:500]
                        instruction_parts.append(f"\nError analysis: {error_summary}")
                else:
                    # Fallback to reading any available file
                    for filename in ["predictions.csv", "results.json", "summary.txt"]:
                        filepath = os.path.join(eval_path, filename)
                        if os.path.exists(filepath):
                            with open(filepath, 'r') as f:
                                eval_content = f.read()[:1500]
                            if eval_content:
                                instruction_parts.append(f"\nEvaluation summary from {filename}:\n{eval_content}")
                            break
            except Exception as e:
                self.log_fn(f"Warning: Could not read eval_path: {e}")
        
        # Add budget information
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 5:
                instruction_parts.append("\n⚠️ WARNING: Low on iterations! Focus on high-impact changes only.")
        
        # Add guidance based on codebase structure
        instruction_parts.append("\n\nGuidance:")
        instruction_parts.append("- Focus on improving grading accuracy, especially distinguishing Partial from Incorrect")
        instruction_parts.append("- The task_agent.py contains the core grading logic")
        instruction_parts.append("- The agent/ directory contains supporting infrastructure")
        instruction_parts.append("- Make targeted, testable changes rather than large refactors")
        
        instruction = "\n".join(instruction_parts)
        self.log_fn(f"Meta agent instruction length: {len(instruction)} chars")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"Meta agent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"Error in meta agent forward: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")
            return []
