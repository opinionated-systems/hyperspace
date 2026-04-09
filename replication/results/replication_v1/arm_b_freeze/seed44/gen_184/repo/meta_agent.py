"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _analyze_codebase(self, repo_path: str) -> dict:
        """Analyze the codebase structure and return summary information.
        
        Args:
            repo_path: path to the agent's repository
            
        Returns:
            Dictionary with codebase analysis results
        """
        analysis = {
            "files": [],
            "total_lines": 0,
            "key_modules": [],
        }
        
        repo = Path(repo_path)
        if not repo.exists():
            return analysis
            
        # Walk through the repository
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text()
                        lines = len(content.split('\n'))
                        analysis["files"].append({
                            "path": str(file_path.relative_to(repo_path)),
                            "lines": lines,
                        })
                        analysis["total_lines"] += lines
                        
                        # Identify key modules by name
                        if file in ['task_agent.py', 'meta_agent.py', 'llm_client.py', 'agentic_loop.py']:
                            analysis["key_modules"].append(str(file_path.relative_to(repo_path)))
                    except Exception:
                        pass
        
        return analysis

    def _read_evaluation_summary(self, eval_path: str) -> dict | None:
        """Attempt to read and summarize evaluation results.
        
        Args:
            eval_path: path to evaluation results
            
        Returns:
            Dictionary with evaluation summary or None if not available
        """
        if not eval_path or not Path(eval_path).exists():
            return None
            
        summary = {
            "has_results": False,
            "error_patterns": [],
            "suggestions": [],
        }
        
        # Look for JSON result files
        eval_dir = Path(eval_path)
        result_files = list(eval_dir.rglob('*.json')) + list(eval_dir.rglob('*.jsonl'))
        
        if result_files:
            summary["has_results"] = True
            summary["result_files"] = [str(f.relative_to(eval_dir)) for f in result_files[:5]]
            
        # Check for staged agent evals
        staged_dir = eval_dir / 'staged' / 'agent_evals'
        if staged_dir.exists():
            staged_files = list(staged_dir.iterdir())
            if staged_files:
                summary["staged_evals_count"] = len(staged_files)
                
        return summary

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
        # Analyze the codebase first
        codebase_info = self._analyze_codebase(repo_path)
        eval_info = self._read_evaluation_summary(eval_path)
        
        # Build a more informative instruction that includes context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add codebase analysis
        if codebase_info["files"]:
            instruction_parts.append(f"\n=== CODEBASE ANALYSIS ===")
            instruction_parts.append(f"Total Python files: {len(codebase_info['files'])}")
            instruction_parts.append(f"Total lines of code: {codebase_info['total_lines']}")
            if codebase_info["key_modules"]:
                instruction_parts.append(f"Key modules: {', '.join(codebase_info['key_modules'])}")
        
        # Add evaluation context if available
        if eval_info and eval_info["has_results"]:
            instruction_parts.append(f"\n=== EVALUATION CONTEXT ===")
            instruction_parts.append(f"Evaluation path: {eval_path}")
            if "result_files" in eval_info:
                instruction_parts.append(f"Result files found: {', '.join(eval_info['result_files'])}")
            if "staged_evals_count" in eval_info:
                instruction_parts.append(f"Staged evaluations: {eval_info['staged_evals_count']}")
            instruction_parts.append("\nReview these results to understand what improvements are needed.")
            instruction_parts.append("Look for patterns in errors - are they related to:")
            instruction_parts.append("  - JSON parsing failures?")
            instruction_parts.append("  - Incorrect grading decisions?")
            instruction_parts.append("  - Edge cases not handled?")
        elif eval_path:
            instruction_parts.append(f"\nNote: Evaluation path provided ({eval_path}) but no results found yet.")
        
        # Add budget context
        if iterations_left is not None:
            instruction_parts.append(f"\n=== BUDGET CONTEXT ===")
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            if iterations_left <= 1:
                instruction_parts.append("⚠️  This is the FINAL iteration - make your most important improvements now!")
            elif iterations_left <= 3:
                instruction_parts.append("⚠️  Limited iterations remaining - focus on high-impact changes.")
        
        instruction_parts.append("\n=== IMPROVEMENT FOCUS AREAS ===")
        instruction_parts.append("1. Improving the task agent's accuracy on grading problems")
        instruction_parts.append("2. Better JSON extraction and validation (handle edge cases)")
        instruction_parts.append("3. More robust error handling and fallback strategies")
        instruction_parts.append("4. Clearer prompting for the LLM with explicit formatting rules")
        instruction_parts.append("5. Early detection of empty/invalid student answers")
        instruction_parts.append("6. Performance optimizations (reduce unnecessary LLM calls)")
        
        instruction_parts.append("\n=== KEY FILES TO CONSIDER ===")
        instruction_parts.append(f"- {repo_path}/task_agent.py: Main grading logic, JSON extraction, validation")
        instruction_parts.append(f"- {repo_path}/agent/llm_client.py: LLM interaction, retry logic, audit logging")
        instruction_parts.append(f"- {repo_path}/agent/agentic_loop.py: Tool calling loop, conversation management")
        instruction_parts.append(f"- {repo_path}/agent/tools/: Editor and bash tool implementations")
        
        instruction_parts.append("\n=== MODIFICATION WORKFLOW ===")
        instruction_parts.append("1. First, explore the codebase structure using the editor tool")
        instruction_parts.append("2. Read key files to understand current implementation details")
        instruction_parts.append("3. If evaluation results exist, analyze them for failure patterns")
        instruction_parts.append("4. Identify the root cause of issues (don't just treat symptoms)")
        instruction_parts.append("5. Make targeted, focused improvements with clear rationale")
        instruction_parts.append("6. Verify your changes maintain backward compatibility")
        instruction_parts.append("7. Consider adding logging or error handling for better debugging")
        
        instruction_parts.append("\n=== BEST PRACTICES ===")
        instruction_parts.append("- Keep changes minimal and focused on specific issues")
        instruction_parts.append("- Add comments explaining complex logic or workarounds")
        instruction_parts.append("- Ensure error messages are informative for debugging")
        instruction_parts.append("- Test edge cases (empty inputs, malformed JSON, etc.)")
        instruction_parts.append("- Maintain the existing code style and patterns")
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
