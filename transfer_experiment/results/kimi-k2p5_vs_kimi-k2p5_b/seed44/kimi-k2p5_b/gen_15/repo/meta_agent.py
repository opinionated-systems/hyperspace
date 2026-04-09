"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses an LLM to analyze and modify the codebase at a given
    repository path. It can be used for automated code improvement,
    refactoring, and bug fixing.
    
    Attributes:
        model: The LLM model to use for code modification.
        temperature: The temperature parameter for the LLM (controls randomness).
        log_fn: The logging function to use for status messages.
        max_retries: Maximum number of retries for failed operations.
    """

    def __init__(
        self, 
        model: str = META_MODEL, 
        temperature: float = 0.0,
        max_retries: int = 3,
        log_fn: Callable | None = None,
    ) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use. Defaults to META_MODEL.
            temperature: The temperature for LLM sampling. Defaults to 0.0.
            max_retries: Maximum retry attempts for failed operations. Defaults to 3.
            log_fn: Custom logging function. Defaults to logger.info.
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max(max_retries, 1)  # Ensure at least 1 retry
        self.log_fn = log_fn if log_fn is not None else logger.info

    def _validate_paths(self, repo_path: str, eval_path: str) -> tuple[Path, Path]:
        """Validate that required paths exist and are accessible.
        
        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results.
            
        Returns:
            Tuple of validated (repo_path, eval_path) as Path objects.
            
        Raises:
            FileNotFoundError: If repo_path does not exist.
            ValueError: If paths are empty or invalid.
        """
        if not repo_path or not isinstance(repo_path, str):
            raise ValueError(f"Invalid repo_path: {repo_path!r}")
        if not eval_path or not isinstance(eval_path, str):
            raise ValueError(f"Invalid eval_path: {eval_path!r}")
        
        repo = Path(repo_path)
        eval_p = Path(eval_path)
        
        if not repo.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {repo_path}"
                f" (resolved to: {repo.resolve()})"
            )
        
        # Log eval_path status but don't require it to exist
        if eval_p.exists():
            self.log_fn(f"Evaluation path found: {eval_path}")
        else:
            self.log_fn(f"Warning: Evaluation path not found: {eval_path}")
        
        return repo, eval_p

    def _analyze_eval_results(self, eval_p: Path) -> dict:
        """Analyze evaluation results to identify improvement opportunities.
        
        Args:
            eval_p: Path to evaluation results directory.
            
        Returns:
            Dictionary containing analysis of evaluation results.
        """
        if not eval_p.exists():
            return {"available": False}
        
        try:
            # Look for result files in the eval directory
            result_files = list(eval_p.rglob("*.json"))
            
            if not result_files:
                return {"available": True, "result_files": 0}
            
            # Basic statistics
            total_files = len(result_files)
            
            # Try to read the most recent results
            latest_results = []
            for rf in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                try:
                    import json
                    with open(rf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        latest_results.append({
                            "file": str(rf.name),
                            "data": data
                        })
                except Exception as e:
                    self.log_fn(f"Warning: Could not parse {rf}: {e}")
            
            return {
                "available": True,
                "result_files": total_files,
                "latest_results": latest_results
            }
        except Exception as e:
            self.log_fn(f"Warning: Error analyzing eval results: {e}")
            return {"available": True, "error": str(e)}

    def _get_repo_summary(self, repo: Path) -> dict:
        """Generate a summary of the repository structure.
        
        Args:
            repo: Path to the repository.
            
        Returns:
            Dictionary containing repository statistics.
        """
        py_files = list(repo.rglob("*.py"))
        total_lines = 0
        file_details = []
        
        for f in py_files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                    line_count = len(lines)
                    total_lines += line_count
                    file_details.append({
                        "path": str(f.relative_to(repo)),
                        "lines": line_count
                    })
            except Exception as e:
                self.log_fn(f"Warning: Could not read {f}: {e}")
                pass
        
        # Sort by line count descending to show most important files first
        file_details.sort(key=lambda x: x["lines"], reverse=True)
        
        return {
            "python_files": len(py_files),
            "total_lines": total_lines,
            "strategies_dir": (repo / "strategies").exists(),
            "evals_dir": (repo / "evals").exists(),
            "file_details": file_details[:10],  # Top 10 files by size
        }

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results.
            iterations_left: Remaining iterations (budget info).

        Returns:
            List of message dictionaries representing the conversation history
            from the agentic loop. Each dictionary contains 'role' and 'text' keys.
            
        Raises:
            FileNotFoundError: If the repo_path does not exist.
            ValueError: If paths are invalid or iterations_left is negative.
            RuntimeError: If the agent fails after max_retries attempts.
        """
        start_time = time.time()
        
        # Validate iterations_left if provided
        if iterations_left is not None and iterations_left < 0:
            raise ValueError(f"iterations_left must be non-negative, got {iterations_left}")
        
        # Validate paths
        repo, eval_p = self._validate_paths(repo_path, eval_path)
        
        # Get repository summary and eval analysis
        summary = self._get_repo_summary(repo)
        eval_analysis = self._analyze_eval_results(eval_p)
        
        self.log_fn(f"Starting meta-agent on repository: {repo_path}")
        self.log_fn(f"Repository stats: {summary['python_files']} Python files, {summary['total_lines']} lines")
        
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        # Build a more informative instruction with context
        file_list = "\n".join([f"  - {f['path']} ({f['lines']} lines)" for f in summary['file_details']])
        
        # Add eval context if available
        eval_context = ""
        if eval_analysis.get("available"):
            eval_context = f"""
Evaluation Results Context:
- Result files found: {eval_analysis.get('result_files', 0)}
"""
            if eval_analysis.get("latest_results"):
                eval_context += "- Recent results available for analysis\n"
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

Repository context:
- {summary['python_files']} Python files
- {summary['total_lines']} total lines of code
- Strategies directory: {'Yes' if summary['strategies_dir'] else 'No'}
- Evaluations directory: {'Yes' if summary['evals_dir'] else 'No'}

Key files (top 10 by size):
{file_list}
{eval_context}
Focus on improving the task_agent.py and meta_agent.py files which are the core components.

Guidelines for modifications:
1. Maintain backward compatibility with existing interfaces
2. Add robust error handling for edge cases
3. Improve JSON parsing and extraction reliability
4. Add comprehensive logging for debugging
5. Follow Python best practices and type hints
6. Consider evaluation results when making improvements"""

        # Attempt with retry logic for transient failures
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    self.log_fn(f"Retry attempt {attempt}/{self.max_retries}...")
                
                msg_history = chat_with_agent(
                    msg=instruction,
                    model=self.model,
                    temperature=self.temperature,
                    msg_history=[],
                    log_fn=self.log_fn,
                    tools_available="all",
                )
                
                elapsed = time.time() - start_time
                self.log_fn(
                    f"Meta-agent completed successfully in {elapsed:.2f}s "
                    f"with {len(msg_history)} messages"
                )
                
                # Log summary of changes made
                self._log_modifications_summary(repo, summary)
                
                return msg_history
                
            except Exception as e:
                last_error = e
                self.log_fn(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)  # Exponential backoff
        
        # All retries exhausted
        raise RuntimeError(
            f"Meta-agent failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    def _log_modifications_summary(self, repo: Path, original_summary: dict) -> None:
        """Log a summary of modifications made to the repository.
        
        Args:
            repo: Path to the repository.
            original_summary: The repository summary before modifications.
        """
        try:
            new_summary = self._get_repo_summary(repo)
            
            # Compare file counts
            file_diff = new_summary['python_files'] - original_summary['python_files']
            line_diff = new_summary['total_lines'] - original_summary['total_lines']
            
            if file_diff != 0:
                self.log_fn(f"Files changed: {file_diff:+d} files")
            if line_diff != 0:
                self.log_fn(f"Lines changed: {line_diff:+d} lines")
            
            # Check for new files
            original_files = {f['path'] for f in original_summary.get('file_details', [])}
            new_files = {f['path'] for f in new_summary.get('file_details', [])}
            
            added_files = new_files - original_files
            removed_files = original_files - new_files
            
            if added_files:
                self.log_fn(f"New files added: {', '.join(added_files)}")
            if removed_files:
                self.log_fn(f"Files removed: {', '.join(removed_files)}")
                
        except Exception as e:
            self.log_fn(f"Warning: Could not generate modifications summary: {e}")
