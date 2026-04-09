"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import json
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
        retry_delay: Base delay between retries in seconds.
    """

    def __init__(
        self, 
        model: str = META_MODEL, 
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        log_fn: Callable | None = None,
    ) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use. Defaults to META_MODEL.
            temperature: The temperature for LLM sampling. Defaults to 0.0.
            max_retries: Maximum retry attempts for failed operations. Defaults to 3.
            retry_delay: Base delay between retries in seconds. Defaults to 0.5.
            log_fn: Custom logging function. Defaults to logger.info.
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max(max_retries, 1)  # Ensure at least 1 retry
        self.retry_delay = max(retry_delay, 0.1)  # Ensure minimum delay
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
            logger.debug("_analyze_eval_results: Eval path does not exist")
            return {"available": False}
        
        try:
            # Look for result files in the eval directory
            result_files = list(eval_p.rglob("*.json"))
            
            if not result_files:
                logger.debug("_analyze_eval_results: No result files found")
                return {"available": True, "result_files": 0}
            
            # Basic statistics
            total_files = len(result_files)
            logger.debug(f"_analyze_eval_results: Found {total_files} result files")
            
            # Try to read the most recent results
            latest_results = []
            report_data = None
            
            # Sort by modification time (most recent first)
            sorted_files = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)
            
            for rf in sorted_files[:10]:  # Check up to 10 most recent files
                try:
                    with open(rf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Prioritize report.json files which contain summary data
                    if rf.name == "report.json" and not report_data:
                        report_data = {
                            "file": str(rf.relative_to(eval_p)),
                            "data": data
                        }
                    else:
                        latest_results.append({
                            "file": str(rf.relative_to(eval_p)),
                            "data": data
                        })
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"_analyze_eval_results: Could not parse JSON from {rf}: {e}")
                except Exception as e:
                    logger.warning(f"_analyze_eval_results: Error reading {rf}: {e}")
            
            # Build analysis summary
            analysis = {
                "available": True,
                "result_files": total_files,
                "latest_results": latest_results[:5]  # Limit to 5 results
            }
            
            # Add report data if available
            if report_data:
                analysis["report_summary"] = report_data["data"]
                
                # Extract key metrics if available
                report = report_data["data"]
                if isinstance(report, dict):
                    if "overall_accuracy" in report:
                        analysis["overall_accuracy"] = report["overall_accuracy"]
                    if "accuracy_by_label" in report:
                        analysis["accuracy_by_label"] = report["accuracy_by_label"]
                        
                        # Identify weak areas
                        weak_labels = []
                        for label, metrics in report["accuracy_by_label"].items():
                            if isinstance(metrics, dict):
                                precision = metrics.get("precision", 0)
                                recall = metrics.get("recall", 0)
                                if precision < 0.5 or recall < 0.5:
                                    weak_labels.append({
                                        "label": label,
                                        "precision": precision,
                                        "recall": recall
                                    })
                        
                        if weak_labels:
                            analysis["weak_areas"] = weak_labels
            
            return analysis
            
        except Exception as e:
            logger.error(f"_analyze_eval_results: Error analyzing eval results: {e}", exc_info=True)
            return {"available": True, "error": str(e)}

    def _get_repo_summary(self, repo: Path) -> dict:
        """Generate a summary of the repository structure.
        
        Args:
            repo: Path to the repository.
            
        Returns:
            Dictionary containing repository statistics.
        """
        if not repo.exists():
            logger.error(f"_get_repo_summary: Repository path does not exist: {repo}")
            return {
                "python_files": 0,
                "total_lines": 0,
                "strategies_dir": False,
                "evals_dir": False,
                "file_details": [],
                "error": "Repository path does not exist"
            }
        
        try:
            py_files = list(repo.rglob("*.py"))
        except Exception as e:
            logger.error(f"_get_repo_summary: Error finding Python files: {e}")
            py_files = []
        
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
            except PermissionError as e:
                logger.warning(f"_get_repo_summary: Permission denied reading {f}: {e}")
            except Exception as e:
                logger.debug(f"_get_repo_summary: Could not read {f}: {e}")
        
        # Sort by line count descending to show most important files first
        file_details.sort(key=lambda x: x["lines"], reverse=True)
        
        # Check for special directories
        strategies_exists = (repo / "strategies").exists()
        evals_exists = (repo / "evals").exists()
        
        summary = {
            "python_files": len(py_files),
            "total_lines": total_lines,
            "strategies_dir": strategies_exists,
            "evals_dir": evals_exists,
            "file_details": file_details[:10],  # Top 10 files by size
        }
        
        # Add strategies info if available
        if strategies_exists:
            try:
                strategies = repo / "strategies"
                strategy_dirs = [d for d in strategies.iterdir() if d.is_dir()]
                summary["strategy_count"] = len(strategy_dirs)
            except Exception as e:
                logger.debug(f"_get_repo_summary: Error counting strategies: {e}")
        
        logger.debug(f"_get_repo_summary: Found {len(py_files)} Python files, {total_lines} total lines")
        
        return summary

    def _build_instruction(
        self, 
        repo_path: str, 
        summary: dict, 
        eval_analysis: dict,
        iterations_left: int | None = None
    ) -> str:
        """Build the instruction for the meta-agent.
        
        Args:
            repo_path: Path to the repository.
            summary: Repository summary dictionary.
            eval_analysis: Evaluation analysis dictionary.
            iterations_left: Remaining iterations (optional).
            
        Returns:
            The formatted instruction string.
        """
        file_list = "\n".join([f"  - {f['path']} ({f['lines']} lines)" for f in summary['file_details']])
        
        # Add eval context if available
        eval_context = ""
        if eval_analysis.get("available"):
            eval_context = f"""
Evaluation Results Context:
- Result files found: {eval_analysis.get('result_files', 0)}
"""
            if eval_analysis.get("overall_accuracy") is not None:
                eval_context += f"- Overall accuracy: {eval_analysis['overall_accuracy']:.2%}\n"
            
            if eval_analysis.get("weak_areas"):
                eval_context += "- Areas needing improvement:\n"
                for area in eval_analysis["weak_areas"]:
                    eval_context += f"  * {area['label']}: precision={area['precision']:.2f}, recall={area['recall']:.2f}\n"
            
            if eval_analysis.get("latest_results"):
                eval_context += "- Recent results available for analysis\n"
        
        # Add iteration context
        iteration_context = ""
        if iterations_left is not None:
            iteration_context = f"\nIterations remaining: {iterations_left}\n"
        
        # Add specific guidance for known weak areas
        weak_area_guidance = ""
        if eval_analysis.get("weak_areas"):
            for area in eval_analysis["weak_areas"]:
                if area['label'] == 'almost' and area['precision'] == 0.0 and area['recall'] == 0.0:
                    weak_area_guidance = """

CRITICAL FOCUS AREA - "almost" grade (0% precision, 0% recall):
The agent is completely failing to correctly identify "almost" grades. This is the most critical issue to fix.

The "almost" grade means:
- Nearly complete solution with only minor issues
- Main proof structure is correct
- Only small computational errors or minor omissions
- Student understood the complete approach

Common confusion: "almost" vs "partial"
- "almost": Complete structure, minor issues only (nearly correct)
- "partial": Significant progress but incomplete (major gaps remain)

To fix this in task_agent.py:
1. Improve the prompt to clearly distinguish "almost" from "partial"
2. Add explicit examples showing the difference
3. Ensure grade extraction prioritizes "almost" detection
4. Add decision criteria in the prompt to help the LLM choose correctly
"""
        
        return f"""Modify any part of the codebase at `{repo_path}`.

Repository context:
- {summary['python_files']} Python files
- {summary['total_lines']} total lines of code
- Strategies directory: {'Yes' if summary['strategies_dir'] else 'No'}
- Evaluations directory: {'Yes' if summary['evals_dir'] else 'No'}

Key files (top 10 by size):
{file_list}
{eval_context}{weak_area_guidance}{iteration_context}
Focus on improving the task_agent.py and meta_agent.py files which are the core components.

Guidelines for modifications:
1. Maintain backward compatibility with existing interfaces
2. Add robust error handling for edge cases
3. Improve JSON parsing and extraction reliability
4. Add comprehensive logging for debugging
5. Follow Python best practices and type hints
6. Consider evaluation results when making improvements
7. Focus on weak areas identified in evaluation analysis
8. Ensure all changes are well-tested and don't break existing functionality"""

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
        try:
            repo, eval_p = self._validate_paths(repo_path, eval_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"forward: Path validation failed: {e}")
            raise
        
        # Get repository summary and eval analysis
        try:
            summary = self._get_repo_summary(repo)
            eval_analysis = self._analyze_eval_results(eval_p)
        except Exception as e:
            logger.error(f"forward: Error gathering context: {e}", exc_info=True)
            # Continue with empty analysis
            summary = {"python_files": 0, "total_lines": 0, "file_details": []}
            eval_analysis = {"available": False}
        
        self.log_fn(f"Starting meta-agent on repository: {repo_path}")
        self.log_fn(f"Repository stats: {summary['python_files']} Python files, {summary['total_lines']} lines")
        
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        # Build instruction
        instruction = self._build_instruction(repo_path, summary, eval_analysis, iterations_left)
        
        # Log weak areas if identified
        if eval_analysis.get("weak_areas"):
            self.log_fn(f"Identified weak areas: {[a['label'] for a in eval_analysis['weak_areas']]}")

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
                logger.error(f"forward: Attempt {attempt} failed: {e}", exc_info=True)
                self.log_fn(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * attempt  # Exponential backoff
                    logger.debug(f"forward: Sleeping for {sleep_time:.2f}s before retry")
                    time.sleep(sleep_time)
        
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
            if not repo.exists():
                logger.warning("_log_modifications_summary: Repository path no longer exists")
                return
                
            new_summary = self._get_repo_summary(repo)
            
            # Handle case where summary generation failed
            if new_summary.get("error"):
                logger.warning(f"_log_modifications_summary: Could not get new summary: {new_summary['error']}")
                return
            
            # Compare file counts
            file_diff = new_summary['python_files'] - original_summary.get('python_files', 0)
            line_diff = new_summary['total_lines'] - original_summary.get('total_lines', 0)
            
            if file_diff != 0:
                self.log_fn(f"Files changed: {file_diff:+d} files")
            if line_diff != 0:
                self.log_fn(f"Lines changed: {line_diff:+d} lines")
            
            if file_diff == 0 and line_diff == 0:
                self.log_fn("No file changes detected")
            
            # Check for new files
            original_files = {f['path'] for f in original_summary.get('file_details', [])}
            new_files = {f['path'] for f in new_summary.get('file_details', [])}
            
            added_files = new_files - original_files
            removed_files = original_files - new_files
            modified_files = new_files & original_files
            
            if added_files:
                self.log_fn(f"New files added: {', '.join(sorted(added_files))}")
            if removed_files:
                self.log_fn(f"Files removed: {', '.join(sorted(removed_files))}")
            
            # Check for modified files by comparing line counts
            if modified_files:
                modified_count = 0
                for f in original_summary.get('file_details', []):
                    if f['path'] in modified_files:
                        # Find matching file in new summary
                        for new_f in new_summary.get('file_details', []):
                            if new_f['path'] == f['path']:
                                if new_f['lines'] != f['lines']:
                                    modified_count += 1
                                break
                if modified_count > 0:
                    self.log_fn(f"Files modified: {modified_count}")
                
        except Exception as e:
            logger.error(f"_log_modifications_summary: Could not generate modifications summary: {e}", exc_info=True)
            self.log_fn(f"Warning: Could not generate modifications summary: {e}")
