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
        
        # Validate model availability
        if not self.model:
            logger.warning("MetaAgent initialized with empty model name")
            self.model = META_MODEL  # Fallback to default

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
        
        if not eval_p.is_dir():
            logger.debug("_analyze_eval_results: Eval path is not a directory")
            return {"available": False, "error": "Not a directory"}
        
        try:
            # Look for result files in the eval directory
            try:
                result_files = list(eval_p.rglob("*.json"))
            except PermissionError as e:
                logger.error(f"_analyze_eval_results: Permission denied searching for result files: {e}")
                return {"available": True, "error": "Permission denied", "result_files": 0}
            except Exception as e:
                logger.error(f"_analyze_eval_results: Error searching for result files: {e}")
                result_files = []
            
            if not result_files:
                logger.debug("_analyze_eval_results: No result files found")
                return {"available": True, "result_files": 0}
            
            # Analyze report.json files for detailed metrics
            report_files = [f for f in result_files if f.name == "report.json"]
            analysis = {
                "available": True,
                "result_files": len(result_files),
                "report_files": len(report_files),
            }
            
            # Parse report files for accuracy metrics
            total_accuracy = 0
            report_count = 0
            grade_issues = {}
            
            for report_file in report_files:
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    if 'overall_accuracy' in report_data:
                        total_accuracy += report_data['overall_accuracy']
                        report_count += 1
                    
                    # Check for grade-specific issues
                    if 'accuracy_by_label' in report_data:
                        for grade, metrics in report_data['accuracy_by_label'].items():
                            if isinstance(metrics, dict):
                                precision = metrics.get('precision', 1.0)
                                recall = metrics.get('recall', 1.0)
                                
                                # Flag grades with low precision or recall
                                if precision < 0.5 or recall < 0.5:
                                    if grade not in grade_issues:
                                        grade_issues[grade] = []
                                    grade_issues[grade].append({
                                        'file': str(report_file),
                                        'precision': precision,
                                        'recall': recall,
                                    })
                                    
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"_analyze_eval_results: Error parsing {report_file}: {e}")
                    continue
            
            if report_count > 0:
                analysis['average_accuracy'] = total_accuracy / report_count
            
            if grade_issues:
                analysis['grade_issues'] = grade_issues
                # Identify critical issues (0% precision or recall)
                critical_grades = [
                    grade for grade, issues in grade_issues.items()
                    if any(i.get('precision', 1.0) == 0.0 or i.get('recall', 1.0) == 0.0 for i in issues)
                ]
                if critical_grades:
                    analysis['critical_grades'] = critical_grades
            
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
        
        if not repo.is_dir():
            logger.error(f"_get_repo_summary: Repository path is not a directory: {repo}")
            return {
                "python_files": 0,
                "total_lines": 0,
                "strategies_dir": False,
                "evals_dir": False,
                "file_details": [],
                "error": "Repository path is not a directory"
            }
        
        try:
            py_files = list(repo.rglob("*.py"))
        except PermissionError as e:
            logger.error(f"_get_repo_summary: Permission denied searching for Python files: {e}")
            py_files = []
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
            except IsADirectoryError:
                logger.debug(f"_get_repo_summary: Skipping directory {f}")
            except UnicodeDecodeError as e:
                logger.debug(f"_get_repo_summary: Unicode error reading {f}: {e}")
            except Exception as e:
                logger.debug(f"_get_repo_summary: Could not read {f}: {e}")
        
        # Sort by line count descending to show most important files first
        file_details.sort(key=lambda x: x["lines"], reverse=True)
        
        # Check for special directories
        try:
            strategies_exists = (repo / "strategies").exists() and (repo / "strategies").is_dir()
        except Exception:
            strategies_exists = False
            
        try:
            evals_exists = (repo / "evals").exists() and (repo / "evals").is_dir()
        except Exception:
            evals_exists = False
        
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
        
        # Add evals info if available
        if evals_exists:
            try:
                evals = repo / "evals"
                eval_dirs = [d for d in evals.iterdir() if d.is_dir()]
                summary["eval_count"] = len(eval_dirs)
            except Exception as e:
                logger.debug(f"_get_repo_summary: Error counting evals: {e}")
        
        logger.debug(f"_get_repo_summary: Found {len(py_files)} Python files, {total_lines} total lines")
        
        return summary

    def _extract_weak_areas(self, eval_analysis: dict) -> list[dict]:
        """Extract weak areas from evaluation analysis with robust error handling.
        
        Args:
            eval_analysis: Evaluation analysis dictionary.
            
        Returns:
            List of weak area dictionaries.
        """
        weak_areas = []
        
        if not eval_analysis or not eval_analysis.get("available"):
            return weak_areas
        
        try:
            # Fix: Use 'accuracy_by_label' which is the actual key in report.json files
            grade_analysis = eval_analysis.get("accuracy_by_label", {})
            if not isinstance(grade_analysis, dict):
                # Fallback: try 'grade_analysis' for backward compatibility
                grade_analysis = eval_analysis.get("grade_analysis", {})
                if not isinstance(grade_analysis, dict):
                    return weak_areas
            
            for grade in ['almost', 'partial', 'incorrect', 'correct']:
                if grade in grade_analysis:
                    grade_data = grade_analysis[grade]
                    if not isinstance(grade_data, dict):
                        continue
                    
                    # Safely extract numeric values with type checking
                    try:
                        precision = float(grade_data.get('precision', 0.0)) if grade_data.get('precision') is not None else 0.0
                        recall = float(grade_data.get('recall', 0.0)) if grade_data.get('recall') is not None else 0.0
                        total = int(grade_data.get('total', 0)) if grade_data.get('total') is not None else 0
                        correct = int(grade_data.get('correct', 0)) if grade_data.get('correct') is not None else 0
                    except (TypeError, ValueError) as e:
                        logger.warning(f"_extract_weak_areas: Invalid numeric values for grade {grade}: {e}")
                        continue
                    
                    # Skip if no data
                    if total == 0:
                        continue
                    
                    # Calculate F1 score safely
                    try:
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    except (TypeError, ValueError):
                        f1 = 0.0
                    
                    # Define thresholds for weak areas - more aggressive thresholds
                    # CRITICAL: "incorrect" with low recall means we're being too generous
                    if grade == 'incorrect':
                        # Low recall on "incorrect" is the MOST critical issue - means we're being too generous
                        if recall < 0.60:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'critical',
                                'message': f"'{grade}' grade has recall={recall:.2f}. CRITICAL: Under-predicting 'incorrect' - being too generous. Need to be stricter about what constitutes 'meaningful progress'"
                            })
                        elif precision < 0.60 or f1 < 0.50:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'high',
                                'message': f"'{grade}' grade has precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}. Need improvement"
                            })
                    elif grade == 'almost':
                        # "almost" is critical - zero tolerance for low performance
                        if precision < 0.50 or recall < 0.50 or f1 < 0.50:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'critical',
                                'message': f"'{grade}' grade has precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}. CRITICAL: Need to improve distinction between 'almost' and 'partial'"
                            })
                        elif precision < 0.60 or recall < 0.60 or f1 < 0.60:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'high',
                                'message': f"'{grade}' grade has precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}. Need improvement"
                            })
                    elif grade == 'partial':
                        if precision < 0.50 or recall < 0.50 or f1 < 0.50:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'high',
                                'message': f"'{grade}' grade has precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}. Confusion with other grades"
                            })
                        elif precision < 0.60 or recall < 0.60 or f1 < 0.60:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'medium',
                                'message': f"'{grade}' grade has precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}. Could be improved"
                            })
                    elif grade == 'correct':
                        # Low precision on "correct" means over-predicting
                        if precision < 0.70 or recall < 0.70 or f1 < 0.70:
                            weak_areas.append({
                                'label': grade,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'total': total,
                                'correct': correct,
                                'severity': 'medium',
                                'message': f"'{grade}' grade has precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}. Over-predicting 'correct' - being too generous"
                            })
        except Exception as e:
            logger.error(f"_extract_weak_areas: Error extracting weak areas: {e}", exc_info=True)
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        weak_areas.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 4))
        
        return weak_areas

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
        # Safely get file details
        file_details = summary.get('file_details', [])
        if not isinstance(file_details, list):
            file_details = []
        
        file_list = "\n".join([f"  - {f.get('path', 'unknown')} ({f.get('lines', 0)} lines)" for f in file_details[:10]])
        
        # Add eval context if available
        eval_context = ""
        if eval_analysis and eval_analysis.get("available"):
            eval_context = f"""
Evaluation Results Context:
- Result files found: {eval_analysis.get('result_files', 0)}
"""
            overall_accuracy = eval_analysis.get("overall_accuracy")
            if overall_accuracy is not None:
                try:
                    eval_context += f"- Overall accuracy: {float(overall_accuracy):.2%}\n"
                except (TypeError, ValueError):
                    eval_context += f"- Overall accuracy: {overall_accuracy}\n"
            
            # Use robust weak area extraction
            weak_areas = self._extract_weak_areas(eval_analysis)
            if weak_areas:
                eval_context += "- Areas needing improvement:\n"
                for area in weak_areas:
                    eval_context += f"  * {area['label']}: precision={area['precision']:.2f}, recall={area['recall']:.2f} (severity: {area.get('severity', 'unknown')})\n"
            
            if eval_analysis.get("latest_results"):
                eval_context += "- Recent results available for analysis\n"
        
        # Add iteration context
        iteration_context = ""
        if iterations_left is not None:
            iteration_context = f"\nIterations remaining: {iterations_left}\n"
        
        # Add specific guidance for known weak areas
        weak_area_guidance = ""
        weak_areas = self._extract_weak_areas(eval_analysis)
        if weak_areas:
            for area in weak_areas:
                if area['label'] == 'almost' and (area['precision'] < 0.5 or area['recall'] < 0.5):
                    weak_area_guidance += """

CRITICAL FOCUS AREA - "almost" grade (low precision/recall):
The agent is failing to correctly identify "almost" grades. This is the most critical issue to fix.

The "almost" grade means:
- Nearly complete solution with only minor issues
- Main proof structure is correct and COMPLETE
- Only small computational errors or minor omissions
- Student understood the complete approach
- Complete proof structure from start to finish INCLUDING the conclusion
- If you fix trivial errors (arithmetic, typos), it becomes correct
- The student has addressed ALL major components

Common confusion: "almost" vs "partial"
- "almost": Complete structure, minor issues only (nearly correct). The student has addressed ALL major components.
- "partial": Significant progress but incomplete (major gaps remain). Needs substantial new work to be complete.

THE KEY TEST: Ask "Does the student have a COMPLETE proof structure?"
- YES (complete structure, minor blemishes) → "almost"
- NO (missing components, incomplete) → "partial"

THE "CONCLUSION" TEST (Critical for distinguishing almost vs partial):
- Did the student REACH the final conclusion/answer? (Even if with minor errors) → Strong indicator for "almost"
- Did the student STOP before the conclusion? → Strong indicator for "partial"

THE "FIXABILITY" TEST (Apply this rigorously!):
- Can you fix the solution by correcting ONLY trivial errors (arithmetic, typos, notation)? → "almost"
- Would you need to add new proof steps, lemmas, or cases? → "partial"

THE "WHAT'S MISSING" TEST:
- "almost": Nothing major is missing - only tiny errors exist
- "partial": Something major is missing - conclusion, cases, lemmas, or proof steps

To fix this in task_agent.py:
1. Improve the prompt to clearly distinguish "almost" from "partial"
2. Add explicit examples showing the difference (at least 70 examples)
3. Ensure grade extraction prioritizes "almost" detection in the conclusion
4. Add decision criteria in the prompt: "If you can fix the solution by correcting only trivial errors → 'almost'"
5. Add the "COMPLETE STRUCTURE" test to the prompt
6. Add the "CONCLUSION" test to the prompt
7. Add a verification checklist at the end of the prompt with explicit YES/NO questions
8. Include more "almost" vs "partial" scenarios in examples covering all problem types
9. Make the prompt emphasize that "almost" requires ALL major components to be present
10. Add examples showing "almost" with complete proofs but wrong final numbers
11. Add examples showing "almost" with complete induction proofs but small errors
12. Add examples showing "almost" with complete geometry proofs but diagram errors
13. Add examples showing "almost" with complete combinatorics but off-by-one errors
14. Add examples showing "almost" with complete number theory but sign errors
15. Emphasize that "almost" is for COMPLETE proofs with MINOR issues only
16. Add ANTI-BIAS checks to prevent being too strict with "almost"
17. Add the "PROGRESS TEST" to distinguish "partial" from "incorrect"
18. Add examples showing "almost" with complete functional equations but domain errors
19. Add examples showing "partial" with missing final connections
20. Add examples showing "almost" with complete proofs but notation switches
21. Add a QUICK DECISION TREE to help the LLM navigate the grading logic
22. Add examples showing "almost" with sequence proofs but index errors
23. Add examples showing "almost" with graph theory proofs but edge count errors
24. Add examples showing "almost" with recurrence solutions but boundary errors
25. Add examples showing "almost" with modular arithmetic but calculation slips
26. Add examples showing "almost" with optimization proofs but derivative errors
27. Add examples showing "almost" with complete proofs but transposed digits
28. Add examples showing "almost" with existence proofs but uniqueness gaps
29. Add examples showing "almost" with pigeonhole proofs but off-by-one errors
30. Add examples showing "partial" with missing existence proofs
31. Add a CRITICAL DECISION FRAMEWORK section to the prompt with exact steps
32. Ensure grade extraction logic checks "almost" before "correct" in conclusion detection
33. Add more "almost" variations to the _normalize_grade function
34. Improve the _extract_grade_from_text function to prioritize "almost" detection
35. Add specific handling for "almost" in the last 500 chars of the response
"""
                elif area['label'] == 'partial' and (area['precision'] < 0.6 or area['recall'] < 0.5):
                    weak_area_guidance += """

FOCUS AREA - "partial" grade (low precision/recall):
The agent is struggling with "partial" grade identification.

The "partial" grade means:
- Significant progress but INCOMPLETE
- Key lemmas or invariants found but not fully utilized
- Major gaps exist in the proof
- Student demonstrated understanding but didn't complete the argument
- The student has NOT addressed all major components
- Missing conclusion or final step is a KEY indicator
- Proof structure is incomplete (stops partway through)
- Needs substantial new work to be complete

Common confusion: "partial" vs "almost"
- "partial": Incomplete structure, missing major components, needs substantial new work
- "almost": Complete structure, only minor issues, trivial fixes needed

THE KEY TEST: Ask "Does the student have a COMPLETE proof structure?"
- YES (complete structure, minor blemishes) → "almost"
- NO (missing components, incomplete) → "partial"

THE "CONCLUSION" TEST (Critical for distinguishing almost vs partial):
- Did the student REACH the final conclusion/answer? (Even if with minor errors) → Strong indicator for "almost"
- Did the student STOP before the conclusion? → Strong indicator for "partial"

THE "PROGRESS TEST" (Critical for distinguishing partial vs incorrect):
- Did the student make MEANINGFUL progress (found lemma, proved intermediate result, established approach)? → "partial"
- Did the student make minimal or no progress (just restated problem, random calculations)? → "incorrect"

To improve:
1. Ensure the prompt clearly defines what constitutes "significant progress"
2. Add examples showing when to award "partial" vs "incorrect"
3. Check that grade extraction correctly identifies "partial" in responses
4. Emphasize that "partial" is for incomplete solutions with meaningful progress
5. Add the "CONCLUSION" test to help distinguish from "almost"
6. Add the "PROGRESS" test to help distinguish from "incorrect"
7. Add examples showing "partial" with missing final connections
8. Add examples showing "partial" with incomplete induction
9. Add examples showing "partial" with incomplete case analysis
10. Add examples showing "partial" with missing sequence convergence
11. Add examples showing "partial" with incomplete graph coloring
12. Add examples showing "partial" with recurrence setup only
13. Add examples showing "partial" with modular setup but no conclusion
14. Add examples showing "partial" with optimization setup only
15. Add examples showing "partial" with missing existence proof
16. Add examples showing "partial" with only uniqueness proof
17. Add examples showing "partial" with construction attempt but invalid
18. Add examples showing "partial" with contradiction setup but no contradiction reached
19. Add examples showing "partial" with pigeonhole setup but wrong application
"""
                elif area['label'] == 'correct' and area['precision'] < 0.7:
                    weak_area_guidance += """

FOCUS AREA - "correct" grade (low precision):
The agent is over-predicting "correct" grades (false positives).

This means:
- Solutions are being marked "correct" when they should be "almost" or "partial"
- The bar for "correct" may be set too low

To improve:
1. Strengthen the definition of "correct" in the prompt (must be FLAWLESS)
2. Emphasize that "correct" requires NO gaps or errors whatsoever
3. Ensure the extraction logic doesn't default to "correct"
4. Add explicit "correct" vs "almost" examples showing the difference
"""
                elif area['label'] == 'incorrect' and area['recall'] < 0.5:
                    weak_area_guidance += """

FOCUS AREA - "incorrect" grade (low recall):
The agent is under-predicting "incorrect" grades (missing false negatives).

This means:
- Solutions that should be "incorrect" are being marked as "partial" or higher
- The agent is being too generous
- "partial" is being awarded for minimal progress when it should be "incorrect"

The "incorrect" grade means:
- No meaningful progress or fundamental errors
- Wrong approach or no valid mathematical progress
- Student failed to demonstrate understanding of key concepts
- Minimal or no substantive work toward the solution
- Just restating the problem without progress is NOT "partial", it's "incorrect"

THE "PROGRESS TEST" (Critical for distinguishing partial vs incorrect):
- Did the student make MEANINGFUL progress (found lemma, proved intermediate result, established valid approach)? → "partial"
- Did the student make minimal or no progress (just restated problem, random calculations, wrong approach)? → "incorrect"

To improve:
1. Strengthen the definition of "incorrect" (no meaningful progress)
2. Ensure "partial" is not awarded for minimal progress
3. Add "partial" vs "incorrect" examples to clarify the boundary
4. Add the "PROGRESS TEST" to the prompt
5. Emphasize that "partial" requires SUBSTANTIVE progress, not just "attempted"
6. Add examples showing "incorrect" for solutions with only problem restatement
7. Add examples showing "incorrect" for solutions with fundamentally wrong approaches
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
            summary = {"python_files": 0, "total_lines": 0, "file_details": [], "strategies_dir": False, "evals_dir": False}
            eval_analysis = {"available": False}
        
        self.log_fn(f"Starting meta-agent on repository: {repo_path}")
        self.log_fn(f"Repository stats: {summary.get('python_files', 0)} Python files, {summary.get('total_lines', 0)} lines")
        
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        # Build instruction
        instruction = self._build_instruction(repo_path, summary, eval_analysis, iterations_left)
        
        # Log weak areas if identified
        weak_areas = self._extract_weak_areas(eval_analysis)
        if weak_areas:
            self.log_fn(f"Identified weak areas: {[a['label'] for a in weak_areas]}")
        
        # Log critical issues if identified
        if eval_analysis.get("critical_issues"):
            for issue in eval_analysis["critical_issues"]:
                self.log_fn(f"Critical issue: {issue.get('message', 'Unknown issue')}")

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
            
            if not isinstance(original_summary, dict):
                logger.warning(f"_log_modifications_summary: Invalid original summary type: {type(original_summary)}")
                return
                
            new_summary = self._get_repo_summary(repo)
            
            # Handle case where summary generation failed
            if new_summary.get("error"):
                logger.warning(f"_log_modifications_summary: Could not get new summary: {new_summary['error']}")
                return
            
            if not isinstance(new_summary, dict):
                logger.warning(f"_log_modifications_summary: Invalid new summary type: {type(new_summary)}")
                return
            
            # Compare file counts
            file_diff = new_summary.get('python_files', 0) - original_summary.get('python_files', 0)
            line_diff = new_summary.get('total_lines', 0) - original_summary.get('total_lines', 0)
            
            if file_diff != 0:
                self.log_fn(f"Files changed: {file_diff:+d} files")
            if line_diff != 0:
                self.log_fn(f"Lines changed: {line_diff:+d} lines")
            
            if file_diff == 0 and line_diff == 0:
                self.log_fn("No file changes detected")
            
            # Check for new files
            original_files = {f['path'] for f in original_summary.get('file_details', []) if isinstance(f, dict) and 'path' in f}
            new_files = {f['path'] for f in new_summary.get('file_details', []) if isinstance(f, dict) and 'path' in f}
            
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
                    if not isinstance(f, dict) or 'path' not in f:
                        continue
                    if f['path'] in modified_files:
                        # Find matching file in new summary
                        for new_f in new_summary.get('file_details', []):
                            if isinstance(new_f, dict) and new_f.get('path') == f['path']:
                                if new_f.get('lines') != f.get('lines'):
                                    modified_count += 1
                                break
                if modified_count > 0:
                    self.log_fn(f"Files modified: {modified_count}")
                
        except Exception as e:
            logger.error(f"_log_modifications_summary: Could not generate modifications summary: {e}", exc_info=True)
            self.log_fn(f"Warning: Could not generate modifications summary: {e}")
