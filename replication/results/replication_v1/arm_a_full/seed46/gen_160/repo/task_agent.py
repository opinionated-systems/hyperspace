"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from typing import Callable

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional fallback for JSON embedded in other text.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON objects directly in the text
    if not results:
        # Look for JSON-like structures with curly braces
        brace_start = text.find("{")
        while brace_start != -1:
            # Try to find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[brace_start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_start:brace_start + i + 1]
                        try:
                            parsed = json.loads(json_str)
                            # Only accept if it has expected fields
                            if any(key in parsed for key in ["response", "grade", "score", "analysis", "reasoning", "understanding"]):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
            brace_start = text.find("{", brace_start + 1)
    
    return results or None


def _log_extraction_details(text: str, results: list[dict] | None, logger_fn: Callable) -> None:
    """Log detailed information about JSON extraction for debugging.
    
    Args:
        text: The raw text that was processed
        results: The extracted JSON results (or None if failed)
        logger_fn: Logging function to use
    """
    if results:
        logger_fn(f"Successfully extracted {len(results)} JSON object(s)")
        for i, obj in enumerate(results):
            keys = list(obj.keys())
            logger_fn(f"  Object {i+1}: keys={keys}")
    else:
        # Log hints about why extraction might have failed
        has_json_tags = "<json>" in text and "</json>" in text
        has_code_blocks = "```" in text
        has_braces = "{" in text and "}" in text
        
        logger_fn("JSON extraction failed - diagnostic info:")
        logger_fn(f"  - Contains <json> tags: {has_json_tags}")
        logger_fn(f"  - Contains code blocks: {has_code_blocks}")
        logger_fn(f"  - Contains curly braces: {has_braces}")
        logger_fn(f"  - Text length: {len(text)} chars")
        
        # Show a snippet of the text for debugging
        snippet = text[:200].replace("\n", " ")
        logger_fn(f"  - Text preview: {snippet}...")


# Pre-compile grade map for faster lookups
_GRADE_MAP = {
    # Correct mappings
    "correct": "Correct",
    "right": "Correct",
    "true": "Correct",
    "yes": "Correct",
    "full": "Correct",
    "full credit": "Correct",
    "complete": "Correct",
    "solved": "Correct",
    "valid": "Correct",
    "acceptable": "Correct",
    "perfect": "Correct",
    "fully correct": "Correct",
    "entirely correct": "Correct",
    "completely correct": "Correct",
    "totally correct": "Correct",
    "100% correct": "Correct",
    "all correct": "Correct",
    "correct answer": "Correct",
    "correct solution": "Correct",
    # Almost mappings
    "almost": "Almost",
    "almost correct": "Almost",
    "nearly correct": "Almost",
    "mostly correct": "Almost",
    "minor errors": "Almost",
    "small mistakes": "Almost",
    "tiny error": "Almost",
    "nearly right": "Almost",
    "almost right": "Almost",
    "mostly right": "Almost",
    "essentially correct": "Almost",
    "fundamentally correct": "Almost",
    "minor issues": "Almost",
    "small error": "Almost",
    "slight error": "Almost",
    "careless mistake": "Almost",
    # Partial mappings
    "partial": "Partial",
    "partially correct": "Partial",
    "partial credit": "Partial",
    "partially": "Partial",
    "some credit": "Partial",
    "incomplete": "Partial",
    "unfinished": "Partial",
    "missing": "Partial",
    "mostly": "Partial",
    "significant progress": "Partial",
    "some understanding": "Partial",
    "partial solution": "Partial",
    "incomplete solution": "Partial",
    "partially solved": "Partial",
    "some progress": "Partial",
    "major errors": "Partial",
    "significant errors": "Partial",
    "substantial progress": "Partial",
    "meaningful progress": "Partial",
    "some correct": "Partial",
    "partly correct": "Partial",
    # Incorrect mappings
    "incorrect": "Incorrect",
    "wrong": "Incorrect",
    "false": "Incorrect",
    "no": "Incorrect",
    "none": "Incorrect",
    "zero": "Incorrect",
    "invalid": "Incorrect",
    "unacceptable": "Incorrect",
    "error": "Incorrect",
    "completely wrong": "Incorrect",
    "totally wrong": "Incorrect",
    "entirely wrong": "Incorrect",
    "fundamentally wrong": "Incorrect",
    "no progress": "Incorrect",
    "no understanding": "Incorrect",
    "no solution": "Incorrect",
    "wrong answer": "Incorrect",
    "incorrect answer": "Incorrect",
    "not correct": "Incorrect",
    "not right": "Incorrect",
    "failed": "Incorrect",
    "failure": "Incorrect",
}

# Pre-compile regex patterns for better performance
_NUMERIC_GRADE_PATTERN = re.compile(r'\b([0-7])\b')
_ALMOST_KEYWORDS = frozenset([
    "almost", "nearly", "minor", "small mistake", "tiny error", "careless",
    "essentially correct", "fundamentally correct", "nearly right", "almost right",
    "mostly right", "minor issues", "small error", "slight error", "trivial error"
])
_PARTIAL_KEYWORDS = frozenset([
    "incomplete", "unfinished", "missing", "significant progress", "some understanding",
    "partial solution", "incomplete solution", "partially solved", "some progress",
    "major errors", "significant errors", "substantial progress", "meaningful progress",
    "some correct", "partly correct", "partial work", "partial credit"
])
_CORRECT_KEYWORDS = frozenset([
    "correct", "right", "true", "full", "complete", "solved", "valid", "acceptable",
    "perfect", "fully correct", "entirely correct", "completely correct", "totally correct",
    "100% correct", "all correct", "correct answer", "correct solution"
])
_INCORRECT_KEYWORDS = frozenset([
    "incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable",
    "completely wrong", "totally wrong", "entirely wrong", "fundamentally wrong",
    "no progress", "no understanding", "no solution", "wrong answer", "incorrect answer",
    "not correct", "not right", "failed", "failure"
])

# Pre-compile regex patterns for grade detection
_GRADE_PATTERNS = {
    "incorrect": re.compile(r'\bincorrect\b', re.IGNORECASE),
    "correct": re.compile(r'\bcorrect\b', re.IGNORECASE),
    "almost": re.compile(r'\balmost\b', re.IGNORECASE),
    "partial": re.compile(r'\bpartial\b', re.IGNORECASE),
    "almost_correct": re.compile(r'\b(almost|nearly|mostly)\s+correct\b', re.IGNORECASE),
    "partial_correct": re.compile(r'\b(partial|partly)\s+correct\b', re.IGNORECASE),
}


@functools.lru_cache(maxsize=2048)
def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    Handles both string grades (Correct/Almost/Partial/Incorrect) and numeric scores (0-7).
    Uses LRU cache for performance optimization.
    
    Mapping:
    - 7 points -> "Correct" (full credit)
    - 5-6 points -> "Almost" (minor issues)
    - 1-4 points -> "Partial" (partial credit)
    - 0 points -> "Incorrect" (no credit)
    """
    if not grade:
        return "None"
    
    # Handle non-string types (int, float, etc.)
    if not isinstance(grade, str):
        grade = str(grade)
    
    grade = grade.strip()
    grade_lower = grade.lower()
    
    # First, check for numeric grades and map to categories
    # 7 = Correct, 5-6 = Almost, 1-4 = Partial, 0 = Incorrect
    if grade_lower == "7":
        return "Correct"
    if grade_lower in ("5", "6"):
        return "Almost"
    if grade_lower in ("1", "2", "3", "4"):
        return "Partial"
    if grade_lower == "0":
        return "Incorrect"
    
    # Check for exact match first (case-insensitive) using pre-compiled map
    if grade_lower in _GRADE_MAP:
        return _GRADE_MAP[grade_lower]
    
    # Try to extract numeric score from strings like "Score: 5" or "Grade: 3"
    numeric_match = _NUMERIC_GRADE_PATTERN.search(grade)
    if numeric_match:
        score = numeric_match.group(1)
        if score == "7":
            return "Correct"
        elif score in ("5", "6"):
            return "Almost"
        elif score == "0":
            return "Incorrect"
        else:
            return "Partial"
    
    # Check for exact grade labels in the text (highest priority)
    # Look for the four main categories as whole words using pre-compiled patterns
    if _GRADE_PATTERNS["incorrect"].search(grade_lower):
        return "Incorrect"
    # Check for "almost correct" or "partially correct" patterns first
    if _GRADE_PATTERNS["almost_correct"].search(grade_lower):
        return "Almost"
    if _GRADE_PATTERNS["partial_correct"].search(grade_lower):
        return "Partial"
    # Then check for standalone "correct" (but not if preceded by modifiers)
    if _GRADE_PATTERNS["correct"].search(grade_lower):
        return "Correct"
    if _GRADE_PATTERNS["almost"].search(grade_lower):
        return "Almost"
    if _GRADE_PATTERNS["partial"].search(grade_lower):
        return "Partial"
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" category first (more specific than partial)
    if any(word in grade_lower for word in _ALMOST_KEYWORDS):
        return "Almost"
    # Check for "partial" category
    if any(word in grade_lower for word in _PARTIAL_KEYWORDS):
        return "Partial"
    # Check for "correct" category - but be careful not to match "incorrect"
    if "incorrect" not in grade_lower and any(word in grade_lower for word in _CORRECT_KEYWORDS):
        return "Correct"
    # Check for "incorrect" category
    if any(word in grade_lower for word in _INCORRECT_KEYWORDS):
        return "Incorrect"
    
    # Default: capitalize first letter if it's one of the four categories
    if grade_lower in ("correct", "almost", "partial", "incorrect"):
        return grade.capitalize()
    
    # If we can't determine, return None to indicate uncertainty
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, problem: str, answer: str, official_solution: str, 
                             rubric: str, student_answer: str) -> str:
        """Build a comprehensive grading prompt with clear instructions and examples."""
        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematics problem.

## Problem Statement
{problem}

## Correct Answer
{answer}

## Official Solution
{official_solution}

## Grading Rubric
{rubric}

## Student's Answer
{student_answer}

## Your Task
Carefully analyze the student's answer and assign a grade using the following structured approach:

### Step 1: Understanding Assessment
- Verify you understand the problem correctly
- Identify the key mathematical concepts required
- Note the expected solution approach

### Step 2: Solution Analysis
- Check if the student's approach is mathematically valid
- Identify any errors, gaps, or incorrect assumptions
- Compare their approach to the official solution
- Check mathematical notation and terminology

### Step 3: Grade Assignment
Based on the rubric, assign ONE of these four grades:

**Correct** (7 points): Complete, correct solution with clear reasoning
- All steps are logically sound and mathematically valid
- Final answer is correct
- Clear explanation of the reasoning process

**Almost** (5-6 points): Minor errors or gaps, but essentially correct approach
- The core approach is correct and would lead to the right answer if executed properly
- Small computational errors, typos, or minor logical gaps
- Missing a trivial final step or simplification
- The student clearly understands the problem but made careless mistakes

**Partial** (1-4 points): Significant progress but incomplete or with major errors
- Shows understanding of the problem but solution is incomplete
- Major logical errors that invalidate parts of the solution
- Correct approach started but abandoned or executed incorrectly
- Some valid mathematical work but missing key insights

**Incorrect** (0 points): No meaningful progress or completely wrong approach
- No valid mathematical work shown
- Completely wrong approach with no redeeming value
- Answer is wrong with no supporting reasoning
- Fundamental misunderstanding of the problem

## Critical Distinctions

**Almost vs Partial**: 
- "Almost" means the solution is fundamentally correct with only minor issues that don't affect the core logic
- "Partial" means there are significant gaps or errors that prevent the solution from being essentially correct

**Partial vs Incorrect**:
- "Partial" requires showing some valid mathematical understanding or progress
- "Incorrect" means no meaningful progress toward a solution

## Important Rules
1. You MUST use exactly one of these four labels: Correct, Almost, Partial, or Incorrect
2. Be strict but fair - partial credit requires meaningful mathematical progress
3. When in doubt between two categories, choose the LOWER grade (be conservative)
4. "Almost" should be rare - only use it when the solution is truly nearly correct
5. "Partial" is for solutions that show genuine effort and some understanding but fall short

Respond in JSON format:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed analysis of the student's answer - what they got right and wrong",
    "reasoning": "Step-by-step reasoning for your grade decision - explain why you chose this category over alternatives",
    "response": "Your final grade: MUST be one of 'Correct', 'Almost', 'Partial', or 'Incorrect'"
}}
</json>"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Try to extract answer from solution if available
        answer = inputs.get("answer", "")
        if not answer and solution:
            # Try to extract final answer from solution
            lines = solution.strip().split('\n')
            for line in reversed(lines):
                if line.strip() and not line.strip().startswith('#'):
                    answer = line.strip()
                    break

        # Use the new prompt builder for better structured prompting
        instruction = self._build_grading_prompt(
            problem=problem,
            answer=answer,
            official_solution=solution,
            rubric=grading_guidelines,
            student_answer=student_answer
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            # Get the last assistant message
            last_assistant_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant" or "text" in msg:
                    last_assistant_msg = msg.get("text", msg.get("content", ""))
                    break
            
            if not last_assistant_msg:
                self.log_fn("Warning: No assistant message found in history")
                return str(prediction), msg_history
            
            extracted = _extract_jsons(last_assistant_msg)
            
            # Log extraction details for debugging
            _log_extraction_details(last_assistant_msg, extracted, self.log_fn)
            
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                
                # Priority order for grade extraction - response field has highest priority
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Validate the prediction is one of the four valid categories
                valid_grades = {"Correct", "Almost", "Partial", "Incorrect"}
                if prediction not in valid_grades and prediction != "None":
                    self.log_fn(f"Warning: Invalid grade '{prediction}', attempting to re-normalize")
                    # Try to extract from reasoning field if available
                    if "reasoning" in last_extract:
                        reasoning_grade = _normalize_grade(last_extract["reasoning"])
                        if reasoning_grade in valid_grades:
                            prediction = reasoning_grade
                            self.log_fn(f"Extracted grade from reasoning: {prediction}")
                
                # Log detailed analysis for debugging
                if "analysis" in last_extract:
                    self.log_fn(f"Analysis: {last_extract['analysis'][:200]}...")
                if "partial_credit_reasoning" in last_extract:
                    self.log_fn(f"Partial Credit: {last_extract['partial_credit_reasoning'][:200]}...")
                if "reasoning" in last_extract:
                    self.log_fn(f"Reasoning: {last_extract['reasoning'][:200]}...")
                if "understanding" in last_extract:
                    self.log_fn(f"Understanding: {last_extract['understanding'][:200]}...")
                
                self.log_fn(f"Extracted grade: {prediction}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
                # Try to extract grade directly from text as last resort
                text_lower = last_assistant_msg.lower()
                if "grade:" in text_lower or "score:" in text_lower:
                    match = re.search(r'(?:grade|score):\s*(.+?)(?:\n|$)', text_lower, re.IGNORECASE)
                    if match:
                        prediction = _normalize_grade(match.group(1).strip())
                        self.log_fn(f"Extracted grade from text: {prediction}")
                
                # Try to find grade labels directly in text with word boundaries
                if prediction == "None":
                    text_lower = last_assistant_msg.lower()
                    # Check in order of specificity (most specific first)
                    grade_patterns = [
                        (r'\balmost\b', "Almost"),
                        (r'\bpartial\b', "Partial"),
                        (r'\bincorrect\b', "Incorrect"),
                        (r'\bcorrect\b', "Correct"),
                    ]
                    for pattern, label in grade_patterns:
                        if re.search(pattern, text_lower):
                            prediction = label
                            self.log_fn(f"Extracted grade from text search: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
