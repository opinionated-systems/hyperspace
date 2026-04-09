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
    "all correct": "Correct",
    "100%": "Correct",
    "full marks": "Correct",
    "excellent": "Correct",
    "flawless": "Correct",
    "no errors": "Correct",
    "no mistakes": "Correct",
    "7": "Correct",
    "7/7": "Correct",
    "seven": "Correct",
    # Almost mappings - expanded
    "almost": "Almost",
    "almost correct": "Almost",
    "nearly correct": "Almost",
    "mostly correct": "Almost",
    "essentially correct": "Almost",
    "essentially right": "Almost",
    "essentially valid": "Almost",
    "minor errors": "Almost",
    "small mistakes": "Almost",
    "minor error": "Almost",
    "small error": "Almost",
    "tiny error": "Almost",
    "trivial error": "Almost",
    "slight error": "Almost",
    "minor issue": "Almost",
    "small issue": "Almost",
    "trivial issue": "Almost",
    "slight issue": "Almost",
    "trivial": "Almost",
    "slight": "Almost",
    "minor": "Almost",
    "minor flaw": "Almost",
    "small flaw": "Almost",
    "trivial flaw": "Almost",
    "slight flaw": "Almost",
    "minor omission": "Almost",
    "small omission": "Almost",
    "slight omission": "Almost",
    "nearly right": "Almost",
    "almost right": "Almost",
    "close": "Almost",
    "very close": "Almost",
    "close to correct": "Almost",
    "minor mistake": "Almost",
    "small mistake": "Almost",
    "slight mistake": "Almost",
    "arithmetic error": "Almost",
    "calculation error": "Almost",
    "computation error": "Almost",
    "notation issue": "Almost",
    "unclear notation": "Almost",
    "missing step": "Almost",
    "small gap": "Almost",
    "minor gap": "Almost",
    "trivial oversight": "Almost",
    "minor oversight": "Almost",
    "small oversight": "Almost",
    "5-6": "Almost",
    "5": "Almost",
    "6": "Almost",
    "5/7": "Almost",
    "6/7": "Almost",
    "five": "Almost",
    "six": "Almost",
    # Partial mappings - expanded
    "partial": "Partial",
    "partially correct": "Partial",
    "partial credit": "Partial",
    "partially": "Partial",
    "some credit": "Partial",
    "incomplete": "Partial",
    "unfinished": "Partial",
    "not complete": "Partial",
    "not finished": "Partial",
    "missing": "Partial",
    "mostly": "Partial",
    "significant progress": "Partial",
    "some progress": "Partial",
    "partial progress": "Partial",
    "half": "Partial",
    "partial solution": "Partial",
    "incomplete solution": "Partial",
    "in progress": "Partial",
    "started": "Partial",
    "attempted": "Partial",
    "some work": "Partial",
    "partial work": "Partial",
    "some correct": "Partial",
    "mixed": "Partial",
    "some right": "Partial",
    "some valid": "Partial",
    "half correct": "Partial",
    "half right": "Partial",
    "major error": "Partial",
    "significant error": "Partial",
    "significant gap": "Partial",
    "major gap": "Partial",
    "couldn't finish": "Partial",
    "didn't finish": "Partial",
    "started correctly": "Partial",
    "attempted but": "Partial",
    "1-4": "Partial",
    "1": "Partial",
    "2": "Partial",
    "3": "Partial",
    "4": "Partial",
    "1/7": "Partial",
    "2/7": "Partial",
    "3/7": "Partial",
    "4/7": "Partial",
    "one": "Partial",
    "two": "Partial",
    "three": "Partial",
    "four": "Partial",
    # Incorrect mappings - expanded
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
    "all wrong": "Incorrect",
    "fundamentally wrong": "Incorrect",
    "completely incorrect": "Incorrect",
    "totally incorrect": "Incorrect",
    "no progress": "Incorrect",
    "no meaningful progress": "Incorrect",
    "no credit": "Incorrect",
    "zero credit": "Incorrect",
    "0%": "Incorrect",
    "failed": "Incorrect",
    "failure": "Incorrect",
    "unsolved": "Incorrect",
    "0": "Incorrect",
    "0/7": "Incorrect",
    "no understanding": "Incorrect",
    "doesn't understand": "Incorrect",
    "wrong approach": "Incorrect",
    "no valid steps": "Incorrect",
}

# Pre-compile regex patterns for better performance
_NUMERIC_GRADE_PATTERN = re.compile(r'\b([0-7])\b')
_ALMOST_KEYWORDS = frozenset([
    "almost", "nearly", "minor", "small mistake", "tiny error", "small error",
    "minor error", "minor issue", "small issue", "trivial", "slight", "nearly correct",
    "mostly correct", "essentially correct", "minor flaw", "small flaw", "trivial flaw", "slight flaw",
    "minor omission", "small omission", "slight omission", "nearly right", "almost right",
    "close", "very close", "close to correct", "minor mistake", "trivial error", "slight error", "slight mistake",
    "arithmetic error", "calculation error", "computation error", "notation issue", "unclear notation",
    "missing step", "small gap", "minor gap", "trivial oversight", "minor oversight", "small oversight"
])
_PARTIAL_KEYWORDS = frozenset([
    "incomplete", "unfinished", "not complete", "not finished", "missing", "significant progress", "some understanding",
    "partial", "partially", "some credit", "half", "partial solution", "incomplete solution", "in progress",
    "started", "attempted", "some work", "partial work", "some correct", "mixed",
    "some right", "some valid", "partially correct", "half correct", "half right",
    "major error", "significant error", "significant gap", "major gap",
    "couldn't finish", "didn't finish", "started correctly", "attempted but"
])
_CORRECT_KEYWORDS = frozenset([
    "correct", "right", "true", "full", "complete", "solved", "valid", "acceptable", "perfect",
    "fully correct", "entirely correct", "completely correct", "totally correct",
    "all correct", "100%", "full marks", "full credit", "excellent", "flawless", "no errors", "no mistakes"
])
_INCORRECT_KEYWORDS = frozenset([
    "incorrect", "wrong", "false", "none", "zero", "error", "invalid", "unacceptable",
    "completely wrong", "no progress", "totally wrong", "entirely wrong", "all wrong",
    "fundamentally wrong", "completely incorrect", "totally incorrect",
    "no credit", "zero credit", "0%", "failed", "failure", "unsolved",
    "no understanding", "doesn't understand", "wrong approach", "no valid steps", "no meaningful progress"
])


@functools.lru_cache(maxsize=1024)
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
    
    # Check if grade contains keywords (order matters - check specific categories first)
    # Check for "almost" category first (more specific than partial)
    if any(word in grade_lower for word in _ALMOST_KEYWORDS):
        return "Almost"
    # Check for "partial" category
    if "partial" in grade_lower:
        return "Partial"
    if any(word in grade_lower for word in _PARTIAL_KEYWORDS):
        return "Partial"
    # Check for "correct" category
    if any(word in grade_lower for word in _CORRECT_KEYWORDS):
        return "Correct"
    # Check for "incorrect" category
    if any(word in grade_lower for word in _INCORRECT_KEYWORDS):
        return "Incorrect"
    
    # Default: capitalize first letter
    return grade.capitalize()


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, problem: str, answer: str, official_solution: str, 
                             rubric: str, student_answer: str) -> str:
        """Build a comprehensive grading prompt with clear instructions."""
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
- **Correct**: Complete, correct solution with clear reasoning (7 points)
- **Almost**: Minor errors or gaps, but essentially correct approach (5-6 points)
- **Partial**: Significant progress but incomplete or with major errors (1-4 points)
- **Incorrect**: No meaningful progress or completely wrong approach (0 points)

## Grade Definitions with Examples

**Correct (7 points)**: The solution is complete and fully correct.
- Example: All steps are valid, reasoning is clear, final answer matches.
- Key: NO errors, NO missing steps, NO unclear notation issues.

**Almost (5-6 points)**: The solution is essentially correct with only MINOR issues.
- Examples:
  - Correct approach but minor calculation error at the end
  - Valid solution but missing one small step in the reasoning
  - Correct answer with slightly unclear notation
  - Nearly complete solution with trivial oversight
  - Right method, minor arithmetic mistake
  - Essentially correct approach with minor flaw
- Key: Would be Correct with only trivial fixes. The core approach is sound.

**Partial (1-4 points)**: Shows understanding but incomplete or has SIGNIFICANT errors.
- Examples:
  - Started correctly but couldn't finish
  - Some valid steps but major conceptual error
  - Partial progress toward solution
  - Understood problem but approach has significant flaws
  - Incomplete solution with meaningful work shown
- Key: Has significant gaps or errors, not just minor issues.

**Incorrect (0 points)**: No meaningful progress or completely wrong.
- Examples:
  - Completely wrong approach
  - No substantial work shown
  - Fundamental misunderstanding of the problem
  - No valid mathematical steps
- Key: Little to no valid mathematical progress.

## Critical Rules
1. You MUST use exactly one of these four labels: Correct, Almost, Partial, or Incorrect
2. "Almost" is for solutions that would be Correct with only MINOR fixes (trivial errors)
3. "Partial" requires meaningful mathematical progress but has SIGNIFICANT gaps or errors
4. The difference between "Almost" and "Partial" is crucial: 
   - "Almost" = essentially correct, minor issues only
   - "Partial" = significant issues, major gaps, or incomplete work
5. Be decisive - choose the single best category that matches the solution quality
6. When in doubt between "Almost" and "Partial", ask: "Would this be correct with just a trivial fix?" If yes, choose "Almost". If significant work is needed, choose "Partial".

Respond in JSON format:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed analysis of the student's answer - what they got right and wrong",
    "reasoning": "Step-by-step reasoning for your grade decision - explicitly state why you chose this category",
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
                
                # Priority order for grade extraction - include reasoning field
                grade_fields = ["response", "grade", "score", "result", "final_grade", "evaluation", "reasoning"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Additional validation: check if reasoning field contains grade indicators
                # This helps catch cases where response field might be wrong
                reasoning_text = last_extract.get("reasoning", "").lower()
                analysis_text = last_extract.get("analysis", "").lower()
                understanding_text = last_extract.get("understanding", "").lower()
                combined_text = reasoning_text + " " + analysis_text + " " + understanding_text
                
                # ENHANCED: Stronger reconsideration logic with weighted indicators
                # Define indicator sets with weights (higher = stronger signal)
                almost_indicators_strong = [
                    "essentially correct", "nearly correct", "almost correct", "mostly correct",
                    "minor error", "small mistake", "tiny error", "trivial error",
                    "minor issue", "small issue", "slight issue", "trivial issue",
                    "minor flaw", "small flaw", "slight flaw", "trivial flaw",
                    "minor omission", "small omission", "slight omission",
                    "nearly right", "almost right", "very close", "close to correct",
                    "minor mistake", "small error", "slight error", "slight mistake",
                    "arithmetic error", "calculation error", "notation issue",
                    "5-6 points", "5 points", "6 points", "5/7", "6/7"
                ]
                
                partial_indicators_strong = [
                    "incomplete", "couldn't finish", "didn't finish", "not complete",
                    "major error", "significant gap", "significant error", "major gap",
                    "fundamental misunderstanding", "completely wrong approach", "wrong approach",
                    "partial credit", "partial solution", "incomplete solution",
                    "some progress", "partial progress", "started but", "attempted but",
                    "1-4 points", "1 point", "2 points", "3 points", "4 points",
                    "1/7", "2/7", "3/7", "4/7"
                ]
                
                incorrect_indicators_strong = [
                    "completely wrong", "totally wrong", "entirely wrong", "all wrong",
                    "fundamentally wrong", "no progress", "no meaningful progress",
                    "no credit", "zero credit", "0 points", "0/7", "failed",
                    "completely incorrect", "totally incorrect", "doesn't understand",
                    "no understanding", "wrong from the start"
                ]
                
                correct_indicators_strong = [
                    "fully correct", "completely correct", "perfect solution",
                    "all steps valid", "no errors", "no mistakes", "flawless",
                    "entirely correct", "totally correct", "100% correct",
                    "full credit", "7 points", "7/7", "full marks"
                ]
                
                # Count indicator matches for each category
                almost_count = sum(1 for ind in almost_indicators_strong if ind in combined_text)
                partial_count = sum(1 for ind in partial_indicators_strong if ind in combined_text)
                incorrect_count = sum(1 for ind in incorrect_indicators_strong if ind in combined_text)
                correct_count = sum(1 for ind in correct_indicators_strong if ind in combined_text)
                
                # Reconsideration logic based on indicator counts and current prediction
                
                # If predicted "Correct" but has Almost indicators, reconsider
                if prediction == "Correct" and almost_count > 0:
                    if correct_count == 0 or almost_count >= 2:
                        self.log_fn(f"Reconsidering: {almost_count} 'Almost' indicators found despite 'Correct' prediction")
                        prediction = "Almost"
                
                # If predicted "Partial" but has strong Almost indicators, reconsider
                if prediction == "Partial" and almost_count > 0:
                    if partial_count == 0 or almost_count >= partial_count:
                        self.log_fn(f"Reconsidering: 'Almost' indicators ({almost_count}) outweigh 'Partial' ({partial_count})")
                        prediction = "Almost"
                
                # If predicted "Almost" but has strong Partial indicators, reconsider
                if prediction == "Almost" and partial_count > 0:
                    if almost_count == 0 or partial_count >= 2:
                        self.log_fn(f"Reconsidering: 'Partial' indicators ({partial_count}) outweigh 'Almost' ({almost_count})")
                        prediction = "Partial"
                
                # If predicted "Correct" or "Almost" but has strong Incorrect indicators, reconsider
                if prediction in ["Correct", "Almost"] and incorrect_count > 0:
                    if correct_count == 0 and almost_count == 0:
                        self.log_fn(f"Reconsidering: Strong 'Incorrect' indicators found ({incorrect_count})")
                        prediction = "Incorrect"
                
                # If predicted "Partial" but has strong Incorrect indicators, reconsider
                if prediction == "Partial" and incorrect_count > partial_count:
                    self.log_fn(f"Reconsidering: 'Incorrect' indicators ({incorrect_count}) outweigh 'Partial' ({partial_count})")
                    prediction = "Incorrect"
                
                # If predicted "Incorrect" but has strong Partial indicators, reconsider
                if prediction == "Incorrect" and partial_count > 0 and incorrect_count == 0:
                    if partial_count >= 2:
                        self.log_fn(f"Reconsidering: 'Partial' indicators ({partial_count}) suggest some progress")
                        prediction = "Partial"
                
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
                
                # Try to find grade labels directly in text - prioritize "Almost" detection
                if prediction == "None":
                    # Check for "Almost" indicators first (since it's often missed)
                    almost_patterns = [
                        "almost correct", "nearly correct", "mostly correct", "essentially correct",
                        "minor error", "small mistake", "trivial error", "slight mistake",
                        "minor issue", "small issue", "trivial issue", "slight issue",
                        "minor flaw", "small flaw", "trivial flaw", "slight flaw",
                        "arithmetic error", "calculation error", "notation issue",
                        "nearly right", "almost right", "very close", "close to correct",
                        "essentially correct approach", "essentially correct solution"
                    ]
                    for pattern in almost_patterns:
                        if pattern in text_lower:
                            prediction = "Almost"
                            self.log_fn(f"Extracted grade from pattern match: {prediction}")
                            break
                    
                    # Check for "Partial" indicators
                    if prediction == "None":
                        partial_patterns = [
                            "partial credit", "partially correct", "incomplete solution",
                            "incomplete answer", "couldn't finish", "didn't finish",
                            "started correctly", "some progress", "partial progress",
                            "significant gap", "major error", "significant error"
                        ]
                        for pattern in partial_patterns:
                            if pattern in text_lower:
                                prediction = "Partial"
                                self.log_fn(f"Extracted grade from pattern match: {prediction}")
                                break
                    
                    # Check for "Incorrect" indicators
                    if prediction == "None":
                        incorrect_patterns = [
                            "completely wrong", "totally wrong", "fundamentally wrong",
                            "no progress", "no meaningful progress", "no valid steps",
                            "doesn't understand", "no understanding", "wrong approach"
                        ]
                        for pattern in incorrect_patterns:
                            if pattern in text_lower:
                                prediction = "Incorrect"
                                self.log_fn(f"Extracted grade from pattern match: {prediction}")
                                break
                    
                    # If still None, check for explicit labels
                    if prediction == "None":
                        for label in ["Correct", "Almost", "Partial", "Incorrect"]:
                            if label.lower() in text_lower:
                                prediction = label
                                self.log_fn(f"Extracted grade from text search: {prediction}")
                                break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
