"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find raw JSON objects if tags are not present.
    """
    results = []
    search_from = 0

    # First try to find <json>...</json> blocks
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
            continue

    # Also try to find ```json code blocks
    search_from = 0
    while True:
        start = text.find("```json", search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        search_from = end + 3
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            continue

    # If no results, try to find raw JSON objects (looking for {...})
    if not results:
        # Find JSON objects by looking for balanced braces
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Try to find the matching closing brace
                brace_count = 1
                j = i + 1
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1

                if brace_count == 0:
                    json_str = text[i:j].strip()
                    try:
                        results.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        pass
                i = j
            else:
                i += 1

    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from various formats in the text."""
    # Priority 1: Look for JSON grade field patterns
    json_patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"grade"\s*:\s*([a-zA-Z]+)',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in ['correct', 'incorrect', 'partial', 'almost']:
                return grade
    
    # Priority 2: Look for explicit grade assignments in text
    text_patterns = [
        r'\bgrade\s*[:=]\s*([a-zA-Z]+)',
        r'\bfinal\s+grade\s*[:=]\s*([a-zA-Z]+)',
        r'\bassigned\s+grade\s*[:=]\s*([a-zA-Z]+)',
        r'\bthe\s+grade\s+is\s+([a-zA-Z]+)',
        r'\bi\s+assign\s+([a-zA-Z]+)',
        r'\bthis\s+is\s+([a-zA-Z]+)\b',
    ]
    
    for pattern in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip().lower()
            if grade in ['correct', 'incorrect', 'partial', 'almost']:
                return grade
    
    # Priority 3: Look for standalone grade keywords in the last 100 words
    # This is where the conclusion typically appears
    words = text.lower().split()
    last_part = ' '.join(words[-100:])
    
    # Check in order of specificity (more specific first)
    if 'almost' in last_part:
        return 'almost'
    elif 'partial' in last_part:
        return 'partial'
    elif 'incorrect' in last_part:
        return 'incorrect'
    elif 'correct' in last_part:
        return 'correct'
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values."""
    if not grade:
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Map to standard grades
    if grade_lower in ["correct", "incorrect", "partial", "almost"]:
        return grade_lower
    
    # Partial matches as fallback
    if "almost" in grade_lower:
        return "almost"
    elif "partial" in grade_lower:
        return "partial"
    elif "incorrect" in grade_lower:
        return "incorrect"
    elif "correct" in grade_lower:
        return "correct"
    
    return "none"


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs for better prompt construction
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's solution to a competition mathematics problem.

## Problem Domain
{domain if domain else "Mathematical Olympiad"}

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grade Definitions (STRICT INTERPRETATION)

**CORRECT**: The solution is COMPLETE and FULLY CORRECT.
- All required steps are present and logically sound
- The proof/solution is complete with no gaps
- Any calculations are correct (or errors are trivial and don't affect the result)
- The student has successfully solved the problem

**INCORRECT**: The solution shows NO MEANINGFUL PROGRESS or contains FUNDAMENTAL ERRORS.
- The approach is fundamentally wrong or misguided
- No valid mathematical progress toward the solution
- Contains critical logical flaws that invalidate the argument
- The student has NOT made significant progress

**PARTIAL**: The student made SIGNIFICANT PROGRESS but the solution is INCOMPLETE.
- Found a useful invariant, lemma, or key insight
- Established a valid approach but didn't complete the proof
- Made meaningful progress but missing critical steps to finish
- Has the right idea but execution is incomplete

**ALMOST**: The solution is NEARLY COMPLETE with only MINOR ISSUES.
- The main proof structure is correct and complete
- Only minor computational errors, small omissions, or slight gaps
- The student essentially solved the problem but with small imperfections
- Much closer to correct than partial - the hard work is done

## Critical Distinctions

**PARTIAL vs INCORRECT**: 
- PARTIAL = meaningful progress (found invariant, proved lemma, valid approach started)
- INCORRECT = no meaningful progress (wrong approach, no valid lemmas, fundamental misunderstanding)

**PARTIAL vs ALMOST**:
- PARTIAL = significant gaps remain, solution is incomplete
- ALMOST = solution is essentially complete, only minor issues

**ALMOST vs CORRECT**:
- ALMOST = minor errors or omissions exist
- CORRECT = completely correct with no meaningful errors

## Grading Process (FOLLOW STRICTLY)

1. **FIRST - Parse the grading guidelines carefully**: The guidelines explicitly list what constitutes each grade level. Look for:
   - (Partial) section: What specific achievements are listed?
   - (Almost) section: What specific criteria are listed?
   - (Correct) section: What complete solution criteria are listed?

2. **SECOND - Check what the student proved against the guidelines**:
   - Did they prove items listed under (Partial)? → Grade PARTIAL
   - Did they prove items listed under (Almost)? → Grade ALMOST  
   - Did they prove items listed under (Correct)? → Grade CORRECT

3. **THIRD - Verify your grade matches the guidelines**:
   - Re-read the guidelines to confirm your grade assignment aligns with the explicit criteria

4. **Final decision**: Choose the grade that matches the grading guidelines criteria.

## Examples of Correct Grading

**Example 1 - PARTIAL grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.
 2. Proved a key lemma.
(Almost)
 1. Nearly complete solution with minor gaps.

Student's answer: Proves the invariant and the key lemma but doesn't complete the main proof.
→ CORRECT grade: "partial" (student met the Partial criteria)

**Example 2 - INCORRECT grading:**
Grading Guidelines say:
(Partial)
 1. Found a useful invariant.

Student's answer: Makes some calculations but no valid approach, no invariant found, fundamental misunderstanding.
→ CORRECT grade: "incorrect" (student did not meet any meaningful criteria)

**Example 3 - ALMOST grading:**
Grading Guidelines say:
(Almost)
 1. Complete proof with minor computational error.

Student's answer: Complete proof with a small arithmetic mistake that doesn't affect the main argument.
→ CORRECT grade: "almost" (student met Almost criteria)

## Response Format

Respond ONLY with a JSON object in the following format. The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces).

<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

Grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "none"
        valid_grades = ["correct", "incorrect", "partial", "almost"]
        
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            
            # Try to find a valid grade in any extracted JSON
            if extracted:
                for json_obj in reversed(extracted):  # Check from last to first
                    if isinstance(json_obj, dict):
                        # Try common grade field names
                        for field in ["grade", "response", "result", "prediction"]:
                            if field in json_obj:
                                val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                                if val in valid_grades:
                                    prediction = val
                                    break
                        if prediction in valid_grades:
                            break
                        # If no standard field found, try any value
                        if prediction == "none":
                            for val in json_obj.values():
                                val_str = str(val).lower().strip().strip('"').strip("'")
                                if val_str in valid_grades:
                                    prediction = val_str
                                    break
            
            # If still no valid grade, try text extraction
            if prediction not in valid_grades:
                extracted_text = _extract_grade_from_text(last_message)
                if extracted_text:
                    prediction = extracted_text
            
            # Last resort: look for quoted grades in the text
            if prediction not in valid_grades:
                text_lower = last_message.lower()
                # Priority order for quoted grades
                if '"almost"' in text_lower or "'almost'" in text_lower:
                    prediction = "almost"
                elif '"partial"' in text_lower or "'partial'" in text_lower:
                    prediction = "partial"
                elif '"incorrect"' in text_lower or "'incorrect'" in text_lower:
                    prediction = "incorrect"
                elif '"correct"' in text_lower or "'correct'" in text_lower:
                    prediction = "correct"
                else:
                    # Look for unquoted grades in the last part of the text
                    last_part = ' '.join(text_lower.split()[-50:])
                    if 'almost' in last_part:
                        prediction = "almost"
                    elif 'partial' in last_part:
                        prediction = "partial"
                    elif 'incorrect' in last_part:
                        prediction = "incorrect"
                    elif 'correct' in last_part:
                        prediction = "correct"
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected values
        prediction = _normalize_grade(prediction)

        return str(prediction), msg_history
