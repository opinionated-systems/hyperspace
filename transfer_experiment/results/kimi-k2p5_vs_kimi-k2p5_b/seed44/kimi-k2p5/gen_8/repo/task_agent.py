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

## Student's Answer:
{student_answer}

## Grading Guidelines (FOLLOW THESE EXPLICITLY):
{grading_guidelines}

## Grading Rubric Reference

**CORRECT (7 points)**: The student's answer is completely correct, complete, and proves the result. The solution contains all necessary steps, valid reasoning, and arrives at the correct conclusion with no significant errors or omissions.

**INCORRECT (0 points)**: The student's answer is wrong, contains fundamental errors, shows no meaningful progress, or demonstrates a fundamental misunderstanding of the problem. The approach is flawed or the conclusion is incorrect.
- No valid approach identified
- Fundamental logical errors
- Wrong conclusion with no valid reasoning
- No substantial progress toward solution

**PARTIAL (1-3 points)**: The student made significant progress toward the solution but the answer is incomplete, has substantial gaps, or misses key steps. The student demonstrates understanding of the problem and makes meaningful progress, but the solution is not close to complete.
- Found a useful invariant or lemma but didn't complete the proof
- Made significant progress on one case but missed others
- Has the right approach but major gaps in the reasoning
- Solution is roughly 25-75% complete
- Missing critical components that prevent the proof from working

**ALMOST (4-6 points)**: The solution is nearly correct with only minor mistakes, small omissions, or slight errors that don't significantly affect the overall correctness. The student has essentially solved the problem but with minor issues.
- Small computational errors that don't affect the main argument
- Minor gaps that could be easily filled by the reader
- Slight misstatements of lemmas or theorems that don't invalidate the proof
- Solution is roughly 85-95% correct, with only cosmetic issues
- The core proof structure is complete and valid

## Critical Grading Rules

**RULE 1 - FOLLOW GUIDELINES EXPLICITLY**: The grading guidelines above explicitly state what criteria correspond to each grade. If the guidelines list specific achievements under "(Partial)", then solutions meeting those criteria MUST be graded as PARTIAL, not INCORRECT.

**RULE 2 - PARTIAL vs INCORRECT**: 
- If the student found a useful invariant, lemma, or made meaningful progress → PARTIAL (not INCORRECT)
- If the student proved something significant but didn't complete the full solution → PARTIAL (not INCORRECT)
- Only grade as INCORRECT if there is NO meaningful progress or the approach is fundamentally wrong

**RULE 3 - ALMOST vs PARTIAL**:
- **PARTIAL**: Significant work remains. The core idea might be there, but substantial development is missing.
- **ALMOST**: The proof is essentially complete. Only minor polishing or correction of small errors is needed.

**RULE 4 - When uncertain between two grades, choose the LOWER grade.**

## Grading Instructions

1. **FIRST - Read the grading guidelines carefully** - they explicitly state what grade to assign based on specific criteria. The guidelines are your PRIMARY reference.

2. **SECOND - Analyze the student's answer**:
   - Check which criteria from the grading guidelines the student meets
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
