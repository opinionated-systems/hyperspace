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
    # Look for grade patterns in the text - order matters, more specific first
    # Prioritize JSON field patterns first - these are most reliable
    patterns = [
        r'"grade"\s*:\s*"([^"]*)"',
        r'"grade"\s*:\s*\'([^\']*)\'',
        r'"response"\s*:\s*"([^"]*)"',
        r'"prediction"\s*:\s*"([^"]*)"',
        r'"result"\s*:\s*"([^"]*)"',
        r'"analysis"[\s\S]*?"grade"\s*:\s*"([^"]*)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip()
            # Validate it's one of the expected grades
            grade_lower = grade.lower()
            if grade_lower in ['correct', 'incorrect', 'partial', 'almost']:
                # Return properly capitalized version
                if grade_lower == 'correct':
                    return 'Correct'
                elif grade_lower == 'incorrect':
                    return 'Incorrect'
                elif grade_lower == 'partial':
                    return 'Partial'
                elif grade_lower == 'almost':
                    return 'Almost'
    
    # Fallback: look for grade keywords in text
    # Search for explicit grade assignments
    grade_patterns = [
        r'grade[\s:]+([\w]+)',
        r'Grade[\s:]+([\w]+)',
        r'final grade[\s:]+([\w]+)',
        r'assigned grade[\s:]+([\w]+)',
        r'assigned[:\s]+([\w]+)',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            grade = match.group(1).strip()
            grade_lower = grade.lower()
            if grade_lower in ['correct', 'incorrect', 'partial', 'almost']:
                if grade_lower == 'correct':
                    return 'Correct'
                elif grade_lower == 'incorrect':
                    return 'Incorrect'
                elif grade_lower == 'partial':
                    return 'Partial'
                elif grade_lower == 'almost':
                    return 'Almost'
    
    # Last resort: look for standalone grade keywords in the last part of text
    # This is where the conclusion typically appears
    text_lower = text.lower()
    last_part = ' '.join(text_lower.split()[-50:])  # Last 50 words
    
    # Check for grades in order of specificity
    if 'almost' in last_part:
        return 'Almost'
    elif 'partial' in last_part:
        return 'Partial'
    elif 'incorrect' in last_part:
        return 'Incorrect'
    elif 'correct' in last_part:
        return 'Correct'
    
    return None


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected values."""
    if not grade:
        return "None"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
    # Map to standard grades - check for exact matches first, then partial
    # Order matters: check longer/more specific matches first
    if grade_lower == "almost":
        return "Almost"
    elif grade_lower == "partial":
        return "Partial"
    elif grade_lower == "incorrect":
        return "Incorrect"
    elif grade_lower == "correct":
        return "Correct"
    
    # Partial matches as fallback
    if "almost" in grade_lower:
        return "Almost"
    elif "partial" in grade_lower:
        return "Partial"
    elif "incorrect" in grade_lower:
        return "Incorrect"
    elif "correct" in grade_lower:
        return "Correct"
    
    # If no match, return the original (capitalized first letter)
    return grade.strip().capitalize() if grade.strip() else "None"


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

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer and assign EXACTLY ONE of the following grades. Be conservative - when in doubt, choose the lower grade.

### Grade Definitions (in order from highest to lowest):

**Correct**: The student's answer is completely correct and proves the result.
- All key steps are present and logically sound
- The proof is rigorous and complete
- No significant gaps or errors
- Only assign this if you are confident the solution is fully correct

**Almost**: The solution is nearly correct with only minor mistakes or omissions.
- The main proof idea/approach is correct
- Minor computational errors or small gaps that don't affect the main argument
- Missing some details but the core logic is sound
- The student understood the key insight but execution has small flaws

**Partial**: The student made significant progress but the solution is incomplete or has major gaps.
- Some correct elements and good observations
- Missing critical steps or key insights
- The approach may be partially correct but incomplete
- Significant gaps in the proof that would need to be filled

**Incorrect**: The student's answer is wrong or contains fundamental errors.
- The approach is flawed or the conclusion is wrong
- Fundamental misunderstanding of the problem
- No substantial progress toward the solution
- Major logical errors that invalidate the proof

### Grading Process:
1. First, carefully read the official solution to understand the correct approach
2. Read the grading guidelines to understand what specific criteria define each grade
3. Analyze the student's answer step by step:
   - Did they understand the problem correctly?
   - What key insights did they identify?
   - What steps are correct vs. incorrect or missing?
   - Are there fundamental flaws or just minor gaps?
4. Compare against the grading guidelines - they often specify what constitutes Partial vs Almost
5. Choose the grade that best matches the student's work

### IMPORTANT Guidelines:
- If the grading guidelines specify criteria for a grade, follow them precisely
- "Partial" typically means significant progress but missing critical components
- "Almost" means the solution is essentially correct but has minor issues
- When uncertain between two grades, choose the more conservative (lower) one
- Be especially careful not to overgrade Incorrect solutions as Partial or Correct

Respond ONLY in the following JSON format. The grade field MUST be exactly one of: "Correct", "Incorrect", "Partial", or "Almost" (case-sensitive, no extra spaces, no quotes around the value).

<json>
{{
    "analysis": "Your detailed analysis of the student's answer. Discuss what they got right, what they missed, and how it compares to the official solution and grading guidelines.",
    "grade": "Correct"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    # Try to get grade first (most specific)
                    if "grade" in last_json:
                        prediction = str(last_json["grade"]).strip()
                    elif "response" in last_json:
                        prediction = str(last_json["response"]).strip()
                    elif "result" in last_json:
                        prediction = str(last_json["result"]).strip()
                    elif "prediction" in last_json:
                        prediction = str(last_json["prediction"]).strip()
                    else:
                        # Try to get any value from the JSON as prediction
                        prediction = str(list(last_json.values())[0]) if last_json else "None"
            
            # Validate the prediction - check if it's a valid grade
            valid_grades = ["Correct", "Incorrect", "Partial", "Almost"]
            pred_normalized = prediction.strip().strip('"').strip("'")
            
            if pred_normalized not in valid_grades:
                # Try to normalize it first
                pred_lower = pred_normalized.lower()
                if pred_lower in ['correct', 'incorrect', 'partial', 'almost']:
                    # Map to proper case
                    for vg in valid_grades:
                        if pred_lower == vg.lower():
                            prediction = vg
                            break
                else:
                    # Fallback: try to extract grade from text patterns
                    extracted_text = _extract_grade_from_text(last_message)
                    if extracted_text:
                        prediction = extracted_text
                    else:
                        # Last resort: look for the keywords directly in the text
                        text_lower = last_message.lower()
                        # Look for quoted grades first
                        if '"almost"' in text_lower or "'almost'" in text_lower:
                            prediction = "Almost"
                        elif '"partial"' in text_lower or "'partial'" in text_lower:
                            prediction = "Partial"
                        elif '"incorrect"' in text_lower or "'incorrect'" in text_lower:
                            prediction = "Incorrect"
                        elif '"correct"' in text_lower or "'correct'" in text_lower:
                            prediction = "Correct"
                        else:
                            # Look for unquoted grades in the last part of the text
                            last_part = ' '.join(text_lower.split()[-50:])
                            if 'almost' in last_part:
                                prediction = "Almost"
                            elif 'partial' in last_part:
                                prediction = "Partial"
                            elif 'incorrect' in last_part:
                                prediction = "Incorrect"
                            elif 'correct' in last_part:
                                prediction = "Correct"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected values
        prediction = _normalize_grade(prediction)

        return str(prediction), msg_history
