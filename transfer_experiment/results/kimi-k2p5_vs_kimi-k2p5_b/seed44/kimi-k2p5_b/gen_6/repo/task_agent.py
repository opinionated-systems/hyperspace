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
    """Extract JSON objects from <json>...</json> blocks or raw JSON."""
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
        i = 0
        while i < len(text):
            if text[i] == '{':
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

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
Evaluate the student's answer and assign EXACTLY ONE of the following grades.

### Grade Definitions (from highest to lowest quality):

**correct**: The student's answer is completely correct and proves the result.
- All key steps are present and logically sound
- The proof is rigorous and complete
- No significant gaps or errors

**almost**: The solution is nearly correct with only minor mistakes or omissions.
- The main proof idea/approach is correct
- Minor computational errors or small gaps that don't affect the main argument
- Missing some details but the core logic is sound
- The student understood the key insight but execution has small flaws

**partial**: The student made significant progress but the solution is incomplete or has major gaps.
- Some correct elements and good observations
- Missing critical steps or key insights
- The approach may be partially correct but incomplete
- Significant gaps in the proof that would need to be filled

**incorrect**: The student's answer is wrong or contains fundamental errors.
- The approach is flawed or the conclusion is wrong
- Fundamental misunderstanding of the problem
- No substantial progress toward the solution
- Major logical errors that invalidate the proof

### Grading Process:
1. First, carefully read the official solution to understand the correct approach
2. Read the grading guidelines carefully - they contain specific criteria for each grade
3. Analyze the student's answer step by step:
   - Did they understand the problem correctly?
   - What key insights did they identify?
   - What steps are correct vs. incorrect or missing?
   - Are there fundamental flaws or just minor gaps?
4. Compare against the grading guidelines - they often specify what constitutes partial vs almost
5. Choose the grade that best matches the student's work

### IMPORTANT Guidelines:
- If the grading guidelines specify criteria for a grade, follow them precisely
- "partial" typically means significant progress but missing critical components
- "almost" means the solution is essentially correct but has minor issues
- When uncertain between two grades, choose the more conservative (lower) one
- Be especially careful not to overgrade incorrect solutions as partial or correct

Respond ONLY with a JSON object in the following format. The grade field MUST be exactly one of: "correct", "incorrect", "partial", or "almost" (all lowercase, no extra spaces).

<json>
{{
    "grade": "correct"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "none"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    # Try to get grade first
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
                        prediction = str(list(last_json.values())[0]) if last_json else "none"
            
            # Validate the prediction - check if it's a valid grade
            valid_grades = ["correct", "incorrect", "partial", "almost"]
            pred_normalized = prediction.lower().strip().strip('"').strip("'")
            
            if pred_normalized not in valid_grades:
                # Try to normalize it first
                if pred_normalized in ['correct', 'incorrect', 'partial', 'almost']:
                    prediction = pred_normalized
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
