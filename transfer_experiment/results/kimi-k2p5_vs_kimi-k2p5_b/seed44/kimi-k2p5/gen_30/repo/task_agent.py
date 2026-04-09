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

VALID_GRADES = ["correct", "incorrect", "partial", "almost"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON."""
    results = []
    
    # Try to find <json>...</json> blocks
    search_from = 0
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
    
    # Try to find ```json code blocks
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
    
    # Try to find raw JSON objects with balanced braces
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


def _extract_grade(text: str) -> str:
    """Extract grade from LLM response text using multiple strategies."""
    if not text:
        return "none"
    
    valid_grades = ["correct", "incorrect", "partial", "almost"]
    text_lower = text.lower()
    
    # Strategy 1: Extract from JSON objects
    extracted = _extract_jsons(text)
    if extracted:
        for json_obj in reversed(extracted):
            if isinstance(json_obj, dict):
                # Try "grade" field first
                if "grade" in json_obj:
                    val = str(json_obj["grade"]).lower().strip().strip('"').strip("'")
                    if val in valid_grades:
                        return val
                # Try other common fields
                for field in ["response", "result", "prediction", "answer", 
                              "evaluation", "assessment", "verdict", "decision"]:
                    if field in json_obj:
                        val = str(json_obj[field]).lower().strip().strip('"').strip("'")
                        if val in valid_grades:
                            return val
                # Try any value in the JSON
                for val in json_obj.values():
                    val_str = str(val).lower().strip().strip('"').strip("'")
                    if val_str in valid_grades:
                        return val_str
    
    # Strategy 2: Look for JSON-like patterns in text
    json_patterns = [
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*\'(correct|incorrect|partial|almost)\'',
        r'"grade"\s*:\s*(correct|incorrect|partial|almost)',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 3: Look for explicit grade declarations
    text_patterns = [
        r'\bgrade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bfinal\s+grade\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bthe\s+grade\s+is\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bi\s+assign\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bconclusion[:]?\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
        r'\bverdict[:]?\s+["\']?(correct|incorrect|partial|almost)["\']?\b',
    ]
    for pattern in text_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Strategy 4: Look for quoted grades
    quoted_patterns = [
        (r'["\']almost["\']', 'almost'),
        (r'["\']partial["\']', 'partial'),
        (r'["\']incorrect["\']', 'incorrect'),
        (r'["\']correct["\']', 'correct'),
    ]
    for pattern, grade in quoted_patterns:
        if re.search(pattern, text_lower):
            return grade
    
    # Strategy 5: Look in the last 100 words (conclusion area)
    words = text_lower.split()
    last_part = ' '.join(words[-100:]) if len(words) > 100 else text_lower
    
    if re.search(r'\balmost\b', last_part):
        return 'almost'
    elif re.search(r'\bpartial\b', last_part):
        return 'partial'
    elif re.search(r'\bincorrect\b', last_part):
        return 'incorrect'
    elif re.search(r'\bcorrect\b', last_part):
        return 'correct'
    
    # Strategy 6: Look anywhere in text as last resort
    for grade in ["almost", "partial", "incorrect", "correct"]:
        if re.search(rf'\b{grade}\b', text_lower):
            return grade
    
    return "none"


def _normalize_grade(grade: str) -> str:
    """Normalize grade to one of the expected lowercase values."""
    if not grade:
        return "none"
    
    grade_lower = grade.lower().strip().strip('"').strip("'")
    
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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical olympiad grader. Evaluate the student's solution and assign exactly one grade: "correct", "incorrect", "partial", or "almost".

## Problem:
{problem}

## Official Solution:
{solution}

## Student's Answer:
{student_answer}

## Grading Guidelines:
{grading_guidelines}

## Grade Definitions

**CORRECT**: Perfect solution (95-100% complete, no errors)

**INCORRECT**: No meaningful progress or fundamentally wrong approach

**PARTIAL**: Significant progress but incomplete (30-75% complete) OR main result NOT proven
- This is the most common grade for incomplete work
- If main theorem is not fully proven → MUST be "partial"

**ALMOST**: Nearly complete with MINOR issues only (75-95% complete)
- Main proof structure IS complete and main result IS proven
- Only small computational errors or minor gaps

## Decision Process

1. Check if there's meaningful progress (no progress → "incorrect")
2. Check if main result is FULLY proven (not proven → "partial")
3. Check completeness (95-100% → "correct", 75-95% → "almost")

## CRITICAL RULE
When in doubt, choose the LOWER grade.

## Response Format (STRICT)

You MUST respond with ONLY a JSON object in this exact format:

<json>
{{
    "grade": "correct" | "incorrect" | "partial" | "almost"
}}
</json>

The grade field MUST be exactly one of the four values above, all lowercase, no extra spaces or punctuation.

Grade:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
                    prediction = _extract_grade(last_message)
                    
                    if prediction == "none":
                        self.log_fn(f"Warning: Could not extract valid grade. Response preview: {last_message[:500]}...")
                        # Fallback: look for any valid grade word
                        text_lower = last_message.lower()
                        for grade in ["almost", "partial", "incorrect", "correct"]:
                            if re.search(rf'\b{grade}\b', text_lower):
                                prediction = grade
                                self.log_fn(f"Fallback extraction found grade: {grade}")
                                break
                else:
                    self.log_fn("Warning: Last message has no text content")
            else:
                self.log_fn("Warning: No message history available")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Normalize prediction
        prediction = _normalize_grade(prediction)
        
        # Final validation
        if prediction not in VALID_GRADES:
            self.log_fn(f"Warning: Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
