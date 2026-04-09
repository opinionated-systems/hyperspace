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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


# Common grade aliases for better normalization
_GRADE_ALIASES = {
    # Correct variations
    "correct": "Correct",
    "right": "Correct",
    "true": "Correct",
    "yes": "Correct",
    "full credit": "Correct",
    "full marks": "Correct",
    "full": "Correct",
    "complete": "Correct",
    "solved": "Correct",
    "valid": "Correct",
    "accepted": "Correct",
    "pass": "Correct",
    "success": "Correct",
    "1": "Correct",
    "1.0": "Correct",
    "100%": "Correct",
    # Incorrect variations
    "incorrect": "Incorrect",
    "wrong": "Incorrect",
    "false": "Incorrect",
    "no": "Incorrect",
    "zero": "Incorrect",
    "no credit": "Incorrect",
    "none": "Incorrect",
    "fail": "Incorrect",
    "failed": "Incorrect",
    "rejected": "Incorrect",
    "invalid": "Incorrect",
    "unsolved": "Incorrect",
    "0": "Incorrect",
    "0.0": "Incorrect",
    "0%": "Incorrect",
    # Partial variations
    "partial": "Partial",
    "partial credit": "Partial",
    "partially correct": "Partial",
    "half credit": "Partial",
    "half": "Partial",
    "incomplete": "Partial",
    "almost": "Partial",
    "mostly correct": "Partial",
    "minor error": "Partial",
    "0.5": "Partial",
    "50%": "Partial",
}


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional heuristics for common LLM output patterns.
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
        
        # Try direct parsing first
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            continue
            
        # Try extracting from within the content
        parsed = _extract_and_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # If no <json> tags found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without "json" specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
            
            # Find the closing ```
            end = text.find("```", start + 3)
            if end == -1:
                break
            
            # Extract content between markers
            inner_start = start + 7 if text[start:start+7] == "```json" else start + 3
            inner = text[inner_start:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
                continue
                
            parsed = _extract_and_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Final fallback: look for any JSON-like structures with brace counting
    if not results:
        results = _extract_any_json(text) or []
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse text as JSON. Returns dict or None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_and_parse_json(text: str) -> dict | None:
    """Extract JSON object from text by finding brace boundaries."""
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            return json.loads(text[json_start:json_end+1])
        except json.JSONDecodeError:
            pass
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces using brace counting
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


def _normalize_prediction(prediction: Any) -> str:
    """Normalize various prediction formats to a string.
    
    Uses comprehensive alias mapping for consistent grade normalization.
    Handles numeric scores, booleans, and string variations.
    """
    if prediction is None:
        return "None"
    if isinstance(prediction, str):
        pred = prediction.strip()
        pred_lower = pred.lower()
        # Check comprehensive alias mapping first
        if pred_lower in _GRADE_ALIASES:
            return _GRADE_ALIASES[pred_lower]
        # Handle percentage strings
        if pred_lower.endswith('%'):
            try:
                pct = float(pred_lower[:-1])
                if pct >= 80:
                    return "Correct"
                if pct <= 20:
                    return "Incorrect"
                return "Partial"
            except ValueError:
                pass
        # Handle fraction strings like "1/2", "3/4"
        if '/' in pred:
            try:
                parts = pred.split('/')
                if len(parts) == 2:
                    num = float(parts[0].strip())
                    den = float(parts[1].strip())
                    if den > 0:
                        ratio = num / den
                        if ratio >= 0.8:
                            return "Correct"
                        if ratio <= 0.2:
                            return "Incorrect"
                        return "Partial"
            except (ValueError, ZeroDivisionError):
                pass
        return pred
    if isinstance(prediction, (int, float)):
        # Normalize numeric scores to categorical grades
        if prediction >= 0.8 or prediction == 1:
            return "Correct"
        if prediction <= 0.2 or prediction == 0:
            return "Incorrect"
        return "Partial"
    if isinstance(prediction, bool):
        return "Correct" if prediction else "Incorrect"
    return str(prediction)


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

GRADE DEFINITIONS:
- Correct: The student's answer is fully correct, complete, and follows the official solution approach. All key steps are present and correct.
- Incorrect: The student's answer is fundamentally wrong, missing critical components, or shows no valid mathematical reasoning.
- Partial: The student shows valid reasoning, correct approach, or partial progress toward the solution, but has errors, gaps, or incomplete work.

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
   - Check if the final answer matches the official solution
5. FINAL DETERMINATION: Assign EXACTLY ONE of these grades: "Correct", "Incorrect", or "Partial"

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "Correct" or "Incorrect" or "Partial"
}}
</json>

Important Grading Rules:
- Award "Partial" when the student shows valid reasoning even if the final answer is incorrect
- Award "Partial" when the student made progress but didn't complete the solution
- Award "Correct" ONLY when the solution is complete and fully correct
- Award "Incorrect" when there's no valid mathematical reasoning or the answer is completely wrong

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines
- You have assigned exactly one of: "Correct", "Incorrect", or "Partial"

OUTPUT FORMAT REQUIREMENTS:
- You MUST wrap your JSON response in <json>...</json> tags
- The JSON must be valid and parseable
- The "response" field MUST contain exactly one of: "Correct", "Incorrect", or "Partial"
- The "reasoning" field must contain your detailed analysis"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                
                # Priority order for prediction fields
                priority_fields = [
                    "response", "grade", "answer", "result", 
                    "evaluation", "score", "verdict", "decision", "prediction"
                ]
                
                for field in priority_fields:
                    if field in last_json:
                        prediction = _normalize_prediction(last_json[field])
                        break
                else:
                    # If no known field, use the first suitable value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and len(value) < 100:
                            prediction = _normalize_prediction(value.strip())
                            break
                        elif isinstance(value, (int, float, bool)):
                            prediction = _normalize_prediction(value)
                            break
            
            # Validate prediction is one of the expected values
            valid_grades = {"Correct", "Incorrect", "Partial"}
            if prediction not in valid_grades and prediction != "None":
                # Try to normalize again with the comprehensive mapping
                pred_lower = str(prediction).lower().strip()
                if pred_lower in _GRADE_ALIASES:
                    prediction = _GRADE_ALIASES[pred_lower]
                else:
                    # If still not valid, fall through to text extraction
                    prediction = "None"
            
            # Last resort: try to find any grade-like text in the response
            if prediction == "None":
                text_lower = last_message.lower()
                
                # Look for explicit grade statements first
                grade_patterns = [
                    r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n]+)',
                    r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\'\n]+)',
                    r'final\s*grade\s*[:=]\s*["\']?([^"\'\n]+)',
                    r'grade\s*[:=]\s*["\']?([^"\'\n]+)',
                ]
                
                for pattern in grade_patterns:
                    match = re.search(pattern, last_message, re.IGNORECASE)
                    if match:
                        extracted_grade = match.group(1).strip().lower()
                        if extracted_grade in _GRADE_ALIASES:
                            prediction = _GRADE_ALIASES[extracted_grade]
                            break
                
                # Fallback to keyword matching if still not found
                if prediction == "None":
                    # Check for explicit grade words with word boundaries
                    if re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
                        prediction = "Correct"
                    elif re.search(r'\bincorrect\b|\bwrong\b', text_lower):
                        prediction = "Incorrect"
                    elif re.search(r'\bpartial\b', text_lower):
                        prediction = "Partial"
                    elif re.search(r'\bfull\s+(credit|marks)\b', text_lower):
                        prediction = "Correct"
                    elif re.search(r'\bno\s+(credit|marks)\b', text_lower):
                        prediction = "Incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
