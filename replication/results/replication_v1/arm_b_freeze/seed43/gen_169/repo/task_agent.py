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
    """Normalize various prediction formats to a valid grade string.
    
    Returns one of: 'Correct', 'Incorrect', 'Partial', or 'None' if unparseable.
    """
    if prediction is None:
        return "None"
    if isinstance(prediction, str):
        cleaned = prediction.strip().lower()
        # Direct match
        if cleaned == "correct":
            return "Correct"
        elif cleaned == "incorrect":
            return "Incorrect"
        elif cleaned == "partial":
            return "Partial"
        # Handle common variations
        elif cleaned in ("true", "yes", "right", "valid", "full credit", "full marks"):
            return "Correct"
        elif cleaned in ("false", "no", "wrong", "invalid", "no credit", "zero"):
            return "Incorrect"
        elif cleaned in ("partial credit", "half", "some"):
            return "Partial"
        # If it contains the grade word somewhere
        elif "correct" in cleaned and "incorrect" not in cleaned and "partial" not in cleaned:
            return "Correct"
        elif "incorrect" in cleaned or "wrong" in cleaned:
            return "Incorrect"
        elif "partial" in cleaned:
            return "Partial"
        return prediction.strip()
    if isinstance(prediction, (int, float)):
        # Numeric scores: 1.0 = Correct, 0.0 = Incorrect, 0.5 = Partial
        if prediction >= 0.9:
            return "Correct"
        elif prediction <= 0.1:
            return "Incorrect"
        else:
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

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate a student's answer with precision, rigor, and consistency.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Follow this systematic evaluation protocol:

STEP 1 - PROBLEM DECOMPOSITION:
- Identify the core mathematical concepts and theorems being tested
- List the essential proof steps required for a complete solution
- Determine what constitutes a valid alternative approach

STEP 2 - SOLUTION MAPPING:
- Extract critical proof steps from the official solution
- Identify acceptable shortcuts or alternative methods
- Note which parts are essential vs. optional

STEP 3 - STUDENT WORK VERIFICATION:
- Map each student claim to the official solution
- Verify correctness of each mathematical statement
- Check if justifications are provided and valid
- Categorize any errors: conceptual, computational, logical, or notational

STEP 4 - PARTIAL CREDIT ANALYSIS:
Award 'Partial' credit when the student demonstrates:
- Correct problem setup or interpretation
- Valid intermediate results with proper reasoning
- Correct method selection even with execution errors
- Substantive progress toward the solution (not just attempting)

STEP 5 - GRADE ASSIGNMENT (STRICT CRITERIA):
- 'Correct': Complete solution with all essential steps correct and justified. Minor notation issues acceptable if mathematics is sound.
- 'Partial': Significant valid reasoning shown but incomplete solution, OR correct approach with substantive errors preventing full credit.
- 'Incorrect': No valid reasoning, fundamentally wrong approach, or answer without supporting work.

DECISION RULES:
1. Correct answer + no valid reasoning = 'Incorrect' (must show work)
2. Correct approach + minor errors = 'Partial' (if errors are minor) or 'Incorrect' (if major)
3. Valid reasoning for significant portion = 'Partial' (even with wrong final answer)
4. Complete and correct solution = 'Correct'

Respond in JSON format:
<json>
{{
    "reasoning": "Detailed analysis following the 5 steps. Be specific about what was correct/incorrect.",
    "response": "MUST be exactly: 'Correct', 'Incorrect', or 'Partial'"
}}
</json>

CRITICAL REMINDERS:
- Grade on mathematical content, not presentation
- 'Partial' requires SUBSTANTIVE valid reasoning, not just an attempt
- When uncertain between 'Partial'/'Incorrect', choose 'Partial' if ANY valid reasoning exists
- When uncertain between 'Partial'/'Correct', choose 'Partial' if ANY gaps remain
- The "response" field MUST contain EXACTLY one of: 'Correct', 'Incorrect', or 'Partial'

VERIFICATION CHECKLIST:
□ Verified each mathematical claim against the official solution?
□ Checked all criteria in the grading guidelines?
□ Grade is consistent with similar cases?
□ Reasoning clearly justifies the assigned grade?
□ JSON is properly formatted with <json> tags?"""

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
                            prediction = value.strip()
                            break
                        elif isinstance(value, (int, float, bool)):
                            prediction = _normalize_prediction(value)
                            break
            
            # Last resort: try to find any grade-like text in the response
            if prediction == "None":
                text_lower = last_message.lower()
                
                # Look for explicit grade statements first (most reliable)
                grade_patterns = [
                    r'["\']?response["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
                    r'["\']?grade["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
                    r'["\']?final grade["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
                    r'["\']?evaluation["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
                    r'grade\s+is\s+["\']?([^"\'\n]+)["\']?',
                    r'assigned\s+grade[\s:]+["\']?([^"\'\n]+)["\']?',
                ]
                
                import re
                for pattern in grade_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        extracted = match.group(1).strip()
                        normalized = _normalize_prediction(extracted)
                        if normalized in ("Correct", "Incorrect", "Partial"):
                            prediction = normalized
                            break
                
                # Fallback to keyword matching with context awareness
                if prediction == "None":
                    # Look for grade words with context to avoid false positives
                    # Check for "partial" first (most specific)
                    if re.search(r'\bpartial\b', text_lower):
                        prediction = "Partial"
                    # Then check for "incorrect" or "wrong" (negative indicators)
                    elif re.search(r'\b(incorrect|wrong|error)\b', text_lower):
                        # Make sure it's not "not incorrect" or similar
                        if not re.search(r'\b(not incorrect|not wrong)\b', text_lower):
                            prediction = "Incorrect"
                    # Finally check for "correct" (but ensure it's not negated)
                    elif re.search(r'\bcorrect\b', text_lower):
                        if not re.search(r'\b(not correct|incorrect)\b', text_lower):
                            prediction = "Correct"
                    # Additional patterns
                    elif re.search(r'\b(full credit|full marks)\b', text_lower):
                        prediction = "Correct"
                    elif re.search(r'\b(no credit|zero|no marks)\b', text_lower):
                        prediction = "Incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, msg_history
