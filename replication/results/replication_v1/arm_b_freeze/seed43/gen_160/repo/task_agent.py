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
    Includes repair for common JSON formatting issues.
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
            continue
        
        # Try repairing common JSON issues
        parsed = _repair_and_parse_json(inner)
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
                continue
            
            # Try repairing common JSON issues
            parsed = _repair_and_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
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


def _repair_and_parse_json(text: str) -> dict | None:
    """Attempt to repair common JSON formatting issues and parse.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    """
    import re
    
    # Extract the JSON-like content between braces
    json_start = text.find('{')
    json_end = text.rfind('}')
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return None
    
    content = text[json_start:json_end+1]
    
    # Try original first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Repair 1: Remove trailing commas before } or ]
    repaired = re.sub(r',(\s*[}\]])', r'\1', content)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Repair 2: Replace single quotes with double quotes (carefully)
    # Only replace quotes that are likely JSON string delimiters
    repaired = re.sub(r"(?<!\\)'", '"', content)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Repair 3: Escape unescaped newlines in string values
    # Find string values and escape newlines within them
    def escape_newlines_in_strings(match):
        s = match.group(0)
        # Escape unescaped newlines
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s
    
    repaired = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines_in_strings, content, flags=re.DOTALL)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Repair 4: Try all repairs combined
    repaired = re.sub(r',(\s*[}\]])', r'\1', content)
    repaired = re.sub(r"(?<!\\)'", '"', repaired)
    repaired = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines_in_strings, repaired, flags=re.DOTALL)
    try:
        return json.loads(repaired)
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
    """Normalize various prediction formats to a string."""
    if prediction is None:
        return "None"
    if isinstance(prediction, str):
        return prediction.strip()
    if isinstance(prediction, (int, float)):
        return str(prediction)
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
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect.

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines

OUTPUT FORMAT REQUIREMENTS:
- You MUST wrap your JSON response in <json>...</json> tags
- The JSON must be valid and parseable
- The "response" field must contain the final grade/prediction
- The "reasoning" field must contain your detailed analysis"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            extraction_method = "json_tags"
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
                extraction_method = "any_json"
            
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
                        extraction_method = f"field:{field}"
                        break
                else:
                    # If no known field, use the first suitable value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and len(value) < 100:
                            prediction = value.strip()
                            extraction_method = f"unknown_field:{key}"
                            break
                        elif isinstance(value, (int, float, bool)):
                            prediction = _normalize_prediction(value)
                            extraction_method = f"unknown_field:{key}"
                            break
            else:
                # Last resort: try to find any grade-like text in the response
                text_lower = last_message.lower()
                extraction_method = "text_heuristic"
                
                # More sophisticated text-based extraction
                # Look for explicit grade statements
                grade_patterns = [
                    (r'\bgrade\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 1),
                    (r'\bfinal\s+(?:grade|score|evaluation)\s*[:=]\s*["\']?([^"\'\n]+)["\']?', 1),
                    (r'\bthe\s+student\s+(?:is|should\s+be)\s+["\']?([^"\'\n]+)["\']?', 1),
                ]
                
                for pattern, group in grade_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        prediction = match.group(group).strip().capitalize()
                        extraction_method = "regex_pattern"
                        break
                else:
                    # Simple keyword-based extraction
                    if "correct" in text_lower and "incorrect" not in text_lower:
                        prediction = "Correct"
                    elif "incorrect" in text_lower or "wrong" in text_lower:
                        prediction = "Incorrect"
                    elif "partial" in text_lower:
                        prediction = "Partial"
                    elif "full" in text_lower or "complete" in text_lower:
                        prediction = "Correct"
                    elif "zero" in text_lower or "0" in text_lower:
                        prediction = "Incorrect"
            
            # Log extraction details for debugging
            self.log_fn(f"Prediction extracted via {extraction_method}: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "Error"

        return prediction, msg_history
