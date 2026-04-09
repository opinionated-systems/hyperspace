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
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes robust error recovery for malformed JSON.
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
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            continue
            
        # Try extracting from nested content
        json_start = inner.find('{')
        json_end = inner.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            parsed = _try_parse_json(inner[json_start:json_end+1])
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
                inner_start = start + 3
            else:
                inner_start = start + 7
            
            # Find the closing ```
            end = text.find("```", inner_start)
            if end == -1:
                break
            
            inner = text[inner_start:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
                continue
                
            # Try extracting from nested content
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                parsed = _try_parse_json(inner[json_start:json_end+1])
                if parsed is not None:
                    results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with various cleanup strategies.
    
    Handles common LLM output issues like trailing commas,
    unescaped newlines in strings, and extra quotes.
    """
    text = text.strip()
    if not text:
        return None
        
    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try fixing common issues
    fixes = [
        # Remove trailing commas before closing braces/brackets
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix unescaped newlines in strings (simple heuristic)
        lambda t: re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', t),
        # Remove BOM if present
        lambda t: t.lstrip('\ufeff'),
        # Fix double-escaped newlines
        lambda t: t.replace('\\n', '\n').replace('\n', '\\n'),
    ]
    
    for fix in fixes:
        try:
            fixed = fix(text)
            return json.loads(fixed)
        except (json.JSONDecodeError, re.error):
            continue
    
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
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
- The JSON must be valid and parseable (no trailing commas, properly escaped strings)
- The "response" field must contain the final grade/prediction
- The "reasoning" field must contain your detailed analysis
- Use simple ASCII characters in your JSON to avoid encoding issues"""

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
                
                # Priority order for grade fields
                grade_fields = ["response", "grade", "answer", "result", 
                               "evaluation", "score", "verdict", "decision",
                               "prediction", "output", "assessment", "mark"]
                
                for field in grade_fields:
                    if field in last_json:
                        value = last_json[field]
                        if isinstance(value, str):
                            prediction = value.strip()
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                        elif isinstance(value, bool):
                            prediction = "Correct" if value else "Incorrect"
                        break
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and len(value) < 100:
                            prediction = value.strip()
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            break
            else:
                # Last resort: try to find any grade-like text in the response
                text_lower = last_message.lower()
                
                # More sophisticated pattern matching
                if re.search(r'\b(correct|right|true|yes|valid)\b', text_lower):
                    # Check if it's negated
                    if not re.search(r'\b(not\s+correct|incorrect|not\s+right|wrong|false|no)\b', text_lower):
                        prediction = "Correct"
                elif re.search(r'\b(incorrect|wrong|false|no|invalid|error)\b', text_lower):
                    prediction = "Incorrect"
                elif re.search(r'\b(partial|partially|incomplete|some|part)\b', text_lower):
                    prediction = "Partial"
                
                # Try to extract numeric scores
                score_match = re.search(r'\b(score|grade|mark)[\s:=]+(\d+(?:\.\d+)?)\b', text_lower)
                if score_match:
                    prediction = score_match.group(2)
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
