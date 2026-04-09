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
    """Extract JSON objects from <json>...</json> blocks with robust fallback.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional cleanup for common LLM output issues.
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
        
        # Try to parse the JSON content
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # If no <json> tags found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            # Look for ```json or just ```
            json_start = text.find("```json", search_from)
            plain_start = text.find("```", search_from)
            
            if json_start != -1:
                start = json_start
                offset = 7  # len("```json")
            elif plain_start != -1:
                start = plain_start
                offset = 3  # len("```")
            else:
                break
            
            # Find the closing ```
            end = text.find("```", start + offset)
            if end == -1:
                break
            
            # Extract content between markers
            inner = text[start + offset:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with various cleanup strategies."""
    text = text.strip()
    
    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from first { to last }
    try:
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return json.loads(text[json_start:json_end+1])
    except json.JSONDecodeError:
        pass
    
    # Try cleaning up common issues: trailing commas, single quotes, etc.
    try:
        cleaned = _clean_json_text(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def _clean_json_text(text: str) -> str:
    """Clean up common JSON formatting issues from LLM output."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Replace single quotes with double quotes (carefully)
    text = re.sub(r"(?<!\\)'", '"', text)
    return text


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
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
                    # Try with cleanup
                    try:
                        cleaned = _clean_json_text(text[start_idx:i+1])
                        obj = json.loads(cleaned)
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
- The JSON must be valid and parseable
- The "response" field must contain the final grade/prediction
- The "reasoning" field must contain your detailed analysis"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with robust fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Args:
            msg_history: List of message dicts with 'role' and 'text' keys
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
        except (KeyError, IndexError):
            return "None"
        
        # Try primary extraction method
        extracted = _extract_jsons(last_message)
        
        # Fallback to generic JSON extraction if primary fails
        if extracted is None:
            extracted = _extract_any_json(last_message)
        
        if extracted:
            return self._extract_from_json_objects(extracted)
        
        # Last resort: pattern-based extraction from text
        return self._extract_from_text_patterns(last_message)
    
    def _extract_from_json_objects(self, extracted: list[dict]) -> str:
        """Extract prediction from list of JSON objects.
        
        Tries multiple field names in order of preference.
        """
        if not extracted:
            return "None"
        
        last_json = extracted[-1]
        
        # Priority order of field names to check
        priority_fields = [
            "response", "grade", "answer", "result", 
            "evaluation", "score", "verdict", "decision",
            "prediction", "output", "conclusion"
        ]
        
        # Check priority fields first
        for field in priority_fields:
            if field in last_json:
                value = last_json[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # If no priority field found, look for any string or numeric value
        for key, value in last_json.items():
            if isinstance(value, str) and len(value) < 200:
                # Skip reasoning-like fields
                if key not in ("reasoning", "analysis", "explanation", "thoughts", "thinking"):
                    return value
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, bool):
                return "Correct" if value else "Incorrect"
        
        return "None"
    
    def _extract_from_text_patterns(self, text: str) -> str:
        """Extract grade from text using pattern matching as last resort."""
        text_lower = text.lower()
        
        # Check for explicit grade statements
        grade_patterns = [
            (r'grade[:\s]+([\w\s-]+)', 1),
            (r'final grade[:\s]+([\w\s-]+)', 1),
            (r'prediction[:\s]+([\w\s-]+)', 1),
            (r'response[:\s]+([\w\s-]+)', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group).strip()
                # Clean up common endings
                grade = re.sub(r'[.\n].*$', '', grade)
                if grade and len(grade) < 50:
                    return grade.capitalize()
        
        # Check for binary correct/incorrect indicators
        # Look for "correct" but not preceded by "in"
        if re.search(r'(?<!in)\bcorrect\b', text_lower) and "incorrect" not in text_lower:
            return "Correct"
        if "incorrect" in text_lower or "wrong" in text_lower:
            return "Incorrect"
        if "partial" in text_lower or "partially" in text_lower:
            return "Partial"
        
        return "None"
