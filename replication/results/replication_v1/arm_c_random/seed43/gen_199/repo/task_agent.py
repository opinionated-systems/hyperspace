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
    Also handles markdown code blocks, raw JSON objects, and nested structures.
    Includes robust error recovery for common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group(1).strip())
            if parsed:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - use a more flexible pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    # Final fallback: try to extract grade directly from text patterns
    if not results:
        # Look for common grade patterns like "Grade: 7" or "Final grade: 6"
        grade_patterns = [
            r'[Gg]rade[:\s]+(\d+)',
            r'[Ff]inal[\s_]*[Gg]rade[:\s]+(\d+)',
            r'[Ss]core[:\s]+(\d+)',
            r'[Ee]valuation[:\s]+(\d+)',
            r'[Rr]esponse[:\s]+["\']?(\d+)["\']?',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, text)
            if match:
                grade = match.group(1)
                results.append({"response": grade, "reasoning": "Extracted from text pattern"})
                break
    
    # Extra fallback: look for standalone numbers 0-7 that might be grades
    if not results:
        # Look for standalone digits 0-7 (common IMO grade range)
        standalone_pattern = r'(?:^|\s)([0-7])(?:\s|$|[^\d])'
        matches = list(re.finditer(standalone_pattern, text))
        if matches:
            # Use the last match as it's likely the final grade
            last_match = matches[-1]
            grade = last_match.group(1)
            results.append({"response": grade, "reasoning": "Extracted from standalone number"})
    
    # NEW: Validate and normalize extracted grades
    validated_results = []
    for result in results:
        if "response" in result:
            normalized = _normalize_grade(result["response"])
            if normalized:
                result["response"] = normalized
                validated_results.append(result)
    
    return validated_results or None


def _normalize_grade(grade: str) -> str | None:
    """Normalize and validate a grade string.
    
    Returns the normalized grade if valid, None otherwise.
    Handles IMO grades (0-7) and common variations.
    """
    if not grade or not isinstance(grade, str):
        return None
    
    grade = grade.strip()
    
    # Direct numeric grades 0-7
    if grade.isdigit():
        num = int(grade)
        if 0 <= num <= 7:
            return str(num)
        return None
    
    # Check for partial credit patterns
    partial_match = re.search(r'[Pp]artial\s*[Cc]redit[:\s]*(\d+)', grade)
    if partial_match:
        return f"Partial credit: {partial_match.group(1)}"
    
    # Check for "X points" pattern
    points_match = re.search(r'(\d+)\s*[Pp]oints?', grade)
    if points_match:
        num = int(points_match.group(1))
        if 0 <= num <= 7:
            return str(num)
    
    # Check for special values
    lower = grade.lower()
    special_map = {
        'correct': '7',
        'full': '7',
        'complete': '7',
        'incorrect': '0',
        'wrong': '0',
        'none': 'None',
        'n/a': 'N/A',
        'incomplete': 'Incomplete',
        'zero': '0',
    }
    if lower in special_map:
        return special_map[lower]
    
    # If it contains a digit 0-7, extract it
    digit_match = re.search(r'([0-7])', grade)
    if digit_match:
        return digit_match.group(1)
    
    return None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict if successful, None otherwise.
    """
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes
    try:
        # Replace single quotes used as JSON delimiters with double quotes
        # Be careful not to replace quotes inside string values
        fixed = re.sub(r"(?<!\\)'", '"', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the fields we need with a more lenient approach
    try:
        # Look for "response" field value (handles both quoted and unquoted numbers)
        response_match = re.search(r'"response"\s*:\s*"?([^"\n,}]*)"?', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL)
        
        if response_match:
            result = {"response": response_match.group(1).strip().strip('"\'')}
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1)
            return result
    except Exception:
        pass
    
    # Strategy 5: Try to find response in a more flexible way
    try:
        # Look for patterns like response: 7 or "response": "Partial credit: 3"
        flexible_response = re.search(r'[Rr]esponse["\']?\s*[:=]\s*["\']?([^"\n,}]+)', text)
        if flexible_response:
            return {"response": flexible_response.group(1).strip().strip('"\'')}
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Evaluation Framework

Follow this systematic approach:

1. **Understanding Check**: Verify you understand the problem and official solution completely.

2. **Student's Approach Analysis**: 
   - Identify the student's key ideas and strategy
   - Note any creative or alternative approaches
   - Check if the approach is valid even if different from official solution

3. **Correctness Verification**:
   - Check each claim and step for logical validity
   - Identify any gaps, errors, or unjustified assertions
   - Verify calculations and algebraic manipulations

4. **Completeness Assessment**:
   - Does the solution cover all cases?
   - Are all conditions from the problem statement addressed?
   - Is the conclusion properly justified?

5. **Grading Decision**:
   - Apply the grading guidelines strictly
   - Consider partial credit for correct ideas with minor gaps
   - Be consistent with IMO standards

## Response Format

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis following the framework above...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction = "None"
        reasoning = ""
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
