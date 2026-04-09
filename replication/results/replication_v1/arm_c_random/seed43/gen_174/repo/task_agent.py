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
    Also handles markdown code blocks and raw JSON objects.
    Includes enhanced error recovery for common JSON formatting issues.
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
        # Look for objects with "response" key - more permissive pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    # Final fallback: try to extract just the grade from plain text
    if not results:
        # Common IMO grade patterns: single digits 0-7, or phrases like "Grade: 5"
        grade_patterns = [
            r'[Gg]rade[:\s]+([0-7])',
            r'[Ss]core[:\s]+([0-7])',
            r'[Ff]inal grade[:\s]+([0-7])',
            r'[Rr]esult[:\s]+([0-7])',
            r'\b([0-7])\s*/\s*7\b',
            r'\b([0-7])\s*points?\b',
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, text)
            if match:
                grade = match.group(1)
                results.append({"response": grade, "reasoning": "Extracted from plain text pattern matching"})
                break
    
    return results or None


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
    
    # Strategy 3: Fix single quotes to double quotes (common LLM error)
    try:
        # Replace single quotes around keys and string values
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the response field if present
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*?)"', text)
        if response_match:
            return {"response": response_match.group(1)}
    except Exception:
        pass
    
    # Strategy 5: Extract reasoning and response fields separately
    try:
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*?)"', text, re.DOTALL)
        response_match = re.search(r'"response"\s*:\s*"([^"]*?)"', text)
        if reasoning_match or response_match:
            result = {}
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1)
            if response_match:
                result["response"] = response_match.group(1)
            return result
    except Exception:
        pass
    
    # Strategy 6: Handle unescaped newlines in JSON strings
    try:
        # Replace newlines within string values with escaped newlines
        # This is a heuristic approach - look for patterns that suggest unescaped newlines
        fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

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

## Instructions

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.

## Grading Scale
IMO problems are typically graded on a scale of 0-7 points:
- 7: Complete, correct solution
- 6: Minor flaw or gap in an otherwise correct solution
- 5-3: Partial credit based on progress made
- 2-1: Significant progress but major gaps
- 0: No meaningful progress or completely incorrect

## Response Format (IMPORTANT)

You MUST respond in the following JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade as a number 0-7 or description"
}}
</json>

Requirements:
- The "reasoning" field must contain your full step-by-step analysis
- The "response" field must contain ONLY the final grade (e.g., "7", "5", "Partial credit: 3")
- Do not include any text outside the <json> tags
- Ensure the JSON is valid with proper escaping of quotes and newlines"""

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

        # Validate the prediction - ensure it's a reasonable grade format
        prediction = self._normalize_grade(prediction)
        
        return str(prediction), msg_history

    def _normalize_grade(self, grade: str) -> str:
        """Normalize the grade to a standard format.
        
        Args:
            grade: Raw grade string from LLM
            
        Returns:
            Normalized grade string
        """
        if not grade or grade == "None":
            return "None"
        
        grade = str(grade).strip()
        
        # Check for numeric grades 0-7
        if grade.isdigit():
            num = int(grade)
            if 0 <= num <= 7:
                return str(num)
        
        # Check for patterns like "Grade: 5" or "Score: 3"
        match = re.search(r'\b([0-7])\b', grade)
        if match:
            return match.group(1)
        
        # Check for partial credit patterns
        partial_match = re.search(r'[Pp]artial\s+[Cc]redit[:\s]+([0-7])', grade)
        if partial_match:
            return f"Partial credit: {partial_match.group(1)}"
        
        # Return original if no normalization possible
        return grade
