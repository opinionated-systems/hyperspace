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
        # Look for objects with "response" key - more flexible pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\}[^{}]*\})?'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict or None if all strategies fail.
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
        cleaned = text.replace("'", '"')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the response field if present
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        if response_match:
            result = {"response": response_match.group(1)}
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1)
            return result
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

        # Validate inputs
        if not problem or not student_answer:
            raise ValueError("Missing required inputs: problem and student_answer are required")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution if solution else "[No official solution provided - use your mathematical expertise to determine correctness]"}

## Grading Guidelines
{grading_guidelines if grading_guidelines else "[No specific grading guidelines - apply standard IMO grading principles]"}

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

## Response Format (CRITICAL)

You MUST respond in valid JSON format wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis following the framework above. Include specific observations about correctness, completeness, and any errors found.",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

IMPORTANT:
- The JSON must be valid (no trailing commas, proper quotes)
- The "response" field must contain ONLY the final grade/evaluation
- The "reasoning" field should contain your full step-by-step analysis
- Do not include any text outside the <json> tags"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        try:
            instruction = self._build_prompt(inputs)
        except ValueError as e:
            self.log_fn(f"Input validation error: {e}")
            return "Error: Invalid inputs", []

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with comprehensive error handling
        prediction = "None"
        reasoning = ""
        extraction_method = "none"
        
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            if not last_msg:
                self.log_fn("Warning: Empty response from LLM")
                return "None", msg_history
            
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                extraction_method = "json_extraction"
                
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning excerpt: {reasoning[:150]}...")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = str(fallback.get("response", "None")).strip()
                        extraction_method = "regex_fallback"
                    except json.JSONDecodeError:
                        pass
                
                # Last resort: look for grade patterns in text
                if prediction == "None":
                    grade_patterns = [
                        r'grade[\s:]+([0-9]+)',
                        r'score[\s:]+([0-9]+)',
                        r'([0-9]+)[\s/]*(?:out of|/)[\s]*7',
                        r'(?:final|grade|score)[\s:]+([0-9]+)',
                    ]
                    for pattern in grade_patterns:
                        match = re.search(pattern, last_msg, re.IGNORECASE)
                        if match:
                            prediction = match.group(1)
                            extraction_method = "pattern_matching"
                            break
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            extraction_method = "error"

        self.log_fn(f"Extraction method: {extraction_method}, Prediction: {prediction}")
        return str(prediction), msg_history
