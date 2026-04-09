"""
Task agent: evaluates student answers against official solutions for IMO grading.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Key improvements in this version:
- Structured 5-step evaluation framework (Understanding, Approach, Execution, Completeness, Partial Credit)
- Input validation for required fields
- Improved extraction with detailed logging of extraction methods
- Better error handling and fallback strategies
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
    """
    results = []
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
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with common fixes
    for attempt in range(max_retries):
        try:
            # Try to find and fix JSON with trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            pass
    
    return None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple fallback strategies.
    
    Tries multiple approaches:
    1. Standard <json>...</json> extraction
    2. Direct JSON object parsing from the entire text
    3. Regex-based extraction of JSON-like structures
    """
    # Strategy 1: Standard extraction
    result = _extract_json_with_retry(text)
    if result:
        return result[-1] if result else None
    
    # Strategy 2: Try to parse the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Look for JSON objects with regex
    # Match content between outermost braces
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    matches = json_pattern.findall(text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _format_grading_prompt(self, inputs: dict) -> str:
        """Format the grading prompt with structured instructions."""
        domain = inputs.get("domain", "mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Evaluate the student's answer carefully using the following structured approach:

### 1. Understanding Assessment
- Does the student demonstrate understanding of the problem?
- Are the key concepts and theorems correctly identified?

### 2. Approach Evaluation
- Is the solution approach valid and appropriate?
- Are there any logical gaps or invalid assumptions?

### 3. Execution Analysis
- Are calculations and derivations correct?
- Is the reasoning clearly presented and well-structured?

### 4. Completeness Check
- Does the answer address all parts of the problem?
- Are edge cases or special conditions considered?

### 5. Partial Credit Determination
- Identify which steps were completed correctly
- Note where errors first occur and their impact
- Award credit proportionally based on correct work shown

## Scoring Rubric
- **Correct (Full Credit)**: Complete, correct solution with valid reasoning
- **Partially Correct**: Significant progress with minor errors or omissions
- **Incorrect**: Fundamental misunderstanding or major errors

Provide your evaluation in the following JSON format:
<json>
{{
    "response": "Your detailed grading feedback here. Structure your response with: (1) Overall assessment (Correct/Partially Correct/Incorrect), (2) Points awarded with justification, (3) Specific strengths identified, (4) Errors or omissions noted with explanations, (5) Suggestions for improvement if applicable."
}}
</json>"""

    def _extract_grading_result(self, msg_history: list[dict]) -> dict:
        """Extract structured grading result from message history.
        
        Returns a dict with 'prediction' (the main response) and 
        'extraction_method' indicating how it was obtained.
        """
        result = {
            "prediction": "None",
            "extraction_method": "unknown",
            "success": False
        }
        
        if not msg_history or len(msg_history) == 0:
            result["extraction_method"] = "no_history"
            return result
            
        last_message = msg_history[-1].get("text", "")
        
        # Try structured JSON extraction first
        extracted = _extract_json_flexible(last_message)
        if extracted and "response" in extracted:
            result["prediction"] = extracted["response"]
            result["extraction_method"] = "json_response_field"
            result["success"] = True
            return result
        
        # Fallback: check if the entire message is a valid grading response
        if len(last_message) > 50:  # Minimum length for a meaningful response
            result["prediction"] = last_message[:2000]  # Increased truncation limit
            result["extraction_method"] = "raw_truncated"
            result["success"] = True
        else:
            result["prediction"] = last_message
            result["extraction_method"] = "raw_full"
            result["success"] = True
            
        return result

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_fields = ["problem", "solution", "student_answer"]
        missing = [f for f in required_fields if not inputs.get(f)]
        if missing:
            error_msg = f"Error: Missing required fields: {', '.join(missing)}"
            self.log_fn(error_msg)
            return error_msg, []
        
        instruction = self._format_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction using structured method
        result = self._extract_grading_result(msg_history)
        prediction = result["prediction"]
        
        if result["success"]:
            self.log_fn(f"Grading extraction successful (method: {result['extraction_method']})")
        else:
            self.log_fn(f"Grading extraction failed (method: {result['extraction_method']})")

        return str(prediction), msg_history
