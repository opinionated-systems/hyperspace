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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}, content: {inner[:100]}...")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction using regex patterns for common formats.
    
    Attempts to extract JSON from code blocks or raw JSON objects.
    Enhanced with better handling for nested structures and common LLM output formats.
    """
    # Try to extract from markdown code blocks (various formats)
    code_block_patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',  # Standard markdown
        r'`\s*(\{.*?\})\s*`',  # Inline code with JSON
        r'<json>\s*(\{.*?\})\s*</json>',  # XML-style tags without proper formatting
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if "response" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Try to find raw JSON objects with improved pattern for nested structures
    # This pattern handles up to 3 levels of nesting
    json_patterns = [
        r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # Nested JSON
        r'\{[^{}]*\}',  # Simple JSON
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "response" in parsed and isinstance(parsed["response"], str):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to extract any key-value pair that looks like a response
    response_pattern = r'["\']?response["\']?\s*[:=]\s*["\']([^"\']+)["\']'
    match = re.search(response_pattern, text, re.IGNORECASE)
    if match:
        return {"response": match.group(1)}
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {"total_calls": 0, "json_extracted": 0, "fallback_used": 0, "failed": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to evaluate a student's answer by comparing it against the correct solution and following the grading guidelines.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
1. Carefully analyze the student's answer against the correct solution
2. Identify any errors, misconceptions, or missing steps
3. Consider partial credit where appropriate based on the grading guidelines
4. Provide a clear, detailed evaluation explaining your reasoning
5. Be specific about what the student did correctly and incorrectly

## Structured Evaluation Rubric:
When evaluating, consider these aspects and assign a score (0-100) for each:

**Correctness (0-100)**: How accurate is the answer mathematically/logically?
- 90-100: Fully correct with proper reasoning
- 70-89: Minor errors but correct approach
- 50-69: Significant errors, partial understanding
- 0-49: Major errors or incorrect approach

**Completeness (0-100)**: Does the answer address all parts of the problem?
- 90-100: All parts addressed thoroughly
- 70-89: Most parts addressed, minor omissions
- 50-69: Some parts missing or incomplete
- 0-49: Major omissions or incomplete work

**Clarity (0-100)**: Is the reasoning clear and well-explained?
- 90-100: Clear, logical, well-structured
- 70-89: Mostly clear, minor confusion
- 50-69: Some clarity issues
- 0-49: Unclear or confusing explanation

**Overall Score**: Average of the three scores above, rounded to nearest integer.

## Common Pitfalls to Check:
- Did the student misinterpret the problem?
- Are there calculation errors or algebraic mistakes?
- Is the reasoning circular or incomplete?
- Did the student skip important steps?
- Are there logical fallacies or invalid assumptions?
- Is the final answer clearly stated?

## Response Format:
Respond ONLY in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here. Structure your response as follows:\n\n## Overall Assessment\n[2-3 sentence summary of the answer quality]\n\n## Score Breakdown\n- Correctness: [score]/100\n- Completeness: [score]/100\n- Clarity: [score]/100\n- Overall Score: [average]/100\n\n## What Was Done Correctly\n- [Point 1]\n- [Point 2]\n\n## Errors and Issues\n- [Error 1: description and impact]\n- [Error 2: description and impact]\n\n## Recommendations\n[Specific suggestions for improvement]\n\n## Final Judgment\n[Clear statement on whether the answer is acceptable, needs revision, or is incorrect]"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly escaped. The response field must contain a single string value."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            self.stats["failed"] += 1
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                
                # Primary extraction method
                extracted = _extract_jsons(last_message)
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "primary"
                    self.stats["json_extracted"] += 1
                else:
                    # Fallback extraction
                    fallback = _extract_json_fallback(last_message)
                    if fallback and "response" in fallback:
                        prediction = fallback["response"]
                        extraction_method = "fallback"
                        self.stats["fallback_used"] += 1
                        self.log_fn(f"Used fallback extraction for response")
                    else:
                        # Last resort: use raw text
                        prediction = last_message[:500]  # Limit length
                        extraction_method = "raw"
                        self.log_fn(f"Using raw text extraction (limited to 500 chars)")
                        
                self.log_fn(f"Extraction method: {extraction_method}, prediction length: {len(str(prediction))}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            self.stats["failed"] += 1

        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return self.stats.copy()
