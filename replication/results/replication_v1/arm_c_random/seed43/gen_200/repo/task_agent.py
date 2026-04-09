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


# Grade patterns for extraction - compiled once for performance
_GRADE_PATTERNS = {
    'fractional': re.compile(r'\b([0-7])\s*/\s*7\b'),
    'numeric': re.compile(r'\b([0-7])\b'),
    'partial': [
        re.compile(r'partial\s*(?:credit)?\s*:?\s*([0-7])', re.I),
        re.compile(r'partial\s*([0-7])\s*(?:points?)?', re.I),
        re.compile(r'([0-7])\s*(?:points?)?\s*partial', re.I),
    ],
    'full': [
        re.compile(r'\bfull\s*(?:credit|points?|score)?\b', re.I),
        re.compile(r'\bcomplete\s*(?:solution|answer|credit)?\b', re.I),
        re.compile(r'\ball\s*(?:points?|credit|marks?)?\b', re.I),
        re.compile(r'\bperfect\s*(?:score|solution)?\b', re.I),
    ],
    'zero': [
        re.compile(r'\bno\s*(?:credit|points?|score|marks?)?\b', re.I),
        re.compile(r'\bzero\s*(?:credit|points?|score|marks?)?\b', re.I),
        re.compile(r'\b0\s*(?:points?|credit|score|marks?)?\b', re.I),
        re.compile(r'\bincorrect\s*(?:solution|answer)?\b', re.I),
        re.compile(r'\bwrong\s*(?:solution|answer)?\b', re.I),
    ],
}


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


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```)."""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    return None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract JSON object using balanced brace counting.
    
    This handles nested JSON objects that regex might miss.
    """
    start_idx = text.find('{')
    while start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = start_idx + i + 1
                    break
        
        if end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        start_idx = text.find('{', start_idx + 1)
    
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with support for:
    - IMO 0-7 point scale
    - Partial credit notation
    - Fractional grades (e.g., 3/7)
    - Descriptive evaluations
    
    Uses pre-compiled regex patterns for performance.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Check for fractional grades like "3/7" or "5 / 7"
    fractional_match = _GRADE_PATTERNS['fractional'].search(pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades (0-7 for IMO problems)
    numeric_match = _GRADE_PATTERNS['numeric'].search(pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for partial credit patterns
    for pattern in _GRADE_PATTERNS['partial']:
        partial_match = pattern.search(pred_lower)
        if partial_match:
            return f"Partial credit: {partial_match.group(1)}", True
    
    # Check for full credit patterns
    for pattern in _GRADE_PATTERNS['full']:
        if pattern.search(pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns
    for pattern in _GRADE_PATTERNS['zero']:
        if pattern.search(pred_lower):
            return "0", True
    
    # Check for other valid grade keywords
    valid_keywords = ['correct', 'partial', 'n/a', 'not applicable']
    for keyword in valid_keywords:
        if keyword in pred_lower:
            return pred_clean, True
    
    # If prediction is very short (1-2 chars), it might be a grade
    if len(pred_clean) <= 2 and pred_clean.isdigit():
        grade_val = int(pred_clean)
        if 0 <= grade_val <= 7:
            return pred_clean, True
    
    # If no clear grade found, mark as invalid but return cleaned prediction
    return pred_clean, False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

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
3. Consider the grading guidelines carefully - IMO problems are typically graded 0-7 points.
4. Provide your reasoning before giving the final grade.
5. The final grade should be a clear numeric value (0-7) or descriptive evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies and better
        handling of edge cases like nested JSON, malformed responses,
        and alternative grading formats.
        
        Uses pre-compiled patterns and helper functions for efficiency.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            if not msg_history:
                return prediction, reasoning
                
            last_msg = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else ""
            if not last_msg:
                return prediction, reasoning
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 3: Try balanced brace JSON extraction
            balanced_json = _extract_balanced_json(last_msg)
            if balanced_json:
                if "response" in balanced_json:
                    prediction = str(balanced_json["response"]).strip()
                if "reasoning" in balanced_json:
                    reasoning = str(balanced_json["reasoning"])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 4: Look for grade patterns in text
            if prediction == "None":
                # Use pre-compiled patterns for efficiency
                grade_patterns = [
                    (re.compile(r'(?:final\s+)?(?:grade|score|mark|points?)\s*:?\s*([0-7]|partial|full|incorrect|correct|zero|none)', re.I), 1),
                    (re.compile(r'(?:assigned|given|awarded)\s*:?\s*([0-7])', re.I), 1),
                    (re.compile(r'(?:worth|value)\s*:?\s*([0-7])\s*(?:points?)?', re.I), 1),
                    (re.compile(r'\b(?:grade|score)\s+of\s+([0-7])\b', re.I), 1),
                    (re.compile(r'\b([0-7])\s*(?:out\s+of\s+7|/\s*7)\b', re.I), 1),
                ]
                
                for pattern, group_idx in grade_patterns:
                    grade_match = pattern.search(last_msg)
                    if grade_match:
                        prediction = grade_match.group(group_idx)
                        pred_lower = prediction.lower()
                        if pred_lower in ['full', 'correct', 'complete']:
                            prediction = '7'
                        elif pred_lower in ['zero', 'none', 'incorrect', 'wrong']:
                            prediction = '0'
                        break
            
            # Strategy 5: Extract reasoning from text if no JSON found
            if not reasoning:
                # Look for reasoning/analysis sections
                reasoning_patterns = [
                    re.compile(r'(?:reasoning|analysis|explanation|thought process)[:\s]+(.*?)(?:\n\n|\Z)', re.I | re.DOTALL),
                    re.compile(r'(?:step by step|detailed) analysis[:\s]+(.*?)(?:\n\n|\Z)', re.I | re.DOTALL),
                ]
                for pattern in reasoning_patterns:
                    reasoning_match = pattern.search(last_msg)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()[:500]  # Limit length
                        break
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
        # If grade is invalid, try to extract from the full response text
        if not is_valid and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")

        return str(validated_grade), msg_history
