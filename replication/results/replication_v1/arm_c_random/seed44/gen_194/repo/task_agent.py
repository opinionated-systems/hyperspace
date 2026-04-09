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


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text
    """
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Try to find raw JSON objects (objects with curly braces)
    # Look for patterns that look like JSON objects
    object_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(object_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Consider partial credit where appropriate - identify:
   - Correct approaches with minor errors
   - Partial solutions or incomplete proofs
   - Alternative valid methods
   - Common misconceptions to watch for
5. Provide your reasoning before giving the final grade
6. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning. Include: (1) What the student did correctly, (2) Any errors or gaps, (3) How partial credit was determined",
    "response": "The final grade/score (a number or specific grade value)",
    "confidence": "High|Medium|Low - your confidence in this grading decision",
    "partial_credit_breakdown": "Optional: explain how partial credit was awarded if applicable"
}}
</json>

Important: 
- The "response" field must contain only the final grade/score
- Use the "reasoning" field to show your work comprehensively
- Be precise and follow the grading guidelines exactly
- Award partial credit generously for correct approaches with minor errors
- Flag uncertain cases with lower confidence scores"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None, str | None, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning, confidence, partial_credit_breakdown)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                confidence = last.get("confidence")
                partial_credit = last.get("partial_credit_breakdown")
                return str(prediction), reasoning, confidence, partial_credit
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None, None, None

    def forward(self, inputs: dict) -> tuple[str, list[dict], dict]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history, metadata)
        """
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        metadata = {
            "reasoning": None,
            "confidence": None,
            "partial_credit_breakdown": None,
            "attempts_used": 0,
        }
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction
                text = msg_history[-1]["text"] if msg_history else ""
                pred, reasoning, confidence, partial_credit = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    metadata["reasoning"] = reasoning
                    metadata["confidence"] = confidence
                    metadata["partial_credit_breakdown"] = partial_credit
                    metadata["attempts_used"] = attempt + 1
                    
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    if confidence:
                        self.log_fn(f"Grading confidence: {confidence}")
                    break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = "Please respond in the required JSON format with 'response', 'reasoning', 'confidence', and 'partial_credit_breakdown' fields."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        return str(prediction), msg_history, metadata
