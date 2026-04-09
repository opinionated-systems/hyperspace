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
    extraction_attempts = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Log the error for debugging but continue trying other blocks
            logger.debug(f"JSON decode error in block {extraction_attempts}: {e}")
            continue
    
    if extraction_attempts > 0 and not results:
        logger.warning(f"Failed to extract JSON from {extraction_attempts} <json> blocks")
    
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text
    """
    results = []
    
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        logger.debug(f"Extracted {len(results)} JSON objects from <json> blocks")
        return results
    
    # Try JSON code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    code_block_success = 0
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
            code_block_success += 1
        except json.JSONDecodeError:
            continue
    
    if results:
        logger.debug(f"Extracted {code_block_success} JSON objects from code blocks")
        return results
    
    # Try to find raw JSON objects (objects with curly braces)
    # Look for patterns that look like JSON objects
    object_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(object_pattern, text, re.DOTALL)
    raw_object_success = 0
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
            raw_object_success += 1
        except json.JSONDecodeError:
            continue
    
    if results:
        logger.debug(f"Extracted {raw_object_success} JSON objects from raw text")
    else:
        logger.warning("All JSON extraction strategies failed")
    
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
4. Identify any errors, omissions, or misconceptions
5. Award partial credit where appropriate based on the guidelines
6. Provide your reasoning before giving the final grade
7. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning. Include: (1) What the student did correctly, (2) Any errors or issues found, (3) How partial credit was determined",
    "response": "The final grade/score (a number or specific grade value)"
}}
</json>

Important: 
- The "response" field must contain ONLY the final grade/score (no explanations, no extra text)
- The "reasoning" field should be thorough and justify the grade assigned
- Be precise and follow the grading guidelines exactly
- If the guidelines specify numeric scores, provide only the number
- If the guidelines specify letter grades, provide only the letter"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text with validation.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                
                # Validate prediction is not empty or None
                if prediction is None or str(prediction).strip() == "":
                    return "None", reasoning
                
                # Clean up prediction - remove extra whitespace and common formatting issues
                prediction_str = str(prediction).strip()
                # Remove quotes if the model wrapped the grade in quotes
                if prediction_str.startswith('"') and prediction_str.endswith('"'):
                    prediction_str = prediction_str[1:-1].strip()
                if prediction_str.startswith("'") and prediction_str.endswith("'"):
                    prediction_str = prediction_str[1:-1].strip()
                
                return prediction_str, reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
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
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, add a follow-up message asking for proper format
                if attempt < self.max_retries - 1:
                    instruction = """Your previous response did not contain a valid JSON object in the required format.

Please respond using this exact format:

<json>
{
    "reasoning": "Your detailed analysis here",
    "response": "The final grade/score here"
}
</json>

Important: The "response" field must contain ONLY the grade value (number or letter), with no additional text or explanation."""
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        return str(prediction), msg_history
