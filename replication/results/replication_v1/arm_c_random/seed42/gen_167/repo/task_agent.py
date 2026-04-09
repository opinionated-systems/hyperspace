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
            logger.debug(f"JSON decode error at position {start}: {e}")
            continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback method to extract JSON using regex patterns.
    
    Used when primary extraction fails to find valid JSON.
    """
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON between curly braces (improved pattern for nested braces)
    brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def _clean_json_text(text: str) -> str:
    """Clean JSON text by removing common formatting issues."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Remove single-line comments
    text = re.sub(r'//[^\n]*', '', text)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def _extract_prediction(self, text: str) -> str | None:
        """Extract prediction from text using multiple methods.
        
        Args:
            text: The text to extract prediction from.
            
        Returns:
            The extracted prediction or None if extraction fails.
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for prediction extraction")
            return None
            
        # Try primary extraction from <json> tags
        extracted = _extract_jsons(text)
        if extracted:
            # Check all extracted JSONs for response field, starting from the last
            for json_obj in reversed(extracted):
                if "response" in json_obj:
                    response = json_obj["response"]
                    if response is not None:
                        logger.info(f"Successfully extracted prediction using primary method")
                        return str(response)
        
        # Try fallback extraction
        fallback = _extract_json_fallback(text)
        if fallback and "response" in fallback:
            response = fallback["response"]
            if response is not None:
                logger.info(f"Successfully extracted prediction using fallback method")
                return str(response)
        
        # Try cleaning and re-parsing
        cleaned = _clean_json_text(text)
        if cleaned != text:
            extracted = _extract_jsons(cleaned)
            if extracted:
                for json_obj in reversed(extracted):
                    if "response" in json_obj:
                        response = json_obj["response"]
                        if response is not None:
                            logger.info(f"Successfully extracted prediction after cleaning")
                            return str(response)
            
            fallback = _extract_json_fallback(cleaned)
            if fallback and "response" in fallback:
                response = fallback["response"]
                if response is not None:
                    logger.info(f"Successfully extracted prediction using cleaned fallback")
                    return str(response)
        
        # Final fallback: try to extract any meaningful text after common patterns
        patterns = [
            r'["\']?response["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'["\']?response["\']?\s*[:=]\s*([\w\s.,;:!?()-]+?)(?:\n|$)',
            r'(?:evaluation|grade|assessment|answer)\s*[:=]\s*([\w\s.,;:!?()-]+?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip()
                if result and len(result) > 2:
                    logger.info(f"Successfully extracted prediction using regex pattern")
                    return result
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            logger.error(f"Invalid inputs type: expected dict, got {type(inputs)}")
            return "Error: Invalid inputs type", []
        
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build a more structured instruction
        instruction = f"""You are an expert grading agent for {domain} problems. Your task is to evaluate the student's answer based on the provided problem, correct solution, and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Evaluate the student's answer carefully. Consider:
1. Correctness of the approach and final answer
2. Partial credit for correct steps even if the final answer is wrong
3. Adherence to the grading guidelines provided

Provide your evaluation in the following JSON format:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) whether the answer is correct/incorrect/partially correct, (2) specific points awarded if applicable, (3) brief explanation of errors if any."
}}
</json>

Important: 
- Ensure your response is valid JSON wrapped in <json> tags
- Be specific and constructive in your feedback
- If assigning points, clearly state the point value"""

        msg_history = []
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if last_message:
                    extracted_prediction = self._extract_prediction(last_message)
                    
                    if extracted_prediction is not None:
                        prediction = extracted_prediction
                    else:
                        logger.warning(f"No valid JSON found in response")
                        self.log_fn(f"Raw response: {last_message[:500]}...")
                        # Fallback: use the raw response if it's not too long
                        if len(last_message) < 1000 and last_message.strip():
                            prediction = last_message.strip()
                else:
                    logger.warning("Empty message text in last history entry")
            else:
                logger.warning("Empty message history received")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
