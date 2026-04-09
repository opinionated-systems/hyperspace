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
        # Try primary extraction from <json> tags
        extracted = _extract_jsons(text)
        if extracted and "response" in extracted[-1]:
            logger.info(f"Successfully extracted prediction using primary method")
            return extracted[-1]["response"]
        
        # Try fallback extraction
        fallback = _extract_json_fallback(text)
        if fallback and "response" in fallback:
            logger.info(f"Successfully extracted prediction using fallback method")
            return fallback["response"]
        
        # Try cleaning and re-parsing
        cleaned = _clean_json_text(text)
        if cleaned != text:
            extracted = _extract_jsons(cleaned)
            if extracted and "response" in extracted[-1]:
                logger.info(f"Successfully extracted prediction after cleaning")
                return extracted[-1]["response"]
            
            fallback = _extract_json_fallback(cleaned)
            if fallback and "response" in fallback:
                logger.info(f"Successfully extracted prediction using cleaned fallback")
                return fallback["response"]
        
        return None

    def _validate_prediction(self, prediction: str, inputs: dict) -> tuple[bool, str]:
        """Validate that the prediction is appropriate for the task.
        
        Args:
            prediction: The extracted prediction string
            inputs: The task inputs containing problem info
            
        Returns:
            (is_valid, reason) tuple
        """
        if not prediction or not isinstance(prediction, str):
            return False, "Prediction is empty or not a string"
        
        # Check for common error indicators
        error_indicators = [
            "error", "failed", "unable to", "cannot", "can't",
            "invalid", "not found", "no valid"
        ]
        prediction_lower = prediction.lower()
        for indicator in error_indicators:
            if indicator in prediction_lower and len(prediction) < 100:
                return False, f"Prediction contains error indicator: '{indicator}'"
        
        # Check minimum length for meaningful response
        if len(prediction.strip()) < 2:
            return False, "Prediction is too short"
        
        return True, "Valid prediction"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt with clear sections
        domain = inputs.get('domain', 'Unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

=== DOMAIN ===
{domain}

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT ANSWER TO EVALUATE ===
{student_answer}

=== INSTRUCTIONS ===
Evaluate the student answer based on the problem, correct solution, and grading guidelines above.
Provide a detailed evaluation that includes:
1. Whether the answer is correct, partially correct, or incorrect
2. Specific feedback on what was done well or what needs improvement
3. A numerical score or grade if applicable

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation/grade here. Be specific and constructive."
}}
</json>

Important: 
- Ensure your response is valid JSON wrapped in <json> tags
- The "response" field should contain your complete evaluation
- Be thorough but concise in your evaluation"""

        max_retries = 3
        prediction = "None"
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return f"Error: LLM call failed after {max_retries} attempts", []
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        # Extract prediction from JSON
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                extracted_prediction = self._extract_prediction(last_message)
                
                if extracted_prediction is not None:
                    prediction = extracted_prediction
                    # Validate the prediction
                    is_valid, reason = self._validate_prediction(prediction, inputs)
                    if not is_valid:
                        logger.warning(f"Prediction validation failed: {reason}")
                        self.log_fn(f"Validation warning: {reason}")
                else:
                    logger.warning(f"No valid JSON found in response")
                    self.log_fn(f"Raw response: {last_message[:500]}...")
                    # Try to use raw response as fallback
                    if last_message and len(last_message.strip()) > 0:
                        prediction = last_message.strip()[:1000]  # Limit length
            else:
                logger.warning("Empty message history received")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
