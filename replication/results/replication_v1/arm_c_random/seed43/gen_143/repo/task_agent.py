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
import time
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Configuration for retry logic
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds between retries


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
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and retry logic."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY

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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def _call_llm_with_retry(self, instruction: str) -> tuple[str, list[dict], dict]:
        """Call LLM with retry logic for transient failures.
        
        Args:
            instruction: The prompt to send to the LLM
            
        Returns:
            (response_text, msg_history, info)
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                return response, msg_history, info
            except Exception as e:
                last_error = e
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.log_fn(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
        
        # All retries exhausted
        raise last_error if last_error else RuntimeError("LLM call failed after all retries")

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Try to extract from the last assistant message
            if not msg_history:
                self.log_fn("Warning: Empty message history")
                return prediction, reasoning
                
            last_msg = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else ""
            if not last_msg:
                self.log_fn("Warning: Last message has no text content")
                return prediction, reasoning
            
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                # Use the last valid JSON object found
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                    # Ensure prediction is a string
                    if not isinstance(prediction, str):
                        prediction = str(prediction)
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                    # Ensure reasoning is a string
                    if not isinstance(reasoning, str):
                        reasoning = str(reasoning)
                
                # Log the reasoning for debugging (truncated)
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to find any JSON-like structure with response field
                json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                        if not isinstance(prediction, str):
                            prediction = str(prediction)
                    except json.JSONDecodeError:
                        # Final fallback: try to extract response value with regex
                        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', last_msg)
                        if response_match:
                            prediction = response_match.group(1)
                        
        except (IndexError, KeyError, TypeError) as e:
            self.log_fn(f"Error extracting prediction (data structure): {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return str(prediction), reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = self._call_llm_with_retry(instruction)
        except Exception as e:
            self.log_fn(f"All LLM retries failed: {e}")
            # Return a fallback response
            return "Error: LLM unavailable", []

        # Extract prediction from JSON with better error handling
        prediction, reasoning = self._extract_prediction(msg_history)

        return prediction, msg_history
