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

# Maximum response length for fallback extraction
MAX_RESPONSE_LENGTH = 1000
# Timeout for LLM calls in seconds
LLM_TIMEOUT = 120


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
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                # If it's a list of dicts, extend results
                for item in parsed:
                    if isinstance(item, dict):
                        results.append(item)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback for malformed responses.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        results.append(item)
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces using balanced brace counting
    if not results:
        # Find all potential JSON starting points
        for start_idx in [m.start() for m in re.finditer(r'\{', text)]:
            try:
                # Count braces to find complete JSON
                brace_count = 0
                for i, char in enumerate(text[start_idx:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[start_idx:start_idx+i+1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict) and any(k in parsed for k in ["response", "reasoning", "answer", "result", "grade"]):
                                    results.append(parsed)
                                    break  # Found a valid one, move to next start
                            except json.JSONDecodeError:
                                break
            except Exception:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        start_time = time.time()
        
        # Build a structured prompt with chain-of-thought reasoning
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        self.log_fn(f"[Call {self.call_count}] Processing {domain} problem")
        
        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the official solution.
2. Identify what the student got correct and what they got wrong.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the response field.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning for the grade",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Ensure your JSON is valid and properly formatted."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"[Call {self.call_count}] LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}", []

        elapsed = time.time() - start_time
        self.log_fn(f"[Call {self.call_count}] LLM call completed in {elapsed:.2f}s")

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        extraction_method = "none"
        
        try:
            if not msg_history:
                self.log_fn(f"[Call {self.call_count}] Empty message history")
                return "Error: Empty message history", msg_history
                
            last_message = msg_history[-1]["text"]
            
            # Try standard extraction first
            extracted = _extract_jsons(last_message)
            if extracted:
                extraction_method = "standard"
            
            # If that fails, try fuzzy extraction
            if not extracted:
                extracted = _extract_json_fuzzy(last_message)
                if extracted:
                    extraction_method = "fuzzy"
            
            if extracted:
                # Prefer response field, but fallback to other common fields
                last_extracted = extracted[-1]
                if "response" in last_extracted:
                    prediction = last_extracted["response"]
                elif "answer" in last_extracted:
                    prediction = last_extracted["answer"]
                elif "result" in last_extracted:
                    prediction = last_extracted["result"]
                elif "grade" in last_extracted:
                    prediction = last_extracted["grade"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_extracted)
                    
                self.log_fn(f"[Call {self.call_count}] Extracted prediction via {extraction_method}: {prediction}")
            else:
                self.log_fn(f"[Call {self.call_count}] No JSON found in response, using raw text")
                # Fallback: use the raw response text (truncated)
                prediction = last_message[:MAX_RESPONSE_LENGTH] if len(last_message) > MAX_RESPONSE_LENGTH else last_message
                extraction_method = "raw"
                
        except Exception as e:
            self.log_fn(f"[Call {self.call_count}] Error extracting prediction: {e}")
            prediction = f"Error: {str(e)}"

        return str(prediction), msg_history

    def reset_call_count(self) -> None:
        """Reset the call counter. Useful for testing."""
        self.call_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "call_count": self.call_count,
            "model": self.model,
        }
