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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, missing quotes,
    and unescaped special characters.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with common fixes
    for attempt in range(max_retries):
        try:
            fixed_text = text
            
            # Fix 1: Remove trailing commas before closing braces/brackets
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
            
            # Fix 2: Fix single quotes to double quotes (common LLM mistake)
            # Only fix if not inside a string
            fixed_text = re.sub(r"(?<!\\)'", '"', fixed_text)
            
            # Fix 3: Remove comments (// style and /* */ style)
            fixed_text = re.sub(r'//.*?\n', '\n', fixed_text)
            fixed_text = re.sub(r'/\*.*?\*/', '', fixed_text, flags=re.DOTALL)
            
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix attempt {attempt + 1} failed: {e}")
            continue
    
    # Log the problematic text for debugging (truncated)
    preview = text[:500] + "..." if len(text) > 500 else text
    logger.warning(f"Failed to extract JSON from text: {preview}")
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required input fields
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs]
        if missing_fields:
            logger.warning(f"Missing input fields: {missing_fields}")
        
        # Extract fields for better prompt construction
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent specializing in {domain}. Your task is to evaluate a student's answer to a problem with careful reasoning.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
1. Analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your evaluation with clear reasoning

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis and comparison",
    "evaluation": "Your final evaluation/grade based on the guidelines",
    "response": "Your complete evaluation result (this will be the final output)"
}}
</json>

Important: 
- Ensure your response is valid JSON with double quotes around keys and string values
- The "response" field should contain your complete evaluation
- Be thorough in your reasoning before providing the final evaluation"""

        self.log_fn(f"Processing task with model: {self.model}")
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Log token usage if available
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                          f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                          f"Total: {usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                extracted = _extract_json_with_retry(text_content)
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        prediction = last_json["response"]
                        self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                    elif "evaluation" in last_json:
                        # Fallback to evaluation field if response not present
                        prediction = last_json["evaluation"]
                        self.log_fn(f"Using 'evaluation' field: {str(prediction)[:100]}...")
                    else:
                        logger.warning(f"JSON missing expected keys. Keys found: {list(last_json.keys())}")
                        prediction = str(last_json)
                else:
                    logger.warning("No valid JSON found in response")
                    # Fallback: return raw text if no JSON found
                    prediction = text_content[:500] if text_content else "None"
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Full traceback:")

        return str(prediction), msg_history
