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
    Also handles markdown code blocks (```json...```) as a fallback.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            logger.debug(f"Failed to parse <json> block: {e}")
            continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse markdown JSON block: {e}")
                continue
    
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, single quotes, etc.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with progressively more aggressive fixes
    fixes = [
        # Fix 1: Remove trailing commas
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix 2: Replace single quotes with double quotes (carefully)
        lambda t: re.sub(r"(?<!\\)'([^']*?)'(?=\s*[:}\],])", r'"\1"', t),
        # Fix 3: Remove comments and fix common issues
        lambda t: re.sub(r'//.*?\n', '\n', re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL)),
    ]
    
    for attempt in range(min(max_retries, len(fixes))):
        try:
            fixed_text = fixes[attempt](text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix attempt {attempt + 1} failed: {e}")
            continue
    
    # Last resort: try to find any JSON-like structure with braces
    try:
        # Find content between outermost braces
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            result = _extract_jsons(potential_json)
            if result:
                logger.debug("JSON extraction succeeded using brace extraction")
                return result
    except Exception as e:
        logger.debug(f"Brace extraction failed: {e}")
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution to a mathematical problem.

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

## Your Task

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the critical steps and insights needed for a correct solution.

3. **Evaluate the Student's Answer**: 
   - Check if the student understood the problem correctly
   - Verify each step of their reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or valid alternative approaches

4. **Assign a Grade**: Based on the grading guidelines, assign a numerical score. Common IMO scoring:
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5: Significant progress with some gaps
   - 3-4: Partial progress
   - 1-2: Minimal progress
   - 0: No meaningful progress or completely wrong

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed analysis of the student's work",
    "reasoning": "Step-by-step evaluation explaining your grading decision",
    "score": 7,
    "response": "7"
}}
</json>

The "response" field should contain only the numerical score as a string."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            # Get the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            if not last_msg:
                logger.warning("No assistant message found in history")
                return "None", msg_history
            
            extracted = _extract_json_with_retry(last_msg)
            if extracted:
                last_extracted = extracted[-1]
                # Try to get the response field first
                if "response" in last_extracted:
                    prediction = last_extracted["response"]
                    logger.debug(f"Extracted prediction from 'response' field: {prediction}")
                # Fallback to score field if response is not present
                elif "score" in last_extracted:
                    prediction = str(last_extracted["score"])
                    logger.debug(f"Extracted prediction from 'score' field: {prediction}")
                else:
                    logger.warning(f"No 'response' or 'score' field found in extracted JSON: {last_extracted.keys()}")
            else:
                logger.warning("Failed to extract JSON from LLM response")
                # Try to extract just a number from the response as last resort
                numbers = re.findall(r'\b([0-7])\b', last_msg)
                if numbers:
                    prediction = numbers[-1]
                    logger.debug(f"Extracted prediction using regex: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Full traceback:")

        return str(prediction), msg_history
