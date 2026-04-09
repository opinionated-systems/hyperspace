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
    Also handles markdown code blocks with json tag.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
    
    # Also try markdown ```json blocks if no results yet
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
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, unquoted keys, etc.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with progressively more aggressive fixes
    fixes = [
        # Fix 1: Remove trailing commas
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix 2: Remove comments (both // and /* */)
        lambda t: re.sub(r'//.*?\n', '\n', re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL)),
        # Fix 3: Try to extract JSON-like structures directly
        lambda t: t,
    ]
    
    for attempt, fix in enumerate(fixes[:max_retries]):
        try:
            fixed_text = fix(text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            continue
    
    # Final fallback: try to find any JSON-like structure with braces
    try:
        # Look for content between outermost braces
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            # Try to parse it
            data = json.loads(potential_json)
            return [data]
    except Exception:
        pass
    
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
            extracted = _extract_json_with_retry(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Try to get the response field first
                if "response" in last_json:
                    prediction = last_json["response"]
                # Fallback to score field if response is not present
                elif "score" in last_json:
                    prediction = str(last_json["score"])
                # Additional fallback: try common field names
                elif "grade" in last_json:
                    prediction = str(last_json["grade"])
                elif "result" in last_json:
                    prediction = str(last_json["result"])
                elif "answer" in last_json:
                    prediction = str(last_json["answer"])
                
                # Validate that prediction is a valid number string
                try:
                    float(prediction)
                except (ValueError, TypeError):
                    # If not a valid number, keep as string but log it
                    self.log_fn(f"Non-numeric prediction extracted: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
