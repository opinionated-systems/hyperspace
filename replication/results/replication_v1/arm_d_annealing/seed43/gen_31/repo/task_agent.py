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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback for malformed responses.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces
    if not results:
        # Look for patterns like {"response": ...} or {"reasoning": ...}
        brace_pattern = r'\{[^{}]*"(?:response|reasoning|answer|result)"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(brace_pattern, text, re.DOTALL):
            try:
                # Try to parse, handling nested braces by extending the match
                candidate = match.group(0)
                # Count braces to find complete JSON
                brace_count = 0
                start_idx = text.find(candidate)
                for i, char in enumerate(text[start_idx:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                results.append(json.loads(text[start_idx:start_idx+i+1]))
                                break
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

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a structured prompt with chain-of-thought reasoning
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
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

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try standard extraction first
            extracted = _extract_jsons(last_message)
            
            # If that fails, try fuzzy extraction
            if not extracted:
                extracted = _extract_json_fuzzy(last_message)
            
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
                    
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn("No JSON found in response, using raw text")
                # Fallback: use the raw response text (truncated)
                prediction = last_message[:500] if len(last_message) > 500 else last_message
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
