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
    """Fuzzy JSON extraction as fallback when tags are missing/malformed.
    
    Tries to find JSON objects by looking for curly braces.
    """
    results = []
    # Find all potential JSON objects by tracking brace depth
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to find a complete JSON object starting here
            brace_count = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found a complete object
                        try:
                            obj = json.loads(text[start:j+1])
                            results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
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
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Based on the grading guidelines, determine the appropriate score or evaluation.
4. Provide your reasoning in the "thinking" field.
5. Provide your final evaluation in the "response" field.

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your step-by-step analysis and reasoning here...",
    "response": "Your final answer/evaluation here..."
}}
</json>"""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and thinking from response text.
        
        Returns:
            (prediction, thinking) tuple
        """
        prediction = "None"
        thinking = ""
        
        # Try strict extraction first
        extracted = _extract_jsons(text)
        if not extracted:
            # Fallback to fuzzy extraction
            extracted = _extract_json_fuzzy(text)
        
        if extracted:
            last = extracted[-1]
            if isinstance(last, dict):
                if "response" in last:
                    prediction = last["response"]
                if "thinking" in last:
                    thinking = last["thinking"]
        
        return str(prediction), str(thinking)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from the last assistant message
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, thinking = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction on attempt {attempt + 1}")
                    break
                
                # If extraction failed and we have retries left, add feedback
                if attempt < self.max_retries:
                    feedback = """Your previous response did not contain a valid JSON object with a "response" field. 

Please ensure you respond in the exact format:
<json>
{
    "thinking": "Your analysis here...",
    "response": "Your final answer here..."
}
</json>"""
                    msg_history.append({"role": "user", "text": feedback})
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    break
        
        return str(prediction), msg_history
