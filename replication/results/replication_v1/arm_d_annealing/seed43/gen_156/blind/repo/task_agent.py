"""
Task agent: solves a given task with chain-of-thought reasoning.

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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries in order:
    1. <json>...</json> blocks
    2. ```json...``` code blocks
    3. Raw JSON objects in text
    """
    # Strategy 1: <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: ```json code blocks
    json_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON-like structures with "response" field
    response_pattern = r'"response"\s*:\s*("[^"]*"|\d+|true|false|null|\{[^}]*\}|\[[^\]]*\])'
    match = re.search(response_pattern, text)
    if match:
        try:
            # Try to construct a valid JSON object
            json_str = '{' + match.group(0) + '}'
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


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
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student answers.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines for partial credit.
4. Provide your reasoning before giving the final grade.
5. Respond in JSON format with your reasoning and final response.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis and reasoning for the grade",
    "response": "Your final grade/assessment"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with flexible parsing
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_json_flexible(last_message)
            if extracted:
                if "response" in extracted:
                    prediction = extracted["response"]
                elif "answer" in extracted:
                    prediction = extracted["answer"]
                elif "grade" in extracted:
                    prediction = extracted["grade"]
                
                # Log reasoning if available
                if "reasoning" in extracted:
                    self.log_fn(f"Agent reasoning: {extracted['reasoning'][:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any numeric or letter grade from the raw text
            try:
                # Look for patterns like "Grade: A" or "Score: 5" or just numbers
                grade_patterns = [
                    r'[Gg]rade[:\s]+([A-F0-9]+)',
                    r'[Ss]core[:\s]+([0-9]+)',
                    r'[Ff]inal[:\s]+([A-F0-9]+)',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_message)
                    if match:
                        prediction = match.group(1)
                        break
            except Exception:
                pass

        return str(prediction), msg_history
