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

# Compile regex patterns once for efficiency
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer|assessment|evaluation)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
# Pattern to find JSON-like objects with nested braces
_JSON_OBJECT_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Finally, tries to find any valid JSON objects in the text.
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
            # Try to extract JSON from within the text if it's wrapped in other content
            try:
                # Look for JSON-like content with braces
                brace_start = inner.find("{")
                brace_end = inner.rfind("}")
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    results.append(json.loads(inner[brace_start:brace_end + 1]))
            except json.JSONDecodeError:
                # Try regex-based extraction for nested braces
                json_matches = _JSON_OBJECT_PATTERN.findall(inner)
                for match in json_matches:
                    try:
                        results.append(json.loads(match))
                    except json.JSONDecodeError:
                        continue
    
    # Fallback 1: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try to extract partial JSON
                json_matches = _JSON_OBJECT_PATTERN.findall(block)
                for match in json_matches:
                    try:
                        results.append(json.loads(match))
                    except json.JSONDecodeError:
                        continue
    
    # Fallback 2: search entire text for JSON objects
    if not results:
        json_matches = _JSON_OBJECT_PATTERN.findall(text)
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
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
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond ONLY in JSON format wrapped in <json>...</json> tags with the following exact schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

## Example Response:
<json>
{{
    "reasoning": "The student correctly identified the key theorem but made an error in the algebraic manipulation at step 3. The final answer is incorrect due to this calculation error.",
    "response": "Partially Correct"
}}
</json>

## Important Notes:
- Your response MUST start with <json> and end with </json>
- The JSON must be valid and properly formatted
- The "response" field should contain a concise grade/assessment
- The "reasoning" field should contain your detailed analysis
- Do not include any text outside the <json>...</json> tags

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        response_text = msg_history[-1]["text"]
        
        try:
            extracted = _extract_jsons(response_text)
            if extracted:
                last_json = extracted[-1]
                # Try multiple possible keys for the response (ordered by priority)
                response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", 
                               "score", "verdict", "conclusion", "decision", "output"]
                for key in response_keys:
                    if key in last_json and last_json[key] is not None:
                        prediction = last_json[key]
                        break
                
                # Log reasoning if available
                reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking", 
                               "rationale", "justification", "notes", "commentary"]
                for key in reasoning_keys:
                    if key in last_json and last_json[key]:
                        reasoning = last_json[key]
                        self.log_fn(f"Reasoning ({key}): {str(reasoning)[:200]}...")
                        break
            else:
                # Fallback: try to extract any meaningful text from the response
                # Look for common patterns like "Grade: X" or "Answer: X"
                grade_match = _GRADE_PATTERN.search(response_text)
                if grade_match:
                    prediction = grade_match.group(1).strip()
                    self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                else:
                    # Try to find standalone grades like "Correct", "Incorrect", "Partially Correct"
                    grade_keywords = ["Correct", "Incorrect", "Partially Correct", "Partial", 
                                    "Full Credit", "No Credit", "Pass", "Fail"]
                    for keyword in grade_keywords:
                        if keyword.lower() in response_text.lower():
                            prediction = keyword
                            self.log_fn(f"Extracted grade via keyword: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
