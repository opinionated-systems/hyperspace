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
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional heuristics for malformed JSON and nested structures.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the text if it's wrapped in other content
        try:
            # Find the first { and last }
            json_start = inner.find("{")
            json_end = inner.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try handling common JSON formatting issues
        try:
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Try to find and extract the outermost JSON object
            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(cleaned[json_start:json_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try extracting just the first complete JSON object
        try:
            brace_count = 0
            in_string = False
            escape_next = False
            json_start = -1
            
            for i, char in enumerate(inner):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start != -1:
                            results.append(json.loads(inner[json_start:i + 1]))
                            break
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try plain ```
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            if text[start:start + 7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            
            # Try direct parsing
            try:
                results.append(json.loads(inner))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try with cleaning for markdown blocks too
            try:
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                json_start = cleaned.find("{")
                json_end = cleaned.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(cleaned[json_start:json_end + 1]))
                    continue
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        try:
            # Look for patterns like {"key": value} or {"key": "value"}
            json_pattern = re.search(r'\{[^{}]*"[^"]+"\s*:\s*[^}]+\}', text)
            if json_pattern:
                results.append(json.loads(json_pattern.group()))
        except (json.JSONDecodeError, AttributeError):
            pass
    
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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

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
1. Analyze the student's answer step by step
2. Compare it against the official solution and grading guidelines
3. Identify any errors, missing steps, or correct approaches
4. Determine the appropriate grade/score

Provide your reasoning first, then respond with your final evaluation in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "score": "The numerical score or grade",
    "response": "The final grade/score as a number or string"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response field first, then score, then any numeric value
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                else:
                    # Fallback: use the first value that's not "reasoning"
                    for key, value in last_json.items():
                        if key != "reasoning" and isinstance(value, (str, int, float)):
                            prediction = value
                            break
                
                # Validate that prediction is a reasonable value
                if prediction is None or prediction == "":
                    prediction = "None"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
