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

__version__ = "1.3.0"

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.

    Falls back to parsing the entire text as JSON if no tags are found.
    Also handles markdown code blocks (```json ... ```) as a secondary fallback.
    Finally tries a broad regex search for any JSON-like object.
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

    # Fallback 1: if no tagged JSON found, try parsing the whole text
    if not results:
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                results.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            pass

    # Fallback 2: try extracting from markdown code blocks
    if not results:
        code_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        for block in code_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    results.append(parsed)
                elif isinstance(parsed, list):
                    results.extend(item for item in parsed if isinstance(item, dict))
            except json.JSONDecodeError:
                continue

    # Fallback 3: broad regex search for JSON objects
    if not results:
        # Match balanced braces up to 2 levels deep
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                pass

    # Fallback 4: try to find and fix common JSON issues (trailing commas, unquoted keys)
    if not results:
        # Try to extract anything that looks like a JSON object and fix common issues
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        for match in json_pattern.finditer(text):
            candidate = match.group()
            # Fix trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
            # Fix unquoted keys
            fixed = re.sub(r'(\w+)\s*:', r'"\1":', fixed)
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    break
            except json.JSONDecodeError:
                continue

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)
            self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grading agent. Your task is to evaluate student answers against the provided solution and grading guidelines.

## Problem
{inputs.get('problem', 'N/A')}

## Official Solution
{inputs.get('solution', 'N/A')}

## Grading Guidelines
{inputs.get('grading_guidelines', 'N/A')}

## Student's Answer
{inputs.get('student_answer', 'N/A')}

Follow these steps carefully:
1. Read and understand the problem statement and the official solution.
2. Review the grading guidelines to understand the scoring rubric.
3. Analyze the student's answer step by step, checking for correctness, completeness, and mathematical rigor.
4. Compare the student's reasoning with the official solution.
5. Apply the grading guidelines to determine the appropriate score.

Your response must be a single JSON object with the following schema:
<json>
{{
    "score": "numerical score (integer) assigned to the student's answer based on the grading guidelines",
    "label": "one of: Correct, Incorrect, Partial",
    "response": "Your detailed grading analysis explaining the score assignment"
}}
</json>

Important:
- The "score" field must be an integer reflecting the points earned.
- The "label" field must be exactly one of: "Correct", "Incorrect", or "Partial" (capitalized).
- The "response" field must contain your complete grading analysis.
- Be thorough in comparing the student's work against the official solution.
- Explicitly state which parts of the grading criteria the student meets or fails.
- Provide a clear final assessment.
- If the student's answer is mathematically equivalent to the official solution but uses a different approach, still award full credit.
- Partial credit should be given for correct intermediate steps even if the final answer is wrong.
- You MUST wrap your JSON response in <json> and </json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            response_text = msg_history[-1]["text"]
            self.log_fn(f"LLM response length: {len(response_text)}")
            extracted = _extract_jsons(response_text)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"Extracted JSON keys: {list(result.keys())}")
                # Return structured prediction with score and label for evaluation
                score = result.get("score", "None")
                label = result.get("label", "None")
                # Normalize label to title case to match expected format (Correct/Incorrect/Partial)
                if isinstance(label, str):
                    label = label.strip().title()
                response_text = result.get("response", "")
                prediction = json.dumps({
                    "score": score,
                    "label": label,
                    "response": response_text
                })
                self.log_fn(f"Prediction: score={score}, label={label}")
            else:
                self.log_fn(f"WARNING: No JSON extracted from response. First 200 chars: {response_text[:200]}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
