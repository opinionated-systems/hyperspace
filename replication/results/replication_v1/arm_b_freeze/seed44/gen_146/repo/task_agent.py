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
    Also handles markdown code blocks as fallback.
    Includes robust nested brace handling for complex JSON structures.
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
            # Try to extract valid JSON by finding matching braces
            try:
                fixed = _extract_valid_json(inner)
                if fixed:
                    results.append(fixed)
            except Exception:
                continue
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try nested brace extraction
                try:
                    fixed = _extract_valid_json(match.strip())
                    if fixed:
                        results.append(fixed)
                except Exception:
                    continue
    
    # Second fallback: try to find any JSON object in the text with nested brace support
    if not results:
        # Use recursive pattern to handle nested braces
        matches = _find_json_objects(text)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _find_json_objects(text: str) -> list[str]:
    """Find JSON objects in text by tracking brace depth."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Found potential start of JSON object
            start = i
            depth = 1
            in_string = False
            escape_next = False
            i += 1
            
            while i < len(text) and depth > 0:
                char = text[i]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                i += 1
            
            if depth == 0:
                # Found a complete JSON object
                json_str = text[start:i]
                # Verify it has at least one key-value pair
                if '"' in json_str or "'" in json_str:
                    results.append(json_str)
        else:
            i += 1
    
    return results


def _extract_valid_json(text: str) -> dict | None:
    """Attempt to extract valid JSON by finding matching braces."""
    # Try to find the outermost valid JSON object
    start = text.find('{')
    if start == -1:
        return None
    
    # Track brace depth to find matching closing brace
    depth = 1
    in_string = False
    escape_next = False
    i = start + 1
    
    while i < len(text) and depth > 0:
        char = text[i]
        if escape_next:
            escape_next = False
        elif char == '\\':
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
        i += 1
    
    if depth == 0:
        json_str = text[start:i]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


def _normalize_score(score: str) -> str:
    """Normalize score to a standard format.
    
    Handles various score formats and converts them to a clean string.
    Supports IMO-style scores (0-7) and handles edge cases.
    """
    if score is None:
        return "None"
    
    score_str = str(score).strip()
    
    # Handle empty string
    if not score_str:
        return "None"
    
    # Remove any explanatory text after the score
    # e.g., "7 (full marks)" -> "7", "7 - excellent" -> "7"
    score_str = re.split(r'\s+\(|\s+-|\s*:|\s+\[', score_str)[0].strip()
    
    # Handle fraction formats like "7/7" -> "7"
    if '/' in score_str:
        parts = score_str.split('/')
        if parts[0].strip().isdigit():
            return parts[0].strip()
    
    # Handle decimal formats like "7.0" -> "7"
    if '.' in score_str:
        try:
            val = float(score_str)
            if val == int(val):
                return str(int(val))
        except ValueError:
            pass
    
    # Extract just the numeric part if there's extra text
    numeric_match = re.search(r'^-?\d+', score_str)
    if numeric_match:
        result = numeric_match.group()
        # Ensure IMO scores are non-negative and reasonable
        try:
            val = int(result)
            if val < 0:
                return "0"
            if val > 7:
                return "7"
            return result
        except ValueError:
            return result
    
    # Check for common non-numeric responses
    lower = score_str.lower()
    if any(word in lower for word in ['none', 'null', 'undefined', 'nan', 'n/a', 'na']):
        return "None"
    
    return score_str


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Validate required inputs
        if not problem:
            self.log_fn("Warning: Empty problem statement")
        if not student_answer:
            self.log_fn("Warning: Empty student answer")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate score based on the official solution and grading guidelines.

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

1. **Analyze the student's answer**: Compare it against the official solution and grading guidelines.
2. **Identify key steps**: Check which steps from the official solution are present in the student's answer.
3. **Evaluate correctness**: Determine if the student's reasoning is valid and complete.
4. **Assign a score**: Based on the grading guidelines, assign the appropriate score.

## Scoring Guidelines

- Read the grading guidelines carefully - they specify exact point allocations
- Award points only for correct mathematical reasoning that leads toward the solution
- Partial credit is given for significant progress even if the final answer is incorrect
- IMO problems are typically scored 0-7 points
- No credit is given for:
  * Incorrect statements or calculations
  * Vague or incomplete arguments
  * Answers that don't address the problem
  * Plagiarized or irrelevant content

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer, comparing it to the official solution and explaining your evaluation. Be specific about what the student got right and wrong.",
    "response": "The final score as a single number (e.g., '0', '1', '2', '7')"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the numeric score, with no additional text or explanation.

Be thorough in your reasoning and precise in your scoring."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "None", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("Warning: Empty message history from LLM")
                return "None", msg_history
            
            last_msg = msg_history[-1]
            if not isinstance(last_msg, dict) or "text" not in last_msg:
                self.log_fn(f"Warning: Invalid message format: {last_msg}")
                return "None", msg_history
            
            extracted = _extract_jsons(last_msg["text"])
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                if not isinstance(last_extract, dict):
                    self.log_fn(f"Warning: Extracted non-dict JSON: {last_extract}")
                    return "None", msg_history
                
                if "response" in last_extract:
                    prediction = last_extract["response"]
                elif "score" in last_extract:
                    prediction = last_extract["score"]
                elif "answer" in last_extract:
                    prediction = last_extract["answer"]
                else:
                    # If no recognized field, use the first string value found
                    for key, value in last_extract.items():
                        if isinstance(value, str) and value.strip():
                            prediction = value
                            break
                    else:
                        prediction = "None"
                
                # Normalize the score
                prediction = _normalize_score(prediction)
                self.log_fn(f"Extracted score: {prediction}")
            else:
                self.log_fn("Warning: No JSON found in LLM response")
                # Try to extract a number directly from the text as last resort
                numbers = re.findall(r'\b([0-7])\b', last_msg["text"])
                if numbers:
                    prediction = numbers[-1]  # Use last number found (likely the score)
                    self.log_fn(f"Fallback extraction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
