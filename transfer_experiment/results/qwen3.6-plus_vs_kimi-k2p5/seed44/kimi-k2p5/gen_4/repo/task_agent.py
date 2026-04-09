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

# Compile regex patterns for grade extraction
_GRADE_PATTERNS = {
    "correct": re.compile(r'\bcorrect\b', re.IGNORECASE),
    "almost": re.compile(r'\balmost\b', re.IGNORECASE),
    "partial": re.compile(r'\bpartial\b', re.IGNORECASE),
    "incorrect": re.compile(r'\bincorrect\b', re.IGNORECASE),
}

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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ``` or ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects by finding matching braces."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find matching closing brace
            count = 1
            j = i + 1
            while j < len(text) and count > 0:
                if text[j] == '{':
                    count += 1
                elif text[j] == '}':
                    count -= 1
                j += 1
            if count == 0:
                try:
                    results.append(json.loads(text[i:j]))
                except json.JSONDecodeError:
                    pass
            i = j
        else:
            i += 1
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the expected values."""
    prediction_str = str(prediction).strip().lower()
    
    # Check for exact matches first (including "almost")
    if prediction_str in ["correct", "incorrect", "partial", "almost"]:
        return prediction_str.capitalize()
    
    # Check for "almost" first - indicates minor mistakes only
    if "almost" in prediction_str:
        return "Almost"
    
    # Check for partial first (to avoid matching "partially correct" as "correct")
    if "partial" in prediction_str:
        return "Partial"
    
    # Check for incorrect/wrong/false
    if any(word in prediction_str for word in ["incorrect", "wrong", "false", "error", "mistake", "flawed", "invalid"]):
        return "Incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in prediction_str and "incorrect" not in prediction_str:
        return "Correct"
    
    # Check for other positive indicators
    if any(word in prediction_str for word in ["right", "true", "valid", "accurate", "complete", "full"]):
        return "Correct"
    
    # Check for negative indicators suggesting fundamental issues
    if any(word in prediction_str for word in ["incomplete", "missing", "lacking", "insufficient"]):
        return "Partial"
    
    # Additional checks for common variations
    # Check for "mostly correct" or "nearly correct" -> Almost
    if any(phrase in prediction_str for phrase in ["mostly correct", "nearly correct", "minor error", "small error", "tiny mistake"]):
        return "Almost"
    
    # Check for "some correct" or "partially right" -> Partial
    if any(phrase in prediction_str for phrase in ["some correct", "partially right", "partly correct", "some understanding", "on the right track"]):
        return "Partial"
    
    # Check for "completely wrong" or "totally incorrect" -> Incorrect
    if any(phrase in prediction_str for phrase in ["completely wrong", "totally incorrect", "fundamentally wrong", "entirely incorrect"]):
        return "Incorrect"
    
    return "None"


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
        # Extract fields from inputs for better prompt construction
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematical problem and assign a grade based on the provided grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric:
Carefully analyze the student's answer against the official solution and grading guidelines. Assign ONE of these grades:

- "Correct" - The answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid.

- "Almost" - The answer is essentially correct with only minor mistakes (e.g., small calculation errors, typos, or notation issues). The core reasoning and approach are sound.

- "Partial" - The answer has some correct elements and shows understanding of key concepts, but is incomplete, missing critical steps, or has significant errors that don't invalidate the entire approach.

- "Incorrect" - The answer is fundamentally wrong, uses incorrect methods, or shows a fundamental misunderstanding of the problem.

## Examples:

Example 1 - Correct answer:
<json>
{{"response": "Correct"}}
</json>

Example 2 - Answer with minor errors:
<json>
{{"response": "Almost"}}
</json>

Example 3 - Partial progress:
<json>
{{"response": "Partial"}}
</json>

Example 4 - Fundamentally wrong:
<json>
{{"response": "Incorrect"}}
</json>

## Your Task:
1. First, identify what the problem is asking and what the official solution provides.
2. Check if the student's answer matches the official solution's approach and conclusion.
3. Look for any grading guidelines that specifically apply to this answer.
4. Determine the appropriate grade based on the rubric above.

IMPORTANT: You must output ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

You MUST respond with a JSON object in EXACTLY this format:
<json>
{{
    "response": "Correct" | "Almost" | "Partial" | "Incorrect"
}}
</json>

The value for "response" must be exactly one of: "Correct", "Almost", "Partial", or "Incorrect" (case-sensitive, with first letter capitalized). Choose the grade that best matches the quality of the student's answer."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple methods
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                text = msg_history[-1].get("text", "")
                
                # Try multiple extraction methods
                extracted = None
                
                # Method 1: <json> tags (most reliable)
                extracted = _extract_jsons(text)
                
                # Method 2: Markdown code blocks
                if not extracted:
                    extracted = _extract_json_from_markdown(text)
                
                # Method 3: Raw JSON with braces
                if not extracted:
                    extracted = _extract_json_braces(text)
                
                if extracted:
                    last_extract = extracted[-1]
                    
                    # Try multiple field names in order of preference
                    pred_value = None
                    field_used = None
                    for field in ["response", "grade", "evaluation", "result", "answer", "verdict", "prediction"]:
                        if field in last_extract:
                            pred_value = last_extract[field]
                            field_used = field
                            break
                    
                    if pred_value is not None:
                        prediction = _normalize_prediction(str(pred_value))
                        self.log_fn(f"Extracted prediction: {prediction} from field: {field_used}")
                    else:
                        # If no recognized field, try to use the whole response if it's a simple string
                        if isinstance(last_extract, str):
                            prediction = _normalize_prediction(last_extract)
                            self.log_fn(f"Extracted prediction from string JSON: {prediction}")
                        else:
                            # Log available keys for debugging
                            self.log_fn(f"No recognized field in JSON. Available keys: {list(last_extract.keys())}")
                            # Try to find any value that looks like a valid grade
                            for key, value in last_extract.items():
                                if isinstance(value, str):
                                    normalized = _normalize_prediction(value)
                                    if normalized != "None":
                                        prediction = normalized
                                        self.log_fn(f"Found valid grade in field '{key}': {prediction}")
                                        break
                else:
                    # No JSON found, try to extract from raw text by looking for grade keywords
                    prediction = _normalize_prediction(text)
                    self.log_fn(f"No JSON found, normalized from text: {prediction}")
                    
                    # If still None, try to find exact grade words in the text
                    if prediction == "None":
                        for grade_name, pattern in _GRADE_PATTERNS.items():
                            if pattern.search(text):
                                prediction = grade_name.capitalize()
                                self.log_fn(f"Found grade keyword in text: {prediction}")
                                break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        return str(prediction), msg_history
