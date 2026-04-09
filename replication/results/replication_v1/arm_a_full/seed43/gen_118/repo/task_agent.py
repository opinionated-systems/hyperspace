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
    Also handles nested JSON objects within the tags.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from the inner content if it's not pure JSON
            # This handles cases where there's extra text before/after the JSON
            try:
                # Find the first '{' and last '}'
                json_start = inner.find('{')
                json_end = inner.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except (json.JSONDecodeError, ValueError):
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    """
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'\{\s*"reasoning".*?\}',  # Raw JSON with reasoning field
        r'\{\s*"response".*?\}',   # Raw JSON with response field
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Handles various edge cases and normalizes common variations.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    
    # Handle common variations of "None"
    if prediction.lower() in ["none", "null", "n/a", "na", "-"]:
        return "None"
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring
        # First try to find a standalone digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Also check for patterns like "score: 7" or "grade: 5"
        match = re.search(r'(?:score|grade)[\s:]*([0-7])\b', prediction.lower())
        if match:
            return match.group(1)
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        pred_lower = prediction.lower()
        # Handle "Correct" variations
        if any(term in pred_lower for term in ["correct", "right", "true", "valid", "yes"]):
            if not any(term in pred_lower for term in ["incorrect", "wrong", "false", "invalid", "not correct"]):
                return "Correct"
        # Handle "Incorrect" variations
        if any(term in pred_lower for term in ["incorrect", "wrong", "false", "invalid", "no", "not correct", "not valid"]):
            return "Incorrect"
    
    # Handle numeric ranges (e.g., "0-10", "1-5")
    range_match = re.search(r'\b(\d+)\s*[-–—]\s*(\d+)\b', grading_guidelines)
    if range_match:
        min_score, max_score = int(range_match.group(1)), int(range_match.group(2))
        # Try to extract a number within the range
        num_match = re.search(r'\b(\d+)\b', prediction)
        if num_match:
            score = int(num_match.group(1))
            if min_score <= score <= max_score:
                return str(score)
    
    # Handle letter grades (A, B, C, D, F)
    if re.search(r'\b[ABCDF][+-]?', grading_guidelines, re.IGNORECASE):
        # Match letter grades with optional +/- (word boundary only at start since +/- aren't word chars)
        letter_match = re.search(r'\b([ABCDF][+-]?)', prediction, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
    
    return prediction


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

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

## Domain
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

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and required techniques
- Note any critical assumptions or constraints

### Step 2: Solution Mapping
- Break down the official solution into key milestones
- Identify which milestones are essential vs. optional
- Note common alternative valid approaches

### Step 3: Student Answer Evaluation
- Map the student's work to solution milestones
- Identify correct steps with clear justification
- Flag any errors, gaps, or logical flaws
- Check for partial credit opportunities per guidelines

### Step 4: Grade Determination
- Apply grading guidelines systematically
- Consider: correctness, completeness, and clarity
- Assign the most appropriate grade/score

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the grade/score (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines. Do not add explanations or extra text in this field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"]
        try:
            extracted = _extract_jsons(last_text)
            if extracted:
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Try fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    if "response" in fallback:
                        prediction = fallback["response"]
                    elif "grade" in fallback:
                        prediction = fallback["grade"]
                    elif "score" in fallback:
                        prediction = fallback["score"]
                    elif "answer" in fallback:
                        prediction = fallback["answer"]
                    else:
                        prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
            
            # Validate and normalize the prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
