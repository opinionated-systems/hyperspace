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


def _extract_json_from_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON objects from ```json...``` code blocks."""
    results = []
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_raw(text: str) -> list[dict] | None:
    """Extract raw JSON objects from text (objects wrapped in {})."""
    results = []
    # Find all JSON-like structures
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results or None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects
    4. Direct label extraction from text
    
    Returns the "response" field value, or None if extraction fails.
    """
    # Valid prediction labels
    VALID_LABELS = {"Correct", "Incorrect", "Partial", "Almost"}
    
    # Try <json> tags first
    extracted = _extract_jsons(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                # Validate the response is one of the expected labels
                if response in VALID_LABELS:
                    return response
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Try raw JSON
    extracted = _extract_json_raw(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Fallback: direct label extraction from text
    # Look for valid labels in the text (case-insensitive, but return proper case)
    text_upper = text.upper()
    for label in VALID_LABELS:
        if label.upper() in text_upper:
            # Verify it's a standalone word, not part of another word
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return label
    
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
        # Extract fields from inputs for better prompt construction
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical problem solver and grader for IMO (International Mathematical Olympiad) problems.

Your task is to grade a student's answer to a mathematical problem based on the provided solution and grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Criteria - READ CAREFULLY:

**"Correct"** - Use ONLY when ALL of these are true:
- The student provides a complete, rigorous proof/solution from start to finish
- All key steps are present and logically sound with no gaps
- The answer matches the official solution's conclusion with proper justification
- No significant errors, missing cases, or logical gaps
- The proof would receive full marks in an IMO competition

**"Incorrect"** - Use when ANY of these are true:
- The student's answer is fundamentally wrong or the approach is completely misguided
- The solution contains critical logical flaws that invalidate the entire argument
- The answer does not address the problem requirements or makes no meaningful progress
- The student misunderstood the problem statement

**"Partial"** - Use when ANY of these are true:
- The student found a correct key insight, invariant, or strategy but didn't complete the proof
- Significant progress was made (e.g., proved one direction, found the right approach) but major gaps remain
- The approach is correct but missing crucial steps, cases, or the conclusion
- The student identified the right strategy but failed to execute it fully or rigorously
- The solution has the main idea but lacks formal proof or verification

**"Almost"** - Use when ALL of these are true:
- The solution is nearly complete and would be "Correct" except for minor technical errors
- The main proof structure is valid but there's a small gap in reasoning
- The proof is valid except for a minor case, edge condition, or small calculation error
- The student demonstrated deep understanding but made one small mistake that doesn't invalidate the main argument
- The error is fixable with a minor adjustment

## Your Task - STEP BY STEP ANALYSIS REQUIRED:

Step 1: Analyze what the problem asks for and what the official solution provides.

Step 2: Analyze the student's answer:
- Did they understand the problem correctly?
- What key insights did they identify?
- Are there logical gaps or errors?
- How complete is the solution?

Step 3: Check against grading guidelines (if provided):
- What specific criteria are mentioned?
- Does the student's answer meet these criteria?

Step 4: Make your classification decision:
- If the solution is complete and rigorous → "Correct"
- If the solution is nearly complete with only minor fixable errors → "Almost"
- If significant progress was made but major gaps remain → "Partial"
- If the approach is wrong or no meaningful progress → "Incorrect"

## IMPORTANT DECISION RULES:
1. Be CONSERVATIVE with "Correct" - if you have ANY doubt, choose "Partial" or "Almost"
2. "Partial" means significant progress but INCOMPLETE - the student didn't finish the proof
3. "Almost" means nearly complete - the student essentially solved it but made a small error
4. If the student found the right invariant/approach but didn't prove sufficiency → "Partial"
5. If the proof structure is complete but has a minor gap → "Almost"
6. If the main idea is wrong or no progress → "Incorrect"

Respond in this exact JSON format:
<json>
{{
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "response": "Incorrect"
}}
</json>

OR

<json>
{{
    "response": "Partial"
}}
</json>

OR

<json>
{{
    "response": "Almost"
}}
</json>

CRITICAL: Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost". Do not add any explanation in the response field - only the label. Include the <json> and </json> tags around your JSON response."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            extracted_prediction = extract_prediction(last_message)
            if extracted_prediction is not None:
                prediction = extracted_prediction
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"Failed to extract prediction from response: {last_message[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
