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

# Valid prediction labels
VALID_LABELS = {"Correct", "Incorrect", "Partial", "Almost"}


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
    # Find all JSON-like structures - improved pattern for nested braces
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_direct_label(text: str) -> str | None:
    """Extract label directly from text by looking for valid labels."""
    # Clean up the text - remove punctuation that might interfere
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}]', ' ', text)
    words = cleaned_text.split()
    
    # Check each word (case-insensitive)
    for word in words:
        word_stripped = word.strip('.,;:!?')
        for label in VALID_LABELS:
            if word_stripped.lower() == label.lower():
                return label
    
    # Also check for labels as substrings in lines
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        for label in VALID_LABELS:
            # Look for label as a whole word
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                return label
    
    return None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects
    4. Direct label extraction from text
    
    Returns the "response" field value, or None if extraction fails.
    """
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
    return _extract_direct_label(text)


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

**"Partial"** - Use when the student made SIGNIFICANT PROGRESS but the solution is INCOMPLETE:
- The student found a correct key insight, invariant, or strategy but didn't complete the proof
- Significant progress was made (e.g., proved one direction, found the right approach) but major gaps remain
- The approach is correct but missing crucial steps, cases, or the conclusion
- The student identified the right strategy but failed to execute it fully or rigorously
- The solution has the main idea but lacks formal proof or verification
- KEY DISTINCTION: The student is missing substantial parts of the solution

**"Almost"** - Use when the solution is NEARLY COMPLETE with only MINOR issues:
- The solution is nearly complete and would be "Correct" except for minor technical errors
- The main proof structure is valid but there's a small gap in reasoning
- The proof is valid except for a minor case, edge condition, or small calculation error
- The student demonstrated deep understanding but made one small mistake that doesn't invalidate the main argument
- The error is fixable with a minor adjustment
- KEY DISTINCTION: The student essentially solved the problem - the structure is there, just needs a small fix

## CRITICAL DISTINCTION: "Almost" vs "Partial"

**Use "Almost" when:**
- The student has 80-95% of a complete solution
- The main proof structure is intact
- Only minor technical details are missing or wrong
- Example: Correct approach, correct main lemma, but a small calculation error or missing one trivial case

**Use "Partial" when:**
- The student has 30-70% of a complete solution  
- Major parts of the proof are missing
- The main idea is there but significant work remains
- Example: Found the right invariant but couldn't prove the key property; proved one direction of an iff but not the other

## Your Task - STEP BY STEP ANALYSIS REQUIRED:

Step 1: Analyze what the problem asks for and what the official solution provides.

Step 2: Analyze the student's answer:
- Did they understand the problem correctly?
- What key insights did they identify?
- Are there logical gaps or errors?
- How complete is the solution? (Estimate percentage: 0%, 25%, 50%, 75%, 90%, 100%)

Step 3: Check against grading guidelines (if provided):
- What specific criteria are mentioned?
- Does the student's answer meet these criteria?

Step 4: Make your classification decision using this decision tree:
- Is the solution complete and rigorous with no gaps? → "Correct"
- Is the solution 80-95% complete with only minor fixable errors? → "Almost"
- Is the solution 30-70% complete with major gaps but correct approach? → "Partial"
- Is the approach wrong or is there no meaningful progress? → "Incorrect"

## IMPORTANT DECISION RULES:
1. Be CONSERVATIVE with "Correct" - only use if you're confident it's 100% complete
2. "Almost" means the student ESSENTIALLY SOLVED IT - just needs a small fix (80-95% complete)
3. "Partial" means SIGNIFICANT PROGRESS but INCOMPLETE - major work remains (30-70% complete)
4. If the student found the right invariant/approach but didn't prove sufficiency → "Partial" (not Almost)
5. If the proof structure is complete but has a minor gap → "Almost" (not Partial)
6. If the main idea is wrong or no progress → "Incorrect"
7. When in doubt between "Almost" and "Partial", ask: "Could this solution be fixed in 5 minutes by an expert?" If yes → "Almost", if no → "Partial"

## RESPONSE FORMAT - CRITICAL:
You MUST respond with ONLY a JSON object in this exact format:
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

CRITICAL INSTRUCTIONS:
1. Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost"
2. The "response" field must contain ONLY the label - no extra text, no explanation
3. You MUST include the <json> and </json> tags around your JSON response
4. Do not include any text after the JSON block
5. The JSON must be valid - use double quotes around the label"""

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
                self.log_fn(f"Failed to extract prediction from response: {last_message[:500]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
