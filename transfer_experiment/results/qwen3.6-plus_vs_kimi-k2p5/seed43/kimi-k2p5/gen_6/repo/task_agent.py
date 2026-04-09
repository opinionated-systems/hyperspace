"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the IMPROVED task agent with better prompting and robust JSON extraction.
The meta agent modifies this file during self-improvement. The evaluation harness 
loads whatever task_agent.py exists at the agent's repo path.
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
    """Extract JSON using multiple fallback methods.
    
    Tries: <json> tags, ```json blocks, raw JSON objects.
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Try ```json code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try finding raw JSON objects with curly braces
    # Look for outermost braces - start from the end to get the last JSON object
    brace_count = 0
    start_idx = -1
    candidates = []
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidates.append((start_idx, i+1))
                start_idx = -1
    
    # Try candidates from the end (last JSON object is usually the answer)
    for start, end in reversed(candidates):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    
    # Try to find and fix common JSON issues
    # Look for patterns like {"response": "..."} or {"response": "..." 
    response_pattern = r'\{\s*"response"\s*:\s*"([^"]*)"\s*\}?'
    match = re.search(response_pattern, text, re.DOTALL)
    if match:
        response_val = match.group(1)
        return {"response": response_val}
    
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
        # Extract key fields for better prompting
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems, specifically trained to evaluate mathematical proofs and solutions.

Your task is to evaluate a student's answer to a problem and assign a grade based on the official solution and grading guidelines.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

EVALUATION INSTRUCTIONS:
1. Carefully compare the student's answer against the official solution
2. Check if the student has:
   - The correct final answer/result
   - Valid reasoning and proof structure
   - All necessary steps and justifications
   - No logical gaps or errors
3. Use these grade categories:
   - "Correct": The answer is completely correct with valid reasoning
   - "Almost": The answer is nearly correct with only minor errors or omissions that don't affect the main result
   - "Partial": The answer has significant correct elements but also major gaps, errors, or missing key steps
   - "Incorrect": The answer is fundamentally wrong or the reasoning is invalid

4. Be precise: "Almost" is for minor issues only; "Partial" is for substantial but incomplete progress

You must respond with a JSON object containing your evaluation in the "response" field.

IMPORTANT: Your response MUST be wrapped in <json> tags like this:
<json>
{{
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "response": "Almost"
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
    "response": "Incorrect"
}}
</json>

Provide ONLY the JSON response wrapped in <json> tags with exactly one of the four category names. Do not include any other text."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw LLM response: {last_message[:500]}...")
            extracted = _extract_json_flexible(last_message)
            self.log_fn(f"Extracted JSON: {extracted}")
            if extracted and "response" in extracted:
                prediction = extracted["response"]
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"No 'response' field found in extracted JSON: {extracted}")
                # Try to extract any of the grade categories directly from text
                text_upper = last_message.upper()
                if '"CORRECT"' in text_upper or 'CORRECT' in text_upper:
                    prediction = "Correct"
                elif '"ALMOST"' in text_upper or 'ALMOST' in text_upper:
                    prediction = "Almost"
                elif '"PARTIAL"' in text_upper or 'PARTIAL' in text_upper:
                    prediction = "Partial"
                elif '"INCORRECT"' in text_upper or 'INCORRECT' in text_upper:
                    prediction = "Incorrect"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
