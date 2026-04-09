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
    Also handles markdown code blocks with json tag.
    Includes robust JSON repair for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        parsed = _parse_json_with_repair(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Also try markdown code blocks ```json ... ```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start = start + 7  # Skip past ```json
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            parsed = _parse_json_with_repair(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _parse_json_with_repair(text: str) -> dict | None:
    """Parse JSON with multiple repair strategies for common LLM issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    """
    text = text.strip()
    if not text:
        return None
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 1: Remove trailing commas before } or ]
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 2: Replace single quotes with double quotes (carefully)
    try:
        # Only replace single quotes that appear to be string delimiters
        # This is a heuristic: single quotes followed by colon or comma or } are likely key delimiters
        fixed = re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*)'\s*(?=[,}])", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 3: Escape unescaped newlines in strings
    try:
        # Find strings and escape newlines within them
        fixed = re.sub(r'(?<!\\)\n', r'\\n', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 4: Combine strategies
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)  # trailing commas
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)  # newlines
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required to solve this problem. Note any critical steps that must be present in a complete solution.

2. OFFICIAL SOLUTION REVIEW: Understand the expected reasoning path, key insights, and any alternative valid approaches mentioned in the official solution.

3. STUDENT ANSWER EVALUATION:
   - Check if the student's final answer is mathematically correct
   - Verify the logical flow of their reasoning
   - Identify any gaps, errors, or misconceptions
   - Note any creative or alternative valid approaches
   - Check for completeness: did they prove all necessary steps?

4. GRADING CRITERIA APPLICATION:
   - Apply the specific grading guidelines provided above
   - Look for partial credit criteria: partial progress, correct methods with calculation errors, etc.
   - Consider the rigor of mathematical proof required
   - Be fair but consistent with IMO standards

5. FINAL GRADE DETERMINATION:
   - Synthesize your analysis into a clear grade
   - Ensure the grade matches the format specified in the guidelines
   - Provide specific justification referencing the student's work

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Be thorough and specific about what the student did right or wrong. Reference specific parts of their answer.",
    "response": "The final grade/prediction. Use the exact format specified in the grading guidelines (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)."
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "verdict" in last_json:
                    prediction = last_json["verdict"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
