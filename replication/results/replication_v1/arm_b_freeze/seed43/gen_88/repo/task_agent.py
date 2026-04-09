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


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-counting approach to find complete JSON objects,
    handling nested braces correctly. Also attempts to fix common JSON
    formatting issues before parsing.
    """
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
                json_str = text[start_idx:i+1]
                try:
                    obj = json.loads(json_str)
                    results.append(obj)
                except json.JSONDecodeError:
                    # Try to fix common issues and retry
                    try:
                        # Fix trailing commas before closing braces
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        # Fix single quotes to double quotes
                        fixed = fixed.replace("'", '"')
                        obj = json.loads(fixed)
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                start_idx = -1
    return results or None


def _extract_from_markdown_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Handles cases where the model outputs JSON in markdown format.
    """
    results = []
    # Find markdown code blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        match = match.strip()
        if not match.startswith('{'):
            continue
        try:
            obj = json.loads(match)
            results.append(obj)
        except json.JSONDecodeError:
            # Try the fallback extraction on the content
            nested = _extract_any_json(match)
            if nested:
                results.extend(nested)
    
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

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis following this structured approach:

STEP 1 - PROBLEM ANALYSIS:
- Identify the key mathematical concepts, theorems, and techniques required
- Determine the problem type (algebra, geometry, number theory, combinatorics)
- Note any special conditions or constraints

STEP 2 - SOLUTION REVIEW:
- Analyze the official solution's approach and key steps
- Identify the critical insights needed for a correct solution
- Note the expected answer format and presentation standards

STEP 3 - STUDENT WORK ANALYSIS:
- Identify the approach the student took (may differ from official solution)
- Note ALL correct steps, valid insights, and creative approaches
- Identify errors, gaps, or misconceptions with specific line references
- Distinguish between computational errors and conceptual misunderstandings

STEP 4 - GRADING CRITERIA CHECK:
- Systematically verify each criterion in the grading guidelines
- Award partial credit for:
  * Correct setup but wrong final answer
  * Valid alternative approaches
  * Correct intermediate results
  * Proper methodology with minor errors
- Document evidence for each criterion met or not met

STEP 5 - FINAL DETERMINATION:
- Assign grade based on: completeness (30%), correctness (40%), methodology (20%), presentation (10%)
- Provide specific justification referencing the grading guidelines
- Note if the grade differs from initial impression and why

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be specific about what the student did right and wrong.",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

Important Grading Principles:
1. Be objective, consistent, and fair
2. Award partial credit generously for valid reasoning even with incorrect final answers
3. Value creative alternative approaches that are mathematically sound
4. Distinguish between minor computational errors and major conceptual gaps
5. Consider the IMO's emphasis on proof structure and logical rigor

SELF-CORRECTION CHECKLIST (verify before finalizing):
☐ Have I identified all correct elements in the student's work?
☐ Did I check for partial credit opportunities at each step?
☐ Is my grade consistent with similar cases in the guidelines?
☐ Have I considered alternative valid approaches?
☐ Would another expert grader reach the same conclusion?
☐ Is my reasoning transparent and defensible?"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (from <json> tags)
            extracted = _extract_jsons(last_message)
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            # Fallback 2: markdown code blocks
            if extracted is None:
                extracted = _extract_from_markdown_code_blocks(last_message)
            
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
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
