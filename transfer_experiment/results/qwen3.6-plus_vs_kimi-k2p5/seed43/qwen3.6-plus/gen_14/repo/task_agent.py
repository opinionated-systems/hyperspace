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

VERSION = "1.4.0"


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
    # Look for outermost braces
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
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    
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
        
        instruction = f"""You are an expert grader for {domain} olympiad problems.

Your task is to carefully evaluate a student's answer and assign a grade category.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

EVALUATION PROCESS:
1. First, understand what the problem is asking and identify the key steps needed for a complete solution.
2. Read the grading guidelines carefully to understand the point breakdown and what earns credit at each stage.
3. Analyze the student's answer step by step:
   - Does the student correctly interpret the problem statement?
   - Are their mathematical definitions, assumptions, and setup correct?
   - Is each logical step valid and well-justified?
   - Do they reach the correct conclusion with proper reasoning?
   - Estimate how many points they would earn based on the grading guidelines.
4. Compare the student's work against the official solution, noting both similarities and differences.
5. Identify any errors: conceptual misunderstandings, computational mistakes, missing cases, or logical gaps.

GRADING CRITERIA:
- "Correct": The student's answer is fully correct with sound reasoning. They demonstrate complete understanding, follow a valid approach, and reach the correct conclusion. They would earn full or near-full points (e.g., 6/7 or 7/7). Minor notational issues or trivial arithmetic slips that don't affect the core reasoning are acceptable.
- "Partial": The student's answer shows genuine understanding and makes meaningful progress toward the solution, but has notable gaps, errors, or is incomplete. They would earn some points but not full credit (e.g., 2/7 to 5/7). This includes: correct approach with computational errors, correct setup but incomplete execution, correct answer with flawed or missing justification, or addressing only part of a multi-part problem.
- "Incorrect": The student's answer is fundamentally flawed. This includes: misunderstanding the problem, using an invalid approach, major logical errors, or earning zero to minimal points (e.g., 0/7 or 1/7). Simply stating an answer without any valid reasoning also falls here.

IMPORTANT RULES:
- Be generous with "Partial" — if the student shows any meaningful mathematical insight or correct partial work, classify as "Partial" rather than "Incorrect".
- If the student's final answer is correct but their reasoning has only minor gaps, classify as "Correct".
- If the student's approach is correct but they made a computational error or didn't fully complete the solution, classify as "Partial".
- Only classify as "Incorrect" if the student's approach is fundamentally wrong or they show no understanding of the problem.
- CRITICAL: Do NOT default to "Correct". Many student answers contain subtle errors, incomplete reasoning, or incorrect conclusions. Carefully verify each step of the student's work against the official solution before assigning "Correct".
- When in doubt between "Correct" and "Partial", prefer "Partial" if there are any gaps in reasoning or missing steps.
- When in doubt between "Partial" and "Incorrect", prefer "Partial" if the student demonstrates any valid mathematical insight.
- Pay special attention to whether the student addresses ALL parts of the problem, not just some.

Provide your step-by-step analysis first, then give your final evaluation as a JSON object with your evaluation in the "response" field. The response must be exactly one of: "Correct", "Incorrect", or "Partial".

IMPORTANT: Your response MUST be wrapped in <json> tags like this:
<json>
{{
    "response": "Correct"
}}
</json>

Provide your reasoning first, then the JSON response at the end."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            extracted = _extract_json_flexible(last_message)
            if extracted and "response" in extracted:
                prediction = extracted["response"]
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"No 'response' field found in extracted JSON: {extracted}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
