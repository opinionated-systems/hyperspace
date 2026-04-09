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
            # Try to fix common JSON issues before giving up
            try:
                # Fix trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (carefully)
                fixed = fixed.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try one more time with more aggressive fixes
                try:
                    # Remove comments
                    fixed = re.sub(r'//.*?\n', '\n', inner)
                    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                    # Fix unquoted keys
                    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
                    results.append(json.loads(fixed))
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
                        # Try more aggressive fixes
                        try:
                            # Remove comments
                            fixed = re.sub(r'//.*?\n', '\n', json_str)
                            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
                            # Fix unquoted keys
                            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
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


def _normalize_grade(prediction: str) -> str:
    """Normalize grade prediction to a standard format.
    
    Converts various grade formats to a consistent set of labels:
    - Correct/Yes/True/1/100/Pass -> 'Correct'
    - Incorrect/No/False/0/0.0/Fail -> 'Incorrect'
    - Partial/Half/Partially -> 'Partial'
    """
    if not prediction or not isinstance(prediction, str):
        return str(prediction) if prediction else "None"
    
    pred_lower = prediction.strip().lower()
    
    # Correct variations
    if any(x in pred_lower for x in ['correct', 'yes', 'true', 'right', 'pass', 'full', 'complete']):
        if 'partial' not in pred_lower and 'incomplete' not in pred_lower:
            return 'Correct'
    
    # Incorrect variations
    if any(x in pred_lower for x in ['incorrect', 'wrong', 'no', 'false', 'fail', 'error', 'invalid']):
        return 'Incorrect'
    
    # Partial variations
    if any(x in pred_lower for x in ['partial', 'half', 'incomplete', 'partially']):
        return 'Partial'
    
    # Numeric scores
    try:
        num = float(prediction)
        if num >= 0.9 or num >= 90:
            return 'Correct'
        elif num <= 0.1 or (num <= 10 and num < 5):
            return 'Incorrect'
        else:
            return 'Partial'
    except (ValueError, TypeError):
        pass
    
    # Return original if no normalization applied
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

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

IMPORTANT FORMATTING RULES:
- Use double quotes for all JSON strings, never single quotes
- Do not include trailing commas
- Ensure the JSON is valid and parseable
- The response field must contain a concise grade label
- Escape any double quotes within string values using backslash (\\")
- Do not include any text outside the <json> tags

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect.

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines
- Your JSON output follows the exact format specified above
- All special characters in your reasoning are properly escaped for JSON"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (from <json> tags)
            extracted = _extract_jsons(last_message)
            if extracted:
                extraction_method = "json_tags"
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
                if extracted:
                    extraction_method = "any_json"
            
            # Fallback 2: markdown code blocks
            if extracted is None:
                extracted = _extract_from_markdown_code_blocks(last_message)
                if extracted:
                    extraction_method = "markdown"
            
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
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and len(value) < 100:
                            prediction = value
                            break
                
                # Normalize the grade to standard format
                prediction = _normalize_grade(prediction)
                self.log_fn(f"Extracted prediction via {extraction_method}: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
