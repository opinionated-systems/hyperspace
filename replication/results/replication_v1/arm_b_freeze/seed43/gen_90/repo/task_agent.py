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
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the inner content
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            # 1. Remove trailing commas before closing braces
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # 2. Handle escaped newlines that might break parsing
            fixed = fixed.replace('\\n', '\n')
            # 3. Try parsing again
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # 4. Try to extract just the first valid JSON object
                try:
                    # Find the first complete JSON object
                    brace_count = 0
                    json_start = -1
                    for i, char in enumerate(fixed):
                        if char == '{':
                            if brace_count == 0:
                                json_start = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and json_start != -1:
                                try:
                                    obj = json.loads(fixed[json_start:i+1])
                                    results.append(obj)
                                    break
                                except json.JSONDecodeError:
                                    continue
                except Exception:
                    continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Handles nested braces, escaped characters, and common formatting issues.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
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
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        # Try to fix common issues
                        try:
                            # Remove trailing commas
                            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            obj = json.loads(fixed)
                            if isinstance(obj, dict):
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

Respond ONLY in JSON format with the following schema. Do not include any text outside the JSON tags:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

Important guidelines:
- Be objective and consistent in your grading
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- Ensure your response is valid JSON with no trailing commas
- The response field should contain a concise grade/prediction, not the full reasoning"""

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
                
                # Priority order for grade fields
                grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                               "prediction", "score", "verdict", "assessment"]
                
                for field in grade_fields:
                    if field in last_json:
                        value = last_json[field]
                        # Handle both string and numeric values
                        if isinstance(value, (str, int, float, bool)):
                            prediction = str(value)
                            break
                        elif isinstance(value, list) and len(value) > 0:
                            # Sometimes grades come as single-item lists
                            prediction = str(value[0])
                            break
                else:
                    # If no known field, use the first string or numeric value found
                    for key, value in last_json.items():
                        if isinstance(value, (str, int, float, bool)):
                            # Skip the reasoning field if it's too long (likely the explanation)
                            if key == "reasoning" and len(str(value)) > 100:
                                continue
                            prediction = str(value)
                            break
                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (str, int, float)):
                            prediction = str(value[0])
                            break
            
            # Clean up the prediction
            prediction = prediction.strip()
            if prediction.lower() in ["none", "null", "undefined", ""]:
                prediction = "None"
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
