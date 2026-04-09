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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback for malformed responses.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces using brace counting
    if not results:
        # Find all potential JSON starting points
        for match in re.finditer(r'\{', text):
            start_idx = match.start()
            # Count braces to find complete JSON object
            brace_count = 0
            for i, char in enumerate(text[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start_idx:start_idx+i+1]
                        try:
                            parsed = json.loads(candidate)
                            # Only include if it has expected fields
                            if any(key in parsed for key in ["response", "reasoning", "answer", "result", "grade"]):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
    
    return results or None


def _normalize_grade(prediction: str) -> str:
    """Normalize grade prediction to standard format.
    
    Converts various grade formats to a consistent representation.
    """
    if not prediction:
        return "None"
    
    pred_lower = prediction.lower().strip()
    
    # Map common variations to standard grades
    correct_variations = ['correct', 'right', 'true', 'yes', '1', 'full', 'full credit', '100%', 'pass']
    incorrect_variations = ['incorrect', 'wrong', 'false', 'no', '0', 'none', 'fail', 'zero']
    partial_variations = ['partial', 'partially', 'partially correct', 'half', '0.5', '50%', 'some']
    
    for var in correct_variations:
        if var in pred_lower:
            return "Correct"
    
    for var in incorrect_variations:
        if var in pred_lower:
            return "Incorrect"
    
    for var in partial_variations:
        if var in pred_lower:
            return "Partially Correct"
    
    # Check for numeric scores
    numeric_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', prediction)
    if numeric_match:
        score = float(numeric_match.group(1))
        total = float(numeric_match.group(2))
        if total > 0:
            ratio = score / total
            if ratio >= 0.9:
                return "Correct"
            elif ratio >= 0.5:
                return "Partially Correct"
            else:
                return "Incorrect"
    
    # Check for percentage
    percent_match = re.search(r'(\d+(?:\.\d+)?)%', prediction)
    if percent_match:
        percent = float(percent_match.group(1))
        if percent >= 90:
            return "Correct"
        elif percent >= 50:
            return "Partially Correct"
        else:
            return "Incorrect"
    
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
        # Build a structured prompt with chain-of-thought reasoning
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the official solution.
2. Identify what the student got correct and what they got wrong.
3. Consider the grading guidelines carefully - these define how partial credit should be awarded.
4. Look for key insights, correct methods, and valid reasoning even if the final answer differs.
5. Provide your reasoning for the grade you will assign.
6. Finally, provide your grade/assessment in the response field using one of these standard formats:
   - "Correct" or "Full Credit" - for complete, correct answers
   - "Partially Correct" or "Partial Credit" - for answers with some correct elements
   - "Incorrect" or "No Credit" - for completely wrong answers
   - Or a numeric score like "7/7" or "3/7" if specified in guidelines

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning for the grade",
    "response": "Your final grade/assessment"
}}
</json>

Ensure your JSON is valid and properly formatted."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try standard extraction first
            extracted = _extract_jsons(last_message)
            
            # If that fails, try fuzzy extraction
            if not extracted:
                extracted = _extract_json_fuzzy(last_message)
            
            if extracted:
                # Prefer response field, but fallback to other common fields
                last_extracted = extracted[-1]
                if "response" in last_extracted:
                    prediction = last_extracted["response"]
                elif "answer" in last_extracted:
                    prediction = last_extracted["answer"]
                elif "result" in last_extracted:
                    prediction = last_extracted["result"]
                elif "grade" in last_extracted:
                    prediction = last_extracted["grade"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_extracted)
                    
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn("No JSON found in response, using raw text")
                # Fallback: use the raw response text (truncated)
                prediction = last_message[:500] if len(last_message) > 500 else last_message
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize the grade to standard format
        normalized_prediction = _normalize_grade(prediction)
        if normalized_prediction != prediction:
            self.log_fn(f"Normalized '{prediction}' to '{normalized_prediction}'")
        
        return str(normalized_prediction), msg_history
