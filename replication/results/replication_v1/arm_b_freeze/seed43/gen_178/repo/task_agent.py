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
                # Try one more aggressive fix: extract just the JSON object
                try:
                    # Find the first { and last }
                    json_start = inner.find('{')
                    json_end = inner.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        fixed = inner[json_start:json_end+1]
                        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                        fixed = fixed.replace("'", '"')
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
    """Normalize various grade formats to a standard set.
    
    Handles common variations like 'Correct'/'correct'/'CORRECT',
    numeric scores, and partial credit indicators.
    Uses a priority-based approach for consistent classification.
    """
    if not prediction or not isinstance(prediction, str):
        return "Incorrect"
    
    pred_lower = prediction.lower().strip()
    
    # Priority 1: Check for explicit incorrect indicators first
    # (must check before "correct" since "incorrect" contains "correct")
    incorrect_indicators = [
        'incorrect', 'wrong', 'false', 'fail', 'failed', 'error', 'invalid',
        'unsatisfactory', 'unacceptable', 'deficient', 'flawed', 'erroneous',
    ]
    for indicator in incorrect_indicators:
        if indicator in pred_lower:
            return "Incorrect"
    
    # Check for standalone "no" or "0" (with word boundaries)
    if re.search(r'\bno\b', pred_lower) or re.search(r'\b0\b', pred_lower) or pred_lower == 'zero':
        return "Incorrect"
    
    # Check for 'none' or empty as incorrect
    if pred_lower == 'none' or pred_lower == '':
        return "Incorrect"
    
    # Priority 2: Check for correct indicators
    correct_indicators = [
        'correct', 'right', 'true', 'yes', 'full credit', '100%', 'pass', 'passed',
        'satisfactory', 'acceptable', 'complete', 'valid', 'accurate', 'perfect',
        'excellent', 'good', 'success', 'successful', 'valid solution',
    ]
    for indicator in correct_indicators:
        if indicator in pred_lower:
            return "Correct"
    
    # Check for exact matches of 1 or 1.0
    if pred_lower in ('1', '1.0', '1.00'):
        return "Correct"
    
    # Priority 3: Check for partial credit indicators
    partial_indicators = [
        'partial', 'partial credit', 'half', 'incomplete', 'partially correct',
        'partially', 'some credit', 'minor errors', 'mostly correct', 'nearly',
        'almost', 'partial success', 'partially valid', 'partially right',
    ]
    for indicator in partial_indicators:
        if indicator in pred_lower:
            return "Partial"
    
    # Check for exact match of 0.5
    if pred_lower in ('0.5', '0.50', '1/2', 'half'):
        return "Partial"
    
    # Priority 4: Numeric score interpretation
    # Extract the first number that looks like a score (0-1 range or 0-100 range)
    numbers = re.findall(r'\b\d+\.?\d*\b', prediction)
    if numbers:
        try:
            num = float(numbers[0])
            # Handle percentage (0-100 scale)
            if num > 1.0 and num <= 100:
                num = num / 100.0
            # Handle score on 0-1 scale
            if 0 <= num <= 1:
                if num >= 0.75:
                    return "Correct"
                elif num >= 0.35:
                    return "Partial"
                else:
                    return "Incorrect"
        except ValueError:
            pass
    
    # Priority 5: Check for negative indicators that might indicate incorrect
    negative_indicators = ['not', 'missing', 'lacking', 'without', 'absent', 'failed to']
    neg_count = sum(1 for ind in negative_indicators if ind in pred_lower)
    if neg_count >= 2:
        return "Incorrect"
    
    # Default: if we can't determine, return the original for manual review
    # but log a warning that normalization failed
    logger.warning(f"Grade normalization uncertain for: '{prediction}' - returning as-is")
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

Think step by step and provide a thorough analysis following this structure:

STEP 1 - PROBLEM ANALYSIS:
Identify the key mathematical concepts, theorems, and techniques required to solve this problem. What makes this problem challenging?

STEP 2 - SOLUTION REVIEW:
Analyze the official solution's approach, key steps, and expected answer format. What are the critical elements that must be present in a correct solution?

STEP 3 - STUDENT WORK ANALYSIS:
- What approach did the student take?
- What correct steps or valid insights did they demonstrate?
- What errors, gaps, or misconceptions are present?
- Did they use appropriate mathematical notation and reasoning?

STEP 4 - GRADING CRITERIA CHECK:
- Does the student's answer meet each criterion in the grading guidelines?
- Is the final answer correct (if applicable)?
- Is the reasoning sound and complete?
- What partial credit should be awarded for incomplete but valid reasoning?

STEP 5 - FINAL DETERMINATION:
Assign a grade based on:
- Correctness of the final answer (if determinable)
- Completeness of the reasoning
- Adherence to the grading guidelines
- Mathematical rigor and clarity

GRADE CATEGORIES (use exactly one of these in your response field):
- "Correct": The answer is fully correct with complete reasoning
- "Incorrect": The answer is wrong, incomplete, or contains critical errors
- "Partial": The answer has valid elements but is incomplete or has minor errors

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be thorough and specific.",
    "response": "Correct" or "Incorrect" or "Partial"
}}
</json>

CRITICAL FORMATTING RULES:
1. Use ONLY double quotes for all JSON strings - NEVER use single quotes
2. Do NOT include trailing commas in JSON objects
3. Ensure the JSON is valid and parseable by standard JSON parsers
4. The response field MUST contain exactly one of: "Correct", "Incorrect", or "Partial"
5. Do not add any text outside the <json>...</json> tags

GRADING PRINCIPLES:
- Be objective, consistent, and fair
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- A correct final answer with no reasoning shown may warrant "Partial" depending on guidelines
- An incorrect final answer with sound partial reasoning may warrant "Partial"
- Mathematical notation and clarity matter for complete solutions

FINAL VERIFICATION: Before outputting your JSON, verify:
- All 5 analysis steps are covered in your reasoning
- Your grade choice aligns with the grading guidelines
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your JSON uses double quotes only and has no trailing commas
- The response field contains exactly Correct, Incorrect, or Partial"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "Incorrect"  # Default to Incorrect on failure
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
            
            # Fallback 3: Look for grade keywords directly in the text
            if extracted is None:
                text_lower = last_message.lower()
                if '"correct"' in text_lower or "'correct'" in text_lower or "correct" in text_lower.split():
                    prediction = "Correct"
                    extraction_method = "keyword_correct"
                elif '"incorrect"' in text_lower or "'incorrect'" in text_lower or "incorrect" in text_lower.split():
                    prediction = "Incorrect"
                    extraction_method = "keyword_incorrect"
                elif '"partial"' in text_lower or "'partial'" in text_lower or "partial" in text_lower.split():
                    prediction = "Partial"
                    extraction_method = "keyword_partial"
            
            if extracted and extraction_method != "keyword":
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
                
                self.log_fn(f"Extracted prediction via {extraction_method}: {prediction}")
                
                # Normalize the grade to standard format
                normalized = _normalize_grade(prediction)
                if normalized != prediction:
                    self.log_fn(f"Normalized grade: {prediction} -> {normalized}")
                    prediction = normalized
            elif extraction_method.startswith("keyword"):
                self.log_fn(f"Extracted prediction via {extraction_method}: {prediction}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "Incorrect"  # Default to Incorrect on any error

        return str(prediction), msg_history
