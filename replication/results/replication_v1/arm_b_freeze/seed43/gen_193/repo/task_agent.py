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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON objects within the tags and multiple JSON objects.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to find JSON object boundaries by brace counting
            # This handles cases where there might be extra text before/after the JSON
            # Also handles multiple JSON objects in the same block
            brace_count = 0
            json_start = -1
            found_in_block = []
            for i, char in enumerate(inner):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        try:
                            obj = json.loads(inner[json_start:i+1])
                            found_in_block.append(obj)
                        except json.JSONDecodeError:
                            continue
                        json_start = -1
            
            if found_in_block:
                results.extend(found_in_block)
            else:
                # If no valid JSON found via brace counting, skip this block
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    This is a fallback for when the model uses markdown formatting instead of <json> tags.
    """
    results = []
    pattern = r'```(?:json)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        content = match.strip()
        if not content:
            continue
        try:
            results.append(json.loads(content))
        except json.JSONDecodeError:
            # Try brace counting approach
            brace_count = 0
            json_start = -1
            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        try:
                            obj = json.loads(content[json_start:i+1])
                            results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        json_start = -1
    
    return results or None


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


def _validate_prediction(prediction: str) -> str:
    """Validate and clean a prediction string.
    
    Removes common artifacts like quotes, extra punctuation, and validates
    that the prediction is one of the expected categories.
    """
    if not isinstance(prediction, str):
        prediction = str(prediction)
    
    # Remove surrounding quotes and whitespace
    cleaned = prediction.strip().strip('"').strip("'").strip()
    
    # Remove trailing punctuation
    cleaned = cleaned.rstrip('.').rstrip(',').rstrip(';').rstrip(':')
    
    # Check if it's a valid category after normalization
    normalized = _normalize_prediction(cleaned)
    
    # If normalization changed it to a standard category, use that
    if normalized in ['Correct', 'Incorrect', 'Partial']:
        return normalized
    
    # Otherwise return the cleaned version
    return cleaned


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like:
    - Case insensitivity
    - Whitespace normalization
    - Common synonyms (e.g., 'correct' vs 'right')
    - Numeric scores mapped to categories
    - Fractions (e.g., "1/2", "3/4")
    - Percentages (e.g., "75%")
    """
    if not isinstance(prediction, str):
        prediction = str(prediction)
    
    # Normalize whitespace and case
    normalized = prediction.strip().lower()
    
    # Map common variations to standard forms
    correct_synonyms = ['correct', 'right', 'true', 'yes', 'valid', 'accurate', '1', 'pass', 'solved', 'full', 'full credit', 'complete', 'success', 'accepted']
    incorrect_synonyms = ['incorrect', 'wrong', 'false', 'no', 'invalid', 'inaccurate', '0', 'fail', 'failed', 'unsolved', 'none', 'zero', 'rejected', 'error']
    partial_synonyms = ['partial', 'partially correct', 'incomplete', 'partial credit', 'half', 'some', 'partially', 'in progress', 'mixed', 'fair']
    
    # Check for exact matches first
    if normalized in correct_synonyms:
        return 'Correct'
    elif normalized in incorrect_synonyms:
        return 'Incorrect'
    elif normalized in partial_synonyms:
        return 'Partial'
    
    # Check for partial match (contains) - be more specific to avoid false positives
    if any(f' {syn} ' in f' {normalized} ' or normalized.startswith(syn + ' ') or normalized.endswith(' ' + syn) or normalized == syn 
           for syn in correct_synonyms):
        return 'Correct'
    elif any(f' {syn} ' in f' {normalized} ' or normalized.startswith(syn + ' ') or normalized.endswith(' ' + syn) or normalized == syn
             for syn in incorrect_synonyms):
        return 'Incorrect'
    elif 'partial' in normalized or 'incomplete' in normalized:
        return 'Partial'
    
    # Try to parse as fraction (e.g., "1/2", "3/4")
    fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', normalized)
    if fraction_match:
        try:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator > 0:
                num = numerator / denominator
                if num >= 0.9:
                    return 'Correct'
                elif num <= 0.1:
                    return 'Incorrect'
                else:
                    return 'Partial'
        except (ValueError, ZeroDivisionError):
            pass
    
    # Try to parse as percentage (e.g., "75%", "100 percent")
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', normalized)
    if percent_match or 'percent' in normalized:
        try:
            if percent_match:
                num = float(percent_match.group(1)) / 100.0
            else:
                # Extract number before "percent"
                percent_num_match = re.search(r'(\d+(?:\.\d+)?)\s*percent', normalized)
                if percent_num_match:
                    num = float(percent_num_match.group(1)) / 100.0
                else:
                    num = None
            
            if num is not None:
                if num >= 0.9:
                    return 'Correct'
                elif num <= 0.1:
                    return 'Incorrect'
                else:
                    return 'Partial'
        except ValueError:
            pass
    
    # Try to parse as numeric score (0-1 or 0-100 scale)
    try:
        # Remove any non-numeric prefix/suffix and try to extract a number
        numeric_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', normalized)
        if numeric_match:
            num_str = numeric_match.group()
            if num_str:
                num = float(num_str)
                # Handle 0-100 scale
                if num > 1 and num <= 100:
                    num = num / 100.0
                # Handle 0-1 scale
                if num >= 0.9:
                    return 'Correct'
                elif num <= 0.1:
                    return 'Incorrect'
                elif 0.1 < num < 0.9:
                    return 'Partial'
    except (ValueError, TypeError):
        pass
    
    # Return original with normalized whitespace if no synonym match
    return prediction.strip()


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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)"
}}
</json>

GRADING STANDARDS:
- Correct: The student's answer is fully correct, complete, and follows the expected solution approach
- Incorrect: The student's answer is fundamentally wrong, missing critical components, or shows major misconceptions
- Partial: The student shows valid reasoning, correct approach, or partial progress but has errors, gaps, or incomplete work

IMPORTANT GUIDELINES:
- Be objective and consistent in your grading
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- Consider the effort and approach, not just the final result
- If the student uses an alternative valid method not in the official solution, it should still be considered correct
- Empty or "I don't know" answers should be marked as Incorrect
- Answers with significant calculation errors but correct approach should be marked as Partial"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (<json> tags)
            extracted = _extract_jsons(last_message)
            self.log_fn(f"Primary extraction found {len(extracted) if extracted else 0} JSON objects")
            
            # Fallback to markdown code blocks
            if extracted is None:
                extracted = _extract_json_from_markdown(last_message)
                self.log_fn(f"Markdown extraction found {len(extracted) if extracted else 0} JSON objects")
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
                self.log_fn(f"Fallback extraction found {len(extracted) if extracted else 0} JSON objects")
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                self.log_fn(f"Using JSON with keys: {list(last_json.keys())}")
                
                # Priority order for prediction fields
                priority_fields = ["response", "grade", "answer", "result", "evaluation", "prediction", "score"]
                found = False
                for field in priority_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        found = True
                        break
                
                if not found:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            self.log_fn(f"Using first string value from key '{key}'")
                            break
                        elif isinstance(value, (int, float)):
                            prediction = str(value)
                            self.log_fn(f"Using first numeric value from key '{key}'")
                            break
            else:
                self.log_fn("No JSON objects found in response")
                
                # Last resort: try to extract any meaningful text after common markers
                markers = ["response:", "grade:", "answer:", "result:", "evaluation:", "prediction:", "score:"]
                text_lower = last_message.lower()
                for marker in markers:
                    if marker in text_lower:
                        idx = text_lower.find(marker) + len(marker)
                        # Extract up to 100 chars after marker
                        snippet = last_message[idx:idx+100].strip()
                        if snippet:
                            # Take first word or line
                            prediction = snippet.split()[0] if snippet.split() else snippet.split('\n')[0]
                            self.log_fn(f"Extracted prediction from text marker: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate and normalize the prediction for consistent output
        validated_prediction = _validate_prediction(prediction)
        normalized_prediction = _normalize_prediction(validated_prediction)
        if normalized_prediction != prediction:
            self.log_fn(f"Normalized prediction: '{prediction}' -> '{normalized_prediction}'")
        
        return normalized_prediction, msg_history
