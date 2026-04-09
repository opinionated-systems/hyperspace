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


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like:
    - Case insensitivity
    - Whitespace normalization
    - Common synonyms (e.g., 'correct' vs 'right')
    """
    if not isinstance(prediction, str):
        prediction = str(prediction)
    
    # Normalize whitespace and case
    normalized = prediction.strip().lower()
    
    # Map common variations to standard forms
    correct_synonyms = ['correct', 'right', 'true', 'yes', 'valid', 'accurate', '1', 'pass']
    incorrect_synonyms = ['incorrect', 'wrong', 'false', 'no', 'invalid', 'inaccurate', '0', 'fail']
    partial_synonyms = ['partial', 'partially correct', 'incomplete', 'partial credit']
    
    if normalized in correct_synonyms:
        return 'Correct'
    elif normalized in incorrect_synonyms:
        return 'Incorrect'
    elif normalized in partial_synonyms or 'partial' in normalized:
        return 'Partial'
    
    # Return original with normalized whitespace if no synonym match
    return prediction.strip()


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

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

        # Validate inputs
        if not problem or not solution:
            self.log_fn("Warning: Missing problem or solution in inputs")
        if not student_answer:
            self.log_fn("Warning: Empty student answer")

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

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect."""

        # Retry loop for LLM calls
        msg_history = []
        prediction = "None"
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON with fallback mechanisms
                prediction = self._extract_prediction(msg_history)
                
                if prediction != "None":
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid prediction extracted, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    self.log_fn("All retry attempts exhausted")
        
        # Normalize the prediction for consistent output
        normalized_prediction = _normalize_prediction(prediction)
        if normalized_prediction != prediction:
            self.log_fn(f"Normalized prediction: '{prediction}' -> '{normalized_prediction}'")
        
        return normalized_prediction, msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Args:
            msg_history: List of message dictionaries
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            self.log_fn("Empty message history")
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            self.log_fn(f"Primary extraction found {len(extracted) if extracted else 0} JSON objects")
            
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
                for field in priority_fields:
                    if field in last_json:
                        return last_json[field]
                
                # If no known field, use the first string or numeric value found
                for key, value in last_json.items():
                    if isinstance(value, str):
                        self.log_fn(f"Using first string value from key '{key}'")
                        return value
                    elif isinstance(value, (int, float)):
                        self.log_fn(f"Using first numeric value from key '{key}'")
                        return str(value)
            
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
                        return prediction
            
            # Final fallback: look for common grade patterns in the text
            text_lower = last_message.lower()
            if any(word in text_lower for word in ['correct', 'right', 'true', 'yes', 'valid']):
                return 'Correct'
            elif any(word in text_lower for word in ['incorrect', 'wrong', 'false', 'no', 'invalid']):
                return 'Incorrect'
            elif any(word in text_lower for word in ['partial', 'partially']):
                return 'Partial'
            
            # Ultra-fallback: check for numeric scores (0-100 or 0-10 scale)
            # Look for patterns like "score: 85", "grade: 7/10", "result: 0.8"
            score_patterns = [
                r'(?:score|grade|result)[\s:=]+(\d+(?:\.\d+)?)\s*/?\s*(\d+)?',
                r'(?:score|grade|result)[\s:=]+(\d+(?:\.\d+)?)',
            ]
            for pattern in score_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    score = float(match.group(1))
                    max_score = float(match.group(2)) if match.group(2) else 100.0
                    # Convert to categorical
                    ratio = score / max_score
                    if ratio >= 0.8:
                        return 'Correct'
                    elif ratio >= 0.5:
                        return 'Partial'
                    else:
                        return 'Incorrect'
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
