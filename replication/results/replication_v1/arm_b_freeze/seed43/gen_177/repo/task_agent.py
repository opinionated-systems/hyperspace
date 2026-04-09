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
    - Punctuation removal
    """
    if not isinstance(prediction, str):
        prediction = str(prediction)
    
    # Normalize whitespace, case, and remove punctuation
    normalized = prediction.strip().lower()
    # Remove common punctuation that might appear in responses
    normalized = normalized.rstrip('.!?')
    
    # Map common variations to standard forms
    correct_synonyms = ['correct', 'right', 'true', 'yes', 'valid', 'accurate', '1', 'pass', 'full', 'full credit']
    incorrect_synonyms = ['incorrect', 'wrong', 'false', 'no', 'invalid', 'inaccurate', '0', 'fail', 'none', 'zero']
    partial_synonyms = ['partial', 'partially correct', 'incomplete', 'partial credit', 'half', 'some']
    
    # Check for exact matches first
    if normalized in correct_synonyms:
        return 'Correct'
    elif normalized in incorrect_synonyms:
        return 'Incorrect'
    elif normalized in partial_synonyms:
        return 'Partial'
    
    # Check for partial matches (substring)
    if 'partial' in normalized or 'incomplete' in normalized:
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

        # Validate inputs with detailed logging
        validation_errors = []
        if not problem:
            validation_errors.append("Missing problem")
        if not solution:
            validation_errors.append("Missing solution")
        if not student_answer:
            validation_errors.append("Empty student answer")
        if not grading_guidelines:
            validation_errors.append("Missing grading guidelines")
            
        if validation_errors:
            self.log_fn(f"Input validation warnings: {', '.join(validation_errors)}")
        else:
            self.log_fn(f"All inputs validated successfully for domain: {domain}")
        
        # Log input sizes for debugging
        self.log_fn(f"Input sizes - Problem: {len(problem)} chars, Solution: {len(solution)} chars, "
                   f"Student answer: {len(student_answer)} chars, Guidelines: {len(grading_guidelines)} chars")

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
    "response": "The final grade/prediction - MUST be one of: 'Correct', 'Incorrect', or 'Partial'"
}}
</json>

Important Grading Rules:
- Be objective and consistent in your evaluation
- Award 'Partial' credit when the student shows valid reasoning even if the final answer is incorrect
- Use 'Correct' only when the answer is fully correct and complete
- Use 'Incorrect' when the answer is wrong or shows no valid reasoning
- Your response field MUST contain exactly one of these three values: 'Correct', 'Incorrect', or 'Partial'"""

        # Retry loop for LLM calls with enhanced error tracking
        msg_history = []
        prediction = "None"
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.log_fn(f"LLM call attempt {attempt + 1}/{self.max_retries}")
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Log token usage if available
                usage = info.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    self.log_fn(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                
                # Extract prediction from JSON with fallback mechanisms
                prediction = self._extract_prediction(msg_history)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction on attempt {attempt + 1}: '{prediction}'")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid prediction extracted, retrying...")
                    
            except Exception as e:
                last_error = str(e)
                self.log_fn(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt == self.max_retries - 1:
                    self.log_fn(f"All {self.max_retries} retry attempts exhausted. Last error: {last_error}")
        
        # Normalize the prediction for consistent output
        normalized_prediction = _normalize_prediction(prediction)
        if normalized_prediction != prediction:
            self.log_fn(f"Normalized prediction: '{prediction}' -> '{normalized_prediction}'")
        
        # Final summary log
        total_messages = len(msg_history)
        self.log_fn(f"Task completed - Final prediction: '{normalized_prediction}', Messages: {total_messages}, Retries used: {attempt + 1}")
        
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
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                self.log_fn("Last message has no text content")
                return "None"
            
            # Log message preview for debugging (first 200 chars)
            preview = last_message[:200].replace('\n', ' ')
            self.log_fn(f"Processing last message (preview): {preview}...")
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            if extracted:
                self.log_fn(f"Primary extraction found {len(extracted)} JSON object(s)")
            else:
                self.log_fn("Primary extraction found no JSON objects")
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
                if extracted:
                    self.log_fn(f"Fallback extraction found {len(extracted)} JSON object(s)")
                else:
                    self.log_fn("Fallback extraction found no JSON objects")
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                available_keys = list(last_json.keys())
                self.log_fn(f"Using JSON with keys: {available_keys}")
                
                # Priority order for prediction fields
                priority_fields = ["response", "grade", "answer", "result", "evaluation", "prediction", "score"]
                for field in priority_fields:
                    if field in last_json:
                        value = last_json[field]
                        # Validate that the value is one of the expected grades
                        if isinstance(value, str):
                            normalized = value.strip()
                            valid_grades = ["Correct", "Incorrect", "Partial"]
                            if normalized in valid_grades:
                                self.log_fn(f"Found valid grade in field '{field}': '{value}'")
                                return value
                            else:
                                # Try to normalize the value
                                lower_val = normalized.lower()
                                if lower_val in ["correct", "right", "true", "yes", "valid"]:
                                    self.log_fn(f"Normalized '{value}' to 'Correct'")
                                    return "Correct"
                                elif lower_val in ["incorrect", "wrong", "false", "no", "invalid"]:
                                    self.log_fn(f"Normalized '{value}' to 'Incorrect'")
                                    return "Incorrect"
                                elif "partial" in lower_val:
                                    self.log_fn(f"Normalized '{value}' to 'Partial'")
                                    return "Partial"
                                else:
                                    self.log_fn(f"Field '{field}' has unexpected value '{value}', continuing search...")
                                    continue
                        self.log_fn(f"Found priority field '{field}' with value: '{value}'")
                        return value
                
                # If no known field, use the first string or numeric value found
                for key, value in last_json.items():
                    if isinstance(value, str):
                        # Validate against expected grades
                        normalized = value.strip()
                        valid_grades = ["Correct", "Incorrect", "Partial"]
                        if normalized in valid_grades:
                            self.log_fn(f"Using valid grade from key '{key}': '{value}'")
                            return value
                        # Try normalization
                        lower_val = normalized.lower()
                        if lower_val in ["correct", "right", "true", "yes", "valid"]:
                            self.log_fn(f"Normalized '{value}' from key '{key}' to 'Correct'")
                            return "Correct"
                        elif lower_val in ["incorrect", "wrong", "false", "no", "invalid"]:
                            self.log_fn(f"Normalized '{value}' from key '{key}' to 'Incorrect'")
                            return "Incorrect"
                        elif "partial" in lower_val:
                            self.log_fn(f"Normalized '{value}' from key '{key}' to 'Partial'")
                            return "Partial"
                        self.log_fn(f"Using first string value from key '{key}': '{value}'")
                        return value
                    elif isinstance(value, (int, float)):
                        self.log_fn(f"Using first numeric value from key '{key}': {value}")
                        return str(value)
            
            self.log_fn("No JSON objects found in response, attempting text-based extraction")
            
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
                        self.log_fn(f"Extracted prediction from text marker '{marker}': '{prediction}'")
                        return prediction
            
            self.log_fn("No text markers found, checking for keyword patterns")
            
            # Final fallback: look for common grade patterns in the text
            text_lower = last_message.lower()
            if any(word in text_lower for word in ['correct', 'right', 'true', 'yes', 'valid']):
                self.log_fn("Keyword match: Found 'correct' pattern in response")
                return 'Correct'
            elif any(word in text_lower for word in ['incorrect', 'wrong', 'false', 'no', 'invalid']):
                self.log_fn("Keyword match: Found 'incorrect' pattern in response")
                return 'Incorrect'
            elif any(word in text_lower for word in ['partial', 'partially']):
                self.log_fn("Keyword match: Found 'partial' pattern in response")
                return 'Partial'
            
            self.log_fn("No keyword patterns found, checking for numeric scores")
            
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
                    self.log_fn(f"Numeric score found: {score}/{max_score} = {ratio:.2%}")
                    if ratio >= 0.8:
                        return 'Correct'
                    elif ratio >= 0.5:
                        return 'Partial'
                    else:
                        return 'Incorrect'
            
            self.log_fn("All extraction methods failed - returning 'None'")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
        
        return "None"
