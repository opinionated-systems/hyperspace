"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the IMPROVED task agent with better prompting, robust JSON extraction,
and chain-of-thought reasoning for more accurate grading.
The meta agent modifies this file during self-improvement. The evaluation harness 
loads whatever task_agent.py exists at the agent's repo path.
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
    
    # Try finding JSON-like structures with response field
    # Look for patterns like "response": "Correct" or "response": "Incorrect"
    response_pattern = r'"response"\s*:\s*"([^"]+)"'
    match = re.search(response_pattern, text)
    if match:
        response_val = match.group(1)
        # Also try to extract reasoning
        reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'
        reasoning_match = re.search(reasoning_pattern, text)
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        return {"response": response_val, "reasoning": reasoning}
    
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
                    start_idx = -1  # Reset start_idx on parse failure
                    continue
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from raw text using pattern matching.
    
    Looks for explicit grade mentions in the text.
    """
    text_lower = text.lower()
    
    # Priority patterns - look for grade in specific contexts first
    priority_patterns = [
        # JSON-like response field
        r'"response"\s*:\s*"?(correct|incorrect|partial)"?',
        # Grade assignment patterns
        r'grade\s*[:=]\s*"?(correct|incorrect|partial)"?',
        r'final grade\s*[:=]\s*"?(correct|incorrect|partial)"?',
        r'assigned grade\s*[:=]\s*"?(correct|incorrect|partial)"?',
        # Response/evaluation patterns
        r'response\s*[:=]\s*"?(correct|incorrect|partial)"?',
        r'evaluation\s*[:=]\s*"?(correct|incorrect|partial)"?',
    ]
    
    for pattern in priority_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).capitalize()
            if grade in ["Correct", "Incorrect", "Partial"]:
                return grade
    
    # Secondary patterns - look for grade at end of text or standalone
    secondary_patterns = [
        r'["\']?(correct|incorrect|partial)["\']?\s*$',  # End of text
        r'\bthe answer is\s+(correct|incorrect|partial)\b',
        r'\bthe grade is\s+(correct|incorrect|partial)\b',
        r'\b(correct|incorrect|partial)\b(?!\s*reasoning)',  # Any occurrence not followed by reasoning
    ]
    
    for pattern in secondary_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).capitalize()
            if grade in ["Correct", "Incorrect", "Partial"]:
                return grade
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required input fields are present and non-empty.
    
    Args:
        inputs: dict with problem inputs
        
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['problem', 'solution', 'grading_guidelines', 'student_answer']
    
    for field in required_fields:
        if field not in inputs:
            return False, f"Missing required field: {field}"
        if not inputs[field] or not str(inputs[field]).strip():
            return False, f"Empty required field: {field}"
    
    return True, ""


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
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        # Extract key fields for better prompting
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems.

Your task is to evaluate a student's answer to a problem and assign a grade.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Follow this step-by-step evaluation process:

1. UNDERSTAND: Briefly summarize what the problem is asking for.

2. ANALYZE SOLUTION: Review the official solution and identify the key concepts, theorems, and steps required.

3. EVALUATE STUDENT WORK: Compare the student's answer against the official solution:
   - Did they use the correct approach/method?
   - Are their calculations correct?
   - Did they justify their steps appropriately?
   - Did they reach the correct final answer?

4. APPLY GRADING GUIDELINES: Use the specific grading criteria provided to determine the appropriate grade.

5. ASSIGN GRADE: Based on your analysis, assign EXACTLY ONE of these three grades:
   - "Correct" - The answer is fully correct with proper reasoning
   - "Incorrect" - The answer is wrong or missing critical elements
   - "Partial" - The answer has some correct elements but is incomplete or has errors

You must respond with a JSON object containing:
- "reasoning": Your step-by-step analysis (brief but thorough)
- "response": The final grade (MUST be exactly "Correct", "Incorrect", or "Partial")

CRITICAL: The "response" field MUST contain ONLY one of these exact strings: "Correct", "Incorrect", or "Partial". Do not add any other text.

Your response MUST be wrapped in <json> tags like this:
<json>
{{
    "reasoning": "Your analysis here",
    "response": "Correct"
}}
</json>

ONLY output the <json>...</json> block. No other text."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        reasoning = ""
        try:
            if not msg_history:
                self.log_fn("Empty message history from LLM")
                return "Error: No response from LLM", msg_history
                
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw LLM response (first 500 chars): {last_message[:500]}...")
            
            extracted = _extract_json_flexible(last_message)
            
            if extracted:
                self.log_fn(f"Successfully extracted JSON: {extracted}")
                if "response" in extracted:
                    prediction = extracted["response"]
                    reasoning = extracted.get("reasoning", "No reasoning provided")
                    self.log_fn(f"Extracted prediction from 'response' field: {prediction}")
                else:
                    self.log_fn(f"No 'response' field found in extracted JSON: {extracted}")
                    # Try to find any string value that looks like a grade
                    for key, value in extracted.items():
                        if isinstance(value, str):
                            value_clean = value.strip().lower()
                            if value_clean in ["correct", "incorrect", "partial"]:
                                prediction = value.strip()
                                self.log_fn(f"Using alternative field '{key}' as prediction: {prediction}")
                                break
                            # Check if value contains a grade
                            for grade in ["correct", "incorrect", "partial"]:
                                if grade in value_clean:
                                    prediction = grade.capitalize()
                                    self.log_fn(f"Found grade '{grade}' in field '{key}': {prediction}")
                                    break
                            if prediction != "None":
                                break
            else:
                self.log_fn(f"Failed to extract JSON from response, trying text extraction...")
                # Fallback: try to find grade keywords in raw text
                text_grade = _extract_grade_from_text(last_message)
                if text_grade:
                    prediction = text_grade
                    self.log_fn(f"Extracted grade from text analysis: {prediction}")
                else:
                    self.log_fn(f"Could not extract grade from text: {last_message[:200]}...")
            
            # Validate and normalize the prediction
            valid_grades = ["Correct", "Incorrect", "Partial"]
            if prediction not in valid_grades:
                self.log_fn(f"Invalid prediction '{prediction}', attempting to normalize...")
                pred_str = str(prediction).lower().strip()
                
                # Remove common prefixes/suffixes that might appear
                pred_str = pred_str.replace('"', '').replace("'", "")
                pred_str = pred_str.replace('grade:', '').replace('final grade:', '')
                pred_str = pred_str.replace('the grade is', '').replace('assigned grade:', '')
                
                # Check for exact matches after cleaning
                for grade in valid_grades:
                    if pred_str == grade.lower():
                        prediction = grade
                        self.log_fn(f"Normalized to: {prediction}")
                        break
                else:
                    # Check if any grade appears as a substring
                    for grade in ["incorrect", "partial", "correct"]:
                        if grade in pred_str:
                            prediction = grade.capitalize()
                            self.log_fn(f"Found grade '{grade}' in prediction: {prediction}")
                            break
                    else:
                        # Last resort: check raw text for grade keywords
                        text_lower = last_message.lower()
                        # Look for grade in quotes or after specific patterns
                        import re
                        grade_patterns = [
                            r'"response"\s*:\s*"?(correct|incorrect|partial)"?',
                            r'grade\s*[:=]\s*"?(correct|incorrect|partial)"?',
                            r'\b(correct|incorrect|partial)\b(?!\s*reasoning)',
                        ]
                        for pattern in grade_patterns:
                            match = re.search(pattern, text_lower)
                            if match:
                                prediction = match.group(1).capitalize()
                                self.log_fn(f"Extracted grade from pattern: {prediction}")
                                break
                        else:
                            # Simple keyword search as final fallback
                            if "incorrect" in text_lower:
                                prediction = "Incorrect"
                            elif "partial" in text_lower:
                                prediction = "Partial"  
                            elif "correct" in text_lower:
                                prediction = "Correct"
                            self.log_fn(f"Final fallback prediction: {prediction}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        return str(prediction), msg_history
