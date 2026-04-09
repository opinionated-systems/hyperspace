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
    Also handles markdown code blocks with json tag and inline JSON.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
                # Fix unescaped newlines in strings
                fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks ```json ... ```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start = start + 7  # Skip past ```json
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                    fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
                    fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Try ``` ... ``` without json tag
    if not results:
        search_from = 0
        while True:
            start = text.find("```", search_from)
            if start == -1:
                break
            start = start + 3  # Skip past ```
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            # Only try if it looks like JSON
            if inner.startswith("{") or inner.startswith("["):
                try:
                    results.append(json.loads(inner))
                except json.JSONDecodeError:
                    try:
                        fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
                        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
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
                json_str = text[start_idx:i+1]
                try:
                    obj = json.loads(json_str)
                    results.append(obj)
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    try:
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
                        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
                        obj = json.loads(fixed)
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with structured chain-of-thought reasoning."""

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

        # Build structured analysis sections
        instruction = self._build_grading_prompt(
            domain, problem, solution, grading_guidelines, student_answer
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        # Validate and normalize the prediction
        prediction = self._validate_prediction(prediction, grading_guidelines)

        return str(prediction), msg_history

    def _build_grading_prompt(
        self, domain: str, problem: str, solution: str, 
        grading_guidelines: str, student_answer: str
    ) -> str:
        """Build a structured grading prompt with clear sections."""
        
        # Determine expected grade format from guidelines
        expected_format = self._infer_grade_format(grading_guidelines)
        
        return f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

=== DOMAIN ===
{domain}

=== PROBLEM STATEMENT ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING INSTRUCTIONS ===
Analyze the student's answer systematically:

1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required.

2. SOLUTION MAPPING: Map the official solution's key steps and verify each is addressed.

3. STUDENT WORK REVIEW: 
   - Check mathematical correctness of each step
   - Verify logical flow and reasoning
   - Identify any gaps, errors, or alternative valid approaches

4. GRADING DECISION:
   - Apply the grading guidelines precisely
   - Consider partial credit for incomplete but correct reasoning
   - Note: Alternative correct solutions should receive full credit

=== OUTPUT FORMAT ===
Your response MUST be valid JSON wrapped in <json> tags.

Expected grade format: {expected_format}

Respond in this exact format:
<json>
{{
    "problem_analysis": "Brief analysis of what the problem requires",
    "student_work_evaluation": "Detailed evaluation of the student's approach and correctness",
    "reasoning": "Your chain-of-thought for the grading decision",
    "response": "The final grade - must match expected format: {expected_format}"
}}
</json>"""

    def _infer_grade_format(self, grading_guidelines: str) -> str:
        """Infer the expected grade format from grading guidelines."""
        guidelines_lower = grading_guidelines.lower()
        
        # Check for specific grade formats
        if "correct" in guidelines_lower and "incorrect" in guidelines_lower:
            if "partial" in guidelines_lower:
                return "Correct, Incorrect, or Partial"
            return "Correct or Incorrect"
        
        if any(x in guidelines_lower for x in ["0", "1", "2", "3", "4", "5", "6", "7"]):
            return "Numeric score (typically 0-7 for IMO)"
        
        if "pass" in guidelines_lower or "fail" in guidelines_lower:
            return "Pass or Fail"
        
        return "Follow the grading guidelines exactly"

    def _validate_prediction(self, prediction: str, grading_guidelines: str) -> str:
        """Validate and normalize the prediction against grading guidelines.
        
        Args:
            prediction: The extracted prediction string
            grading_guidelines: The grading guidelines to validate against
            
        Returns:
            Validated and normalized prediction string
        """
        if not prediction or prediction == "None":
            return "None"
        
        prediction_clean = prediction.strip()
        guidelines_lower = grading_guidelines.lower()
        
        # Normalize common variations
        prediction_lower = prediction_clean.lower()
        
        # Check for binary correct/incorrect with normalization
        if prediction_lower in ["correct", "right", "true", "yes", "valid", "accurate"]:
            if "correct" in guidelines_lower:
                return "Correct"
        
        if prediction_lower in ["incorrect", "wrong", "false", "no", "invalid", "inaccurate", "error"]:
            if "incorrect" in guidelines_lower:
                return "Incorrect"
        
        if prediction_lower in ["partial", "partially correct", "partial credit", "incomplete"]:
            if "partial" in guidelines_lower:
                return "Partial"
        
        # Check for numeric grades (IMO-style 0-7)
        try:
            num = int(prediction_clean)
            if 0 <= num <= 7 and any(str(x) in grading_guidelines for x in range(8)):
                return str(num)
        except ValueError:
            pass
        
        # If no normalization applied, return original
        return prediction_clean

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with robust fallback mechanisms.
        
        Args:
            msg_history: List of message dicts from LLM conversation
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history, returning 'None'")
            return "None"
        
        try:
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                self.log_fn("Warning: Last message has no text content")
                return "None"
            
            # Try primary extraction method (tagged JSON blocks)
            extracted = _extract_jsons(last_message)
            self.log_fn(f"Primary JSON extraction found {len(extracted) if extracted else 0} objects")
            
            # Fallback to generic JSON extraction if primary fails
            if not extracted:
                self.log_fn("Primary extraction failed, trying fallback extraction")
                extracted = _extract_any_json(last_message)
                if extracted:
                    self.log_fn(f"Fallback extraction found {len(extracted)} objects")
            
            if extracted:
                prediction = self._get_prediction_from_json(extracted[-1])
                self.log_fn(f"Successfully extracted prediction: '{prediction}'")
                return prediction
            
            # Final fallback: look for common grade patterns in raw text
            self.log_fn("JSON extraction failed, trying raw text extraction")
            raw_prediction = self._extract_from_raw_text(last_message)
            self.log_fn(f"Raw text extraction result: '{raw_prediction}'")
            return raw_prediction
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            return "None"

    def _get_prediction_from_json(self, json_obj: dict) -> str:
        """Extract prediction value from a JSON object.
        
        Tries multiple common field names in order of preference.
        """
        # Ordered list of field names to try (most specific first)
        field_names = [
            "response", "grade", "evaluation", "verdict", 
            "result", "answer", "score", "prediction", "output"
        ]
        
        for field in field_names:
            if field in json_obj:
                value = json_obj[field]
                if isinstance(value, str):
                    return value.strip()
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, bool):
                    return "Correct" if value else "Incorrect"
        
        # If no known field, use the first string or numeric value found
        for key, value in json_obj.items():
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, (int, float)):
                return str(value)
        
        return "None"

    def _extract_from_raw_text(self, text: str) -> str:
        """Extract grade from raw text when JSON parsing fails.
        
        Looks for common grade patterns and keywords with improved accuracy.
        """
        text_lower = text.lower()
        
        # Check for explicit grade statements with more specific patterns
        grade_patterns = [
            (r'final\s+grade[:\s]+([\w\s-]+?)(?:\n|$|\.)', 1),
            (r'grade[:\s]+([\w\s-]+?)(?:\n|$|\.)', 1),
            (r'evaluation[:\s]+([\w\s-]+?)(?:\n|$|\.)', 1),
            (r'response[:\s]+([\w\s-]+?)(?:\n|$|\.)', 1),
            (r'verdict[:\s]+([\w\s-]+?)(?:\n|$|\.)', 1),
            (r'score[:\s]+([\d]+(?:\.\d+)?)', 1),
            (r'(?:^|\n)\s*([\w\s-]+?)(?:\s*=\s*|\s*is\s+)(?:correct|incorrect|partial)', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(group).strip()
                # Clean up common artifacts
                result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
                if result and result not in ['the', 'a', 'an', 'this', 'that']:
                    return result
        
        # Check for binary correct/incorrect indicators with context awareness
        # Look for patterns like "is correct", "is incorrect", "answer is correct"
        correct_patterns = [
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+correct\b',
            r'\b(?:the\s+)?(?:answer|response|solution)\s+is\s+right\b',
            r'\bcorrect\s+(?:answer|response|solution)\b',
        ]
        incorrect_patterns = [
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+incorrect\b',
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+wrong\b',
            r'\bincorrect\s+(?:answer|response|solution)\b',
            r'\bwrong\s+(?:answer|response|solution)\b',
        ]
        partial_patterns = [
            r'\b(?:answer|response|solution|grade|evaluation)\s+is\s+partial\b',
            r'\bpartial\s+(?:credit|score|grade)\b',
            r'\bpartially\s+correct\b',
        ]
        
        # Count matches for each category
        correct_count = sum(1 for p in correct_patterns if re.search(p, text_lower))
        incorrect_count = sum(1 for p in incorrect_patterns if re.search(p, text_lower))
        partial_count = sum(1 for p in partial_patterns if re.search(p, text_lower))
        
        # Return the most specific match
        if partial_count > 0:
            return "Partial"
        if incorrect_count > correct_count:
            return "Incorrect"
        if correct_count > 0 and incorrect_count == 0:
            return "Correct"
        
        # Last resort: simple keyword matching
        has_correct = re.search(r'\bcorrect\b', text_lower) is not None
        has_incorrect = re.search(r'\bincorrect\b', text_lower) is not None
        has_wrong = re.search(r'\bwrong\b', text_lower) is not None
        has_partial = re.search(r'\bpartial\b', text_lower) is not None
        
        if has_partial:
            return "Partial"
        if has_incorrect or has_wrong:
            return "Incorrect"
        if has_correct and not has_incorrect:
            return "Correct"
        
        return "None"
