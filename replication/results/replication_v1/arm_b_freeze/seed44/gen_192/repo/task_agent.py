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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects in text (with nested brace handling)
    4. Look for JSON with "reasoning" and "response" keys
    5. LLM may output JSON with single quotes - try to fix
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            fixed = _fix_json(match.group(1).strip())
            if fixed:
                return fixed
            continue
    
    # Strategy 3: Look for JSON-like structures with balanced braces
    # This handles nested JSON objects properly
    def find_json_objects(s: str) -> list[str]:
        """Find all JSON-like objects with balanced braces."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(s) and brace_count > 0:
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    for obj_str in find_json_objects(text):
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            fixed = _fix_json(obj_str)
            if fixed and "response" in fixed:
                return fixed
            continue
    
    # Strategy 4: Look for JSON with "response" key (simple pattern)
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    return None


def _fix_json(text: str) -> dict | None:
    """Attempt to fix common JSON formatting issues.
    
    Handles:
    - Single quotes instead of double quotes
    - Trailing commas
    - Missing quotes around keys
    - Unescaped newlines in strings
    - Comments in JSON
    - Extra whitespace
    """
    import ast
    
    # Try Python literal eval (handles single quotes)
    try:
        # Replace single quotes with double quotes for JSON compatibility
        # First, try ast.literal_eval which handles Python dict syntax
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Manual fix: replace single quotes with double quotes
    # Be careful with apostrophes in text
    try:
        # Simple approach: replace ' with " for key/value delimiters
        # This is a heuristic and may not work for all cases
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Fix trailing commas before closing braces/brackets
    try:
        # Remove trailing commas: ,} -> } and ,] -> ]
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Fix missing quotes around keys (simple pattern: word followed by :)
    try:
        # Add quotes around unquoted keys
        fixed = re.sub(r'(\w+):', r'"\1":', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Remove comments (both // and /* */)
    try:
        # Remove single-line comments
        fixed = re.sub(r'//[^\n]*', '', text)
        # Remove multi-line comments
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Fix unescaped newlines in strings
    try:
        # Replace newlines within string values with escaped newlines
        fixed = re.sub(r'(?<=")\n(?=")', '\\n', text)
        fixed = re.sub(r'(?<=")\r\n(?=")', '\\r\\n', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Build a structured grading prompt with clear instructions."""
        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING INSTRUCTIONS ===
Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Important grading criteria:
- The answer must be mathematically correct
- The reasoning must be sound and complete
- Partial credit is NOT given - the answer is either fully correct (1) or incorrect (0)
- If the student answer is empty, blank, or says "I don't know", mark as incorrect (0)

=== RESPONSE FORMAT ===
You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain why the answer is wrong.",
    "response": 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect). No other values are accepted."""

    def _validate_prediction(self, prediction: any) -> str | None:
        """Validate and normalize prediction value.
        
        Returns normalized prediction string or None if invalid.
        """
        # Handle various formats
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        if isinstance(prediction, (int, float)):
            if prediction == 1 or prediction == 1.0:
                return "1"
            elif prediction == 0 or prediction == 0.0:
                return "0"
        if isinstance(prediction, str):
            pred_clean = prediction.strip().lower()
            if pred_clean in ("1", "true", "correct", "yes", "valid", "right"):
                return "1"
            elif pred_clean in ("0", "false", "incorrect", "no", "invalid", "wrong"):
                return "0"
        return None

    def _calculate_confidence(self, extracted: dict | None, response_text: str) -> float:
        """Calculate confidence score for the prediction.
        
        Returns a score between 0.0 and 1.0 based on:
        - Presence of reasoning in the response
        - Length and quality of reasoning
        - JSON format compliance
        - Response value validity
        """
        if extracted is None:
            return 0.0
        
        confidence = 0.4  # Base confidence for having valid JSON
        
        # Check for reasoning field
        reasoning = extracted.get("reasoning", "")
        if reasoning:
            confidence += 0.15
            reasoning_str = str(reasoning)
            # Bonus for detailed reasoning (at least 50 chars)
            if len(reasoning_str) >= 50:
                confidence += 0.1
            # Bonus for very detailed reasoning (at least 200 chars)
            if len(reasoning_str) >= 200:
                confidence += 0.1
            # Bonus for extremely detailed reasoning (at least 500 chars)
            if len(reasoning_str) >= 500:
                confidence += 0.05
        
        # Check if response uses proper <json> tags
        if "<json>" in response_text and "</json>" in response_text:
            confidence += 0.1
        
        # Check if response value is valid (1 or 0)
        response_val = extracted.get("response")
        if response_val in (1, 0, "1", "0", True, False):
            confidence += 0.1
        
        return min(1.0, confidence)

    def _check_empty_answer(self, student_answer: str) -> bool:
        """Check if student answer is empty or indicates no attempt."""
        if not student_answer:
            return True
        student_clean = str(student_answer).strip().lower()
        empty_indicators = (
            "", "i don't know", "i dont know", "idk", "n/a", "none", 
            "null", "unknown", "no answer", "not sure", "unsure",
            "can't solve", "cannot solve", "no solution", "blank",
            "empty", "skip", "skipped", "pass", "?", "???"
        )
        return student_clean in empty_indicators or not student_clean

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Quick check for empty/invalid student answers
        if self._check_empty_answer(student_answer):
            self.log_fn("Empty or invalid student answer detected, returning 0")
            return "0", []

        # Build structured prompt
        instruction = self._build_grading_prompt(
            domain=domain,
            problem=problem,
            solution=solution,
            grading_guidelines=grading_guidelines,
            student_answer=student_answer,
        )

        msg_history = []
        best_prediction = None
        best_confidence = 0.0
        last_error = None
        
        # Try with retries for robustness, tracking best prediction by confidence
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = self._validate_prediction(prediction)
                    
                    if validated is not None:
                        confidence = self._calculate_confidence(extracted, response_text)
                        self.log_fn(f"Attempt {attempt + 1}: Valid prediction {validated} with confidence {confidence:.2f}")
                        
                        # If high confidence, return immediately
                        if confidence >= 0.85:
                            return validated, msg_history
                        
                        # Track best prediction
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_prediction = validated
                    else:
                        self.log_fn(f"Attempt {attempt + 1}: Invalid prediction value: {prediction}")
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response")
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
        
        # Return best prediction if we have one with reasonable confidence
        if best_prediction is not None and best_confidence >= 0.5:
            self.log_fn(f"Returning best prediction: {best_prediction} (confidence: {best_confidence:.2f})")
            return best_prediction, msg_history
        
        # If we have a prediction but low confidence, still use it as last resort
        if best_prediction is not None:
            self.log_fn(f"Returning low-confidence prediction: {best_prediction} (confidence: {best_confidence:.2f})")
            return best_prediction, msg_history
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history
