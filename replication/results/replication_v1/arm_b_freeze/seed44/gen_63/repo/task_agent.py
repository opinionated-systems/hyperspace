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
import string

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
    4. Look for JSON with "response" key (simple pattern)
    5. Try to fix common JSON formatting issues
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
            continue
    
    # Strategy 3: Look for JSON-like structures with balanced braces
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
            continue
    
    # Strategy 4: Look for JSON with "response" key (simple pattern)
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Try to fix common JSON issues and re-parse
    # Look for patterns like {response: 1} (missing quotes)
    fixed_pattern = r'\{\s*response\s*:\s*(\d+)\s*\}'
    for match in re.finditer(fixed_pattern, text, re.DOTALL):
        try:
            response_val = int(match.group(1))
            return {"response": response_val, "reasoning": "Extracted from malformed JSON", "confidence": "low"}
        except (ValueError, json.JSONDecodeError):
            continue
    
    # Look for text patterns like "The answer is correct" or "The answer is incorrect"
    text_lower = text.lower()
    if "answer is correct" in text_lower or "mark as correct" in text_lower:
        return {"response": 1, "reasoning": "Detected correct answer from text", "confidence": "medium"}
    if "answer is incorrect" in text_lower or "mark as incorrect" in text_lower:
        return {"response": 0, "reasoning": "Detected incorrect answer from text", "confidence": "medium"}
    
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
        # Check for common invalid answer patterns
        student_clean = str(student_answer).strip().lower()
        invalid_patterns = [
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null",
            "not sure", "unsure", "no answer", "blank", "empty", "unknown"
        ]
        is_invalid = student_clean in invalid_patterns or not student_clean

        invalid_note = ""
        if is_invalid:
            invalid_note = """
⚠️ NOTE: The student's answer appears to be empty or invalid. 
Please verify this and mark as incorrect (0) if confirmed."""

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
{invalid_note}

=== GRADING INSTRUCTIONS ===
Think step by step:
1. Analyze what the problem is asking for - identify the key requirements
2. Review the correct solution approach - understand the expected answer format
3. Compare the student's answer to the correct solution - check for equivalence
4. Check if the student followed the grading guidelines - apply any specific rules
5. Determine if the student's answer is correct (1) or incorrect (0)

Important grading criteria:
- The answer must be mathematically/logically correct
- The reasoning must be sound and complete (if reasoning is provided)
- Partial credit is NOT given - the answer is either fully correct (1) or incorrect (0)
- If the student answer is empty, blank, says "I don't know", or is clearly invalid, mark as incorrect (0)
- Check for common errors: off-by-one, sign errors, calculation mistakes, missing steps
- Consider equivalent forms: 0.5 = 1/2, 2x = x*2, etc. are all correct
- If the student provides the correct final answer but with wrong reasoning, still mark as correct (1)

=== RESPONSE FORMAT ===
You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must be valid and parseable.

Example for a correct answer:
<json>
{{
    "reasoning": "The student's answer matches the correct solution. They correctly calculated...",
    "response": 1,
    "confidence": "high"
}}
</json>

Example for an incorrect answer:
<json>
{{
    "reasoning": "The student's answer is incorrect because...",
    "response": 0,
    "confidence": "high"
}}
</json>

CRITICAL RULES:
- The "response" field MUST be exactly 1 (correct) or 0 (incorrect). No other values.
- The "confidence" field MUST be "high", "medium", or "low".
- Do not include any text outside the <json>...</json> tags.
- Ensure the JSON is properly formatted with double quotes around keys and string values."""

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
            if pred_clean in ("1", "true", "correct", "yes"):
                return "1"
            elif pred_clean in ("0", "false", "incorrect", "no"):
                return "0"
        return None

    def _extract_confidence(self, extracted: dict | None) -> str:
        """Extract confidence score from LLM response.
        
        Returns "high", "medium", or "low" based on confidence field.
        Defaults to "medium" if not specified.
        """
        if not extracted or "confidence" not in extracted:
            return "medium"
        
        confidence = str(extracted["confidence"]).strip().lower()
        if confidence in ("high", "medium", "low"):
            return confidence
        
        # Map similar terms
        if confidence in ("very high", "certain", "sure", "confident"):
            return "high"
        elif confidence in ("uncertain", "unsure", "guess"):
            return "low"
        
        return "medium"

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

        # Pre-check for obviously invalid answers
        student_clean = str(student_answer).strip().lower()
        invalid_patterns = [
            "", "i don't know", "i dont know", "idk", "n/a", "none", "null",
            "not sure", "unsure", "no answer", "blank", "empty", "unknown",
            "na", "n/a", "nil", "undefined", "nan", "skip", "pass", "?",
            "???", "...", "-", "--", "---", "~", "~~", "~~~~"
        ]
        # Also check for answers that are just whitespace or punctuation
        if not student_clean or student_clean in invalid_patterns or len(student_clean) < 1:
            self.log_fn(f"Empty or invalid student answer detected: '{student_answer}', returning 0")
            return "0", []
        # Check for answers that are just whitespace characters
        if student_clean.strip(string.whitespace + string.punctuation) == "":
            self.log_fn(f"Whitespace-only student answer detected: '{student_answer}', returning 0")
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
        
        # Try with retries for robustness
        last_error = None
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
                    confidence = self._extract_confidence(extracted)
                    
                    if validated is not None:
                        self.log_fn(f"Valid prediction: {validated} (raw: {prediction}, confidence: {confidence}, attempt: {attempt + 1})")
                        return validated, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction} (type: {type(prediction).__name__}), retrying... (attempt {attempt + 1}/{self.max_retries})")
                        last_error = f"Invalid prediction value: {prediction}"
                else:
                    self.log_fn(f"No valid JSON found in response (extracted: {extracted}), retrying... (attempt {attempt + 1}/{self.max_retries})")
                    # Log a snippet of the response for debugging
                    snippet = response_text[:200] if response_text else "(empty)"
                    self.log_fn(f"Response snippet: {snippet}...")
                    last_error = "No valid JSON found"
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Final fallback: return "0" if all retries failed
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction 0")
        return "0", msg_history
