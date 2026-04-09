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
    Also handles nested JSON objects by tracking brace depth.
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
            # Try to extract JSON by finding balanced braces
            json_obj = _extract_balanced_json(inner)
            if json_obj:
                results.append(json_obj)
            continue
    
    # Also try to find JSON in markdown code blocks if no <json> tags found
    if not results:
        code_block_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                json_obj = _extract_balanced_json(match)
                if json_obj:
                    results.append(json_obj)
    
    return results or None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract a JSON object by tracking brace depth.
    
    This handles cases where the JSON content contains nested braces
    that might confuse simple regex-based extraction.
    """
    start_idx = -1
    brace_depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
            continue
        if char == '"' and in_string:
            in_string = False
            continue
        if not in_string:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        continue
    return None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using multiple strategies for malformed responses."""
    results = []
    
    # Strategy 1: Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            # Try to parse the matched content
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try balanced brace extraction
            balanced = _extract_balanced_json(match)
            if balanced:
                results.append(balanced)
    
    # Strategy 2: Look for any JSON-like structure with balanced braces
    if not results:
        balanced = _extract_balanced_json(text)
        if balanced:
            results.append(balanced)
    
    # Strategy 3: Extract key-value pairs for response and reasoning
    if not results:
        response_match = re.search(r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        
        extracted = {}
        if response_match:
            extracted["response"] = response_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        if reasoning_match:
            extracted["reasoning"] = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        
        if extracted:
            results.append(extracted)
    
    # Strategy 4: Look for grade/assessment mentions in plain text
    if not results:
        extracted = _extract_grade_from_text(text)
        if extracted:
            results.append(extracted)
    
    return results or None


def _extract_grade_from_text(text: str) -> dict | None:
    """Extract grade information from plain text when JSON is not available.
    
    This is a last-resort extraction strategy for cases where the model
    outputs grading information in natural language without proper JSON formatting.
    """
    text_lower = text.lower()
    extracted = {"reasoning": text[:500] + "..." if len(text) > 500 else text}
    
    # Look for explicit grade statements
    grade_patterns = [
        (r'grade\s*:\s*(correct|partial|incorrect)', 1),
        (r'assessment\s*:\s*(correct|partial|incorrect)', 1),
        (r'(?:the\s+)?(?:answer|response|grade)\s+is\s+(correct|partial|incorrect)', 1),
        (r'(?:i\s+)?(?:would\s+)?(?:rate|grade|assess)\s+(?:this\s+)?(?:as\s+)?(correct|partial|incorrect)', 1),
        (r'(?:this\s+is\s+)?(correct|partial|incorrect)', 0),
    ]
    
    for pattern, group_idx in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(group_idx).capitalize()
            if grade in ["Correct", "Partial", "Incorrect"]:
                extracted["response"] = grade
                return extracted
    
    # Look for numeric scores
    score_pattern = r'(?:score|grade|mark)s?\s*:?\s*(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*(\d+(?:\.\d+)?)'
    score_match = re.search(score_pattern, text_lower)
    if score_match:
        try:
            score = float(score_match.group(1))
            total = float(score_match.group(2))
            if total > 0:
                ratio = score / total
                if ratio >= 0.9:
                    extracted["response"] = "Correct"
                elif ratio >= 0.5:
                    extracted["response"] = "Partial"
                else:
                    extracted["response"] = "Incorrect"
                return extracted
        except (ValueError, TypeError):
            pass
    
    # Check for keywords indicating the grade
    if any(word in text_lower for word in ['perfect', 'excellent', 'complete', 'fully correct']):
        extracted["response"] = "Correct"
        return extracted
    elif any(word in text_lower for word in ['partially correct', 'partial credit', 'some correct', 'incomplete']):
        extracted["response"] = "Partial"
        return extracted
    elif any(word in text_lower for word in ['wrong', 'incorrect', 'error', 'mistake', 'invalid']):
        extracted["response"] = "Incorrect"
        return extracted
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison.
    
    Handles common variations like case differences, extra whitespace,
    punctuation differences, and numeric grades.
    """
    if not prediction:
        return "None"
    
    # Strip whitespace and normalize case
    normalized = prediction.strip()
    
    # Handle common grade variations
    grade_map = {
        "correct": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "partially": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "right": "Correct",
        "true": "Correct",
        "false": "Incorrect",
        "yes": "Correct",
        "no": "Incorrect",
        "pass": "Correct",
        "fail": "Incorrect",
        "full": "Correct",
        "full marks": "Correct",
        "full credit": "Correct",
        "zero": "Incorrect",
        "0": "Incorrect",
        "1": "Correct",
        "100%": "Correct",
        "0%": "Incorrect",
        "50%": "Partial",
        "half": "Partial",
        "mostly correct": "Partial",
        "mostly incorrect": "Incorrect",
        "incomplete": "Partial",
        "complete": "Correct",
        "valid": "Correct",
        "invalid": "Incorrect",
        "acceptable": "Correct",
        "unacceptable": "Incorrect",
        "satisfactory": "Correct",
        "unsatisfactory": "Incorrect",
    }
    
    lower_pred = normalized.lower()
    if lower_pred in grade_map:
        return grade_map[lower_pred]
    
    # Handle numeric grades (0-100 scale)
    try:
        numeric_val = float(lower_pred)
        if numeric_val >= 90:
            return "Correct"
        elif numeric_val >= 50:
            return "Partial"
        else:
            return "Incorrect"
    except (ValueError, TypeError):
        pass
    
    # Handle fraction grades
    fraction_map = {
        "1/1": "Correct",
        "0/1": "Incorrect",
        "1/2": "Partial",
        "0.5": "Partial",
        "0.5/1": "Partial",
        "2/2": "Correct",
        "1/3": "Partial",
        "2/3": "Partial",
        "3/3": "Correct",
    }
    if lower_pred in fraction_map:
        return fraction_map[lower_pred]
    
    # Try to parse fraction format "x/y"
    if "/" in lower_pred:
        try:
            parts = lower_pred.split("/")
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator > 0:
                    ratio = numerator / denominator
                    if ratio >= 0.9:
                        return "Correct"
                    elif ratio >= 0.4:
                        return "Partial"
                    else:
                        return "Incorrect"
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    # Try to parse percentage format with % sign
    if "%" in lower_pred:
        try:
            pct_val = float(lower_pred.replace("%", "").strip())
            if pct_val >= 90:
                return "Correct"
            elif pct_val >= 40:
                return "Partial"
            else:
                return "Incorrect"
        except (ValueError, TypeError):
            pass
    
    return normalized


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 3) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = max_retries
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
        self._log_file = log_file
    
    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics."""
        return dict(self._extraction_stats)
    
    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"success": 0, "fallback": 0, "failure": 0}
    
    def save_extraction_stats(self) -> None:
        """Save extraction statistics to the log file if configured."""
        if self._log_file:
            try:
                import json
                from datetime import datetime
                stats = {
                    "timestamp": datetime.now().isoformat(),
                    "extraction_stats": dict(self._extraction_stats),
                    "total_attempts": sum(self._extraction_stats.values()),
                }
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(stats) + "\n")
            except Exception as e:
                self.log_fn(f"Failed to save extraction stats: {e}")
    
    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate the input dictionary for required fields.
        
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["problem", "solution", "student_answer"]
        missing_fields = []
        
        for field in required_fields:
            if field not in inputs:
                missing_fields.append(field)
            elif not inputs[field] or not str(inputs[field]).strip():
                missing_fields.append(f"{field} (empty)")
        
        if missing_fields:
            return False, f"Missing or empty required fields: {', '.join(missing_fields)}"
        
        # Validate field types and content
        for field in ["problem", "solution", "student_answer", "grading_guidelines"]:
            if field in inputs and inputs[field] is not None:
                if not isinstance(inputs[field], str):
                    return False, f"Field '{field}' must be a string, got {type(inputs[field]).__name__}"
        
        return True, ""

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:

**Correct**: The answer must satisfy ALL of the following:
- The final result/answer matches the correct solution exactly
- The reasoning/logic is mathematically sound and complete
- All necessary steps are present and correctly executed
- No significant errors or omissions in the derivation
- Alternative valid approaches are accepted if they lead to the correct answer

**Partial**: The answer satisfies SOME but not all of the following:
- The final result is correct or nearly correct (minor arithmetic errors)
- The overall approach is correct but some steps are missing or unclear
- There are minor logical gaps that don't invalidate the main argument
- The student shows understanding of key concepts but execution is imperfect
- Partial credit applies when the student is "on the right track" but incomplete

**Incorrect**: The answer satisfies ANY of the following:
- The final result is wrong or fundamentally misunderstood
- The approach/method is completely wrong for the problem type
- Major logical errors invalidate the reasoning
- The answer is irrelevant or doesn't address the problem
- Critical steps are missing or severely flawed

## Examples of Grading Decisions

Example 1 - Correct:
- Student provides a complete proof with all steps justified
- Final answer matches the solution exactly
- Reasoning is clear and mathematically rigorous
→ Grade: Correct

Example 2 - Partial:
- Student uses the right approach but skips some intermediate steps
- Final answer is correct but justification is incomplete
- Minor calculation error that doesn't affect the overall approach
→ Grade: Partial

Example 3 - Incorrect:
- Student uses a completely wrong method (e.g., tries to use calculus on a combinatorics problem)
- Final answer is wrong and reasoning is flawed
- Student misunderstands the problem statement
→ Grade: Incorrect

## Response Format (REQUIRED - READ CAREFULLY)
You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

The JSON must have exactly these two fields:
- "reasoning": A string containing your detailed step-by-step analysis
- "response": A string containing ONLY one of: "Correct", "Partial", or "Incorrect"

Example of a VALID response:
<json>
{{
    "reasoning": "The student correctly identified the approach and executed all steps properly. The final answer matches the solution exactly.",
    "response": "Correct"
}}
</json>

Example of an INVALID response (DO NOT DO THIS):
<json>
{{
    "reasoning": "The student made some errors.",
    "response": "The answer is incorrect because..."
}}
</json>

CRITICAL: The "response" field must contain ONLY the grade word (Correct, Partial, or Incorrect), NOT a sentence or explanation.

IMPORTANT: 
- Ensure your JSON is valid and properly formatted
- Use double quotes for strings, not single quotes
- Do not include any text outside the <json> tags
- The 'response' field should contain ONLY the grade, not the reasoning"""

        if is_retry:
            return f"""ERROR: Your previous response did not contain valid JSON with a 'response' field, or the JSON was malformed, or the response field contained text instead of just the grade.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

CRITICAL RULES:
1. The "response" field must contain ONLY one of: "Correct", "Partial", or "Incorrect"
2. Do NOT write sentences like "The answer is correct" in the response field
3. Do NOT include explanations in the response field
4. Use double quotes, not single quotes

Now try again with the original task:

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning, method) tuple where method indicates extraction success
        """
        prediction = "None"
        reasoning = ""
        method = "failure"
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted:
            method = "success"
        else:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            if extracted:
                method = "fallback"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Post-process prediction to extract just the grade word if it contains a sentence
        prediction = self._extract_grade_from_response(prediction)
        
        # Normalize the prediction
        prediction = _normalize_prediction(prediction)
        
        return prediction, reasoning, method
    
    def _extract_grade_from_response(self, response: str) -> str:
        """Extract just the grade word from a response that might contain a sentence.
        
        Sometimes models return "The answer is correct" instead of just "Correct".
        This method extracts just the grade word.
        """
        if not response:
            return "None"
        
        response_lower = response.lower().strip()
        
        # If it's already just a single word grade, return it capitalized
        if response_lower in ["correct", "partial", "incorrect"]:
            return response_lower.capitalize()
        
        # Look for grade words in the response
        # Check for "correct" first, but avoid false positives like "incorrect"
        if re.search(r'\bcorrect\b', response_lower) and not re.search(r'\bincorrect\b', response_lower):
            return "Correct"
        if re.search(r'\bpartial\b', response_lower) or re.search(r'\bpartially\b', response_lower):
            return "Partial"
        if re.search(r'\bincorrect\b', response_lower) or re.search(r'\bwrong\b', response_lower):
            return "Incorrect"
        
        # If no grade word found, return the original response for normalization
        return response

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
        instruction = self._build_grading_prompt(inputs, is_retry=False)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Track all attempts for better debugging
        attempt_details = []
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, method = self._extract_prediction(last_text)
                
                # Update statistics
                self._extraction_stats[method] += 1
                
                # Log detailed attempt info
                attempt_info = {
                    "attempt": attempt + 1,
                    "method": method,
                    "prediction": prediction,
                    "response_length": len(last_text),
                    "has_reasoning": bool(reasoning),
                }
                attempt_details.append(attempt_info)
                
                if prediction != "None":
                    self.log_fn(f"[Attempt {attempt + 1}] Successfully extracted prediction: {prediction} (method: {method})")
                    if reasoning:
                        self.log_fn(f"[Attempt {attempt + 1}] Reasoning preview: {reasoning[:200]}...")
                    # Log extraction stats summary
                    self.log_fn(f"Extraction stats: {dict(self._extraction_stats)}")
                    break
                else:
                    self.log_fn(f"[Attempt {attempt + 1}] Failed to extract prediction (method: {method})")
                    # Log a preview of the problematic response for debugging
                    preview = last_text[:300].replace('\n', ' ')
                    self.log_fn(f"[Attempt {attempt + 1}] Response preview: {preview}...")
                    
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(inputs, is_retry=True)
                    
            except Exception as e:
                error_msg = f"[Attempt {attempt + 1}] Error: {type(e).__name__}: {str(e)[:100]}"
                self.log_fn(error_msg)
                attempt_details.append({"attempt": attempt + 1, "error": str(e)})
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Log summary of all attempts if we failed
        if prediction == "None" or str(prediction).startswith("Error:"):
            self.log_fn(f"All {self.max_retries} attempts failed. Details: {attempt_details}")
        
        # Save extraction statistics to log file
        self.save_extraction_stats()
        
        return str(prediction), msg_history
