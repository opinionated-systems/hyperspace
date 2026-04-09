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
    3. Raw JSON objects in text
    4. Relaxed JSON pattern matching for malformed responses
    5. Line-by-line JSON object detection for fragmented responses
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
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Relaxed pattern - look for any JSON object with response field
    # Handles cases where JSON might span multiple lines with extra whitespace
    relaxed_pattern = r'\{[^}]*"response"\s*:\s*(?:1|0|"1"|"0")[^}]*\}'
    for match in re.finditer(relaxed_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Extract JSON from nested braces with balanced bracket counting
    # This handles cases where JSON spans multiple lines with nested structures
    def extract_balanced_json(text: str) -> list[str]:
        """Extract JSON objects using balanced brace counting."""
        results = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                count = 1
                i += 1
                in_string = False
                escape_next = False
                while i < len(text) and count > 0:
                    char = text[i]
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            count += 1
                        elif char == '}':
                            count -= 1
                    i += 1
                if count == 0:
                    results.append(text[start:i])
            else:
                i += 1
        return results
    
    for json_str in extract_balanced_json(text):
        try:
            parsed = json.loads(json_str)
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def _normalize_prediction(prediction: Any) -> str:
    """Normalize prediction value to '0' or '1'.
    
    Handles various formats: integers, strings, booleans, floats.
    Returns '0' as default for invalid values.
    """
    if prediction is None:
        return "0"
    
    # Handle boolean type directly
    if isinstance(prediction, bool):
        return "1" if prediction else "0"
    
    # Handle numeric types directly
    if isinstance(prediction, (int, float)):
        # Handle NaN and infinity
        if isinstance(prediction, float):
            if prediction != prediction:  # NaN check
                logger.warning(f"Prediction is NaN, defaulting to 0")
                return "0"
            if prediction == float('inf') or prediction == float('-inf'):
                logger.warning(f"Prediction is infinity, defaulting to 0")
                return "0"
        return "1" if prediction == 1 else "0"
    
    # Convert to string and clean
    pred_str = str(prediction).strip().lower()
    
    # Handle boolean-like values
    if pred_str in ("true", "1", "yes", "correct", "right", "valid", "pass"):
        return "1"
    if pred_str in ("false", "0", "no", "incorrect", "wrong", "invalid", "fail"):
        return "0"
    
    # Try numeric conversion
    try:
        num = int(float(pred_str))
        return "1" if num == 1 else "0"
    except (ValueError, TypeError):
        pass
    
    # Default to 0 for unparseable values
    logger.warning(f"Could not normalize prediction: {prediction!r}, defaulting to 0")
    return "0"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._last_msg_history: list[dict] = []
        self._total_calls = 0
        self._successful_extractions = 0
        self._confidence_scores: list[float] = []  # Track confidence for analysis

    def _build_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from input fields."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Truncate very long inputs to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        guidelines = grading_guidelines[:max_len] if len(grading_guidelines) > max_len else grading_guidelines
        answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer

        return f"""You are an expert {domain} grader evaluating student solutions for mathematical olympiad problems.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines precisely.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{guidelines}

STUDENT'S ANSWER:
{answer}

Think step by step and be thorough:
1. Analyze what the problem is asking for - identify key requirements and expected outcomes
2. Review the correct solution approach - understand the logic, methods, and final answer format
3. Compare the student's answer to the correct solution:
   - Check if the final answer matches exactly (numerical values, expressions, formats)
   - Verify if the reasoning/method is sound and leads to the correct result
   - Look for partial correctness - even if the final answer is wrong, check if the approach is valid
4. Apply the grading guidelines strictly:
   - If guidelines specify exact answer matching, verify character-by-character where needed
   - If guidelines allow for equivalent forms, check mathematical equivalence
   - Consider common acceptable variations in notation or format
5. Determine if the student's answer is correct (1) or incorrect (0):
   - Mark as 1 (correct) if the answer is fully correct OR if the core reasoning is sound and the answer is essentially correct
   - Mark as 0 (incorrect) only if the answer is clearly wrong or the reasoning is fundamentally flawed
   - When in doubt, lean toward 1 if the student demonstrates understanding

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your comparison and why you determined the answer is correct or incorrect.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Be fair and accurate in your assessment."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._total_calls += 1
        instruction = self._build_prompt(inputs)

        # Try with retries for robustness
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self._last_msg_history = msg_history
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1].get("text", "") if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    normalized = _normalize_prediction(prediction)
                    self._successful_extractions += 1
                    
                    # Compute and track confidence score
                    confidence = self._compute_confidence(extracted, response_text)
                    self._confidence_scores.append(confidence)
                    
                    self.log_fn(f"Successfully extracted prediction: {prediction} -> {normalized} (confidence: {confidence:.2f})")
                    return normalized, msg_history
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response")
                    if attempt == 0:
                        # Log first few characters of response for debugging
                        preview = response_text[:200] if response_text else "(empty)"
                        self.log_fn(f"Response preview: {preview!r}")
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call or parsing: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return "0" if all retries failed
        if last_error:
            self.log_fn(f"All retries failed with error: {last_error}")
        else:
            self.log_fn("All retries failed - could not extract valid prediction")
        
        return "0", self._last_msg_history if self._last_msg_history else []

    def _compute_confidence(self, extracted: dict | None, response_text: str) -> float:
        """Compute confidence score based on extraction quality and response characteristics.
        
        Returns a score between 0.0 and 1.0 indicating confidence in the prediction.
        """
        if extracted is None or "response" not in extracted:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence if reasoning is present and substantial
        reasoning = extracted.get("reasoning", "")
        if reasoning and len(reasoning) > 50:
            confidence += 0.2
        
        # Boost if response format is clean (JSON tags present)
        if "<json>" in response_text and "</json>" in response_text:
            confidence += 0.2
        
        # Boost if reasoning contains key analytical phrases
        analytical_phrases = [
            "correct", "incorrect", "matches", "does not match",
            "solution", "answer", "therefore", "conclusion"
        ]
        reasoning_lower = reasoning.lower()
        phrase_matches = sum(1 for phrase in analytical_phrases if phrase in reasoning_lower)
        confidence += min(0.1, phrase_matches * 0.02)
        
        return min(1.0, confidence)

    def get_stats(self) -> dict:
        """Return extraction statistics."""
        stats = {
            "total_calls": self._total_calls,
            "successful_extractions": self._successful_extractions,
            "extraction_rate": self._successful_extractions / max(1, self._total_calls),
        }
        
        # Add confidence statistics if available
        if self._confidence_scores:
            stats["avg_confidence"] = sum(self._confidence_scores) / len(self._confidence_scores)
            stats["min_confidence"] = min(self._confidence_scores)
            stats["max_confidence"] = max(self._confidence_scores)
        
        return stats
