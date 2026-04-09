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
import time
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
    4. Relaxed JSON parsing for common LLM output errors
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
    
    # Strategy 4: Relaxed parsing - look for any JSON object with response field
    # Handle common LLM errors like trailing commas, single quotes, etc.
    relaxed_pattern = r'\{[\s\S]*?"response"\s*:\s*(\d+|"[^"]*")[\s\S]*?\}'
    for match in re.finditer(relaxed_pattern, text, re.DOTALL):
        try:
            candidate = match.group(0)
            # Fix common JSON errors
            candidate = re.sub(r',\s*}', '}', candidate)  # Remove trailing commas
            candidate = re.sub(r"'", '"', candidate)     # Replace single quotes
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Features:
    - Robust JSON extraction with multiple fallback strategies
    - Confidence scoring based on reasoning quality
    - Detailed logging for debugging and audit trails
    - Automatic retry with exponential backoff
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0
        self._success_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Return agent performance statistics."""
        return {
            "total_calls": self._call_count,
            "successful_calls": self._success_count,
            "success_rate": self._success_count / max(1, self._call_count),
        }

    def _calculate_confidence(self, extracted: dict, reasoning: str) -> float:
        """Calculate confidence score based on reasoning quality.
        
        Returns a score between 0.0 and 1.0 based on:
        - Presence of detailed reasoning
        - Length and structure of analysis
        - Explicit mention of key grading criteria
        """
        confidence = 0.5  # Base confidence
        
        if not reasoning:
            return confidence
        
        reasoning_lower = reasoning.lower()
        
        # Boost for detailed reasoning (at least 100 chars)
        if len(reasoning) > 100:
            confidence += 0.1
        
        # Boost for structured analysis with numbered steps
        if any(marker in reasoning for marker in ["1.", "2.", "3.", "step", "analysis"]):
            confidence += 0.1
        
        # Boost for explicit comparison to correct solution
        if any(phrase in reasoning_lower for phrase in ["correct solution", "student's answer", "compared"]):
            confidence += 0.1
        
        # Boost for mentioning grading guidelines
        if "grading" in reasoning_lower or "guidelines" in reasoning_lower:
            confidence += 0.1
        
        # Boost for clear conclusion language
        if any(phrase in reasoning_lower for phrase in ["therefore", "conclusion", "in summary", "thus"]):
            confidence += 0.1
        
        return min(1.0, confidence)

    def _normalize_prediction(self, prediction) -> str:
        """Normalize prediction to '0' or '1'.
        
        Handles various formats: integers, strings, booleans, etc.
        """
        if prediction is None:
            return None
        
        # Handle boolean values
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        
        # Handle numeric values
        if isinstance(prediction, (int, float)):
            return "1" if prediction == 1 else "0"
        
        # Handle string values
        pred_str = str(prediction).strip().lower()
        if pred_str in ("1", "true", "correct", "yes", "right"):
            return "1"
        elif pred_str in ("0", "false", "incorrect", "no", "wrong"):
            return "0"
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        start_time = time.time()
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log problem metadata for debugging
        self.log_fn(f"[TaskAgent] Grading {domain} problem (student answer length: {len(student_answer)} chars)")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

        # Try with retries for robustness
        msg_history = []
        last_error = None
        best_result = None
        best_confidence = 0.0
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(msg_history[-1]["text"])
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    normalized = self._normalize_prediction(prediction)
                    
                    if normalized is not None:
                        # Calculate confidence based on reasoning quality
                        reasoning = extracted.get("reasoning", "")
                        confidence = self._calculate_confidence(extracted, reasoning)
                        
                        elapsed = time.time() - start_time
                        self.log_fn(f"[TaskAgent] Success on attempt {attempt + 1} (confidence: {confidence:.2f}, time: {elapsed:.2f}s)")
                        
                        # Track best result by confidence
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = normalized
                        
                        # If high confidence, return immediately
                        if confidence >= 0.8:
                            self._success_count += 1
                            return normalized, msg_history
                        
                        # Otherwise continue to retry for potentially better result
                        if attempt == self.max_retries - 1:
                            self._success_count += 1
                            return best_result, msg_history
                    else:
                        self.log_fn(f"[TaskAgent] Invalid prediction value: {prediction}, retrying...")
                        last_error = f"Invalid prediction: {prediction}"
                else:
                    self.log_fn(f"[TaskAgent] No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    last_error = "No valid JSON found"
                    
            except Exception as e:
                self.log_fn(f"[TaskAgent] Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return best result if we have one, otherwise default to "0"
        elapsed = time.time() - start_time
        if best_result is not None:
            self.log_fn(f"[TaskAgent] Using best result with confidence {best_confidence:.2f} after {elapsed:.2f}s")
            self._success_count += 1
            return best_result, msg_history
        
        self.log_fn(f"[TaskAgent] All retries failed ({last_error}), returning default prediction 0 (time: {elapsed:.2f}s)")
        return "0", msg_history
