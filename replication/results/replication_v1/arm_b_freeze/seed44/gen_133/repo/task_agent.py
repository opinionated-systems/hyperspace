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
from collections import Counter
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
    5. Line-by-line JSON object detection
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
    
    # Strategy 5: Find any JSON object containing "response" key
    # This handles cases with nested braces by using a stack-based approach
    brace_pattern = r'\{[^{}]*"response"[^{}]*\}'
    for match in re.finditer(brace_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 6: Look for response field in plain text (last resort)
    # Extract just the response value if clearly stated
    response_patterns = [
        r'"response"\s*:\s*1\b',
        r'"response"\s*:\s*0\b',
        r'"response"\s*:\s*"1"',
        r'"response"\s*:\s*"0"',
        r'response["\']?\s*[:=]\s*1\b',
        r'response["\']?\s*[:=]\s*0\b',
    ]
    for pattern in response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if '1' in match.group(0):
                return {"response": 1}
            else:
                return {"response": 0}
    
    return None


def _normalize_prediction(prediction: Any) -> str:
    """Normalize prediction value to '0' or '1'.
    
    Handles various formats: integers, strings, booleans.
    Returns '0' as default for invalid values.
    """
    if prediction is None:
        return "0"
    
    # Convert to string and clean
    pred_str = str(prediction).strip().lower()
    
    # Handle boolean-like values
    if pred_str in ("true", "1", "yes", "correct"):
        return "1"
    if pred_str in ("false", "0", "no", "incorrect"):
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
        self.max_retries = 5  # Increased from 3 for better robustness
        self._last_msg_history: list[dict] = []
        self._confidence_threshold = 0.8  # Threshold for high-confidence predictions

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

        instruction = f"""You are an expert {domain} grader evaluating student solutions for mathematical olympiad problems.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines. Be thorough, precise, and objective in your evaluation.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. PROBLEM ANALYSIS: Identify what the problem is asking for - key mathematical concepts, required proof structure, and expected answer format
2. SOLUTION REVIEW: Understand the correct solution's logical steps, key insights, and final answer
3. STUDENT ANSWER ANALYSIS: Carefully read the student's answer, identifying their approach, reasoning, and final result
4. COMPARISON: Compare the student's answer against the correct solution:
   - Does the final answer match (numerically, algebraically, or conceptually)?
   - Is the reasoning logically sound and mathematically valid?
   - Did they use appropriate methods and theorems?
   - Are there any gaps, errors, or unjustified claims?
5. GUIDELINES CHECK: Verify compliance with grading guidelines:
   - Are all required elements present?
   - Does it meet format requirements?
   - Are partial credit conditions satisfied?
6. DECISION: Based on all above, determine if the answer is correct (1) or incorrect (0)

IMPORTANT GRADING CRITERIA:
- A student answer is CORRECT (1) if it:
  * Arrives at the correct final answer/result (even if expressed differently but equivalent)
  * Uses valid mathematical reasoning (alternative approaches are acceptable if correct)
  * Meets all mandatory requirements from the grading guidelines
  * Contains no significant mathematical errors that invalidate the solution
  * Has sufficient rigor for an olympiad-level proof

- A student answer is INCORRECT (0) if it:
  * Arrives at the wrong final answer or no answer
  * Contains critical mathematical errors that break the logic
  * Fails to meet mandatory requirements from the grading guidelines
  * Is incomplete, missing essential components, or lacks necessary justification
  * Uses invalid reasoning or incorrect theorems

EDGE CASES TO CONSIDER:
- Partial solutions: If the student made progress but didn't complete, check if partial credit applies per guidelines
- Notation differences: Different but valid notation is acceptable
- Alternative proofs: Different proof methods are valid if mathematically sound
- Trivial errors: Minor arithmetic errors may or may not matter depending on the problem

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis. Structure as: (1) Problem understanding, (2) Student's approach, (3) Key comparisons made, (4) Specific findings, (5) Final decision rationale",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Be decisive - avoid uncertain or ambiguous assessments."""

        # Try with retries for robustness, with self-consistency voting
        last_error = None
        predictions = []  # Track predictions for self-consistency
        reasoning_history = []  # Track reasoning for analysis
        
        for attempt in range(self.max_retries):
            try:
                # Add variation to prompt on retries to get diverse perspectives
                varied_instruction = instruction
                if attempt > 0:
                    varied_instruction = instruction + f"\n\n[Attempt {attempt + 1}] Please provide your evaluation with careful attention to the grading criteria."
                
                response, msg_history, info = get_response_from_llm(
                    msg=varied_instruction,
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
                    predictions.append(normalized)
                    
                    # Store reasoning if available
                    reasoning = extracted.get("reasoning", "")
                    if reasoning:
                        reasoning_history.append(reasoning[:500])  # Truncate for logging
                    
                    self.log_fn(f"Attempt {attempt + 1}: Extracted prediction: {prediction} -> {normalized}")
                    
                    # If we have consistent predictions, return early
                    if len(predictions) >= 3:
                        # Check for consensus
                        vote_counts = Counter(predictions)
                        most_common = vote_counts.most_common(1)[0]
                        if most_common[1] >= 2:  # At least 2 agree
                            self.log_fn(f"Self-consensus reached: {most_common[0]} (votes: {dict(vote_counts)})")
                            return most_common[0], msg_history
                    
                    # On first success, try to return immediately if confident
                    if attempt == 0:
                        return normalized, msg_history
                else:
                    self.log_fn(f"Attempt {attempt + 1}: No valid JSON found in response")
                    if attempt == 0:
                        # Log first few characters of response for debugging
                        preview = response_text[:300] if response_text else "(empty)"
                        self.log_fn(f"Response preview: {preview!r}")
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call or parsing: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # If we have any predictions, use majority voting
        if predictions:
            vote_counts = Counter(predictions)
            most_common = vote_counts.most_common(1)[0]
            self.log_fn(f"Using majority vote: {most_common[0]} (predictions: {predictions}, votes: {dict(vote_counts)})")
            return most_common[0], self._last_msg_history if self._last_msg_history else []
        
        # Fallback: return "0" if all retries failed
        if last_error:
            self.log_fn(f"All retries failed with error: {last_error}")
        else:
            self.log_fn("All retries failed - could not extract valid prediction")
        
        return "0", self._last_msg_history if self._last_msg_history else []
