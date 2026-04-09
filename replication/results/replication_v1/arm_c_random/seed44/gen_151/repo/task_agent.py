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


def _extract_json_with_retry(text: str, max_retries: int = 2) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with common fixes
    for attempt in range(max_retries):
        try:
            # Try to find and fix JSON with trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception:
            pass
    
    return None


def _validate_imo_score(score: int | float | str | None) -> tuple[int, float]:
    """Validate and normalize IMO score.
    
    Args:
        score: Raw score value from LLM output
        
    Returns:
        Tuple of (normalized_score, confidence)
        - normalized_score: Integer 0-7 (standard IMO scoring)
        - confidence: Float 0.0-1.0 indicating confidence in the score
    """
    if score is None:
        return 0, 0.0
    
    try:
        # Convert to number
        if isinstance(score, str):
            # Handle common text patterns
            score_str = score.strip().lower()
            # Extract number from patterns like "score: 7", "7/7", "7 points"
            match = re.search(r'(\d+(?:\.\d+)?)', score_str)
            if match:
                score_val = float(match.group(1))
            else:
                return 0, 0.0
        else:
            score_val = float(score)
        
        # IMO scores are integers 0-7
        # Round to nearest integer if fractional
        normalized = int(round(max(0, min(7, score_val))))
        
        # Calculate confidence based on:
        # 1. Whether the original value was already an integer
        # 2. Whether the value is within valid range
        confidence = 1.0
        
        # Penalize if we had to round
        if score_val != normalized:
            confidence *= 0.8
        
        # Penalize if outside valid range
        if score_val < 0 or score_val > 7:
            confidence *= 0.5
        
        # Penalize if very far from nearest integer
        if abs(score_val - round(score_val)) > 0.5:
            confidence *= 0.7
            
        return normalized, round(confidence, 2)
        
    except (ValueError, TypeError):
        return 0, 0.0


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution to a mathematical problem.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the critical steps and insights needed for a correct solution.

3. **Evaluate the Student's Answer**:
   - Check if the student understood the problem correctly
   - Verify each step of their reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or valid alternative approaches

4. **Assign a Grade**: Based on the grading guidelines, assign a numerical score. Common IMO scoring:
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5: Significant progress with some gaps
   - 3-4: Partial progress
   - 1-2: Minimal progress
   - 0: No meaningful progress or completely wrong

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed analysis of the student's work",
    "reasoning": "Step-by-step evaluation explaining your grading decision",
    "score": 7,
    "response": "7"
}}
</json>

Important guidelines:
- The "score" field must be an integer between 0 and 7 (inclusive)
- The "response" field must be the same integer as a string (e.g., "7")
- IMO scoring: 7=complete correct, 6=minor flaw, 5=significant progress, 3-4=partial, 1-2=minimal, 0=no progress
- Be precise with your scoring - partial credit should reflect actual mathematical progress made
- If the student uses a valid alternative approach not in the official solution, award appropriate credit"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        confidence = 0.0
        try:
            extracted = _extract_json_with_retry(msg_history[-1]["text"])
            if extracted:
                # Try to get the response field first
                raw_score = None
                if "response" in extracted[-1]:
                    raw_score = extracted[-1]["response"]
                # Fallback to score field if response is not present
                elif "score" in extracted[-1]:
                    raw_score = extracted[-1]["score"]
                
                # Validate and normalize the score
                if raw_score is not None:
                    normalized_score, confidence = _validate_imo_score(raw_score)
                    prediction = str(normalized_score)
                    
                    # Log confidence for monitoring
                    if confidence < 0.5:
                        self.log_fn(f"Low confidence score ({confidence}): raw={raw_score}, normalized={prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
