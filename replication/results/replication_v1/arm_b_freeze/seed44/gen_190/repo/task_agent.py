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


def _extract_score_from_text(text: str) -> str | None:
    """Extract a numeric score from plain text as fallback.
    
    Looks for patterns like "score: 7", "final score: 0", "grade: 2", etc.
    """
    # Look for score patterns
    patterns = [
        r'(?:final\s+)?score[:\s]+(\d+)',
        r'(?:grade|points?)[:\s]+(\d+)',
        r'(?:score\s+of|grade\s+of)[:\s]+(\d+)',
        r'(?:assigned|given|awarded)\s+(?:a\s+)?score[:\s]+(\d+)',
        r'response[:\s]+["\']?(\d+)["\']?',
        r'^(\d+)$',  # Just a number on its own line
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1]  # Return last match (usually the final conclusion)
    
    return None


def _validate_score(score: str, grading_guidelines: str = "") -> tuple[str, bool]:
    """Validate and normalize the extracted score.
    
    Returns (normalized_score, is_valid).
    """
    if not score or score == "None":
        return "0", False
    
    # Clean the score string
    score = str(score).strip().strip('"\'')
    
    # Try to extract just the numeric part
    numeric_match = re.search(r'(\d+)', score)
    if not numeric_match:
        return "0", False
    
    score_val = numeric_match.group(1)
    
    # IMO scores are typically 0-7, but could be other ranges
    # Check if score is a reasonable number
    try:
        val = int(score_val)
        if val < 0:
            return "0", False
        if val > 100:  # Unreasonably high
            return str(min(val, 7)), False
        return str(val), True
    except ValueError:
        return "0", False


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate score based on the official solution and grading guidelines.

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

Follow this structured evaluation process:

### Step 1: Problem Understanding
- Identify the key mathematical concepts and techniques required
- Note the critical steps in the official solution
- Understand the scoring rubric from the grading guidelines
- Identify what constitutes a complete, correct solution

### Step 2: Student Answer Analysis
- Check if the student stated the final answer correctly
- Identify which solution steps the student completed correctly
- Note any missing, incorrect, or incomplete steps
- Evaluate the logical flow and mathematical rigor
- Check for computational errors vs conceptual errors

### Step 3: Partial Credit Assessment
- Award points for each correct and complete step
- Award partial credit for partially correct steps
- Deduct points for logical gaps or significant errors
- Consider alternative valid approaches - they may be different but equally correct
- Be generous with partial credit when reasoning is sound, even if not fully rigorous

### Step 4: Final Score Determination
- Sum the points earned across all steps
- Verify against the grading guidelines
- Ensure consistency with the official scoring rubric
- The final score MUST be a non-negative integer

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis following the steps above. Include: (1) key concepts identified, (2) steps completed by student with specific point awards, (3) partial credit breakdown, (4) justification for final score",
    "response": "The final score as a single non-negative integer (e.g., '0', '1', '2', '7')"
}}
</json>

IMPORTANT: The "response" field must contain ONLY a numeric score (like "7" or "3"), not text or explanations.

Be thorough in your reasoning, generous with partial credit for correct reasoning, and precise in your final scoring."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction with multiple fallback strategies
        prediction = "0"  # Default to 0 instead of "None"
        extraction_method = "default"
        
        try:
            # Strategy 1: Try to extract from JSON blocks
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_extract = extracted[-1]
                if "response" in last_extract:
                    prediction = str(last_extract["response"])
                    extraction_method = "json_response_field"
                elif "score" in last_extract:
                    prediction = str(last_extract["score"])
                    extraction_method = "json_score_field"
                elif "answer" in last_extract:
                    prediction = str(last_extract["answer"])
                    extraction_method = "json_answer_field"
                elif last_extract:
                    # Try to find any numeric-looking value
                    for key, value in last_extract.items():
                        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                            prediction = str(value)
                            extraction_method = f"json_field_{key}"
                            break
                    else:
                        prediction = str(list(last_extract.values())[0])
                        extraction_method = "json_first_value"
            else:
                # Strategy 2: Try to extract from text patterns
                text_score = _extract_score_from_text(msg_history[-1]["text"])
                if text_score:
                    prediction = text_score
                    extraction_method = "text_pattern"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Strategy 3: Last resort - try text extraction on raw response
            try:
                text_score = _extract_score_from_text(response)
                if text_score:
                    prediction = text_score
                    extraction_method = "text_pattern_fallback"
            except:
                pass

        # Validate and normalize the score
        normalized_score, is_valid = _validate_score(prediction, grading_guidelines)
        
        if not is_valid:
            self.log_fn(f"Score validation failed for '{prediction}' (method: {extraction_method}), normalized to '{normalized_score}'")
        else:
            self.log_fn(f"Score extracted: '{normalized_score}' (method: {extraction_method})")

        return str(normalized_score), msg_history
