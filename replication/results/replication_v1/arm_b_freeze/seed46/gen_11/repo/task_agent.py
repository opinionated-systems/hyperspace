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
    Also attempts to extract raw JSON objects as a fallback.
    """
    results = []
    search_from = 0
    
    # First try to find JSON in <json> tags
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
            # Try to clean up common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find raw JSON objects (for cases without tags)
    if not results:
        # Look for JSON-like structures with curly braces
        json_pattern = re.compile(r'\{[^{}]*"[^"]+"[^{}]*\}', re.DOTALL)
        for match in json_pattern.finditer(text):
            try:
                candidate = match.group()
                # Ensure it has required fields
                if '"reasoning"' in candidate or '"score"' in candidate:
                    results.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _validate_score(score: any) -> str:
    """Validate and normalize the score value.
    
    Args:
        score: The score value from JSON extraction
        
    Returns:
        Validated score as string, or "None" if invalid
    """
    if score is None:
        return "None"
    
    # Handle numeric types
    if isinstance(score, (int, float)):
        return str(score)
    
    # Handle string scores
    if isinstance(score, str):
        score = score.strip()
        
        # Check for common non-numeric responses first
        if score.lower() in ['none', 'null', 'nan', '', 'n/a', 'undefined']:
            return "None"
        
        # Try to extract numeric portion (handles cases like "7 points", "3/7", "~5")
        # Look for decimal number pattern
        numeric_match = re.search(r'[-+]?\d+\.?\d*', score)
        if numeric_match:
            extracted = numeric_match.group()
            # Validate it's a reasonable number
            try:
                val = float(extracted)
                # IMO scores are typically 0-7, but be flexible
                if -100 <= val <= 100:
                    return extracted
            except ValueError:
                pass
        
        # Check for fraction patterns like "3/7" - extract numerator
        fraction_match = re.search(r'(\d+)\s*/\s*\d+', score)
        if fraction_match:
            return fraction_match.group(1)
        
        return "None"
    
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

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

2. **Analyze the Official Solution**: Identify the critical steps, key insights, and the final answer in the official solution.

3. **Review Grading Guidelines**: Note the specific criteria and point allocation scheme.

4. **Evaluate Student's Answer**: 
   - Check if the final answer matches the official solution
   - Assess the reasoning and proof structure
   - Identify any gaps, errors, or creative valid approaches
   - Compare against the grading guidelines

5. **Determine Score**: Based on your analysis, assign a numerical score that reflects the student's work.

**IMPORTANT SCORING INSTRUCTIONS:**
- The score MUST be a single number (integer or decimal)
- Do NOT include units, explanations, or text in the score field
- Examples of valid scores: "7", "3.5", "0", "2"
- Examples of INVALID scores: "7 points", "3/7", "approximately 5", "N/A"

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering steps 1-4 above",
    "score": "NUMERICAL_SCORE_ONLY",
    "response": "NUMERICAL_SCORE_ONLY"
}}
</json>

Be thorough in your reasoning and fair in your grading."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                result = extracted[-1]
                # Prefer "response" field, fallback to "score" field
                if "response" in result:
                    prediction = _validate_score(result["response"])
                elif "score" in result:
                    prediction = _validate_score(result["score"])
                else:
                    # If neither field exists, log the available fields
                    self.log_fn(f"Warning: No 'response' or 'score' field found. Available fields: {list(result.keys())}")
            else:
                self.log_fn("Warning: No JSON found in response")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
