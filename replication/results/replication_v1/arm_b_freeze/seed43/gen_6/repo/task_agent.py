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
    Also attempts to extract JSON from markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> blocks
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
    
    # Fallback: Extract from markdown code blocks
    if not results:
        json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', block.strip())
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: Try to find any JSON object in the text
    if not results:
        # Look for patterns like {"key": value} or {"key": "value"}
        json_pattern = re.search(r'\{[^{}]*"[^"]+"\s*:[^}]+\}', text)
        if json_pattern:
            try:
                results.append(json.loads(json_pattern.group()))
            except json.JSONDecodeError:
                pass
    
    return results or None


def _normalize_prediction(prediction: str | int | float) -> str:
    """Normalize prediction to a consistent string format.
    
    Handles various score formats (numbers, fractions, ranges) and
    converts them to a standardized string representation.
    
    For IMO grading, preserves the X/7 format as it's the standard.
    """
    if prediction is None:
        return "None"
    
    # Convert to string and strip whitespace
    pred_str = str(prediction).strip()
    
    # Handle empty strings
    if not pred_str:
        return "None"
    
    # Handle IMO-specific fraction format: X/7
    # Keep this format as-is since it's the standard IMO scoring format
    imo_match = re.match(r'^(\d+)\s*/\s*7$', pred_str)
    if imo_match:
        score = int(imo_match.group(1))
        # Validate IMO score range (0-7)
        if 0 <= score <= 7:
            return f"{score}/7"
    
    # Normalize other fraction formats
    # e.g., "7/7" -> "1", "0/7" -> "0"
    fraction_match = re.match(r'^(\d+)\s*/\s*(\d+)$', pred_str)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator != 0:
            # Return as simplified fraction or decimal
            if numerator == denominator:
                return "1"
            elif numerator == 0:
                return "0"
            else:
                return f"{numerator}/{denominator}"
    
    # Normalize decimal scores (round to reasonable precision)
    try:
        float_val = float(pred_str)
        # If it's effectively an integer, return as int
        if float_val == int(float_val):
            return str(int(float_val))
        # Otherwise return with max 2 decimal places
        return f"{float_val:.2f}".rstrip('0').rstrip('.')
    except ValueError:
        pass
    
    # Handle range formats like "0-1" or "0 to 1"
    range_match = re.match(r'^(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)$', pred_str, re.IGNORECASE)
    if range_match:
        # Return the lower bound as the score (conservative grading)
        return range_match.group(1)
    
    return pred_str


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
        # Extract fields with defaults for safety
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Build a structured prompt with chain-of-thought reasoning
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign a score based on the official grading guidelines.

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

## Instructions

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the key steps, techniques, and results that constitute a complete correct solution.

3. **Review Grading Guidelines**: Note the specific criteria for partial credit and what constitutes a complete solution.

4. **Evaluate Student's Answer**: 
   - Check if the final answer is correct
   - Verify if the reasoning is sound and complete
   - Identify any gaps, errors, or missing steps
   - Compare against the grading guidelines for partial credit
   - **IMPORTANT**: IMO problems are typically scored out of 7 points. A complete correct solution earns 7/7.
   - Partial credit (1-6 points) is awarded for significant progress toward a solution
   - 0 points for no meaningful progress or completely incorrect answers

5. **Assign Score**: Based on your analysis, assign the appropriate score according to the grading guidelines.

## Output Format (CRITICAL)

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must contain exactly these three fields:

<json>
{{
    "analysis": "Your detailed analysis of the student's answer, including what they did correctly and any errors or omissions",
    "reasoning": "Step-by-step reasoning for your grading decision",
    "response": "The final score (a number or string as specified in the grading guidelines)"
}}
</json>

Requirements:
- The "response" field MUST contain ONLY the final score, with no additional text
- Use the exact score format specified in the grading guidelines (e.g., "7/7", "0", "1", "2/7")
- For IMO problems, common formats are: "7/7" (full credit), "0" (no credit), or "X/7" (partial credit)
- Do not include explanations in the response field - put those in the reasoning field
- Ensure the JSON is valid (no trailing commas, proper quotes, etc.)
- If no specific format is given in guidelines, use "X/7" format for IMO problems

Be thorough in your analysis but concise in your reasoning."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        extraction_error = None
        
        try:
            # Try to extract from the assistant's response
            assistant_text = msg_history[-1].get("text", "") if msg_history else ""
            extracted = _extract_jsons(assistant_text)
            
            if extracted:
                # Try to get response field, fallback to other common fields
                last_json = extracted[-1]
                
                # Priority order for score fields
                score_fields = ["response", "score", "grade", "answer", "result", "value"]
                for field in score_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        break
                else:
                    # If no recognized field, use the first string/number value
                    for val in last_json.values():
                        if isinstance(val, (str, int, float)):
                            prediction = val
                            break
                    else:
                        prediction = list(last_json.values())[0] if last_json else "None"
                
                self.log_fn(f"Extracted prediction: {prediction} (from field: {field if 'field' in dir() else 'unknown'})")
            else:
                extraction_error = "No JSON found in response"
                # Fallback: try to extract a number from the text
                number_match = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b', assistant_text)
                if number_match:
                    prediction = f"{number_match.group(1)}/{number_match.group(2)}"
                    self.log_fn(f"Fallback extraction (fraction): {prediction}")
                else:
                    number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', assistant_text)
                    if number_match:
                        prediction = number_match.group(1)
                        self.log_fn(f"Fallback extraction (number): {prediction}")
        except Exception as e:
            extraction_error = str(e)
            self.log_fn(f"Error extracting prediction: {e}")
        
        # Normalize the prediction to a consistent format
        normalized_prediction = _normalize_prediction(prediction)
        
        if extraction_error:
            self.log_fn(f"Extraction warning: {extraction_error}, normalized to: {normalized_prediction}")
        
        return normalized_prediction, msg_history
