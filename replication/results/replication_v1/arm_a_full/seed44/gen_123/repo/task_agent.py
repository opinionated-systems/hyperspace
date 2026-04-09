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
    Also handles nested JSON objects within the response field.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON issues
        # Handle cases where response contains nested braces that weren't properly closed
        try:
            # Find the "response" field and try to extract it
            response_match = re.search(r'"response"\s*:\s*(.+?)(?:,\s*"|$)', inner, re.DOTALL)
            if response_match:
                response_val = response_match.group(1).strip()
                # Try to parse the response value
                try:
                    parsed_val = json.loads(response_val)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as string
                    parsed_val = response_val.strip('"')
                results.append({"response": parsed_val})
                continue
        except Exception:
            pass
        
        # Try extracting just the numeric value if present
        try:
            numeric_match = re.search(r'"response"\s*:\s*(\d+(?:\.\d+)?)', inner)
            if numeric_match:
                num_val = numeric_match.group(1)
                # Try int first, then float
                try:
                    parsed_num = int(num_val)
                except ValueError:
                    parsed_num = float(num_val)
                results.append({"response": parsed_num})
                continue
        except Exception:
            pass
            
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed outputs."""
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to extract response field from malformed JSON
            try:
                response_match = re.search(r'"response"\s*:\s*(\d+(?:\.\d+)?|"[^"]*"|\[[^\]]*\])', match, re.DOTALL)
                if response_match:
                    val_str = response_match.group(1)
                    # Try to parse the value
                    try:
                        val = json.loads(val_str)
                    except json.JSONDecodeError:
                        val = val_str.strip('"')
                    results.append({"response": val})
            except Exception:
                pass
            continue
    
    # Try to find JSON objects with curly braces - improved pattern for nested content
    if not results:
        # Look for response field with various value types
        response_patterns = [
            r'"response"\s*:\s*(\d+(?:\.\d+)?)',  # numeric
            r'"response"\s*:\s*"([^"]*)"',  # string
            r'"response"\s*:\s*(\[[^\]]*\])',  # array
        ]
        for pattern in response_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Try to parse as number first
                    try:
                        val = int(match)
                    except ValueError:
                        try:
                            val = float(match)
                        except ValueError:
                            val = match
                    results.append({"response": val})
                except Exception:
                    continue
            if results:
                break
    
    # Last resort: look for standalone numbers that might be scores
    if not results:
        # Look for patterns like "Score: 7" or "Final score: 7" or just "7"
        score_patterns = [
            r'[Ss]core\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'[Ff]inal\s*(?:score|grade)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'[Gg]rade\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'[Rr]esponse\s*[:=]\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in score_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    val = int(match)
                except ValueError:
                    val = float(match)
                results.append({"response": val})
            if results:
                break
    
    return results or None


def _validate_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction to ensure it's a valid score.
    
    Args:
        prediction: The extracted prediction value
        grading_guidelines: The grading guidelines to infer valid score range
        
    Returns:
        Validated prediction string
    """
    if prediction is None or prediction == "None":
        return "None"
    
    # Convert to string if not already
    pred_str = str(prediction).strip()
    
    # Try to extract numeric value
    try:
        # Direct numeric conversion
        if isinstance(prediction, (int, float)):
            return str(prediction)
        
        # Try parsing as number
        numeric_val = float(pred_str)
        # Check if it's effectively an integer
        if numeric_val == int(numeric_val):
            return str(int(numeric_val))
        return str(numeric_val)
    except (ValueError, TypeError):
        pass
    
    # If it's a string that looks like a valid grade (e.g., "Correct", "Incorrect", "Partial")
    valid_grade_keywords = ["correct", "incorrect", "partial", "full", "zero", "none", "pass", "fail"]
    pred_lower = pred_str.lower()
    for keyword in valid_grade_keywords:
        if keyword in pred_lower:
            return pred_str
    
    # If we can't validate it, return as-is but log a warning
    return pred_str


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
        # Extract key fields for better prompt construction
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

Your task is to grade a student's answer to an IMO-level mathematics problem with precision and consistency. You must carefully analyze all provided materials before making any grading decision.

PROBLEM DOMAIN: {domain}

PROBLEM STATEMENT:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES (RUBRIC):
```
{grading_guidelines}
```

STUDENT'S ANSWER:
```
{student_answer}
```

GRADING PROTOCOL - Follow these steps in order:

STEP 1 - PROBLEM ANALYSIS:
- Identify the key mathematical concepts and techniques required
- Note the critical steps that must be present for a complete solution
- Understand what constitutes a correct proof/answer

STEP 2 - SOLUTION MAPPING:
- Break down the official solution into logical steps
- Identify which steps are essential vs. optional
- Note alternative valid approaches that could also solve the problem

STEP 3 - RUBRIC INTERPRETATION:
- Parse the grading guidelines carefully
- Identify point allocations for each component
- Note partial credit conditions explicitly

STEP 4 - STUDENT WORK ANALYSIS:
- Read the student's answer completely before judging
- Map their solution steps against the official solution
- Check for:
  * Correct problem interpretation
  * Valid approach selection
  * Mathematical accuracy in calculations
  * Logical flow and rigor in proofs
  * Completeness of the solution
- Identify any errors: conceptual, computational, or logical
- Note any creative or alternative valid approaches

STEP 5 - SCORE DETERMINATION:
- Start from zero and award points for correct elements
- OR start from full credit and deduct for errors (per rubric)
- Ensure the score aligns precisely with the grading guidelines
- Double-check: would another expert grader reach the same conclusion?

STEP 6 - FINAL VERIFICATION:
- Verify your score matches the rubric's specifications exactly
- Confirm you haven't missed any partial credit opportunities
- Ensure numerical scores are formatted as numbers, not strings

CRITICAL OUTPUT INSTRUCTIONS:
You MUST respond ONLY in the following JSON format. Do not include any other text, explanations, or markdown outside the JSON block:

<json>
{{
    "response": <your numerical score or evaluation>
}}
</json>

The "response" field must contain ONLY your final grading decision:
- If the rubric specifies numeric points, provide a number (e.g., 7, 3.5, 0)
- If the rubric uses categorical grades, use the exact string specified (e.g., "Correct", "Incorrect", "Partial")
- Do NOT include explanations, reasoning, or additional text in the response field
- Do NOT wrap the number in quotes - use bare numbers for numeric scores

Examples of CORRECT responses:
<json>
{{"response": 7}}
</json>

<json>
{{"response": 3.5}}
</json>

<json>
{{"response": "Correct"}}
</json>

Examples of INCORRECT responses:
<json>
{{"response": "7 points"}}  <!-- Don't include text with numbers -->
</json>

<json>
{{"response": "The answer is correct"}}  <!-- Don't include explanations -->
</json>

Be precise and objective in your assessment."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        raw_response_text = msg_history[-1]["text"]
        extraction_attempts = [
            lambda: _extract_jsons(raw_response_text),
            lambda: _extract_json_with_regex(raw_response_text),
        ]
        
        for attempt_fn in extraction_attempts:
            try:
                extracted = attempt_fn()
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    break
            except Exception as e:
                self.log_fn(f"Extraction attempt failed: {e}")
                continue
        
        # Validate and normalize the prediction
        grading_guidelines = inputs.get("grading_guidelines", "")
        validated_prediction = _validate_prediction(prediction, grading_guidelines)
        
        if validated_prediction == "None":
            self.log_fn(f"Failed to extract valid prediction from response. Raw text: {raw_response_text[:500]}")
        elif validated_prediction != str(prediction):
            self.log_fn(f"Prediction normalized: {prediction} -> {validated_prediction}")

        return validated_prediction, msg_history
