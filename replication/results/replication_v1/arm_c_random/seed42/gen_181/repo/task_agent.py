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
    Also attempts to extract JSON from markdown code blocks as fallback.
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> tags
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
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Fallback: Extract from markdown code blocks ```json ... ```
    if not results:
        pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from markdown block: {e}")
                continue
    
    # Last resort: Try to find any JSON object in the text
    if not results:
        # Look for content between first { and last }
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json = text[start_idx:end_idx + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError:
            pass
    
    return results or None


def _safe_extract_prediction(extracted: list[dict] | None) -> Any:
    """Safely extract prediction from extracted JSON objects.
    
    Tries multiple common keys for the response field.
    Also attempts to extract numeric values from string responses.
    """
    if not extracted:
        return None
    
    # Try the last extracted object first (most recent)
    for obj in reversed(extracted):
        if not isinstance(obj, dict):
            continue
        # Try common response keys
        for key in ["response", "answer", "result", "output", "prediction", "grade", "score", "points"]:
            if key in obj:
                value = obj[key]
                # If it's a string that looks like a number, try to extract it
                if isinstance(value, str):
                    # Try to extract just the numeric part
                    numeric_match = re.search(r'\b(\d+(?:\.\d+)?)\b', value.strip())
                    if numeric_match:
                        try:
                            num_val = float(numeric_match.group(1))
                            # Return as int if it's a whole number
                            if num_val == int(num_val):
                                return int(num_val)
                            return num_val
                        except ValueError:
                            pass
                    return value.strip()
                return value
    
    return None


def _extract_numeric_grade(text: str) -> int | None:
    """Extract a numeric grade from text.
    
    Looks for patterns like "Score: 7", "Grade: 3", "7/10", etc.
    Returns the numeric value as an integer, or None if not found.
    """
    # Pattern: Score/Grade/Mark followed by number
    patterns = [
        r'[Ss]core\s*[:=]\s*(\d+)',
        r'[Gg]rade\s*[:=]\s*(\d+)',
        r'[Mm]ark\s*[:=]\s*(\d+)',
        r'[Pp]oints?\s*[:=]\s*(\d+)',
        r'\b(\d+)\s*/\s*\d+\s*(?:points?)?',
        r'(?:^|\s)(\d+)(?:\s*$|\s+points?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # Look for standalone numbers that could be grades (0-10 range)
    numbers = re.findall(r'\b([0-9]|10)\b', text)
    if numbers:
        # Return the last number found (often the final score)
        return int(numbers[-1])
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build an improved prompt with chain-of-thought reasoning, few-shot examples, and clearer instructions
        instruction = f"""You are an expert grading agent for mathematical problems. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

## Task Input:
```json
{json.dumps(inputs, indent=2, default=str)}
```

## Grading Instructions:

Follow these steps carefully to evaluate the student's answer:

1. **Understand the Problem**: Read the problem statement carefully. Identify what is being asked, the key concepts involved, and the expected solution approach.

2. **Review the Solution**: Study the provided solution thoroughly. Understand the correct approach, key steps, and final answer.

3. **Analyze the Grading Guidelines**: Pay close attention to the grading guidelines. They specify how points should be awarded for:
   - Complete correct solutions
   - Partial credit for correct approaches with minor errors
   - Partial credit for significant progress
   - No credit for incorrect or irrelevant answers

4. **Evaluate the Student's Answer**: 
   - Compare the student's answer to the correct solution step-by-step
   - Identify what the student did correctly
   - Identify errors, misconceptions, or missing steps
   - Check if the final answer matches the expected answer
   - Consider partial credit based on the grading guidelines

5. **Determine the Final Score**: Assign a score that accurately reflects the student's work based on the grading guidelines.

## Output Format:

You MUST respond in valid JSON format with exactly these two fields:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis. Explain what the student did correctly and incorrectly. Reference specific parts of their answer and compare to the solution. Justify why you are assigning the particular score.",
    "response": "The final score or grade. This should be ONLY a numeric score (e.g., '7', '3', '0') or a grade label (e.g., 'Correct', 'Partial', 'Incorrect') based on the grading guidelines. No explanation here."
}}
</json>

## Examples:

Example 1 - Full credit:
<json>
{{
    "reasoning": "The student correctly identified the problem as a quadratic equation. They applied the quadratic formula accurately: x = (-b ± √(b²-4ac))/2a. They calculated the discriminant correctly as 25 - 24 = 1. They found both roots: x = (5+1)/2 = 3 and x = (5-1)/2 = 2. The final answer matches the solution exactly.",
    "response": "7"
}}
</json>

Example 2 - Partial credit:
<json>
{{
    "reasoning": "The student set up the equation correctly and identified the quadratic formula, but made an arithmetic error when calculating the discriminant (used 16 instead of 25-24=1). However, they showed correct methodology and would have found the right answer with correct arithmetic. According to the grading guidelines, this deserves partial credit.",
    "response": "4"
}}
</json>

Example 3 - No credit:
<json>
{{
    "reasoning": "The student completely misunderstood the problem. They attempted to use linear equation methods on a quadratic problem and arrived at an answer that doesn't satisfy the original equation. No meaningful progress was made toward the solution.",
    "response": "0"
}}
</json>

## Important Rules:
- Provide thorough reasoning in the "reasoning" field
- The "response" field must contain ONLY the final score/grade, no explanatory text
- Ensure your output is valid JSON
- Follow the grading guidelines precisely for awarding partial credit
- Be consistent and fair in your evaluation"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                extracted = _extract_jsons(text_content)
                pred_value = _safe_extract_prediction(extracted)
                
                if pred_value is not None:
                    prediction = pred_value
                    self.log_fn(f"Successfully extracted prediction: {repr(str(prediction)[:100])}")
                else:
                    # Fallback: try to extract any numeric value or grade from the text
                    pred_value = _extract_numeric_grade(text_content)
                    if pred_value is not None:
                        prediction = pred_value
                        self.log_fn(f"Fallback extraction successful: {repr(str(prediction)[:100])}")
                    else:
                        self.log_fn(f"No valid prediction found in response. Raw text: {repr(text_content[:200])}")
            else:
                self.log_fn("Empty message history received")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
            # Log the raw response for debugging
            if msg_history:
                self.log_fn(f"Raw last message: {repr(msg_history[-1].get('text', 'N/A')[:200])}")

        return str(prediction), msg_history


