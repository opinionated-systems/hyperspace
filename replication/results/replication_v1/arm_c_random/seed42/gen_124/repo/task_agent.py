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
    """
    if not extracted:
        return None
    
    # Try the last extracted object first (most recent)
    for obj in reversed(extracted):
        if not isinstance(obj, dict):
            continue
        # Try common response keys
        for key in ["response", "answer", "result", "output", "prediction"]:
            if key in obj:
                return obj[key]
    
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
        # Build an improved prompt with chain-of-thought reasoning and clearer instructions
        instruction = f"""You are an expert grading agent for mathematical problems. Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```json
{json.dumps(inputs, indent=2, default=str)}
```

Follow these steps to evaluate the student's answer:

1. **Understand the Problem**: Carefully read the problem statement and identify what is being asked.

2. **Review the Solution**: Study the provided solution to understand the correct approach and expected answer.

3. **Analyze the Grading Guidelines**: Pay close attention to the grading guidelines to understand how points should be awarded.

4. **Evaluate the Student's Answer**: 
   - Compare the student's answer to the correct solution
   - Check for partial credit based on the grading guidelines
   - Identify any errors or misconceptions
   - Note any correct steps or reasoning

5. **Determine the Final Score**: Based on your analysis, assign an appropriate score or evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, including what they did correctly and incorrectly",
    "response": "The final evaluation result - either a numeric score (e.g., '7', '3', '0') or a grade label (e.g., 'Correct', 'Partial', 'Incorrect') based on the grading guidelines"
}}
</json>

Important:
- Provide thorough reasoning before giving the final response
- The response field should contain ONLY the final score/grade, not explanatory text
- Ensure your response is valid JSON with both fields populated"""

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
                    pred_value = self._extract_fallback_prediction(text_content)
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

    def _extract_fallback_prediction(self, text: str) -> Any:
        """Fallback method to extract prediction when JSON parsing fails.
        
        Looks for common patterns like scores (numbers) or grade labels.
        """
        import re
        
        # Look for patterns like "Score: 7" or "Grade: 3/7" or "Final score: 5"
        score_patterns = [
            r'[Ss]core[:\s]+(\d+)',
            r'[Gg]rade[:\s]+(\d+)',
            r'[Ff]inal\s+[Ss]core[:\s]+(\d+)',
            r'[Rr]esponse[:\s]+["\']?(\d+)["\']?',
            r'[Aa]nswer[:\s]+["\']?(\d+)["\']?',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Look for grade labels
        grade_patterns = [
            r'\b(Correct|Incorrect|Partial|Pass|Fail|Full|Zero)\b',
        ]
        
        for pattern in grade_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
