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
    Enhanced to handle nested braces and multiple JSON objects.
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
            # Try to fix common JSON issues
            try:
                # Try removing trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
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
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: Try to find any JSON object in the text using brace matching
    if not results:
        # Find all potential JSON objects by matching braces
        start_idx = 0
        while True:
            start_idx = text.find('{', start_idx)
            if start_idx == -1:
                break
            
            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(text[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = start_idx + i
                        break
            
            if end_idx > start_idx:
                potential_json = text[start_idx:end_idx + 1]
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    try:
                        fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        pass
            start_idx += 1
    
    return results or None


def _safe_extract_prediction(extracted: list[dict] | None) -> Any:
    """Safely extract prediction from extracted JSON objects.
    
    Tries multiple common keys for the response field.
    Enhanced to handle various response formats and normalize values.
    """
    if not extracted:
        return None
    
    # Try the last extracted object first (most recent)
    for obj in reversed(extracted):
        if not isinstance(obj, dict):
            continue
        # Try common response keys in order of preference
        for key in ["response", "answer", "result", "output", "prediction", "grade", "score", "evaluation"]:
            if key in obj:
                value = obj[key]
                # Normalize the value
                if isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, str):
                    # Strip whitespace and quotes
                    value = value.strip().strip('"\'')
                    if value:
                        return value
                elif value is not None:
                    return str(value)
    
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
        # Extract key fields for better prompt construction
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        # Build a more structured and focused prompt
        instruction = f"""You are an expert mathematics grader. Your task is to evaluate a student's answer to a mathematical problem.

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== YOUR TASK ===
Evaluate the student's answer by following these steps:

1. **Analyze the Problem**: Identify the key concepts, theorems, and techniques required to solve this problem.

2. **Review the Solution**: Understand the correct approach and the expected final answer or proof structure.

3. **Study the Grading Guidelines**: Note the specific criteria for awarding points (e.g., partial credit for correct methods, deductions for errors).

4. **Evaluate the Student's Answer**:
   - Check if the student understood the problem correctly
   - Verify if their approach aligns with the correct solution
   - Identify any mathematical errors, logical gaps, or missing steps
   - Note any creative or alternative valid approaches
   - Assess the completeness and clarity of their work

5. **Assign a Score**: Based on the grading guidelines, assign the most appropriate score.

=== RESPONSE FORMAT ===
You MUST respond in valid JSON format with exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis here. Explain what the student did correctly, what errors they made, and how you applied the grading guidelines. Be specific about mathematical concepts.",
    "response": "The final score/grade here - ONLY the score value, no explanation. Examples: '7', '3/7', '0', 'Correct', 'Partial', 'Incorrect'"
}}
</json>

IMPORTANT RULES:
- The "response" field must contain ONLY the score/grade value, nothing else
- Use the exact format specified in the grading guidelines
- Be objective and consistent with the grading criteria
- If partial credit is allowed, calculate it precisely based on the guidelines"""

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
        Enhanced to handle fraction scores (e.g., "3/7") and various formats.
        """
        import re
        
        # Look for patterns like "Score: 7" or "Grade: 3/7" or "Final score: 5"
        # Enhanced to capture fraction scores like "3/7"
        score_patterns = [
            r'[Ss]core[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
            r'[Gg]rade[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
            r'[Ff]inal\s+[Ss]core[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
            r'[Rr]esponse[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
            r'[Aa]nswer[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
            r'[Ff]inal\s+[Gg]rade[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
            r'[Pp]oints?[:\s]+["\']?(\d+(?:/\d+)?)["\']?',
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Look for standalone fraction patterns like "3/7" or "5/10" (but not dates like "3/7/2024")
        fraction_pattern = r'\b(\d+/\d+)\b(?!\d)'
        match = re.search(fraction_pattern, text)
        if match:
            return match.group(1)
        
        # Look for standalone numbers that could be scores (1-2 digits, not part of larger numbers)
        # Avoid matching years or other large numbers
        number_pattern = r'\b(\d{1,2})\b'
        matches = re.findall(number_pattern, text)
        if matches:
            # Return the last single/double digit number found (often the score)
            return matches[-1]
        
        # Look for grade labels
        grade_patterns = [
            r'\b(Correct|Incorrect|Partial|Pass|Fail|Full|Zero|True|False|Yes|No)\b',
        ]
        
        for pattern in grade_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
