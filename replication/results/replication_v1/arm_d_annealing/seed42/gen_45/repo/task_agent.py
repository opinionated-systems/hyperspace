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
    Also handles markdown code blocks with json and inline JSON objects.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
                search_from = end + 3
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
                search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = _clean_json_string(inner)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
            break  # Only process first valid JSON block
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
            # Use a more robust approach: find all potential JSON objects
            brace_start = text.find('{')
            while brace_start != -1:
                # Try to find matching closing brace
                brace_count = 0
                for i, char in enumerate(text[brace_start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[brace_start:brace_start + i + 1]
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    results.append(parsed)
                            except json.JSONDecodeError:
                                # Try to clean up common JSON issues
                                try:
                                    cleaned = _clean_json_string(json_str)
                                    parsed = json.loads(cleaned)
                                    if isinstance(parsed, dict):
                                        results.append(parsed)
                                except json.JSONDecodeError:
                                    pass
                            break
                brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
    return results or None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix single quotes used as JSON delimiters (but not within strings)
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    # Remove comments
    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    return cleaned


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem. Follow these steps carefully:

1. **Understand the Problem**: Read the problem statement carefully and identify what is being asked.

2. **Review the Official Solution**: Study the provided official solution to understand the expected approach and key insights.

3. **Analyze the Student's Answer**: Examine the student's solution step by step, checking:
   - Mathematical correctness of each step
   - Logical soundness and justification
   - Completeness (does it address all parts of the problem?)
   - Clarity of presentation

4. **Apply Grading Guidelines**: Use the provided grading guidelines to assess the student's work objectively.

5. **Provide Your Evaluation**: Give a clear, definitive assessment.

---

**Domain**: {domain}

**Problem**:
```
{problem}
```

**Official Solution**:
```
{solution}
```

**Grading Guidelines**:
```
{grading_guidelines}
```

**Student's Answer**:
```
{student_answer}
```

---

**Your Task**:

First, think through your evaluation step by step (chain-of-thought reasoning). Consider:
- Does the student's approach align with the official solution?
- Are the mathematical steps correct?
- Is the logic sound and well-justified?
- Does the student address all parts of the problem?
- What score would you assign based on the grading guidelines?

Then, provide your final assessment in the following JSON format. Be precise and concise:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation and reasoning process",
    "assessment": "Exactly one of: 'Correct', 'Partially correct', or 'Incorrect'",
    "response": "Exactly one of: 'Correct', 'Partially correct', or 'Incorrect' - must match the assessment field"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- Both "assessment" and "response" fields must contain EXACTLY one of these three values: 'Correct', 'Partially correct', or 'Incorrect'.
- Do not add any other text or explanation in these fields - only the exact label.
- The evaluation system extracts these fields to determine the grade, so they must be exact."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = _extract_prediction(msg_history)
        
        return prediction, msg_history


def _extract_prediction(msg_history: list[dict]) -> str:
    """Extract and normalize prediction from message history.
    
    Args:
        msg_history: List of message dictionaries from LLM conversation
        
    Returns:
        Normalized prediction string ("Correct", "Partially correct", or "Incorrect")
    """
    prediction = "None"
    
    try:
        if msg_history and len(msg_history) > 0:
            # Handle both "text" (paper format) and "content" (OpenAI format) fields
            last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
            
            # Try to extract JSON first
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                # Try assessment field first (contains categorical label)
                if "assessment" in last_json:
                    prediction = last_json["assessment"]
                # Fallback: try response field if assessment is not available
                elif "response" in last_json:
                    prediction = last_json["response"]
                # Fallback: try reasoning field if neither is available
                elif "reasoning" in last_json:
                    prediction = last_json["reasoning"]
                # Fallback: try any string field
                else:
                    for key, value in last_json.items():
                        if isinstance(value, str) and value.strip():
                            prediction = value
                            break
            else:
                # If no JSON found, try to extract the last non-empty line as a fallback
                lines = last_message.strip().split('\n')
                for line in reversed(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith('<') and not stripped.startswith('{') and not stripped.startswith('`'):
                        prediction = stripped
                        break
                
                # If still no prediction, try to find any text after common markers
                if prediction == "None":
                    markers = ["Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:", 
                               "Final Answer:", "Conclusion:", "Verdict:", "Decision:",
                               "Evaluation:", "Judgment:", "Rating:", "Status:"]
                    for marker in markers:
                        if marker in last_message:
                            parts = last_message.split(marker, 1)
                            if len(parts) > 1:
                                candidate = parts[1].strip().split('\n')[0].strip()
                                if candidate and not candidate.startswith('{'):
                                    prediction = candidate
                                    break
    except Exception as e:
        logger.warning(f"Error extracting prediction: {e}")
    
    # Normalize prediction to expected format
    return _normalize_prediction(prediction)


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the expected assessment values.
    
    Args:
        prediction: Raw prediction string
        
    Returns:
        Normalized prediction: "Correct", "Partially correct", or "Incorrect"
    """
    prediction = str(prediction).strip()
    prediction_lower = prediction.lower()
    
    # First check for exact matches (case-insensitive)
    exact_matches = {
        "correct": "Correct",
        "partially correct": "Partially correct",
        "incorrect": "Incorrect",
    }
    if prediction_lower in exact_matches:
        return exact_matches[prediction_lower]
    
    # Check for "partially correct" first (before "correct" and "incorrect")
    # to avoid misclassifying it as either
    if "partially correct" in prediction_lower:
        return "Partially correct"
    
    # Check for "incorrect" variants BEFORE "correct" variants
    # to avoid misclassifying "not correct" or "incorrect" as "correct"
    incorrect_variants = [
        "incorrect", "wrong", "false", "no", "fail", "rejected", 
        "zero", "0", "invalid", "erroneous", "unsound", "flawed",
        "incorrect solution", "wrong answer", "not correct"
    ]
    for variant in incorrect_variants:
        if variant in prediction_lower:
            return "Incorrect"
    
    # Check for "partial" variants (but not "partially correct" which we already checked)
    partial_variants = [
        "partial credit", "partially right", "incomplete", "mostly correct", 
        "some correct", "partial solution", "partially valid", "partial success", 
        "half correct", "partial marks"
    ]
    for variant in partial_variants:
        if variant in prediction_lower:
            return "Partially correct"
    
    # Check for "correct" variants (but not "partially correct" or "incorrect")
    correct_variants = [
        "correct", "right", "true", "yes", "pass", "accepted", 
        "full marks", "full credit", "valid", "accurate", "proper",
        "sound", "complete solution", "correct solution"
    ]
    for variant in correct_variants:
        if variant in prediction_lower:
            return "Correct"
    
    # If no match found, return the original (may need manual review)
    return prediction
