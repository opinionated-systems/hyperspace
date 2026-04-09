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


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Fix single quotes used as JSON delimiters
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    # Fix unescaped newlines in strings
    cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
    # Fix double-escaped quotes
    cleaned = re.sub(r'\\\\"', r'\\"', cleaned)
    # Fix unescaped control characters
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', lambda m: f'\\u{ord(m.group(0)):04x}', cleaned)
    return cleaned


def _try_parse_json(json_str: str) -> dict | None:
    """Try to parse a JSON string, with cleanup on failure.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            cleaned = _clean_json_string(json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json and inline JSON objects.
    
    Args:
        text: Text to extract JSON from
        
    Returns:
        List of parsed JSON dicts or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None
        
    results = []
    max_iterations = 100
    
    # First, try to find <json>...</json> blocks
    search_from = 0
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        # Skip nested <json> tags
        next_start = text.find("<json>", start + 6)
        if next_start != -1 and next_start < end:
            search_from = start + 6
            continue
        inner = text[start + 6:end].strip()
        search_from = end + 7
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Try markdown json blocks ```json...```
    if not results:
        search_from = 0
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            start = text.find("```json", search_from)
            if start == -1:
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
            parsed = _try_parse_json(inner)
            if parsed:
                results.append(parsed)
                break  # Only process first valid JSON block
            search_from = end + 3 if start == -1 or text.find("```json", search_from) == -1 else end + 3
    
    # Try inline JSON objects as last resort
    if not results:
        brace_start = text.find('{')
        brace_searches = 0
        max_brace_searches = 50
        while brace_start != -1 and brace_searches < max_brace_searches:
            brace_searches += 1
            brace_count = 0
            in_string = False
            escape_next = False
            for i, char in enumerate(text[brace_start:]):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[brace_start:brace_start + i + 1]
                            parsed = _try_parse_json(json_str)
                            if parsed and isinstance(parsed, dict):
                                results.append(parsed)
                            break
            brace_start = text.find('{', brace_start + 1)
    
    return results if results else None


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
    "assessment": "A clear summary: 'Correct', 'Partially correct', or 'Incorrect'",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field should be one of: 'Correct', 'Partially correct', or 'Incorrect'.
- The "response" field should contain the final answer that will be used for evaluation."""

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
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_json = extracted[-1]
                    # Try assessment field first (contains categorical label like "Correct", "Partially correct", "Incorrect")
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
                                     "Final Answer:", "Conclusion:", "Verdict:", "Decision:"]
                        for marker in markers:
                            if marker in last_message:
                                parts = last_message.split(marker, 1)
                                if len(parts) > 1:
                                    candidate = parts[1].strip().split('\n')[0].strip()
                                    if candidate and not candidate.startswith('{'):
                                        prediction = candidate
                                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Normalize prediction to expected format
        prediction = str(prediction).strip()
        
        # Normalize common variations of assessment values
        prediction_lower = prediction.lower()
        
        # Expanded list of correct variations
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted", 
            "full marks", "full credit", "perfect", "excellent", 
            "valid", "accurate", "proper", "appropriate", "sound",
            "well done", "good", "satisfactory", "complete"
        ]
        
        # Expanded list of partially correct variations
        partial_variations = [
            "partially correct", "partial", "partial credit", 
            "partially right", "incomplete", "mostly correct",
            "some correct", "partial success", "half correct",
            "partially valid", "partially accurate", "partially sound",
            "needs improvement", "minor errors", "some errors"
        ]
        
        # Expanded list of incorrect variations
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected", 
            "zero", "0", "invalid", "inaccurate", "unsatisfactory",
            "poor", "inadequate", "deficient", "flawed", "erroneous",
            "not correct", "not right", "not valid", "not accurate"
        ]
        
        # Check for exact matches first
        if prediction_lower in correct_variations:
            prediction = "Correct"
        elif prediction_lower in partial_variations:
            prediction = "Partially correct"
        elif prediction_lower in incorrect_variations:
            prediction = "Incorrect"
        else:
            # Check for partial matches (contains keywords)
            if any(var in prediction_lower for var in correct_variations):
                prediction = "Correct"
            elif any(var in prediction_lower for var in partial_variations):
                prediction = "Partially correct"
            elif any(var in prediction_lower for var in incorrect_variations):
                prediction = "Incorrect"
        
        return prediction, msg_history
