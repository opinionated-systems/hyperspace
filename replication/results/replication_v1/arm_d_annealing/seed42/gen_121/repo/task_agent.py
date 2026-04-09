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
    if not text or not isinstance(text, str):
        return None
    
    # Strip common markdown formatting that might interfere with JSON parsing
    text = text.strip()
    results = []
    
    # Try to find <json>...</json> blocks
    results.extend(_extract_tagged_json(text, "<json>", "</json>"))
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        results.extend(_extract_markdown_json(text))
    
    # Try to find inline JSON objects as a last resort
    if not results:
        results.extend(_extract_inline_json(text))
    
    return results if results else None


def _extract_tagged_json(text: str, start_tag: str, end_tag: str) -> list[dict]:
    """Extract JSON from tagged blocks like <json>...</json>."""
    results = []
    search_from = 0
    max_iterations = 100
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        start = text.find(start_tag, search_from)
        if start == -1:
            break
        end = text.find(end_tag, start)
        if end == -1:
            break
        # Ensure we don't have nested tags
        next_start = text.find(start_tag, start + len(start_tag))
        if next_start != -1 and next_start < end:
            search_from = start + len(start_tag)
            continue
        inner = text[start + len(start_tag):end].strip()
        search_from = end + len(end_tag)
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    return results


def _extract_markdown_json(text: str) -> list[dict]:
    """Extract JSON from markdown code blocks."""
    results = []
    search_from = 0
    max_iterations = 100
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
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
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
            break  # Only process first valid JSON block
    
    return results


def _extract_inline_json(text: str) -> list[dict]:
    """Extract inline JSON objects from text."""
    results = []
    brace_start = text.find('{')
    max_brace_searches = 50
    brace_searches = 0
    
    while brace_start != -1 and brace_searches < max_brace_searches:
        brace_searches += 1
        # Try to find matching closing brace
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
                            return results  # Return first valid inline JSON
                        break
        brace_start = text.find('{', brace_start + 1)
    
    return results


def _try_parse_json(json_str: str) -> dict | None:
    """Try to parse JSON string, with cleanup on failure."""
    try:
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        try:
            cleaned = _clean_json_string(json_str)
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


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
    # Fix unescaped newlines in strings
    cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
    # Fix unescaped carriage returns
    cleaned = re.sub(r'(?<!\\)\r', r'\\r', cleaned)
    # Fix unescaped tabs
    cleaned = re.sub(r'(?<!\\)\t', r'\\t', cleaned)
    # Remove BOM if present
    cleaned = cleaned.lstrip('\ufeff')
    # Remove any leading/trailing whitespace
    cleaned = cleaned.strip()
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
    "assessment": "A clear summary: 'Correct', 'Partially correct', or 'Incorrect'",
    "response": "The final grading decision or score as specified in the guidelines"
}}
</json>

Important: 
- Ensure your response is valid JSON without trailing commas.
- The "assessment" field MUST be exactly one of: 'Correct', 'Partially correct', or 'Incorrect' (case-sensitive).
- The "response" field should contain the final answer that will be used for evaluation.
- Do not include any text outside the JSON block.
- The JSON must be properly formatted with double quotes around all strings.

Examples of valid assessments:
- "Correct" - The student's solution is fully correct and complete
- "Partially correct" - The student's solution has some correct elements but is incomplete or has errors
- "Incorrect" - The student's solution is wrong or fundamentally flawed"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        # Retry logic for LLM calls with exponential backoff
        max_retries = 3
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log_fn(f"LLM call failed after {max_retries} attempts: {e}")
                    return "Error: LLM call failed", []
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                # Handle both "text" (paper format) and "content" (OpenAI format) fields
                last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
                
                # Log the raw response for debugging
                if len(last_message) > 500:
                    self.log_fn(f"Raw LLM response (truncated): {last_message[:500]}...")
                else:
                    self.log_fn(f"Raw LLM response: {last_message}")
                
                extracted = _extract_jsons(last_message)
                if extracted:
                    last_json = extracted[-1]
                    self.log_fn(f"Extracted JSON: {last_json}")
                    
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
                                     "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:"]
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
        if prediction_lower in ["correct", "right", "true", "yes", "pass", "accepted", "full marks", "full credit", "valid", "accurate"]:
            prediction = "Correct"
        elif prediction_lower in ["partially correct", "partial", "partial credit", "partially right", "incomplete", "partially accurate", "mostly correct"]:
            prediction = "Partially correct"
        elif prediction_lower in ["incorrect", "wrong", "false", "no", "fail", "rejected", "zero", "0", "invalid", "inaccurate", "error"]:
            prediction = "Incorrect"
        
        # Validate that prediction is one of the expected values
        valid_predictions = {"Correct", "Partially correct", "Incorrect"}
        if prediction not in valid_predictions and prediction != "None":
            self.log_fn(f"Warning: Unexpected prediction value '{prediction}', attempting to normalize")
            # Try to extract assessment from the prediction text
            if "correct" in prediction_lower and "partial" not in prediction_lower and "in" not in prediction_lower:
                prediction = "Correct"
            elif "partial" in prediction_lower or ("correct" in prediction_lower and "in" in prediction_lower):
                prediction = "Partially correct"
            elif "incorrect" in prediction_lower or "wrong" in prediction_lower or "error" in prediction_lower:
                prediction = "Incorrect"
        
        self.log_fn(f"Final prediction: {prediction}")
        return prediction, msg_history
