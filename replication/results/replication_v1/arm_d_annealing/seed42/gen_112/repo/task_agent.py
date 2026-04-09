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
        
    results = []
    search_from = 0
    max_iterations = 100  # Prevent infinite loops on malformed input
    iterations = 0
    
    def _clean_and_parse_json(json_str: str) -> dict | None:
        """Try to clean and parse a JSON string with common fixes."""
        if not json_str or not isinstance(json_str, str):
            return None
            
        # Strip common leading/trailing whitespace and markers
        json_str = json_str.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to clean up common JSON issues
            try:
                cleaned = json_str
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                # Fix single quotes used as JSON delimiters (but not within values)
                cleaned = re.sub(r"'([^']*?)':\s*", r'"\1": ', cleaned)
                cleaned = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', cleaned)
                # Fix unescaped newlines in strings
                cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
                # Fix unescaped tabs in strings
                cleaned = re.sub(r'(?<!\\)\t', r'\\t', cleaned)
                # Fix unescaped carriage returns
                cleaned = re.sub(r'(?<!\\)\r', r'\\r', cleaned)
                # Remove BOM if present
                cleaned = cleaned.lstrip('\ufeff')
                # Remove any leading/trailing whitespace
                cleaned = cleaned.strip()
                # Remove any leading/trailing backticks
                cleaned = cleaned.strip('`')
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
    
    # First, try to find <json>...</json> blocks
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        # Ensure we don't have nested <json> tags
        next_start = text.find("<json>", start + 6)
        if next_start != -1 and next_start < end:
            # Nested tag found, skip this one
            search_from = start + 6
            continue
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = _clean_and_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Also try to find markdown json blocks ```json...```
    if not results:
        search_from = 0
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            start = text.find("```json", search_from)
            if start != -1:
                # Found ```json, look for closing ```
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
                search_from = end + 3
            else:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
                search_from = end + 3
            
            # Skip if this looks like a non-JSON code block
            if inner.startswith('python') or inner.startswith('bash') or inner.startswith('shell'):
                continue
            
            parsed = _clean_and_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
                break  # Only process first valid JSON block
    
    # Try to find inline JSON objects as a last resort
    if not results:
        try:
            # Look for JSON-like structures with curly braces
            brace_start = text.find('{')
            max_brace_searches = 50  # Limit brace searches
            brace_searches = 0
            found_valid_json = False
            while brace_start != -1 and brace_searches < max_brace_searches and not found_valid_json:
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
                                parsed = _clean_and_parse_json(json_str)
                                if parsed is not None and isinstance(parsed, dict):
                                    results.append(parsed)
                                    found_valid_json = True
                                break
                brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
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

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

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

**Evaluation Process** (think step by step):

1. **Understand the Problem**: Identify what is being asked and the key mathematical concepts involved.

2. **Review the Official Solution**: Note the expected approach, key insights, and critical steps.

3. **Analyze the Student's Answer**:
   - Check mathematical correctness of each step
   - Verify logical soundness and proper justification
   - Assess completeness (all parts of the problem addressed?)
   - Evaluate clarity and rigor of presentation

4. **Compare with Guidelines**: Match the student's work against the grading criteria.

5. **Determine Assessment**:
   - **Correct**: Fully correct solution with proper reasoning
   - **Partially correct**: Has some correct elements but incomplete or has errors
   - **Incorrect**: Fundamentally wrong or no meaningful progress

---

**Output Format** (STRICT JSON):

You MUST output ONLY a valid JSON object in this exact format:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation explaining the strengths and weaknesses of the student's solution",
    "assessment": "Correct",
    "response": "Correct"
}}
</json>

**CRITICAL REQUIREMENTS**:
1. The "assessment" field MUST be EXACTLY one of: 'Correct', 'Partially correct', or 'Incorrect' (case-sensitive, no extra spaces)
2. The "response" field should match the assessment and follow any specific format in the grading guidelines
3. Output ONLY the JSON block - no additional text before or after
4. Ensure valid JSON: use double quotes, no trailing commas
5. Be objective and consistent with the official solution and grading guidelines"""

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
                                     "Final Answer:", "Conclusion:", "Verdict:", "Decision:", 
                                     "Evaluation:", "Judgment:", "Rating:"]
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
        
        # Check for exact matches first (case-sensitive)
        if prediction in ["Correct", "Partially correct", "Incorrect"]:
            return prediction, msg_history
        
        # Then check for common variations
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted", 
            "full marks", "full credit", "100%", "perfect", "valid",
            "accurate", "proper", "appropriate", "satisfactory",
            "excellent", "complete", "fully correct", "all correct"
        ]
        partially_correct_variations = [
            "partially correct", "partial", "partial credit", 
            "partially right", "incomplete", "mostly correct",
            "some correct", "partial success", "half correct",
            "partially valid", "partially accurate", "partially satisfactory",
            "partly correct", "partially complete", "some progress"
        ]
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected", 
            "zero", "0", "invalid", "unsatisfactory", "inaccurate",
            "erroneous", "flawed", "deficient", "inadequate",
            "not correct", "not right", "not valid", "none"
        ]
        
        # Check for exact matches in variations
        if prediction_lower in correct_variations:
            prediction = "Correct"
        elif prediction_lower in partially_correct_variations:
            prediction = "Partially correct"
        elif prediction_lower in incorrect_variations:
            prediction = "Incorrect"
        # Check for partial matches (prefix/suffix)
        elif prediction_lower.startswith("correct") or prediction_lower.endswith("correct"):
            prediction = "Correct"
        elif prediction_lower.startswith("partial") or prediction_lower.startswith("partly"):
            prediction = "Partially correct"
        elif prediction_lower.startswith("incorrect") or prediction_lower.startswith("wrong"):
            prediction = "Incorrect"
        elif prediction_lower.startswith("not ") and ("correct" in prediction_lower or "right" in prediction_lower):
            prediction = "Incorrect"
        elif prediction_lower.startswith("mostly") and "correct" in prediction_lower:
            prediction = "Partially correct"
        
        return prediction, msg_history
