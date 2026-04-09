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
        self._validate_model()

    def _validate_model(self) -> None:
        """Validate that the model is properly configured."""
        if not self.model or not isinstance(self.model, str):
            raise ValueError(f"Invalid model specified: {self.model}")
        logger.debug(f"TaskAgent initialized with model: {self.model}")

    def _validate_inputs(self, inputs: dict) -> None:
        """Validate that required input fields are present and non-empty."""
        if not isinstance(inputs, dict):
            raise TypeError(f"Expected dict for inputs, got {type(inputs).__name__}")
        
        required_fields = ["problem", "solution", "student_answer"]
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in inputs:
                missing_fields.append(field)
            elif not inputs[field] or not str(inputs[field]).strip():
                empty_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required input fields: {missing_fields}")
        if empty_fields:
            raise ValueError(f"Empty required input fields: {empty_fields}")

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
- Output ONLY the JSON block above. Do not include any text after the JSON block.
- The assessment value must be one of the three exact strings: 'Correct', 'Partially correct', or 'Incorrect'."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
            
        Raises:
            TypeError: If inputs is not a dict
            ValueError: If required input fields are missing or empty
            RuntimeError: If LLM call fails
        """
        # Validate inputs before processing
        self._validate_inputs(inputs)
        
        instruction = self._build_grading_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Failed to get response from LLM: {e}") from e

        # Extract prediction from JSON with improved error handling
        prediction = self._extract_prediction(msg_history)
        
        # Normalize and return the prediction
        return self._normalize_prediction(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Args:
            msg_history: List of message dictionaries from LLM response
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        # Handle both "text" (paper format) and "content" (OpenAI format) fields
        last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
        if not last_message:
            return "None"
        
        # Try to extract JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            last_json = extracted[-1]
            # Try assessment field first (contains categorical label)
            if "assessment" in last_json and isinstance(last_json["assessment"], str):
                return last_json["assessment"].strip()
            # Fallback: try response field
            elif "response" in last_json and isinstance(last_json["response"], str):
                return last_json["response"].strip()
            # Fallback: try reasoning field
            elif "reasoning" in last_json and isinstance(last_json["reasoning"], str):
                return last_json["reasoning"].strip()
            # Fallback: try any string field
            else:
                for key, value in last_json.items():
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        
        # If no JSON found, try to extract from text
        return self._extract_from_text(last_message)

    def _extract_from_text(self, text: str) -> str:
        """Extract prediction from plain text when JSON extraction fails.
        
        Args:
            text: Raw text response from LLM
            
        Returns:
            Extracted prediction string
        """
        # Try to find text after common markers
        markers = [
            "Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:", 
            "Final Answer:", "Conclusion:", "Verdict:", "Decision:", "Evaluation:"
        ]
        
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().split('\n')[0].strip()
                    if candidate and not candidate.startswith('{') and not candidate.startswith('<'):
                        return candidate
        
        # Try to extract the last non-empty line as a fallback
        lines = text.strip().split('\n')
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('<') and not stripped.startswith('{') and not stripped.startswith('`'):
                return stripped
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to one of the three standard assessment values.
        
        Args:
            prediction: Raw prediction string from extraction
            
        Returns:
            Normalized prediction: "Correct", "Partially correct", "Incorrect", or original string
        """
        if not prediction or prediction == "None":
            return "None"
        
        # Normalize prediction to expected format
        prediction = str(prediction).strip()
        
        # Check for exact matches first (case-sensitive)
        if prediction in ["Correct", "Partially correct", "Incorrect"]:
            return prediction
        
        # Normalize common variations of assessment values
        prediction_lower = prediction.lower()
        
        # Define variation lists for maintainability
        correct_variations = [
            "correct", "right", "true", "yes", "pass", "accepted", 
            "full marks", "full credit", "100%", "perfect", "valid",
            "accurate", "proper", "appropriate", "satisfactory", "excellent"
        ]
        partially_correct_variations = [
            "partially correct", "partial", "partial credit", 
            "partially right", "incomplete", "mostly correct",
            "some correct", "partial success", "half correct",
            "partially valid", "partially accurate", "partially satisfactory",
            "partial marks", "some credit"
        ]
        incorrect_variations = [
            "incorrect", "wrong", "false", "no", "fail", "rejected", 
            "zero", "0", "invalid", "unsatisfactory", "inaccurate",
            "erroneous", "flawed", "deficient", "inadequate", "poor"
        ]
        
        # Check for exact matches in variation lists
        if prediction_lower in correct_variations:
            return "Correct"
        if prediction_lower in partially_correct_variations:
            return "Partially correct"
        if prediction_lower in incorrect_variations:
            return "Incorrect"
        
        # Check for prefix/suffix matches
        if prediction_lower.startswith("correct") or prediction_lower.endswith("correct"):
            return "Correct"
        if prediction_lower.startswith("partial"):
            return "Partially correct"
        if prediction_lower.startswith("incorrect") or prediction_lower.startswith("wrong"):
            return "Incorrect"
        
        # Check for numeric scores that might indicate correctness
        if prediction_lower in ["1", "2", "3", "4", "5", "7", "10"]:
            # These could be scores - treat as correct if high
            try:
                score = int(prediction_lower)
                if score >= 7:
                    return "Correct"
                elif score >= 3:
                    return "Partially correct"
                else:
                    return "Incorrect"
            except ValueError:
                pass
        
        # Return original if no normalization applied
        return prediction
