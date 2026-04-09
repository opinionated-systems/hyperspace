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
            # Use a more robust approach that respects string boundaries
            brace_start = text.find('{')
            while brace_start != -1:
                # Try to find matching closing brace, respecting string boundaries
                brace_count = 0
                in_string = False
                escape_next = False
                for i, char in enumerate(text[brace_start:]):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\' and in_string:
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
                                try:
                                    parsed = json.loads(json_str)
                                    if isinstance(parsed, dict):
                                        # Only add if it has relevant fields
                                        if any(k in parsed for k in ['assessment', 'response', 'reasoning', 'answer', 'result', 'grade', 'score']):
                                            results.append(parsed)
                                except json.JSONDecodeError:
                                    # Try to clean up common JSON issues
                                    try:
                                        cleaned = _clean_json_string(json_str)
                                        parsed = json.loads(cleaned)
                                        if isinstance(parsed, dict):
                                            if any(k in parsed for k in ['assessment', 'response', 'reasoning', 'answer', 'result', 'grade', 'score']):
                                                results.append(parsed)
                                    except json.JSONDecodeError:
                                        pass
                                break
                brace_start = text.find('{', brace_start + 1)
        except Exception:
            pass
    
    # Try to find JSON objects with common field patterns (assessment, response, etc.)
    if not results:
        try:
            # Look for patterns like {"assessment": ...} or {"response": ...}
            # Use a more flexible pattern that can handle various value types (strings, numbers, nested objects)
            pattern = r'\{\s*"(assessment|response|reasoning|answer|result|grade|score)"\s*:\s*([^}]+|"[^"]*"|\{[^}]*\})[^}]*\}'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Try to extract the full JSON object around the match
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    json_str = match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        try:
                            cleaned = _clean_json_string(json_str)
                            parsed = json.loads(cleaned)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
    
    # Final fallback: try to extract any valid JSON object from the text
    if not results:
        try:
            # Find all potential JSON start positions
            for start in re.finditer(r'\{', text):
                start_pos = start.start()
                # Try to find a valid JSON object starting at this position
                for end_pos in range(start_pos + 2, min(start_pos + 2000, len(text) + 1)):
                    candidate = text[start_pos:end_pos]
                    if candidate.count('{') == candidate.count('}'):
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and any(k in parsed for k in ['assessment', 'response', 'reasoning', 'answer', 'result', 'grade', 'score']):
                                results.append(parsed)
                                break  # Found a valid one, move to next start
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass
    
    # NEW: Try to find JSON with single quotes (common LLM output issue)
    if not results:
        try:
            # Look for patterns like {'assessment': ...} or {'response': ...}
            pattern = r"\{\s*'(assessment|response|reasoning|answer|result|grade|score)'\s*:\s*([^}]+|'[^']*'|\{[^}]*\})[^}]*\}"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                json_str = match.group(0)
                # Convert single quotes to double quotes for valid JSON
                try:
                    # Replace single-quoted keys with double-quoted keys
                    fixed = re.sub(r"'([^']+)'\s*:", r'"\1":', json_str)
                    # Replace single-quoted string values with double-quoted values
                    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                except (json.JSONDecodeError, re.error):
                    pass
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
    # Remove control characters (except whitespace)
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    # Fix unescaped newlines and tabs in string values (common LLM issue)
    # Use a safer approach: find string content and escape special chars within
    def escape_special_in_strings(match):
        content = match.group(1)
        # Only escape if not already escaped
        escaped = content.replace('\\n', '\x00NEWLINE\x00').replace('\\t', '\x00TAB\x00')
        escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        escaped = escaped.replace('\x00NEWLINE\x00', '\\n').replace('\x00TAB\x00', '\\t')
        return '"' + escaped + '"'
    cleaned = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_special_in_strings, cleaned)
    # Fix common unicode issues
    cleaned = cleaned.replace('\u0000', '')
    # Fix missing quotes around keys (simple cases only)
    cleaned = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
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
    prediction = None
    
    try:
        if msg_history and len(msg_history) > 0:
            # Handle both "text" (paper format) and "content" (OpenAI format) fields
            last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
            
            if not last_message:
                logger.warning("Last message is empty")
                return _normalize_prediction(None)
            
            # Try to extract JSON first
            extracted = _extract_jsons(last_message)
            if extracted:
                last_json = extracted[-1]
                # Try assessment field first (contains categorical label)
                if "assessment" in last_json:
                    prediction = last_json["assessment"]
                    logger.info(f"Extracted assessment from JSON: {prediction}")
                # Fallback: try response field if assessment is not available
                elif "response" in last_json:
                    prediction = last_json["response"]
                    logger.info(f"Extracted response from JSON: {prediction}")
                # Fallback: try reasoning field if neither is available
                elif "reasoning" in last_json:
                    prediction = last_json["reasoning"]
                    logger.info(f"Extracted reasoning from JSON: {prediction}")
                # Fallback: try any string field
                else:
                    for key, value in last_json.items():
                        if isinstance(value, str) and value.strip():
                            prediction = value
                            logger.info(f"Extracted {key} from JSON: {prediction}")
                            break
            
            # If no JSON found or no valid prediction extracted, try fallback methods
            if prediction is None:
                logger.info("No JSON found, trying fallback extraction methods")
                
                # Look for assessment labels directly in the text
                text_lower = last_message.lower()
                
                # Check for exact phrases first (more specific patterns)
                if '"assessment":' in text_lower or "'assessment':" in text_lower:
                    # Try to extract value after assessment field
                    import re
                    match = re.search(r'["\']assessment["\']\s*:\s*["\']([^"\']+)["\']', last_message, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        logger.info(f"Extracted assessment via regex: {prediction}")
                
                if prediction is None and '"response":' in text_lower or "'response':" in text_lower:
                    import re
                    match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', last_message, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        logger.info(f"Extracted response via regex: {prediction}")
                
                # If still no prediction, try to find any text after common markers
                if prediction is None:
                    markers = ["Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:", 
                               "Final Answer:", "Conclusion:", "Verdict:", "Decision:",
                               "Evaluation:", "Judgment:", "Rating:", "Status:",
                               "The answer is", "I conclude", "Therefore"]
                    for marker in markers:
                        if marker in last_message:
                            parts = last_message.split(marker, 1)
                            if len(parts) > 1:
                                candidate = parts[1].strip().split('\n')[0].strip()
                                # Clean up the candidate
                                candidate = candidate.strip('"\'.,;:!?`{}[]')
                                if candidate and len(candidate) < 100:  # Reasonable length for a label
                                    prediction = candidate
                                    logger.info(f"Extracted prediction via marker '{marker}': {prediction}")
                                    break
                
                # Last resort: look for the three target labels directly in the text
                if prediction is None:
                    # Search for the exact labels in order of priority
                    if "partially correct" in text_lower:
                        prediction = "Partially correct"
                        logger.info("Extracted 'Partially correct' from text")
                    elif text_lower.count("incorrect") > text_lower.count("correct"):
                        # More "incorrect" mentions than "correct"
                        prediction = "Incorrect"
                        logger.info("Extracted 'Incorrect' from text (frequency)")
                    elif "incorrect" in text_lower and "correct" not in text_lower.replace("incorrect", ""):
                        prediction = "Incorrect"
                        logger.info("Extracted 'Incorrect' from text")
                    elif "correct" in text_lower and "incorrect" not in text_lower:
                        prediction = "Correct"
                        logger.info("Extracted 'Correct' from text")
                    elif "correct" in text_lower:
                        # Both present, need to determine which is the final assessment
                        # Look at the last occurrence
                        last_correct = text_lower.rfind("correct")
                        last_incorrect = text_lower.rfind("incorrect")
                        if last_incorrect > last_correct:
                            prediction = "Incorrect"
                        else:
                            # Check if "partially" appears near "correct"
                            nearby_text = text_lower[max(0, last_correct-20):last_correct+30]
                            if "partially" in nearby_text or "partial" in nearby_text:
                                prediction = "Partially correct"
                            else:
                                prediction = "Correct"
                        logger.info(f"Extracted '{prediction}' from text (last occurrence)")
    except Exception as e:
        logger.warning(f"Error extracting prediction: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    # Normalize prediction to expected format
    normalized = _normalize_prediction(prediction)
    logger.info(f"Final normalized prediction: {normalized}")
    return normalized


def _normalize_prediction(prediction: str | None) -> str:
    """Normalize prediction to one of the expected assessment values.
    
    Args:
        prediction: Raw prediction string or None
        
    Returns:
        Normalized prediction: "Correct", "Partially correct", or "Incorrect"
    """
    # Handle None or empty prediction
    if prediction is None:
        return "Incorrect"  # Default to Incorrect when no prediction could be extracted
    
    prediction = str(prediction).strip()
    if not prediction or prediction.lower() == "none":
        return "Incorrect"
    
    # Remove common punctuation and quotes that might surround the value
    prediction = prediction.strip('"\'.,;:!?')
    prediction_lower = prediction.lower()
    
    # First check for exact matches (case-insensitive)
    exact_matches = {
        "correct": "Correct",
        "partially correct": "Partially correct",
        "partially": "Partially correct",
        "partial": "Partially correct",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
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
        "incorrect solution", "wrong answer", "not correct", "not right",
        "not valid", "not accurate", "not proper", "not sound",
        "does not pass", "did not pass", "fails", "failure",
        "error", "mistake", "faulty", "defective"
    ]
    for variant in incorrect_variants:
        if variant in prediction_lower:
            return "Incorrect"
    
    # Check for "partial" variants (but not "partially correct" which we already checked)
    partial_variants = [
        "partial credit", "partially right", "incomplete", "mostly correct", 
        "some correct", "partial solution", "partially valid", "partial success", 
        "half correct", "partial marks", "partially accurate", "somewhat correct",
        "mostly right", "partial pass", "partial acceptance", "incomplete solution",
        "missing parts", "lacking", "needs improvement", "minor errors",
        "small mistakes", "almost correct", "nearly correct", "close but not quite"
    ]
    for variant in partial_variants:
        if variant in prediction_lower:
            return "Partially correct"
    
    # Check for "correct" variants (but not "partially correct" or "incorrect")
    correct_variants = [
        "correct", "right", "true", "yes", "pass", "accepted", 
        "full marks", "full credit", "valid", "accurate", "proper",
        "sound", "complete solution", "correct solution", "perfect",
        "excellent", "good", "well done", "success", "successful",
        "passes", "passing", "meets requirements", "satisfactory",
        "appropriate", "fitting", "suitable", "adequate"
    ]
    for variant in correct_variants:
        if variant in prediction_lower:
            return "Correct"
    
    # If no match found, default to Incorrect for safety
    # This is more conservative than returning the original
    return "Incorrect"
