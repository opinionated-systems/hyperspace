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
    
    # Try to find JSON objects with common field patterns (assessment, response, etc.)
    if not results:
        try:
            # Look for patterns like {"assessment": ...} or {"response": ...}
            pattern = r'\{\s*"(assessment|response|reasoning|answer|result|grade|score)"\s*:\s*"[^"]*"[^}]*\}'
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
    
    return results if results else None


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Args:
        json_str: Raw JSON string that may have formatting issues
        
    Returns:
        Cleaned JSON string
    """
    if not json_str or not isinstance(json_str, str):
        return "{}"
    
    cleaned = json_str.strip()
    
    # Remove markdown code block markers if present
    cleaned = re.sub(r'^```\w*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```$', '', cleaned)
    
    # Remove <json> and </json> tags if present
    cleaned = re.sub(r'^<json>\s*', '', cleaned)
    cleaned = re.sub(r'\s*</json>$', '', cleaned)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes used as JSON delimiters (but not within strings)
    # This is a simplified approach - more robust would require parsing
    cleaned = re.sub(r"'([^']*?)':", r'"\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
    
    # Remove comments
    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Remove control characters except whitespace
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    # Fix unescaped newlines in string values (common LLM issue)
    # Use a more careful approach that respects string boundaries
    result = []
    in_string = False
    escape_next = False
    for char in cleaned:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
        elif in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\t':
            result.append('\\t')
        elif in_string and char == '\r':
            result.append('\\r')
        else:
            result.append(char)
    cleaned = ''.join(result)
    
    # Fix common issues with unclosed strings
    # Count quotes to check for odd number (unclosed string)
    quote_count = cleaned.count('"')
    if quote_count % 2 == 1:
        # Try to find and close the unclosed string
        # This is a heuristic - look for the last colon or comma followed by text
        last_colon = cleaned.rfind('":')
        if last_colon != -1:
            # Check if there's an unclosed string after the last colon
            after_colon = cleaned[last_colon + 2:].strip()
            if after_colon and not after_colon.startswith('"') and not after_colon.startswith('{'):
                # Try to close it
                cleaned = cleaned + '"'
    
    # Ensure the JSON starts with { and ends with }
    cleaned = cleaned.strip()
    if not cleaned.startswith('{'):
        # Try to find the first {
        first_brace = cleaned.find('{')
        if first_brace != -1:
            cleaned = cleaned[first_brace:]
    if not cleaned.endswith('}'):
        # Try to find the last }
        last_brace = cleaned.rfind('}')
        if last_brace != -1:
            cleaned = cleaned[:last_brace + 1]
    
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

Your task is to evaluate a student's solution to a mathematical problem and provide a definitive grade.

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

First, analyze the student's answer step by step:
1. Does the student's approach align with the official solution?
2. Are the mathematical steps correct?
3. Is the logic sound and well-justified?
4. Does the student address all parts of the problem?
5. What score would you assign based on the grading guidelines?

Then, provide your final assessment in the following JSON format:

<json>
{{
    "reasoning": "Your detailed step-by-step evaluation explaining what the student did right/wrong",
    "assessment": "Correct",
    "response": "Correct"
}}
</json>

**CRITICAL INSTRUCTIONS:**
- The "assessment" and "response" fields MUST contain EXACTLY one of these three values (case-sensitive):
  - "Correct" - if the solution is fully correct and complete
  - "Partially correct" - if the solution has some correct elements but is incomplete or has errors
  - "Incorrect" - if the solution is wrong or fundamentally flawed
- Both fields must have the SAME value.
- Do NOT add quotes around the values in the JSON.
- Do NOT add any explanation after the JSON block.
- The evaluation system will extract the "assessment" field to determine the grade.

Example of a correct response:
<json>
{{
    "reasoning": "The student correctly identified the pattern and provided a valid proof.",
    "assessment": "Correct",
    "response": "Correct"
}}
</json>"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        # Log the problem being evaluated
        domain = inputs.get("domain", "unknown")
        logger.info(f"Evaluating problem in domain: {domain}")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )
        
        # Log token usage for monitoring
        usage = info.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        # Extract prediction from JSON with improved error handling
        prediction = _extract_prediction(msg_history)
        
        # Log the final prediction
        logger.info(f"Final prediction for {domain}: {prediction}")
        
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
        if not msg_history:
            logger.warning("Empty message history, defaulting to Incorrect")
            return _normalize_prediction(None)
            
        # Handle both "text" (paper format) and "content" (OpenAI format) fields
        last_message = msg_history[-1].get("text") or msg_history[-1].get("content", "")
        
        if not last_message:
            logger.warning("Empty last message, defaulting to Incorrect")
            return _normalize_prediction(None)
        
        # Try to extract JSON first
        extracted = _extract_jsons(last_message)
        if extracted:
            last_json = extracted[-1]
            logger.debug(f"Extracted JSON: {last_json}")
            
            # Try assessment field first (contains categorical label)
            if "assessment" in last_json:
                prediction = last_json["assessment"]
                logger.debug(f"Extracted prediction from 'assessment' field: {prediction}")
            # Fallback: try response field if assessment is not available
            elif "response" in last_json:
                prediction = last_json["response"]
                logger.debug(f"Extracted prediction from 'response' field: {prediction}")
            # Fallback: try answer field
            elif "answer" in last_json:
                prediction = last_json["answer"]
                logger.debug(f"Extracted prediction from 'answer' field: {prediction}")
            # Fallback: try result field
            elif "result" in last_json:
                prediction = last_json["result"]
                logger.debug(f"Extracted prediction from 'result' field: {prediction}")
            # Fallback: try grade field
            elif "grade" in last_json:
                prediction = last_json["grade"]
                logger.debug(f"Extracted prediction from 'grade' field: {prediction}")
            # Fallback: try score field
            elif "score" in last_json:
                prediction = last_json["score"]
                logger.debug(f"Extracted prediction from 'score' field: {prediction}")
            # Fallback: try evaluation field
            elif "evaluation" in last_json:
                prediction = last_json["evaluation"]
                logger.debug(f"Extracted prediction from 'evaluation' field: {prediction}")
            # Fallback: try verdict field
            elif "verdict" in last_json:
                prediction = last_json["verdict"]
                logger.debug(f"Extracted prediction from 'verdict' field: {prediction}")
            # Fallback: try any string field (excluding reasoning/explanation fields)
            else:
                for key, value in last_json.items():
                    if isinstance(value, str) and value.strip():
                        # Skip reasoning/explanation fields
                        if key.lower() not in ['reasoning', 'explanation', 'thought', 'thinking', 'analysis', 'commentary']:
                            prediction = value
                            logger.debug(f"Extracted prediction from field '{key}': {prediction}")
                            break
        
        # If no JSON found or no valid prediction extracted, try fallback methods
        if prediction is None:
            logger.debug("No JSON found, trying fallback extraction methods")
            
            # Try to find text after common markers first (more reliable than last line)
            markers = ["Assessment:", "Response:", "Answer:", "Grade:", "Score:", "Result:", 
                       "Final Answer:", "Conclusion:", "Verdict:", "Decision:",
                       "Evaluation:", "Judgment:", "Rating:", "Status:",
                       "The student's answer is", "This is", "Grade is"]
            for marker in markers:
                if marker in last_message:
                    parts = last_message.split(marker, 1)
                    if len(parts) > 1:
                        candidate = parts[1].strip().split('\n')[0].strip()
                        # Clean up common punctuation
                        candidate = candidate.rstrip('.').rstrip(',').strip()
                        if candidate and not candidate.startswith('{'):
                            prediction = candidate
                            logger.debug(f"Extracted prediction from marker '{marker}': {prediction}")
                            break
            
            # If still no prediction, try to extract the last non-empty line
            if prediction is None:
                lines = last_message.strip().split('\n')
                for line in reversed(lines):
                    stripped = line.strip()
                    # Skip lines that are clearly not predictions
                    if stripped and not stripped.startswith('<') and not stripped.startswith('{') and not stripped.startswith('`'):
                        # Skip common non-prediction lines
                        skip_prefixes = ['```', '}', '<json>', '</json>', 'Note:', 'Please', 'Here', 'The ']
                        if not any(stripped.startswith(prefix) for prefix in skip_prefixes):
                            prediction = stripped
                            logger.debug(f"Extracted prediction from last non-empty line: {prediction}")
                            break
    except Exception as e:
        logger.warning(f"Error extracting prediction: {e}")
    
    # Normalize prediction to expected format
    normalized = _normalize_prediction(prediction)
    logger.info(f"Final prediction: '{prediction}' -> '{normalized}'")
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
        logger.debug("Prediction is None, defaulting to Incorrect")
        return "Incorrect"  # Default to Incorrect when no prediction could be extracted
    
    prediction = str(prediction).strip()
    if not prediction or prediction.lower() == "none":
        logger.debug("Prediction is empty or 'none', defaulting to Incorrect")
        return "Incorrect"
    
    # Remove common punctuation and quotes
    prediction_clean = prediction.strip('"').strip("'").rstrip('.').rstrip(',').strip()
    prediction_lower = prediction_clean.lower()
    
    logger.debug(f"Normalizing prediction: '{prediction}' (cleaned: '{prediction_clean}')")
    
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
        "incorrect solution", "wrong answer", "not correct", "not right",
        "not valid", "not accurate", "not proper", "not sound",
        "incomplete solution", "incomplete answer", "missing",
        "does not", "doesn't", "did not", "didn't", "cannot", "can't",
        "unable to", "failed to", "error", "mistake", "mistaken"
    ]
    for variant in incorrect_variants:
        if variant in prediction_lower:
            logger.debug(f"Matched incorrect variant: '{variant}'")
            return "Incorrect"
    
    # Check for "partial" variants (but not "partially correct" which we already checked)
    partial_variants = [
        "partial credit", "partially right", "incomplete", "mostly correct", 
        "some correct", "partial solution", "partially valid", "partial success", 
        "half correct", "partial marks", "partially accurate", "partially proper",
        "some right", "some valid", "some accurate", "some proper",
        "mostly right", "mostly valid", "mostly accurate", "mostly proper",
        "partial understanding", "partial progress", "partial completion"
    ]
    for variant in partial_variants:
        if variant in prediction_lower:
            logger.debug(f"Matched partial variant: '{variant}'")
            return "Partially correct"
    
    # Check for "correct" variants (but not "partially correct" or "incorrect")
    correct_variants = [
        "correct", "right", "true", "yes", "pass", "accepted", 
        "full marks", "full credit", "valid", "accurate", "proper",
        "sound", "complete solution", "correct solution", "right answer",
        "correct answer", "valid solution", "accurate solution", "proper solution",
        "sound solution", "complete answer", "valid answer", "accurate answer",
        "proper answer", "sound answer", "well done", "good job",
        "excellent", "perfect", "flawless", "complete", "full"
    ]
    for variant in correct_variants:
        if variant in prediction_lower:
            logger.debug(f"Matched correct variant: '{variant}'")
            return "Correct"
    
    # Check for numeric scores that might indicate partial credit
    # e.g., "7/7" or "100%" would be correct, "3/7" or "50%" would be partial
    import re
    
    # Pattern for fractions like "3/7" or "5 / 7"
    fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', prediction_lower)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator > 0:
            ratio = numerator / denominator
            if ratio >= 0.9:  # 90% or higher is correct
                logger.debug(f"Matched high fraction score: {numerator}/{denominator}")
                return "Correct"
            elif ratio >= 0.4:  # 40-89% is partially correct
                logger.debug(f"Matched medium fraction score: {numerator}/{denominator}")
                return "Partially correct"
            else:  # Below 40% is incorrect
                logger.debug(f"Matched low fraction score: {numerator}/{denominator}")
                return "Incorrect"
    
    # Pattern for percentages like "75%" or "100%"
    percent_match = re.search(r'(\d+)%', prediction_lower)
    if percent_match:
        percent = int(percent_match.group(1))
        if percent >= 90:  # 90% or higher is correct
            logger.debug(f"Matched high percentage: {percent}%")
            return "Correct"
        elif percent >= 40:  # 40-89% is partially correct
            logger.debug(f"Matched medium percentage: {percent}%")
            return "Partially correct"
        else:  # Below 40% is incorrect
            logger.debug(f"Matched low percentage: {percent}%")
            return "Incorrect"
    
    # If no match found, default to "Incorrect" for safety
    # This ensures we always return one of the three expected values
    logger.debug(f"No match found for '{prediction}', defaulting to Incorrect")
    return "Incorrect"
