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
    """
    results = []
    search_from = 0
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
            continue
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON objects at start/end of text
    4. JSON-like structures with relaxed parsing
    5. Repair common JSON syntax errors
    """
    results = []
    
    # Strategy 1: <json> tags (original)
    search_from = 0
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
            # Try to repair common JSON errors
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
            continue
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
                continue
    
    # Strategy 3: Look for JSON objects directly
    if not results:
        # Try to find JSON objects between curly braces
        pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                # Expand to capture nested structures
                start = match.start()
                brace_count = 0
                end = start
                for i, char in enumerate(text[start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = start + i + 1
                            break
                if end > start:
                    obj_str = text[start:end]
                    try:
                        results.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        repaired = _repair_json(obj_str)
                        if repaired:
                            results.append(repaired)
            except (json.JSONDecodeError, ValueError):
                continue
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    """
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    # This is a best-effort approach
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    
    # Try parsing the repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Last resort: try to extract just the first complete JSON object
        try:
            # Find the first { and matching }
            start = repaired.find('{')
            if start == -1:
                return None
            
            brace_count = 0
            for i, char in enumerate(repaired[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete object
                        return json.loads(repaired[start:start+i+1])
            return None
        except Exception:
            return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability.
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (original format)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with repair
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Look for any JSON-like structure
    # Try to find content between outermost braces
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            # Try to repair and parse
            repaired = _repair_json(candidate)
            if repaired:
                return [repaired]
    except Exception:
        pass
    
    # Strategy 4: Extract key-value pairs from malformed JSON
    # Last resort: try to extract reasoning and response fields directly
    try:
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\.[^"]*)*)"', text, re.DOTALL)
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
        
        if reasoning_match or response_match:
            extracted = {}
            if reasoning_match:
                extracted["reasoning"] = reasoning_match.group(1).replace('\\n', '\n').replace('\\"', '"')
            if response_match:
                extracted["response"] = response_match.group(1)
            elif not response_match and reasoning_match:
                # Try to infer response from text patterns
                text_lower = text.lower()
                if "correct" in text_lower and "incorrect" not in text_lower:
                    extracted["response"] = "correct"
                elif "incorrect" in text_lower:
                    extracted["response"] = "incorrect"
                elif "partial" in text_lower:
                    extracted["response"] = "partial"
            
            if extracted:
                return [extracted]
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to International Mathematical Olympiad (IMO) style competition problems.

Your task is to carefully analyze the student's answer and provide a rigorous evaluation according to the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

Think through this step-by-step:

1. **Understand the Problem**: What is being asked? What are the key concepts and theorems involved?

2. **Analyze the Official Solution**: What is the correct approach? What is the final answer? What are the critical steps that must be present?

3. **Review the Student's Answer**: What approach did the student take? What is their final answer? Did they show all necessary work?

4. **Compare and Evaluate**: Does the student's answer match the official solution? Consider:
   - Is the final answer numerically/algebraically equivalent to the official solution?
   - Did the student demonstrate correct mathematical reasoning?
   - Are there any logical gaps or errors in the student's work?
   - Did the student use appropriate methods and theorems?
   - Is the solution complete or partial?

5. **Assign Grade**: Based on your analysis, provide your evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here, including specific observations about the student's work",
    "response": "Your final evaluation here - should be one of: 'correct', 'incorrect', 'partial', or a specific score if applicable"
}}
</json>

The "response" field must contain a clear, concise final determination. Use:
- "correct" if the answer is fully correct with proper reasoning
- "incorrect" if the answer is wrong or has critical errors
- "partial" if the answer has some correct elements but is incomplete or has minor errors
- A specific score (e.g., "7" or "3/7") if the problem uses point-based scoring"""
        
        return prompt

    # Priority order for JSON field extraction
    _PREDICTION_FIELDS = [
        "grade", "score", "evaluation", "response", "answer",
        "result", "conclusion", "verdict"
    ]
    
    # Regex patterns for field extraction (field_name, pattern, group_index)
    _REGEX_PATTERNS = [
        ("grade", r'"grade"\s*:\s*"([^"]+)"', 1),
        ("score", r'"score"\s*:\s*"?([^"},\s]+)"?', 1),
        ("evaluation", r'"evaluation"\s*:\s*"([^"]+)"', 1),
        ("response", r'"response"\s*:\s*"([^"]+)"', 1),
        ("answer", r'"answer"\s*:\s*"([^"]+)"', 1),
        ("verdict", r'"verdict"\s*:\s*"([^"]+)"', 1),
        ("correct", r'"correct"\s*:\s*(true|false)', 1),
    ]
    
    # Score extraction patterns
    _SCORE_PATTERNS = [
        r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'(\d+)\s*/\s*\d+\s*(?:points?)?',
        r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?',
        r'(?:score|grade)\s+of\s+(\d+)',
    ]
    
    # Verdict patterns for text-based extraction
    _VERDICT_PATTERNS = [
        (r'\bthe\s+answer\s+is\s+(correct|incorrect|wrong|right)\b', 1),
        (r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', 1),
        (r'\b(correct|incorrect|partial)\s+answer\b', 1),
        (r'\banswer\s+is\s+(correct|incorrect|partial)\b', 1),
    ]
    
    # Partial credit keywords
    _PARTIAL_KEYWORDS = ["partial", "partially", "some credit", "incomplete", "partial credit"]

    def _extract_from_json_object(self, obj: dict) -> str | None:
        """Extract prediction value from a JSON object.
        
        Checks fields in priority order and handles type conversions.
        Returns None if no valid field found.
        """
        # Check standard fields in priority order
        for key in self._PREDICTION_FIELDS:
            if key in obj:
                value = obj[key]
                if isinstance(value, (str, int, float, bool)):
                    return str(value)
                elif isinstance(value, (list, dict)):
                    return json.dumps(value)
        
        # Check for boolean correctness field
        if "correct" in obj:
            correct_val = obj["correct"]
            if isinstance(correct_val, bool):
                return "correct" if correct_val else "incorrect"
            return str(correct_val)
        
        # Check for points field
        if "points" in obj:
            return f"points:{obj['points']}"
        
        return None

    def _extract_with_regex(self, text: str) -> str | None:
        """Extract prediction using regex patterns on raw text."""
        for field_name, pattern, group_idx in self._REGEX_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(group_idx)
                if field_name == "correct":
                    return "correct" if value.lower() == "true" else "incorrect"
                return value.lower()
        return None

    def _extract_score_from_text(self, text: str) -> str | None:
        """Extract numeric score from text using score patterns."""
        text_lower = text.lower()
        for pattern in self._SCORE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        return None

    def _extract_verdict_from_text(self, text: str) -> str | None:
        """Extract verdict (correct/incorrect/partial) from text."""
        text_lower = text.lower()
        
        # Check explicit verdict patterns first
        for pattern, group_idx in self._VERDICT_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                verdict = match.group(group_idx).lower()
                if verdict in ["right", "correct"]:
                    return "correct"
                elif verdict in ["wrong", "incorrect"]:
                    return "incorrect"
                elif verdict == "partial":
                    return "partial"
        
        # Check for correctness indicators
        if "correct" in text_lower:
            if "incorrect" in text_lower or "not correct" in text_lower:
                return "incorrect"
            return "correct"
        
        # Check for partial credit
        if any(term in text_lower for term in self._PARTIAL_KEYWORDS):
            return "partial"
        
        return None

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
        
        Uses a structured approach with separate extraction methods for
        different strategies, ordered by reliability.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Strategy 1: Extract from JSON objects
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                last_obj = extracted[-1]
                result = self._extract_from_json_object(last_obj)
                if result:
                    return result
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"JSON extraction failed: {e}")
        
        # Strategy 2: Regex-based extraction on raw text
        try:
            result = self._extract_with_regex(last_text)
            if result:
                return result
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Strategy 3: Score extraction
        result = self._extract_score_from_text(last_text)
        if result:
            return result
        
        # Strategy 4: Verdict extraction
        result = self._extract_verdict_from_text(last_text)
        if result:
            return result
        
        # Strategy 5: Return truncated text as fallback
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score format (e.g., "score:7", "7/7")
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Normalize correct variations
        if any(term in pred_lower for term in ["correct", "right", "true", "valid", "accepted"]):
            if "incorrect" not in pred_lower and "not correct" not in pred_lower:
                return "correct"
        
        # Normalize incorrect variations
        if any(term in pred_lower for term in ["incorrect", "wrong", "false", "invalid", "rejected", "error"]):
            return "incorrect"
        
        # Normalize partial variations
        if any(term in pred_lower for term in ["partial", "partially", "incomplete", "some credit"]):
            return "partial"
        
        # Return original if no normalization applied
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Processing problem: {problem_preview}...")

        # Retry loop for transient failures
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Don't retry on certain errors
                if "invalid api key" in error_str or "authentication" in error_str:
                    self.log_fn(f"Authentication error on attempt {attempt + 1}, not retrying: {e}")
                    break
                
                if "context length" in error_str or "too long" in error_str:
                    self.log_fn(f"Context length exceeded on attempt {attempt + 1}, not retrying: {e}")
                    break
                
                if attempt < max_retries - 1:
                    self.log_fn(f"LLM call failed on attempt {attempt + 1}/{max_retries}: {e}. Retrying...")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.log_fn(f"All {max_retries} retry attempts exhausted. Last error: {e}")
        else:
            # All retries exhausted
            error_msg = f"Error: LLM call failed after {max_retries} attempts: {last_error}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        # Check if we actually got a response (in case we broke from auth error)
        if 'response' not in locals():
            error_msg = f"Error: LLM call failed: {last_error}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract prediction from JSON
        raw_prediction = self._extract_prediction(msg_history)
        
        # Normalize prediction to standard format
        prediction = self._normalize_prediction(raw_prediction)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history
