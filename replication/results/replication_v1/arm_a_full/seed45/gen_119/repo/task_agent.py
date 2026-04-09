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
    
    Args:
        text: The text containing <json> tags to parse.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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
            # Try to repair common JSON errors before giving up
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
            continue
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. ```json code blocks
    2. Raw JSON objects at start/end of text
    3. Repair common JSON syntax errors
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    results = []
    
    # Strategy 1: ```json code blocks
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
    
    # Strategy 2: Look for JSON objects directly with brace counting
    if not results:
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in start_indices:
            try:
                brace_count = 0
                in_string = False
                escape_next = False
                end = start
                
                for i, char in enumerate(text[start:]):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = start + i + 1
                                break
                
                if end > start and brace_count == 0:
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
    - Comments in JSON (// and /* */)
    - BOM and other invisible characters
    - Unescaped quotes in strings
    - Missing quotes around keys
    
    Args:
        text: The potentially malformed JSON string to repair.
        
    Returns:
        A parsed JSON dict if repair succeeds, None otherwise.
    """
    if not text or not text.strip():
        return None
    
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove BOM and other invisible characters at the start
    repaired = text.lstrip('\ufeff\u200b\u200c\u200d\ufe00-\ufe0f')
    
    # Remove comments (// and /* */)
    repaired = re.sub(r'//[^\n]*(?:\n|$)', '\n', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Try parsing the repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: try to extract just the first complete JSON object
    try:
        start = repaired.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(repaired[start:]):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = repaired[start:start+i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Try additional repairs on the candidate
                            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                # Try to fix unescaped newlines in strings
                                candidate = re.sub(r'(?<!\\)\n', '\\n', candidate)
                                try:
                                    return json.loads(candidate)
                                except json.JSONDecodeError:
                                    pass
                        break
        return None
    except Exception:
        pass
    
    # Third attempt: try to find and parse any valid JSON-like structure
    try:
        # Look for patterns like "key": "value" or 'key': 'value'
        pattern = r'[\{\[][^\{\[\]\}]*[\}\]]'
        matches = re.findall(pattern, repaired, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent evaluates student solutions to International Mathematical Olympiad (IMO)
    style competition problems. It uses a structured prompt with chain-of-thought
    reasoning to analyze student answers against official solutions and grading guidelines.
    
    The agent supports flexible JSON extraction with multiple fallback strategies
    to handle various LLM response formats.
    
    Attributes:
        model: The LLM model to use for evaluation.
        log_fn: Logging function for agent activity.
    
    Example:
        >>> agent = TaskAgent()
        >>> inputs = {
        ...     "problem": "Find the sum of 1+2+3+...+10",
        ...     "solution": "The sum is 55 using n(n+1)/2 formula",
        ...     "student_answer": "55",
        ...     "grading_guidelines": "Correct answer: 55"
        ... }
        >>> prediction, history = agent.forward(inputs)
    """

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
   - Did the student justify their steps with clear reasoning?

5. **Assign Grade**: Based on your analysis, provide your evaluation using the rubric below.

## Grading Rubric

- **Correct**: Fully correct with proper reasoning, all steps justified, final answer matches official solution.
- **Incorrect**: Wrong answer, critical errors, incorrect methods, or final answer doesn't match.
- **Partial**: Some correct elements but incomplete, minor errors, lacks justification, or only partially matches.

## Response Format

Respond ONLY in this exact JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be thorough and mention specific mathematical steps, theorems used, and any errors found.",
    "response": "correct" | "incorrect" | "partial"
}}
</json>

CRITICAL: The "response" field must contain ONLY one of these three exact lowercase values: "correct", "incorrect", or "partial". No other text, explanations, or formatting allowed in this field."""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
        
        Improvements:
        - Better handling of nested JSON structures
        - Support for confidence scores in responses
        - Improved error logging for debugging
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Try robust extraction first
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                last_obj = extracted[-1]
                
                # Check for structured grading response (priority order)
                priority_keys = ["response", "grade", "evaluation", "result", "prediction", "answer", "score"]
                for key in priority_keys:
                    if key in last_obj:
                        value = last_obj[key]
                        if isinstance(value, str):
                            # Clean up the value
                            cleaned = value.strip().lower()
                            # Handle common variations
                            if cleaned in ["correct", "incorrect", "partial"]:
                                return cleaned
                            return value.strip()
                        elif isinstance(value, (int, float)):
                            # Convert numeric scores to categories
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                if value >= 0.9:
                                    return "correct"
                                elif value >= 0.5:
                                    return "partial"
                                else:
                                    return "incorrect"
                            return str(value)
                        elif isinstance(value, bool):
                            return "correct" if value else "incorrect"
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    correct_val = last_obj["correct"]
                    if isinstance(correct_val, bool):
                        return "correct" if correct_val else "incorrect"
                    elif isinstance(correct_val, str):
                        cleaned = correct_val.strip().lower()
                        if cleaned in ["true", "yes", "1"]:
                            return "correct"
                        elif cleaned in ["false", "no", "0"]:
                            return "incorrect"
                    return str(correct_val)
                
                # Check for nested structures that might contain the answer
                if "analysis" in last_obj and isinstance(last_obj["analysis"], dict):
                    analysis = last_obj["analysis"]
                    for key in ["verdict", "conclusion", "assessment", "grade"]:
                        if key in analysis:
                            return str(analysis[key]).strip()
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"JSON extraction failed: {e}")
        
        # Fallback: Look for explicit keywords in the raw text
        text_lower = last_text.lower()
        
        # Check for explicit grading keywords in the text
        if '"response":' in text_lower or '"grade":' in text_lower or '"evaluation":' in text_lower:
            # Try to extract value after these keywords
            import re
            pattern = r'"(?:response|grade|evaluation)":\s*"([^"]+)"'
            match = re.search(pattern, last_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip().lower()
                if value in ["correct", "incorrect", "partial"]:
                    return value
        
        # Last resort: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _calculate_confidence(self, prediction: str, msg_history: list[dict]) -> float:
        """Calculate confidence score for the prediction.
        
        Returns a confidence score between 0.0 and 1.0.
        
        Improvements:
        - Better handling of edge cases
        - Considers reasoning quality in confidence
        - Handles empty or malformed predictions
        """
        if not msg_history:
            return 0.0
        
        if prediction == "None" or not prediction:
            return 0.0
        
        confidence = 0.5  # Base confidence
        last_text = msg_history[-1].get("text", "")
        
        if not last_text:
            return 0.1  # Very low confidence if no text
        
        # Boost confidence if we got valid JSON
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                confidence += 0.2
                last_obj = extracted[-1]
                
                # Check for standard response field
                if "response" in last_obj:
                    response_val = str(last_obj["response"]).lower()
                    if response_val in ["correct", "incorrect", "partial"]:
                        confidence += 0.2
                
                # Check for reasoning field (indicates thorough analysis)
                if "reasoning" in last_obj:
                    reasoning = last_obj["reasoning"]
                    if isinstance(reasoning, str) and len(reasoning) > 50:
                        confidence += 0.1  # Bonus for detailed reasoning
                
                # Check for confidence score in response
                if "confidence" in last_obj:
                    try:
                        resp_confidence = float(last_obj["confidence"])
                        if 0 <= resp_confidence <= 1:
                            # Blend our confidence with model's confidence
                            confidence = (confidence + resp_confidence) / 2
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        
        # Boost confidence for clear, normalized predictions
        pred_lower = prediction.lower()
        if pred_lower in ["correct", "incorrect", "partial"]:
            confidence += 0.15
        
        # Reduce confidence for ambiguous predictions
        if len(prediction) > 50:
            confidence -= 0.2
        
        # Reduce confidence for very short predictions (might be incomplete)
        if len(prediction) < 3:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        
        Improvements:
        - Handles numeric scores (0-1 scale)
        - Better handling of compound phrases
        - Support for IMO-specific grading terms
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        pred_clean = pred_lower.strip('"\'.,;: ')
        
        # Try to parse as numeric score (0-1 scale)
        try:
            score = float(pred_clean)
            if 0 <= score <= 1:
                if score >= 0.9:
                    return "correct"
                elif score >= 0.5:
                    return "partial"
                else:
                    return "incorrect"
        except ValueError:
            pass
        
        # Try to parse as percentage
        if pred_clean.endswith('%'):
            try:
                score = float(pred_clean[:-1]) / 100
                if score >= 0.9:
                    return "correct"
                elif score >= 0.5:
                    return "partial"
                else:
                    return "incorrect"
            except ValueError:
                pass
        
        # Exact matches
        exact_matches = {
            "correct": "correct",
            "incorrect": "incorrect", 
            "partial": "partial",
            "true": "correct",
            "false": "incorrect",
            "right": "correct",
            "wrong": "incorrect",
            "valid": "correct",
            "invalid": "incorrect",
            "accepted": "correct",
            "rejected": "incorrect",
            "incomplete": "partial",
            "partially correct": "partial",
            "half correct": "partial",
            "mostly correct": "partial",
            "full marks": "correct",
            "full credit": "correct",
            "no credit": "incorrect",
            "zero": "incorrect",
            "full": "correct",
            "none": "incorrect",
            "pass": "correct",
            "fail": "incorrect",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Keyword-based detection (priority order - most specific first)
        # Check for compound phrases first
        compound_phrases = [
            ("not correct", "incorrect"),
            ("not right", "incorrect"),
            ("not valid", "incorrect"),
            ("not accepted", "incorrect"),
            ("partially correct", "partial"),
            ("mostly correct", "partial"),
            ("half correct", "partial"),
            ("partial credit", "partial"),
            ("some credit", "partial"),
        ]
        for phrase, result in compound_phrases:
            if phrase in pred_lower:
                return result
        
        # Single keyword detection
        incorrect_indicators = ["incorrect", "wrong", "false", "invalid", "rejected", "error", "mistake"]
        for indicator in incorrect_indicators:
            if indicator in pred_lower:
                return "incorrect"
        
        partial_indicators = ["partial", "incomplete", "half", "mostly", "some", "minor"]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        correct_indicators = ["correct", "right", "true", "valid", "accepted", "accurate", "proper"]
        for indicator in correct_indicators:
            if indicator in pred_lower:
                return "correct"
        
        return prediction

    def _call_llm_with_retry(
        self,
        msg: str,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> tuple[str, list[dict], dict]:
        """Call LLM with retry logic for transient failures.
        
        Args:
            msg: The message to send to the LLM.
            max_retries: Maximum number of retry attempts.
            temperature: Temperature for LLM sampling.
            
        Returns:
            Tuple of (response_text, msg_history, info).
            
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=msg,
                    model=self.model,
                    msg_history=[],
                )
                return response, msg_history, info
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Don't retry on authentication or context length errors
                if "invalid api key" in error_str or "authentication" in error_str:
                    self.log_fn(f"Authentication error, not retrying: {e}")
                    raise
                if "context length" in error_str or "too long" in error_str:
                    self.log_fn(f"Context length exceeded, not retrying: {e}")
                    raise
                
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.log_fn(f"All {max_retries} retry attempts exhausted. Last error: {e}")
        
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

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

        try:
            response, msg_history, info = self._call_llm_with_retry(
                msg=instruction,
                max_retries=3,
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract prediction from JSON
        raw_prediction = self._extract_prediction(msg_history)
        
        # Normalize prediction to standard format
        prediction = self._normalize_prediction(raw_prediction)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(prediction, msg_history)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
        else:
            # Log successful extraction with confidence
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview} (confidence: {confidence:.2f})")

        return str(prediction), msg_history
