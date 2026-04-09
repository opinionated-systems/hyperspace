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
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON objects at start/end of text
    4. JSON-like structures with relaxed parsing
    5. Repair common JSON syntax errors
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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
    
    # Strategy 3: Look for JSON objects directly with improved nested structure handling
    if not results:
        # Find all potential JSON starting points
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in start_indices:
            try:
                # Use brace counting to find the matching end
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
    - Unescaped quotes within strings
    - BOM and other invisible characters
    
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
    
    # Remove comments (// and /* */) - handle end-of-string comments too
    repaired = re.sub(r'//[^\n]*(?:\n|$)', '\n', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not in the overall structure)
    # This is tricky - we need to be careful not to break valid JSON
    # Only escape newlines that appear to be inside string values
    def escape_newlines_in_strings(match):
        # Escape unescaped newlines and tabs within the matched string
        content = match.group(1)
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\t', r'\\t', content)
        content = re.sub(r'(?<!\\)"', r'\\"', content)
        return f'"{content}"'
    
    # Try to find and fix strings with unescaped characters
    # Match content between quotes that contains newlines or tabs
    repaired = re.sub(r'"([^"]*(?:\n|\t)[^"]*)"', escape_newlines_in_strings, repaired)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Remove extra closing braces from the end
        for _ in range(-open_braces):
            last_brace = repaired.rfind('}')
            if last_brace > 0 and last_brace == len(repaired) - 1:
                repaired = repaired[:last_brace]
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        # Remove extra closing brackets from the end
        for _ in range(-open_brackets):
            last_bracket = repaired.rfind(']')
            if last_bracket > 0 and last_bracket == len(repaired) - 1:
                repaired = repaired[:last_bracket]
    
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
        for i, char in enumerate(repaired[start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete object
                    candidate = repaired[start:start+i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try one more repair on just this object
                        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                        return json.loads(candidate)
        return None
    except Exception:
        pass
    
    # Final attempt: try to find and parse any valid JSON subset
    try:
        # Look for patterns like {"key": "value"} or {"key": value}
        pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
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
        numeric scores, and partial credit scenarios. Includes confidence scoring
        and improved edge case handling.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Try robust extraction first (includes all strategies)
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                
                # Check for structured grading response first
                for key in ["grade", "score", "evaluation", "response", "answer", "result", "conclusion", "prediction"]:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # Check for correctness boolean
                if "correct" in last_obj:
                    correct_val = last_obj["correct"]
                    if isinstance(correct_val, bool):
                        return "correct" if correct_val else "incorrect"
                    return str(correct_val)
                
                # Check for points/partial credit
                if "points" in last_obj:
                    return f"points:{last_obj['points']}"
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                r'"grade"\s*:\s*"([^"]+)"',
                r'"score"\s*:\s*"?([^"},\s]+)"?',
                r'"evaluation"\s*:\s*"([^"]+)"',
                r'"response"\s*:\s*"([^"]+)"',
                r'"answer"\s*:\s*"([^"]+)"',
                r'"correct"\s*:\s*(true|false)',
                r'"prediction"\s*:\s*"([^"]+)"',
                r'"result"\s*:\s*"([^"]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    return match.group(1).lower() if len(match.groups()) > 0 else match.group(0)
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Enhanced text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores (e.g., "score: 7", "7/7", "7 points")
        score_patterns = [
            r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*/\s*\d+\s*(?:points?)?',
            r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?',
            r'(?:score|grade)\s+of\s+(\d+)',
            r'(?:score|grade)\s+is\s+(\d+)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        
        # Check for explicit grading statements
        grading_statements = [
            (r'(?:the\s+)?student\s+(?:answer|solution)\s+is\s+(correct|incorrect|partial|wrong|right)', 1),
            (r'(?:this\s+)?(?:answer|solution)\s+is\s+(correct|incorrect|partial|wrong|right)', 1),
            (r'(?:i\s+)?(?:would\s+)?(?:grade|rate|score|evaluate)\s+(?:this\s+)?(?:as\s+)?(correct|incorrect|partial|wrong|right)', 1),
            (r'(?:the\s+)?(?:grade|score|evaluation)\s+(?:is\s+)?(correct|incorrect|partial|wrong|right)', 1),
        ]
        for pattern, group in grading_statements:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(group).lower()
                # Normalize common variations
                if result in ["right", "valid", "accepted", "full credit"]:
                    return "correct"
                elif result in ["wrong", "invalid", "rejected", "no credit"]:
                    return "incorrect"
                return result
        
        # Check for correctness indicators with negation handling
        if "correct" in text_lower:
            # Distinguish between "correct" and "incorrect" / "not correct"
            if re.search(r'\b(not\s+correct|incorrect|not\s+right)\b', text_lower):
                return "incorrect"
            return "correct"
        
        # Check for partial credit indicators
        partial_indicators = ["partial", "partially", "some credit", "incomplete", "half credit", "partial credit"]
        for indicator in partial_indicators:
            if indicator in text_lower:
                return "partial"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _calculate_confidence(self, prediction: str, msg_history: list[dict]) -> float:
        """Calculate confidence score for the prediction.
        
        Returns a confidence score between 0.0 and 1.0 based on:
        - Whether prediction was extracted from valid JSON
        - Presence of reasoning in the response
        - Clarity of the prediction value
        
        Args:
            prediction: The extracted prediction string
            msg_history: The message history from the LLM
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not msg_history or prediction == "None":
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        last_text = msg_history[-1].get("text", "")
        
        # Boost confidence if we got valid JSON
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                confidence += 0.2
                last_obj = extracted[-1]
                # Extra boost if we have reasoning
                if "reasoning" in last_obj and last_obj["reasoning"]:
                    confidence += 0.1
                # Extra boost if response field is present and valid
                if "response" in last_obj:
                    response_val = str(last_obj["response"]).lower()
                    if response_val in ["correct", "incorrect", "partial"]:
                        confidence += 0.2
        except Exception:
            pass
        
        # Boost confidence for clear, normalized predictions
        pred_lower = prediction.lower()
        if pred_lower in ["correct", "incorrect", "partial"]:
            confidence += 0.15
        elif pred_lower.startswith("score:"):
            confidence += 0.1
        
        # Reduce confidence for ambiguous predictions
        if len(prediction) > 50:  # Long text suggests extraction failed
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Uses a priority-based approach to handle ambiguous cases.
        Enhanced with additional edge case handling and scoring normalization.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Remove common punctuation and whitespace variations
        pred_clean = pred_lower.strip('"\'.,;: ')
        
        # Exact matches first (highest priority)
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
            "partially incorrect": "partial",
            "half": "partial",
            "half correct": "partial",
            "mostly correct": "partial",
            "mostly incorrect": "partial",
            "some correct": "partial",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format (e.g., "score:7", "7/7")
        if pred_lower.startswith("score:") or pred_lower.startswith("points:"):
            # Extract numeric value and normalize
            score_match = re.search(r'(?:score|points):\s*(\d+(?:\.\d+)?)', pred_lower)
            if score_match:
                score_val = float(score_match.group(1))
                # Normalize based on typical scoring ranges
                if score_val >= 7:  # Full marks
                    return "correct"
                elif score_val <= 1:  # Very low score
                    return "incorrect"
                else:
                    return "partial"
            return prediction
        
        # Check for numeric patterns that might indicate scoring
        numeric_match = re.search(r'(\d+)\s*/\s*(\d+)', pred_lower)
        if numeric_match:
            num, denom = int(numeric_match.group(1)), int(numeric_match.group(2))
            if denom == 0:
                return prediction  # Avoid division by zero
            ratio = num / denom
            if ratio >= 0.9:  # 90% or higher
                return "correct"
            elif ratio <= 0.1:  # 10% or lower
                return "incorrect"
            else:
                return "partial"
        
        # Check for standalone numbers (assume out of typical max)
        standalone_num = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', pred_clean)
        if standalone_num:
            num = float(standalone_num.group(1))
            # Handle different scoring scales
            if num >= 7:  # Full marks on typical scale
                return "correct"
            elif num >= 4:  # Partial credit
                return "partial"
            elif num > 0:  # Some credit but low
                return "partial"
            else:
                return "incorrect"
        
        # Priority-based keyword detection
        # Check for incorrect first (to catch "not correct" patterns)
        incorrect_indicators = [
            "incorrect", "wrong", "false", "invalid", "rejected", "error", 
            "not correct", "not right", "not valid", "no credit", "zero"
        ]
        for indicator in incorrect_indicators:
            if indicator in pred_lower:
                return "incorrect"
        
        # Check for partial
        partial_indicators = [
            "partial", "partially", "incomplete", "some credit", "half", 
            "partial credit", "half credit", "mostly", "somewhat",
            "partially complete", "in progress"
        ]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        # Check for correct (after checking for "not correct")
        correct_indicators = [
            "correct", "right", "true", "valid", "accepted", "full credit", 
            "complete", "perfect", "excellent", "full marks"
        ]
        for indicator in correct_indicators:
            if indicator in pred_lower:
                return "correct"
        
        # Return original if no normalization applied
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
