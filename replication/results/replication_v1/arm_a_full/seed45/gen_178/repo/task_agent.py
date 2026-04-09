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
    
    Args:
        text: The potentially malformed JSON string to repair.
        
    Returns:
        A parsed JSON dict if repair succeeds, None otherwise.
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
        """Extract prediction from message history with optimized fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios. Uses prioritized extraction
        for better performance and accuracy.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        # Try to find the last assistant message with content
        last_text = ""
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                last_text = msg.get("text", "")
                if last_text:
                    break
        
        if not last_text:
            self.log_fn("Warning: No assistant message with text content found")
            return "None"
        
        # Try robust extraction first (includes all strategies)
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                last_obj = extracted[-1]
                
                # Priority-ordered field extraction for grading responses
                priority_fields = [
                    ("response", None),
                    ("grade", None),
                    ("evaluation", None),
                    ("answer", None),
                    ("result", None),
                    ("conclusion", None),
                    ("correct", lambda v: "correct" if v else "incorrect"),
                    ("points", lambda v: f"points:{v}"),
                ]
                
                for field, transform in priority_fields:
                    if field in last_obj:
                        value = last_obj[field]
                        if transform:
                            return transform(value)
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)
                        elif isinstance(value, (list, dict)):
                            return json.dumps(value)
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"JSON extraction failed: {e}")
        
        # Optimized regex extraction - single pass with combined pattern
        try:
            combined_pattern = r'"(?:grade|score|evaluation|response|answer)"\s*:\s*"?([^"},\s]+)"?'
            match = re.search(combined_pattern, last_text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
            
            # Boolean correct field
            bool_match = re.search(r'"correct"\s*:\s*(true|false)', last_text, re.IGNORECASE)
            if bool_match:
                return "correct" if bool_match.group(1).lower() == "true" else "incorrect"
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Text-based extraction with early termination
        text_lower = last_text.lower()
        
        # Check for numeric scores first (most specific)
        score_match = re.search(r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', text_lower)
        if score_match:
            return f"score:{score_match.group(1)}"
        
        # Check for fraction scores like "7/7"
        fraction_match = re.search(r'(\d+)\s*/\s*\d+', text_lower)
        if fraction_match:
            return f"score:{fraction_match.group(1)}"
        
        # Priority-based keyword detection - check incorrect first to avoid false positives
        if re.search(r'\b(?:in\s*correct|not\s+correct|wrong|error|false|invalid)\b', text_lower):
            return "incorrect"
        
        # Check for partial credit
        if re.search(r'\b(?:partial\w*|some\s+credit|incomplete|partially\s+correct|half\s+credit)\b', text_lower):
            return "partial"
        
        # Check for correct (after checking for incorrect patterns)
        if re.search(r'\bcorrect\b', text_lower):
            return "correct"
        
        # Fallback: return truncated text
        stripped = last_text.strip()
        if stripped:
            return stripped[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Uses a priority-based approach to handle ambiguous cases.
        Includes improved handling of edge cases and compound statements.
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
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith("score:") or pred_lower.startswith("points:"):
            try:
                # Extract the numeric value
                num_match = re.search(r'\d+', pred_lower)
                if num_match:
                    num = int(num_match.group())
                    if num >= 7:  # Assuming 7 is max for IMO
                        return "correct"
                    elif num == 0:
                        return "incorrect"
                    else:
                        return "partial"
            except (ValueError, IndexError):
                pass
            return prediction
        
        # Check for numeric patterns that might indicate scoring (e.g., "7/7", "3/7")
        numeric_match = re.search(r'(\d+)\s*/\s*(\d+)', pred_lower)
        if numeric_match:
            try:
                num, denom = int(numeric_match.group(1)), int(numeric_match.group(2))
                if denom > 0:  # Avoid division by zero
                    ratio = num / denom
                    if ratio >= 0.9:  # 90% or more is correct
                        return "correct"
                    elif ratio <= 0.1:  # 10% or less is incorrect
                        return "incorrect"
                    else:
                        return "partial"
            except (ValueError, ZeroDivisionError):
                pass
        
        # Check for standalone numbers (assume out of some max)
        standalone_num = re.search(r'^\s*(\d+)\s*$', pred_clean)
        if standalone_num:
            try:
                num = int(standalone_num.group(1))
                if num >= 7:  # Assuming 7 is max for IMO
                    return "correct"
                elif num == 0:
                    return "incorrect"
                else:
                    return "partial"
            except ValueError:
                pass
        
        # Priority-based keyword detection with word boundaries
        # Check for incorrect first (to catch "not correct" patterns)
        incorrect_patterns = [
            r'\bin\s*correct\b',
            r'\bnot\s+correct\b',
            r'\bwrong\b',
            r'\berror\b',
            r'\bfalse\b',
            r'\binvalid\b',
            r'\brejected\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, pred_lower):
                return "incorrect"
        
        # Check for partial with word boundaries
        partial_patterns = [
            r'\bpartial\w*\b',
            r'\bsome\s+credit\b',
            r'\bincomplete\b',
            r'\bpartially\s+correct\b',
            r'\bhalf\s+credit\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, pred_lower):
                return "partial"
        
        # Check for correct with word boundaries (after checking for "not correct")
        correct_patterns = [
            r'\bcorrect\b',
            r'\bright\b',
            r'\btrue\b',
            r'\bvalid\b',
            r'\baccepted\b',
            r'\bfull\s+credit\b',
            r'\bcomplete\b',
        ]
        for pattern in correct_patterns:
            if re.search(pattern, pred_lower):
                return "correct"
        
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
        
        # Validate input types
        for key in required_keys:
            value = inputs.get(key)
            if value is not None and not isinstance(value, str):
                error_msg = f"Error: Input '{key}' must be a string, got {type(value).__name__}"
                self.log_fn(error_msg)
                return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        self.log_fn(f"Processing problem: {problem_preview}...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract prediction from JSON
        raw_prediction = self._extract_prediction(msg_history)
        
        # Normalize prediction to standard format
        prediction = self._normalize_prediction(raw_prediction)
        
        # Validate the final prediction
        valid_predictions = {"correct", "incorrect", "partial"}
        if prediction not in valid_predictions and prediction != "None":
            self.log_fn(f"Warning: Prediction '{prediction}' not in standard format, attempting additional normalization")
            # Try one more normalization pass
            prediction = self._normalize_prediction(prediction)
        
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
