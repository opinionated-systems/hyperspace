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
        """Extract prediction from message history with unified extraction strategy.
        
        Uses a single-pass approach that prioritizes structured JSON extraction
        with intelligent fallback to pattern matching for robustness.
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        # Unified extraction: try JSON first, then pattern matching
        text_lower = last_text.lower()
        
        # Step 1: Try structured JSON extraction
        try:
            extracted = _extract_json_robust(last_text)
            if extracted and extracted[-1]:
                last_obj = extracted[-1]
                
                # Priority-ordered field extraction
                priority_fields = [
                    ("response", None),      # Primary field from our prompt
                    ("grade", None),
                    ("evaluation", None),
                    ("answer", None),
                    ("result", None),
                    ("conclusion", None),
                    ("correct", lambda v: "correct" if v else "incorrect" if isinstance(v, bool) else str(v)),
                    ("points", lambda v: f"points:{v}"),
                    ("score", lambda v: f"score:{v}"),
                ]
                
                for field, transform in priority_fields:
                    if field in last_obj:
                        value = last_obj[field]
                        if transform:
                            return transform(value)
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)
                        return json.dumps(value)
        except Exception as e:
            self.log_fn(f"JSON extraction failed: {e}")
        
        # Step 2: Pattern-based extraction for common response formats
        # Combined regex patterns for efficiency
        patterns = [
            # JSON-style patterns
            (r'"response"\s*:\s*"([^"]+)"', 1),
            (r'"grade"\s*:\s*"([^"]+)"', 1),
            (r'"evaluation"\s*:\s*"([^"]+)"', 1),
            (r'"correct"\s*:\s*(true|false)', 1),
            # Score patterns
            (r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', 1),
            (r'(\d+)\s*/\s*\d+\s*(?:points?)?', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(group)
                # Handle boolean strings
                if result in ("true", "false"):
                    return "correct" if result == "true" else "incorrect"
                return result
        
        # Step 3: Keyword-based classification with priority ordering
        # Check for incorrect first (catches "not correct" patterns)
        if re.search(r'\b(incorrect|wrong|false|invalid|rejected|error|not\s+correct|not\s+right)\b', text_lower):
            return "incorrect"
        
        # Check for partial credit
        if re.search(r'\b(partial|partially|incomplete|some\s+credit|half|partial\s+credit)\b', text_lower):
            return "partial"
        
        # Check for correct (after checking for negations)
        if re.search(r'\b(correct|right|true|valid|accepted|full\s+credit|complete)\b', text_lower):
            return "correct"
        
        # Step 4: Return truncated text as last resort
        stripped = last_text.strip()
        if stripped:
            return stripped[:200] if len(stripped) > 200 else stripped
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Uses a priority-based approach to handle ambiguous cases.
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
        
        # Handle score format (e.g., "score:7", "7/7")
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Check for numeric patterns that might indicate scoring
        numeric_match = re.search(r'(\d+)\s*/\s*(\d+)', pred_lower)
        if numeric_match:
            num, denom = int(numeric_match.group(1)), int(numeric_match.group(2))
            if num == denom:
                return "correct"
            elif num == 0:
                return "incorrect"
            else:
                return "partial"
        
        # Check for standalone numbers (assume out of some max)
        standalone_num = re.search(r'^\s*(\d+)\s*$', pred_clean)
        if standalone_num:
            num = int(standalone_num.group(1))
            if num >= 7:  # Assuming 7 is max for IMO
                return "correct"
            elif num == 0:
                return "incorrect"
            else:
                return "partial"
        
        # Priority-based keyword detection
        # Check for incorrect first (to catch "not correct" patterns)
        incorrect_indicators = ["incorrect", "wrong", "false", "invalid", "rejected", "error", "not correct", "not right"]
        for indicator in incorrect_indicators:
            if indicator in pred_lower:
                return "incorrect"
        
        # Check for partial
        partial_indicators = ["partial", "partially", "incomplete", "some credit", "half", "partial credit"]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        # Check for correct (after checking for "not correct")
        correct_indicators = ["correct", "right", "true", "valid", "accepted", "full credit", "complete"]
        for indicator in correct_indicators:
            if indicator in pred_lower:
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
