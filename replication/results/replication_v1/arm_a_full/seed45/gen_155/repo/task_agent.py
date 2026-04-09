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
    - Comments in JSON (// and /* */)
    - Unescaped quotes within strings
    
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
    
    # Remove comments (// and /* */) - handle end of string case for //
    repaired = re.sub(r'//[^\n]*(?:\n|$)', '\n', text)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not in already-escaped contexts)
    # Use a more careful approach: only escape newlines that are between quotes
    def escape_newlines_in_json(m):
        # m.group(1) is the content between quotes
        content = m.group(1)
        # Escape unescaped newlines and tabs
        content = re.sub(r'(?<!\\)\n', r'\\n', content)
        content = re.sub(r'(?<!\\)\t', r'\\t', content)
        content = re.sub(r'(?<!\\)"', r'\\"', content)
        return '"' + content + '"'
    
    # Match quoted strings and process them
    repaired = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_newlines_in_json, repaired)
    
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
    
    # Second attempt: more aggressive cleaning
    # Remove control characters except tab, newline, carriage return
    cleaned = ''.join(char for char in repaired if ord(char) >= 32 or char in '\t\n\r')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to extract just the first complete JSON object
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
        
        # Truncate very long inputs to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer
        
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
   - For partial credit: identify which specific steps were correct vs incorrect

5. **Assign Grade**: Based on your analysis, provide your evaluation using the rubric below.

## Grading Rubric

- **Correct**: Fully correct with proper reasoning, all steps justified, final answer matches official solution. Award full marks (7/7 for IMO problems).
- **Incorrect**: Wrong answer, critical errors, incorrect methods, or final answer doesn't match. Award zero or minimal marks (0-1/7).
- **Partial**: Some correct elements but incomplete, minor errors, lacks justification, or only partially matches. Award partial marks (2-6/7 depending on completeness).

## Response Format

You MUST respond in this exact JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be thorough and mention specific mathematical steps, theorems used, and any errors found.",
    "response": "correct"
}}
</json>

The "response" field MUST be exactly one of: "correct", "incorrect", or "partial" (all lowercase, no quotes around the value in the field).

## Examples of Valid Responses

Example 1 - Correct solution:
<json>
{{
    "reasoning": "The student correctly applied the AM-GM inequality to prove the statement. All steps are justified and the final answer matches the official solution exactly.",
    "response": "correct"
}}
</json>

Example 2 - Incorrect solution:
<json>
{{
    "reasoning": "The student made a critical error in the base case of the induction proof. The conclusion does not follow from the given reasoning.",
    "response": "incorrect"
}}
</json>

Example 3 - Partial solution:
<json>
{{
    "reasoning": "The student correctly identified the approach using modular arithmetic but failed to complete the final calculation. The reasoning is sound up to step 3 but incomplete.",
    "response": "partial"
}}
</json>

CRITICAL RULES:
1. The "response" field must contain ONLY one of these three exact lowercase values: "correct", "incorrect", or "partial"
2. No other text, explanations, or formatting allowed in the "response" field
3. Use "partial" when the student has some correct work but not a complete solution
4. Use "incorrect" when the answer is fundamentally wrong or missing critical components
5. Use "correct" only when the solution is fully correct with proper justification
6. Do not include any text outside the <json>...</json> tags"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
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
                
                # Check for structured grading response first - prioritize "response" field
                for key in ["response", "grade", "score", "evaluation", "answer", "result", "conclusion", "verdict"]:
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
                
                # Check for marks (common in IMO grading)
                if "marks" in last_obj:
                    return f"marks:{last_obj['marks']}"
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            # Prioritize response field
            patterns = [
                r'"response"\s*:\s*"([^"]+)"',
                r'"grade"\s*:\s*"([^"]+)"',
                r'"score"\s*:\s*"?([^"},\s]+)"?',
                r'"evaluation"\s*:\s*"([^"]+)"',
                r'"answer"\s*:\s*"([^"]+)"',
                r'"verdict"\s*:\s*"([^"]+)"',
                r'"correct"\s*:\s*(true|false)',
                r'"marks"\s*:\s*(\d+(?:\.\d+)?)',
                r'"points"\s*:\s*(\d+(?:\.\d+)?)',
            ]
            for pattern in patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    return match.group(1).lower() if len(match.groups()) > 0 else match.group(0)
        except Exception as e:
            self.log_fn(f"Regex extraction failed: {e}")
        
        # Fallback 2: Enhanced text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores (e.g., "score: 7", "7/7", "7 points", "marks: 5")
        score_patterns = [
            r'(?:score|grade|points?|marks?)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*\d+\s*(?:points?|marks?)?',
            r'(?:awarded|given|assigned)\s+(\d+(?:\.\d+)?)\s*(?:points?|marks?)?',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        
        # Check for correctness indicators - check for "incorrect" first to avoid false positives
        # Use word boundaries to avoid matching substrings
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bnot correct\b',
            r'\bwrong\b',
            r'\bnot right\b',
            r'\bfalse\b',
            r'\binvalid\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, text_lower):
                return "incorrect"
        
        # Check for correct (but be careful about "not correct" which we already checked)
        if re.search(r'\bcorrect\b', text_lower):
            return "correct"
        
        # Check for partial credit indicators
        partial_patterns = [
            r'\bpartial\b',
            r'\bpartially\b',
            r'\bsome credit\b',
            r'\bincomplete\b',
            r'\bpartial credit\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "partial"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Uses a priority-based approach to handle ambiguous cases.
        Enhanced for IMO grading with better handling of edge cases.
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
            "mostly correct": "partial",
            "mostly incorrect": "partial",
            "full marks": "correct",
            "no marks": "incorrect",
            "zero": "incorrect",
            "full": "correct",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith("score:") or pred_lower.startswith("points:"):
            # Extract the numeric value
            num_match = re.search(r'(?:score|points):\s*(\d+(?:\.\d+)?)', pred_lower)
            if num_match:
                score = float(num_match.group(1))
                # IMO problems are typically out of 7 points
                if score >= 6.5:  # Full or near-full credit
                    return "correct"
                elif score <= 0.5:  # Minimal or no credit
                    return "incorrect"
                else:
                    return "partial"
            return prediction
        
        # Check for numeric patterns that might indicate scoring (e.g., "7/7", "3/7")
        numeric_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', pred_lower)
        if numeric_match:
            num, denom = float(numeric_match.group(1)), float(numeric_match.group(2))
            if denom > 0:
                ratio = num / denom
                if ratio >= 0.9:  # 90% or more
                    return "correct"
                elif ratio <= 0.1:  # 10% or less
                    return "incorrect"
                else:
                    return "partial"
        
        # Check for standalone numbers (assume out of 7 for IMO)
        standalone_num = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', pred_clean)
        if standalone_num:
            num = float(standalone_num.group(1))
            # IMO problems are typically graded out of 7
            if num >= 6.5:  # 7/7 or 6.5/7
                return "correct"
            elif num <= 0.5:  # 0/7 or 0.5/7
                return "incorrect"
            else:
                return "partial"
        
        # Check for percentage patterns
        percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', pred_lower)
        if percent_match:
            percent = float(percent_match.group(1))
            if percent >= 90:
                return "correct"
            elif percent <= 10:
                return "incorrect"
            else:
                return "partial"
        
        # Priority-based keyword detection with negation handling
        # First check for negated correct patterns (e.g., "not correct", "isn't right")
        negation_patterns = [
            r'\bnot\s+correct\b',
            r'\bnot\s+right\b',
            r'\bisn\'t\s+correct\b',
            r'\bisn\'t\s+right\b',
            r'\bnot\s+valid\b',
            r'\bnot\s+accepted\b',
            r'\bnot\s+true\b',
        ]
        for pattern in negation_patterns:
            if re.search(pattern, pred_lower):
                return "incorrect"
        
        # Check for incorrect indicators (higher priority than correct)
        incorrect_indicators = [
            "incorrect", "wrong", "false", "invalid", "rejected", "error",
            "mistake", "flawed", "erroneous", "unsound", "fallacious",
            "does not match", "doesn't match", "mismatch", "contradiction",
            "no credit", "zero credit", "full marks deducted"
        ]
        for indicator in incorrect_indicators:
            if indicator in pred_lower:
                return "incorrect"
        
        # Check for partial indicators
        partial_indicators = [
            "partial", "partially", "incomplete", "some credit", "half",
            "partial credit", "minor error", "small mistake", "mostly",
            "partially correct", "partial marks", "partial solution",
            "some correct", "some valid", "partial success", "in progress"
        ]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        # Check for correct indicators (lowest priority to avoid false positives)
        correct_indicators = [
            "correct", "right", "true", "valid", "accepted", "full credit",
            "complete", "perfect", "accurate", "proper", "sound", "flawless",
            "excellent", "matches", "equivalent", "consistent", "verified",
            "full marks", "all correct", "fully correct", "entirely correct"
        ]
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
        
        # Validate input types
        for key in required_keys:
            if not isinstance(inputs.get(key), str):
                error_msg = f"Error: Input '{key}' must be a string"
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
            # Try one more extraction attempt with the raw response text directly
            if response:
                backup_extraction = self._extract_prediction([{"role": "assistant", "text": response}])
                if backup_extraction != "None":
                    prediction = self._normalize_prediction(backup_extraction)
                    self.log_fn(f"Backup extraction succeeded: {prediction}")
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history
