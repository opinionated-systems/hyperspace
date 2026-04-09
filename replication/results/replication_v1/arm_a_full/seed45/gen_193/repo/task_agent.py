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
    Also handles nested JSON structures within the tags.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        try:
            parsed = json.loads(inner)
            results.append(parsed)
        except json.JSONDecodeError as e:
            # Log detailed error for debugging
            logger.debug(f"JSON parse error in block {extraction_attempts}: {e}")
            
            # Try nested extraction for complex JSON structures
            if '{' in inner:
                nested_parsed, _ = _extract_nested_json(inner, inner.find('{'))
                if nested_parsed:
                    results.append(nested_parsed)
                    logger.debug(f"Extracted nested JSON from block {extraction_attempts}")
                    continue
            
            # Try to extract partial data if possible
            try:
                # Attempt to find and extract just the prediction field
                pred_match = re.search(r'"prediction"\s*:\s*"([^"]*)"', inner)
                if pred_match:
                    partial = {"prediction": pred_match.group(1)}
                    results.append(partial)
                    logger.debug(f"Extracted partial JSON with prediction field")
                    continue
                
                # Try to extract response field
                resp_match = re.search(r'"response"\s*:\s*"([^"]*)"', inner)
                if resp_match:
                    partial = {"response": resp_match.group(1)}
                    results.append(partial)
                    logger.debug(f"Extracted partial JSON with response field")
            except Exception:
                pass
            continue
    
    logger.debug(f"Extracted {len(results)} valid JSON objects from {extraction_attempts} attempts")
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON objects using nested extraction
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
    
    # Strategy 3: Look for JSON objects using robust nested extraction
    if not results:
        search_start = 0
        while True:
            start = text.find('{', search_start)
            if start == -1:
                break
            parsed, end = _extract_nested_json(text, start)
            if parsed:
                results.append(parsed)
                search_start = end
            else:
                search_start = start + 1
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Unescaped quotes in strings
    - Control characters
    """
    if not text or not text.strip():
        return None
    
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove control characters except for valid whitespace
    repaired = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    # This is a best-effort approach
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not in already-escaped sequences)
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)
    repaired = re.sub(r'(?<!\\)\r', r'\\r', repaired)
    repaired = re.sub(r'(?<!\\)\t', r'\\t', repaired)
    
    # Fix unescaped quotes inside strings (complex case - try to fix common patterns)
    # This handles cases like: "key": "value with "quotes" inside"
    # We try to escape quotes that appear to be inside string values
    
    # Try parsing the repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Try to balance braces using the nested extraction logic
    try:
        start = repaired.find('{')
        if start == -1:
            return None
        
        # Use the nested extraction helper to find a complete object
        parsed, _ = _extract_nested_json(repaired, start)
        if parsed:
            return parsed
    except Exception:
        pass
    
    # Last resort: try to extract just the first complete JSON object
    try:
        # Find the first { and matching }
        start = repaired.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        in_string = False
        escaped = False
        
        for i, char in enumerate(repaired[start:]):
            if escaped:
                escaped = False
                continue
            if char == '\\' and in_string:
                escaped = True
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
                        # Found complete object
                        return json.loads(repaired[start:start+i+1])
        return None
    except Exception:
        return None


def _extract_nested_json(text: str, start_idx: int) -> tuple[dict | None, int]:
    """Extract a single JSON object starting at the given index.
    
    Handles nested braces correctly by counting brace depth.
    
    Args:
        text: The text to parse
        start_idx: Index where '{' starts
        
    Returns:
        Tuple of (parsed_dict or None, end_index or -1)
    """
    if start_idx >= len(text) or text[start_idx] != '{':
        return None, -1
    
    brace_count = 0
    in_string = False
    escaped = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escaped:
            escaped = False
            continue
            
        if char == '\\' and in_string:
            escaped = True
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
                    # Found complete JSON object
                    try:
                        obj_str = text[start_idx:i+1]
                        parsed = json.loads(obj_str)
                        return parsed, i + 1
                    except json.JSONDecodeError:
                        # Try to repair
                        repaired = _repair_json(obj_str)
                        if repaired:
                            return repaired, i + 1
                        return None, i + 1
    
    return None, -1


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
    
    # Strategy 3: Look for any JSON-like structure using nested extraction
    # This handles complex nested JSON better than simple find/rfind
    results = []
    search_start = 0
    while True:
        start = text.find('{', search_start)
        if start == -1:
            break
        parsed, end = _extract_nested_json(text, start)
        if parsed:
            results.append(parsed)
            search_start = end
        else:
            search_start = start + 1
    
    if results:
        return results
    
    # Strategy 4: Last resort - try to find content between outermost braces
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

5. **Assign Grade**: Based on your analysis, provide your evaluation.

## Grading Rubric

When evaluating, use these criteria:
- **Correct**: The answer is fully correct with proper reasoning, all steps are justified, and the final answer matches the official solution.
- **Incorrect**: The answer is wrong, has critical errors, uses incorrect methods, or the final answer does not match the official solution.
- **Partial**: The answer has some correct elements but is incomplete, has minor errors, lacks proper justification, or only partially matches the official solution.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here, including specific observations about the student's work. Be thorough and mention specific mathematical steps, theorems used, and any errors found.",
    "response": "Your final evaluation here - must be exactly one of: 'correct', 'incorrect', or 'partial'"
}}
</json>

IMPORTANT: The "response" field must contain ONLY one of these three exact values: "correct", "incorrect", or "partial". Do not include any other text, explanations, or formatting in this field."""
        
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
                
                # Check for structured grading response first
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
                
                # If no known field, return the whole object as string
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"Flexible extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            patterns = [
                r'"response"\s*:\s*"([^"]+)"',
                r'"grade"\s*:\s*"([^"]+)"',
                r'"score"\s*:\s*"?([^"},\s]+)"?',
                r'"evaluation"\s*:\s*"([^"]+)"',
                r'"answer"\s*:\s*"([^"]+)"',
                r'"correct"\s*:\s*(true|false)',
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
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        
        # Check for correctness indicators with better negation handling
        # Look for explicit statements first
        if "the answer is correct" in text_lower or "this is correct" in text_lower:
            if "not correct" not in text_lower and "incorrect" not in text_lower:
                return "correct"
        
        if "the answer is incorrect" in text_lower or "this is incorrect" in text_lower:
            return "incorrect"
        
        if "partially correct" in text_lower or "partial credit" in text_lower:
            return "partial"
        
        # Check for standalone correctness keywords
        if "correct" in text_lower:
            # Distinguish between "correct" and "incorrect"
            if "incorrect" in text_lower or "not correct" in text_lower:
                return "incorrect"
            return "correct"
        
        # Check for partial credit indicators
        if any(term in text_lower for term in ["partial", "partially", "some credit", "incomplete"]):
            return "partial"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        # Log extraction failure details before returning None
        self._log_extraction_failure(last_text, "_extract_prediction")
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
            "yes": "correct",
            "no": "incorrect",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith("score:") or pred_lower.startswith("points:"):
            # Extract the numeric value
            num_match = re.search(r'\d+', pred_lower)
            if num_match:
                num = int(num_match.group())
                # Assume max score of 7 for IMO problems
                if num >= 7:
                    return "correct"
                elif num == 0:
                    return "incorrect"
                else:
                    return "partial"
            return prediction
        
        # Check for numeric patterns that might indicate scoring
        numeric_match = re.search(r'(\d+)\s*/\s*(\d+)', pred_lower)
        if numeric_match:
            num, denom = int(numeric_match.group(1)), int(numeric_match.group(2))
            if denom > 0:  # Avoid division by zero
                ratio = num / denom
                if ratio >= 0.9:  # 90% or more is correct
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
        incorrect_indicators = ["incorrect", "wrong", "false", "invalid", "rejected", "error", "not correct", "not right", "not valid"]
        for indicator in incorrect_indicators:
            if indicator in pred_lower:
                return "incorrect"
        
        # Check for partial
        partial_indicators = ["partial", "partially", "incomplete", "some credit", "half", "partial credit", "partially right"]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        # Check for correct (after checking for "not correct")
        correct_indicators = ["correct", "right", "true", "valid", "accepted", "full credit", "complete", "accurate"]
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
        
        # Validate input types and content
        for key in required_keys:
            value = inputs.get(key)
            if not isinstance(value, str):
                error_msg = f"Error: Input '{key}' must be a string, got {type(value).__name__}"
                self.log_fn(error_msg)
                return error_msg, [{"role": "assistant", "text": error_msg}]
            if len(value.strip()) == 0:
                error_msg = f"Error: Input '{key}' cannot be empty or whitespace only"
                self.log_fn(error_msg)
                return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        # Log the problem being processed (truncated for privacy)
        problem_preview = inputs.get("problem", "")[:100]
        student_preview = inputs.get("student_answer", "")[:50]
        self.log_fn(f"Processing problem: {problem_preview}...")
        self.log_fn(f"Student answer preview: {student_preview}...")

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

        # Extract prediction from JSON with multiple fallback strategies
        raw_prediction = self._extract_prediction(msg_history)
        
        # If primary extraction fails, try emergency extraction
        if raw_prediction == "None" and msg_history:
            raw_prediction = self._emergency_extract_prediction(msg_history)
        
        # Normalize prediction to standard format
        prediction = self._normalize_prediction(raw_prediction)
        
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response")
            # Log detailed debugging information
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self._log_extraction_failure(raw_text, "forward")
                # Try to extract any meaningful text as last resort
                prediction = self._last_resort_extraction(raw_text)
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")

        return str(prediction), msg_history

    def _emergency_extract_prediction(self, msg_history: list[dict]) -> str:
        """Emergency extraction when standard methods fail.
        
        Uses aggressive text analysis to find any indication of grading.
        """
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return "None"
            
        text_lower = last_text.lower()
        
        # Look for explicit statements about correctness (highest priority)
        explicit_correct = [
            "the answer is correct",
            "this is correct",
            "the student's answer is correct",
            "the solution is correct",
            "this answer is correct",
            "grading: correct",
            "evaluation: correct",
            "verdict: correct",
        ]
        for pattern in explicit_correct:
            if pattern in text_lower:
                # Make sure it's not negated
                if "not correct" not in text_lower and "incorrect" not in text_lower:
                    return "correct"
        
        explicit_incorrect = [
            "the answer is incorrect",
            "this is incorrect",
            "the student's answer is incorrect",
            "the solution is incorrect",
            "this answer is incorrect",
            "grading: incorrect",
            "evaluation: incorrect",
            "verdict: incorrect",
        ]
        for pattern in explicit_incorrect:
            if pattern in text_lower:
                return "incorrect"
        
        explicit_partial = [
            "partially correct",
            "partial credit",
            "grading: partial",
            "evaluation: partial",
            "verdict: partial",
            "the answer is partially",
        ]
        for pattern in explicit_partial:
            if pattern in text_lower:
                return "partial"
        
        # Look for conclusion/summary sections with more context
        conclusion_patterns = [
            r'conclusion[\s:]*([^\n]{1,100})',
            r'evaluation[\s:]*([^\n]{1,100})',
            r'grade[\s:]*([^\n]{1,100})',
            r'verdict[\s:]*([^\n]{1,100})',
            r'summary[\s:]*([^\n]{1,100})',
            r'final[\s:]*([^\n]{1,100})',
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                conclusion = match.group(1).strip()
                # Check for incorrect first (higher priority to avoid false positives)
                if any(word in conclusion for word in ["incorrect", "wrong", "false", "invalid"]):
                    return "incorrect"
                if "partial" in conclusion:
                    return "partial"
                if any(word in conclusion for word in ["correct", "right", "true", "valid", "accurate"]):
                    # Make sure it's not negated
                    if "not" not in conclusion:
                        return "correct"
        
        # Look for standalone grading keywords in the last few sentences
        sentences = last_text.split('.')
        last_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
        last_text_snippet = ' '.join(last_sentences).lower()
        
        # Check last sentences for grading indicators
        if any(word in last_text_snippet for word in ["incorrect", "wrong", "false"]):
            return "incorrect"
        if "partial" in last_text_snippet:
            return "partial"
        if any(word in last_text_snippet for word in ["correct", "right", "true", "valid"]):
            if "not" not in last_text_snippet:
                return "correct"
        
        return "None"

    def _log_extraction_failure(self, raw_text: str, stage: str) -> None:
        """Log detailed information when extraction fails for debugging.
        
        Args:
            raw_text: The text that failed extraction
            stage: Which extraction stage failed (for context)
        """
        if not raw_text:
            self.log_fn(f"Extraction failure at {stage}: Empty text provided")
            return
        
        # Log summary statistics
        text_len = len(raw_text)
        json_tags = raw_text.count("<json>")
        brace_count = raw_text.count("{")
        
        self.log_fn(f"Extraction failure at {stage}: text_len={text_len}, json_tags={json_tags}, braces={brace_count}")
        
        # Log a snippet of the text for context (first and last 200 chars)
        preview_start = raw_text[:200].replace("\n", " ")
        preview_end = raw_text[-200:].replace("\n", " ") if text_len > 200 else ""
        self.log_fn(f"Text start: {preview_start}...")
        if preview_end:
            self.log_fn(f"Text end: ...{preview_end}")
    
    def _last_resort_extraction(self, raw_text: str) -> str:
        """Last resort: extract any meaningful grading indicator from text.
        
        Returns the most likely prediction based on overall text sentiment.
        Uses weighted scoring and focuses on the last part of the text.
        """
        if not raw_text:
            self.log_fn(f"Last resort extraction: Empty text provided")
            return "None"
        
        text_lower = raw_text.lower()
        
        # Weight indicators by importance
        positive_indicators = {
            "correct": 3, "right": 2, "valid": 2, "properly": 1, 
            "accurate": 2, "full credit": 3, "accepted": 2, "true": 2
        }
        negative_indicators = {
            "incorrect": 3, "wrong": 3, "invalid": 2, "error": 1, 
            "mistake": 2, "no credit": 3, "rejected": 2, "false": 2
        }
        partial_indicators = {
            "partial": 3, "incomplete": 2, "some credit": 3, 
            "partially": 2, "half": 1, "partial credit": 3
        }
        
        # Calculate weighted scores
        pos_score = sum(weight for ind, weight in positive_indicators.items() if ind in text_lower)
        neg_score = sum(weight for ind, weight in negative_indicators.items() if ind in text_lower)
        part_score = sum(weight for ind, weight in partial_indicators.items() if ind in text_lower)
        
        # Check for negation of positive indicators (heavy penalty)
        negation_patterns = ["not correct", "not right", "not valid", "not accurate"]
        for pattern in negation_patterns:
            if pattern in text_lower:
                pos_score = max(0, pos_score - 4)
                neg_score += 2
        
        # Boost scores based on position (later mentions are more important)
        # Focus on the last 500 characters
        last_part = text_lower[-500:] if len(text_lower) > 500 else text_lower
        
        for ind, weight in positive_indicators.items():
            if ind in last_part:
                pos_score += weight * 1.5  # 1.5x weight for mentions in last part
        
        for ind, weight in negative_indicators.items():
            if ind in last_part:
                neg_score += weight * 1.5
        
        for ind, weight in partial_indicators.items():
            if ind in last_part:
                part_score += weight * 1.5
        
        self.log_fn(f"Last resort extraction - pos:{pos_score:.1f} neg:{neg_score:.1f} part:{part_score:.1f}")
        
        # Determine based on weighted scores with thresholds
        if neg_score > pos_score and neg_score > part_score:
            return "incorrect"
        elif part_score > pos_score and part_score > neg_score:
            return "partial"
        elif pos_score > 0 and pos_score >= neg_score:
            return "correct"
        
        return "None"
