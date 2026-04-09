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
    if not text or not isinstance(text, str):
        return None
        
    results = []
    search_from = 0
    max_iterations = 100  # Safety limit to prevent infinite loops
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug("Found opening <json> tag but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty JSON blocks
        if not inner:
            continue
            
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                # Flatten list results into individual objects if they are dicts
                for item in parsed:
                    if isinstance(item, dict):
                        results.append(item)
            else:
                # Wrap non-dict values in a dict
                results.append({"value": parsed})
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            # Try to repair before giving up
            try:
                repaired = _repair_json(inner)
                if repaired and isinstance(repaired, dict):
                    results.append(repaired)
            except Exception as repair_err:
                logger.debug(f"JSON repair failed: {repair_err}")
            continue
    
    if iterations >= max_iterations:
        logger.warning(f"Reached max iterations ({max_iterations}) in _extract_jsons")
        
    return results if results else None


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
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    if not text:
        return None
        
    results = []
    max_iterations = 50
    
    # Strategy 1: <json> tags (original) - use the improved _extract_jsons
    json_tag_results = _extract_jsons(text)
    if json_tag_results:
        results.extend(json_tag_results)
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            if not match.strip():
                continue
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict):
                    results.append(parsed)
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            results.append(item)
            except json.JSONDecodeError:
                repaired = _repair_json(match.strip())
                if repaired:
                    results.append(repaired)
    
    # Strategy 3: Look for JSON objects directly with improved brace matching
    if not results:
        # Find all potential JSON object starts
        start_indices = [m.start() for m in re.finditer(r'\{[\s]*"', text)]
        
        for start in start_indices[:max_iterations]:  # Limit iterations
            try:
                # Use proper brace counting to find the matching end
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
                        parsed = json.loads(obj_str)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        repaired = _repair_json(obj_str)
                        if repaired:
                            results.append(repaired)
            except Exception:
                continue
    
    # Strategy 4: Try to find JSON at the very beginning or end of text
    if not results:
        # Try beginning
        if text.startswith('{'):
            try:
                # Find first complete object
                brace_count = 0
                for i, char in enumerate(text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[:i+1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    results.append(parsed)
                                    break
                            except json.JSONDecodeError:
                                repaired = _repair_json(candidate)
                                if repaired:
                                    results.append(repaired)
                                    break
                            break
            except Exception:
                pass
        
        # Try end
        if not results and text.rstrip().endswith('}'):
            try:
                # Find last complete object from the end
                rev_text = text[::-1]
                brace_count = 0
                for i, char in enumerate(rev_text):
                    if char == '}':
                        brace_count += 1
                    elif char == '{':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[-(i+1):]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    results.append(parsed)
                                    break
                            except json.JSONDecodeError:
                                repaired = _repair_json(candidate)
                                if repaired:
                                    results.append(repaired)
                                    break
                            break
            except Exception:
                pass
    
    return results if results else None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Comments in JSON (// and /* */)
    - Unescaped quotes in strings
    - Unicode escape sequences
    
    Args:
        text: The potentially malformed JSON string to repair.
        
    Returns:
        A parsed JSON dict if repair succeeds, None otherwise.
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    if not text:
        return None
    
    # First, try parsing as-is
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0]
        else:
            return {"value": result}
    except json.JSONDecodeError:
        pass
    
    # Remove comments (// and /* */)
    repaired = re.sub(r'//.*?\n', '\n', text)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not in already-escaped contexts)
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)
    
    # Escape unescaped tabs in strings
    repaired = re.sub(r'(?<!\\)\t', r'\\t', repaired)
    
    # Escape unescaped carriage returns
    repaired = re.sub(r'(?<!\\)\r', r'\\r', repaired)
    
    # Fix unescaped quotes inside strings (complex - handle common cases)
    # This is a simplified approach - look for quotes that appear to be inside values
    repaired = re.sub(r'(?<=[^\\])"(?=[^:,}\]]*["\'])', r'\\"', repaired)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Remove extra closing braces from the end
        for _ in range(-open_braces):
            last_brace = repaired.rfind('}')
            if last_brace > 0 and last_brace == len(repaired) - 1:
                repaired = repaired[:last_brace].rstrip()
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        # Remove extra closing brackets from the end
        for _ in range(-open_brackets):
            last_bracket = repaired.rfind(']')
            if last_bracket > 0 and last_bracket == len(repaired) - 1:
                repaired = repaired[:last_bracket].rstrip()
    
    # Try parsing the repaired version
    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0]
        else:
            return {"value": result}
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
                    candidate = repaired[start:start+i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try one more repair on just this object
                        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                        return json.loads(candidate)
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
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: <json> tags (original format)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with repair
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Look for any JSON-like structure with proper brace matching
    try:
        # Find all potential starting points
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in start_indices[:20]:  # Limit to first 20 candidates
            # Use proper brace counting with string awareness
            brace_count = 0
            in_string = False
            escape_next = False
            
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
                            candidate = text[start:start+i+1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    return [parsed]
                            except json.JSONDecodeError:
                                repaired = _repair_json(candidate)
                                if repaired:
                                    return [repaired]
                            break
    except Exception as e:
        logger.debug(f"Strategy 3 failed: {e}")
    
    # Strategy 4: Last resort - try to extract the largest JSON-like block
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

CRITICAL RULES:
1. The "response" field must contain ONLY one of these three exact lowercase values: "correct", "incorrect", or "partial"
2. No other text, explanations, or formatting allowed in the "response" field
3. Use "partial" when the student has some correct work but not a complete solution
4. Use "incorrect" when the answer is fundamentally wrong or missing critical components
5. Use "correct" only when the solution is fully correct with proper justification"""
        
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
                for key in ["response", "grade", "score", "evaluation", "answer", "result", "conclusion"]:
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
            # Prioritize response field
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
        
        # Check for correctness indicators - check for "incorrect" first to avoid false positives
        if "incorrect" in text_lower or "not correct" in text_lower or "wrong" in text_lower:
            return "incorrect"
        
        if "correct" in text_lower:
            return "correct"
        
        # Check for partial credit indicators
        if any(term in text_lower for term in ["partial", "partially", "some credit", "incomplete"]):
            return "partial"
        
        # Fallback 3: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
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
