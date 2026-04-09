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
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    results = []
    search_from = 0
    max_iterations = 100  # Prevent infinite loops on malformed input
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
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
            # Try to repair the JSON before giving up
            try:
                repaired = _repair_json(inner)
                if repaired:
                    results.append(repaired)
            except Exception:
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
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    results = []
    
    # Strategy 1: <json> tags (original)
    try:
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
    except Exception as e:
        logger.debug(f"Strategy 1 (<json> tags) in flexible extraction failed: {e}")
    
    # Strategy 2: ```json code blocks
    if not results:
        try:
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
        except Exception as e:
            logger.debug(f"Strategy 2 (code blocks) in flexible extraction failed: {e}")
    
    # Strategy 3: Look for JSON objects directly
    if not results:
        try:
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
        except Exception as e:
            logger.debug(f"Strategy 3 (direct JSON) in flexible extraction failed: {e}")
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Control characters in strings
    - Invalid escape sequences
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
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
    
    # Remove control characters (except common whitespace)
    repaired = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', repaired)
    
    # Fix common invalid escape sequences
    repaired = repaired.replace('\\', '\\\\')  # Double-escape backslashes first
    repaired = re.sub(r'\\\\"', r'\\"', repaired)  # Fix over-escaped quotes
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Too many closing braces - try to find valid subset
        repaired = repaired[:repaired.rfind('}') + 1]
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        # Too many closing brackets
        repaired = repaired[:repaired.rfind(']') + 1]
    
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


def _safe_json_loads(text: str, max_depth: int = 10) -> dict | None:
    """Safely parse JSON with depth limiting and error recovery.
    
    Args:
        text: JSON string to parse
        max_depth: Maximum nesting depth to prevent stack overflow
        
    Returns:
        Parsed dict or None if parsing fails
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Quick depth check by counting braces
    depth = 0
    max_found = 0
    for char in text:
        if char == '{':
            depth += 1
            max_found = max(max_found, depth)
        elif char == '}':
            depth -= 1
    
    if max_found > max_depth:
        logger.warning(f"JSON exceeds max depth {max_depth}, attempting truncated parse")
        # Try to extract only top-level structure
        try:
            return json.loads(text, parse_constant=lambda x: None)
        except json.JSONDecodeError:
            return None
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try repair as last resort
        return _repair_json(text)


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: <json> tags (original format)
    try:
        results = _extract_jsons(text)
        if results:
            return results
    except Exception as e:
        logger.debug(f"Strategy 1 (<json> tags) failed: {e}")
    
    # Strategy 2: Flexible extraction with repair
    try:
        results = _extract_json_flexible(text)
        if results:
            return results
    except Exception as e:
        logger.debug(f"Strategy 2 (flexible extraction) failed: {e}")
    
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
    except Exception as e:
        logger.debug(f"Strategy 3 (outermost braces) failed: {e}")
    
    # Strategy 4: Try safe JSON loads as last resort
    try:
        result = _safe_json_loads(text)
        if result:
            return [result]
    except Exception as e:
        logger.debug(f"Strategy 4 (safe JSON loads) failed: {e}")
    
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
        
        # Build guidelines section only if guidelines exist
        guidelines_section = ""
        if grading_guidelines and grading_guidelines.strip():
            guidelines_section = f"\n## Grading Guidelines\n{grading_guidelines}\n"
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to IMO-style competition problems.

## Problem Statement
{problem}

## Official Solution
{solution}
{guidelines_section}
## Student's Answer
{student_answer}

## Evaluation Instructions

Analyze the student's answer step-by-step:

1. **Problem Understanding**: Identify the core question and required mathematical concepts.
2. **Official Solution Analysis**: Note the canonical approach and definitive answer.
3. **Student Answer Review**: Examine the student's approach, final answer, and reasoning.
4. **Comparison**: Evaluate against correctness, reasoning quality, completeness, and presentation.

## Grade Assignment

Assign exactly one of these grades:
- **correct**: Fully correct with proper reasoning, matches official solution.
- **incorrect**: Critical errors, wrong methods, or wrong final answer.
- **partial**: Valid elements but incomplete, minor errors, or lacks justification.

## Examples

**Example 1 - Correct:**
Problem: Find 2 + 3. Student: 2 + 3 = 5.
<json>{{"reasoning": "Correct arithmetic and answer.", "response": "correct"}}</json>

**Example 2 - Incorrect:**
Problem: Find 2 + 3. Student: 2 + 3 = 6.
<json>{{"reasoning": "Arithmetic error: 2+3=5, not 6.", "response": "incorrect"}}</json>

**Example 3 - Partial:**
Problem: Solve x² - 5x + 6 = 0 for both roots. Student: x = 2.
<json>{{"reasoning": "Found one root but missed x = 3.", "response": "partial"}}</json>

## Response Format

Respond ONLY with a JSON object in <json> tags:
<json>{{"reasoning": "Your analysis here.", "response": "correct|incorrect|partial"}}</json>

Requirements:
1. "response" must be exactly: "correct", "incorrect", or "partial" (lowercase)
2. "reasoning" must explain your evaluation
3. No text outside the JSON tags
4. Valid JSON only"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios.
        
        This method uses a multi-stage extraction approach:
        1. JSON extraction with robust parsing
        2. Regex-based field extraction
        3. Text-based keyword detection
        4. Fallback to raw text truncation
        """
        if not msg_history:
            self.log_fn("Warning: Empty message history")
            return "None"
        
        # Validate message history structure
        if not isinstance(msg_history, list):
            self.log_fn(f"Warning: msg_history is not a list, got {type(msg_history).__name__}")
            return "None"
        
        last_message = msg_history[-1]
        if not isinstance(last_message, dict):
            self.log_fn(f"Warning: Last message is not a dict, got {type(last_message).__name__}")
            return "None"
        
        last_text = last_message.get("text", "")
        if not last_text:
            self.log_fn("Warning: Last message has no text content")
            return "None"
        
        if not isinstance(last_text, str):
            self.log_fn(f"Warning: Text content is not a string, got {type(last_text).__name__}")
            return "None"
        
        # Log the raw response length for debugging
        self.log_fn(f"Processing response of length {len(last_text)} characters")
        
        # Stage 1: Try robust JSON extraction first (includes all strategies)
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                
                # Validate extracted object
                if not isinstance(last_obj, dict):
                    self.log_fn(f"Warning: Extracted JSON is not a dict, got {type(last_obj).__name__}")
                    return str(last_obj) if last_obj is not None else "None"
                
                # Log successful JSON extraction
                self.log_fn(f"Successfully extracted JSON with {len(last_obj)} fields")
                
                # Check for structured grading response first - prioritize 'response' field
                if "response" in last_obj:
                    value = last_obj["response"]
                    if isinstance(value, str):
                        return value.strip().lower()
                    return str(value).lower()
                
                # Check other common fields in priority order
                for key in ["grade", "evaluation", "answer", "result", "conclusion", "score"]:
                    if key in last_obj:
                        value = last_obj[key]
                        # Handle different value types
                        if isinstance(value, str):
                            return value.strip()
                        elif isinstance(value, bool):
                            return "correct" if value else "incorrect"
                        elif isinstance(value, (int, float)):
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
            self.log_fn(f"Stage 1 - JSON extraction failed: {e}")
        
        # Fallback 1: try to find any JSON-like structure with regex
        try:
            # Look for patterns like "response": "..." or 'response': '...'
            # Prioritize response field
            response_patterns = [
                r'"response"\s*:\s*"([^"]+)"',
                r"'response'\s*:\s*'([^']+)'",
                r'"grade"\s*:\s*"([^"]+)"',
                r'"evaluation"\s*:\s*"([^"]+)"',
                r'"answer"\s*:\s*"([^"]+)"',
                r'"result"\s*:\s*"([^"]+)"',
                r'"score"\s*:\s*"?([^"},\s]+)"?',
                r'"correct"\s*:\s*(true|false)',
            ]
            for pattern in response_patterns:
                match = re.search(pattern, last_text, re.IGNORECASE)
                if match:
                    result = match.group(1).lower() if len(match.groups()) > 0 else match.group(0)
                    return result
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
        
        # Check for correctness indicators with priority ordering
        # Check for "incorrect" first to avoid false positives from "not correct"
        incorrect_indicators = ["incorrect", "not correct", "not right", "wrong answer", "false"]
        for indicator in incorrect_indicators:
            if indicator in text_lower:
                return "incorrect"
        
        # Check for "correct" after checking for negations
        if "correct" in text_lower:
            return "correct"
        
        # Check for partial credit indicators
        partial_indicators = ["partial", "partially correct", "some credit", "incomplete"]
        for indicator in partial_indicators:
            if indicator in text_lower:
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
        
        # Handle score format (e.g., "score:7", "7/7", "points:5")
        if pred_lower.startswith("score:") or pred_lower.startswith("points:") or "/" in pred_lower:
            # Try to extract numeric score
            score_match = re.search(r'(?:score|points)[:\s]*(\d+(?:\.\d+)?)', pred_lower)
            if score_match:
                score = float(score_match.group(1))
                # For IMO-style scoring (0-7 scale typically)
                if score >= 6.5:  # Near perfect score
                    return "correct"
                elif score <= 0.5:  # Near zero score
                    return "incorrect"
                else:
                    return "partial"
            
            # Handle fraction format like "7/7" or "3/7"
            fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', pred_lower)
            if fraction_match:
                num, denom = float(fraction_match.group(1)), float(fraction_match.group(2))
                if denom > 0:
                    ratio = num / denom
                    if ratio >= 0.9:  # 90% or higher
                        return "correct"
                    elif ratio <= 0.1:  # 10% or lower
                        return "incorrect"
                    else:
                        return "partial"
            
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
        standalone_num = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', pred_clean)
        if standalone_num:
            num = float(standalone_num.group(1))
            # For IMO-style problems (typically 0-7 scale)
            if num >= 6.5:  # Near perfect
                return "correct"
            elif num <= 0.5:  # Near zero
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
        partial_indicators = ["partial", "partially", "incomplete", "some credit", "half", "partial credit", "partially correct"]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        # Check for correct (after checking for "not correct")
        correct_indicators = ["correct", "right", "true", "valid", "accepted", "full credit", "complete", "perfect"]
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
            if not isinstance(inputs[key], str):
                error_msg = f"Error: Input '{key}' must be a string, got {type(inputs[key]).__name__}"
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
            
            # Log reasoning if available
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                try:
                    extracted = _extract_json_robust(raw_text)
                    if extracted and "reasoning" in extracted[-1]:
                        reasoning = extracted[-1]["reasoning"]
                        reasoning_preview = reasoning[:200] if len(reasoning) > 200 else reasoning
                        self.log_fn(f"Reasoning preview: {reasoning_preview}...")
                except Exception:
                    pass  # Reasoning extraction is optional

        return str(prediction), msg_history
