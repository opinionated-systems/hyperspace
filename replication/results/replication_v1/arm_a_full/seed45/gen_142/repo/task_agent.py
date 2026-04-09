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


def _validate_json_structure(obj: dict, required_fields: list[str] | None = None) -> tuple[bool, str]:
    """Validate that a JSON object has the expected structure.
    
    Args:
        obj: The JSON object to validate
        required_fields: List of required field names (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(obj, dict):
        return False, f"Expected dict, got {type(obj).__name__}"
    
    if required_fields:
        missing = [f for f in required_fields if f not in obj]
        if missing:
            return False, f"Missing required fields: {missing}"
    
    # Validate response field if present
    if "response" in obj:
        response = obj["response"]
        if not isinstance(response, str):
            return False, f"'response' must be a string, got {type(response).__name__}"
        valid_responses = {"correct", "incorrect", "partial"}
        if response.lower() not in valid_responses:
            return False, f"'response' must be one of {valid_responses}, got '{response}'"
    
    return True, "Valid"


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
    
    return None


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that a grading response has the required fields and valid values.
    
    Returns:
        (is_valid, error_message)
    """
    # Check for required fields
    if "response" not in response and "grade" not in response:
        return False, "Missing required field: 'response' or 'grade'"
    
    # Get the grade value
    grade_value = response.get("response") or response.get("grade")
    if not grade_value:
        return False, "Empty grade value"
    
    # Normalize to string
    if isinstance(grade_value, str):
        grade_str = grade_value.strip().lower()
    else:
        grade_str = str(grade_value).lower()
    
    # Validate against allowed values
    valid_grades = {"correct", "incorrect", "partial", "true", "false"}
    if grade_str not in valid_grades and not any(v in grade_str for v in valid_grades):
        return False, f"Invalid grade value: '{grade_str}'. Expected one of: correct, incorrect, partial"
    
    # Check reasoning field exists and is non-empty
    reasoning = response.get("reasoning", "")
    if not reasoning or not str(reasoning).strip():
        return False, "Missing or empty 'reasoning' field"
    
    return True, "Valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with enhanced chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Build guidelines section only if guidelines exist
        guidelines_section = ""
        if grading_guidelines and grading_guidelines.strip():
            guidelines_section = f"""
## Grading Guidelines
{grading_guidelines}
"""
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to International Mathematical Olympiad (IMO) style competition problems.

Your task is to carefully analyze the student's answer and provide a rigorous evaluation according to the official solution.

## Problem Statement
{problem}

## Official Solution
{solution}
{guidelines_section}

## Student's Answer
{student_answer}

## Instructions

Think through this step-by-step, providing detailed analysis at each stage:

### Stage 1: Problem Understanding
- What is the core question being asked?
- What are the key mathematical concepts, theorems, and techniques required?
- What constraints or conditions must be satisfied?

### Stage 2: Official Solution Analysis
- What is the canonical approach to solving this problem?
- What is the definitive final answer (in exact form)?
- What are the critical proof steps or logical deductions that must be present?
- Are there alternative valid approaches?

### Stage 3: Student Answer Review
- What approach did the student attempt?
- What is the student's final answer (exactly as stated)?
- What key steps did the student include or omit?
- What mathematical reasoning did the student demonstrate?

### Stage 4: Detailed Comparison
Evaluate the student's answer against these criteria:

**Answer Correctness:**
- Is the student's final answer mathematically equivalent to the official solution?
- Did the student arrive at the correct numerical/algebraic result?
- Are there any sign errors, calculation mistakes, or algebraic errors?

**Reasoning Quality:**
- Did the student demonstrate sound mathematical logic?
- Are the proof steps valid and well-justified?
- Did the student cite appropriate theorems and apply them correctly?
- Are there logical gaps or circular reasoning?

**Completeness:**
- Did the student address all parts of the problem?
- Is the solution fully worked out or are there missing steps?
- Did the student show sufficient work to justify their conclusion?

**Presentation:**
- Is the solution clearly organized and easy to follow?
- Did the student define variables and explain their notation?
- Are there any ambiguous or unclear statements?

### Stage 5: Grade Assignment
Based on your comprehensive analysis, assign one of these grades:

- **correct**: The answer is fully correct with proper reasoning, all critical steps are justified, the final answer matches the official solution, and the solution is complete.
- **incorrect**: The answer contains critical errors, uses fundamentally incorrect methods, has a wrong final answer, or demonstrates major logical flaws.
- **partial**: The answer has valid elements but is incomplete, contains minor errors, lacks proper justification for key steps, or only partially matches the official solution.

## Response Format

You must respond with a valid JSON object enclosed in <json> tags:

<json>
{{
    "reasoning": "Provide your detailed step-by-step analysis here. Include specific observations about the student's mathematical reasoning, any errors found, comparison with the official solution, and justification for your grade assignment. Be thorough and cite specific elements from the student's work.",
    "response": "correct"
}}
</json>

CRITICAL REQUIREMENTS:
1. The "response" field MUST contain exactly one of: "correct", "incorrect", or "partial" (all lowercase, no quotes around the value in the field)
2. The "reasoning" field must contain your complete analysis
3. Do not include any text outside the JSON tags
4. Ensure the JSON is valid and properly formatted"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Enhanced for IMO grading: specifically handles correctness evaluation,
        numeric scores, and partial credit scenarios. Includes validation of
        response format to ensure quality.
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
                
                # Validate the response structure
                is_valid, validation_msg = _validate_grading_response(last_obj)
                if not is_valid:
                    self.log_fn(f"Response validation warning: {validation_msg}")
                
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
            self.log_fn(f"Flexible extraction failed: {e}")
        
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
