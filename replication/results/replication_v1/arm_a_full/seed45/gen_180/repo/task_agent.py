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
    1. <json>...</json> tags (with repair)
    2. ```json code blocks
    3. Raw JSON objects at start/end of text
    """
    results = []
    
    # Strategy 1: <json> tags with repair fallback
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
        
        # Try direct parse first, then repair
        for parser in [json.loads, _repair_json]:
            try:
                parsed = parser(inner)
                if parsed:
                    results.append(parsed)
                    break
            except (json.JSONDecodeError, TypeError):
                continue
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.findall(pattern, text, re.DOTALL):
            for parser in [json.loads, _repair_json]:
                try:
                    parsed = parser(match.strip())
                    if parsed:
                        results.append(parsed)
                        break
                except (json.JSONDecodeError, TypeError):
                    continue
    
    # Strategy 3: Look for JSON objects directly using brace balancing
    if not results:
        start = text.find('{')
        while start != -1:
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_str = text[start:start + i + 1]
                        for parser in [json.loads, _repair_json]:
                            try:
                                parsed = parser(obj_str)
                                if parsed and any(k in parsed for k in ["response", "reasoning", "grade", "evaluation"]):
                                    results.append(parsed)
                                    break
                            except (json.JSONDecodeError, TypeError):
                                continue
                        break
            # Find next potential start
            next_start = text.find('{', start + 1)
            if next_start == start:
                break
            start = next_start
    
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


def _safe_json_loads(text: str, max_depth: int = 10) -> dict | None:
    """Safely parse JSON with depth limiting and error recovery.
    
    Args:
        text: JSON string to parse
        max_depth: Maximum nesting depth to prevent stack overflow
        
    Returns:
        Parsed dict or None if parsing fails
    """
    if not text or not text.strip():
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
    
    # Strategy 2: Flexible extraction with repair and code blocks
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Last resort - try to find content between outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        repaired = _repair_json(candidate)
        if repaired:
            return [repaired]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with enhanced chain-of-thought reasoning and few-shot examples."""
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

## Few-Shot Examples

Here are examples of how to evaluate different types of student answers:

### Example 1: Correct Answer
**Problem:** Find the sum of 2 + 3.
**Official Solution:** The sum is 5.
**Student Answer:** 2 + 3 = 5

<json>
{{
    "reasoning": "The student correctly computed the sum of 2 and 3, arriving at the answer 5 which matches the official solution. The reasoning is straightforward and correct.",
    "response": "correct"
}}
</json>

### Example 2: Incorrect Answer
**Problem:** Find the sum of 2 + 3.
**Official Solution:** The sum is 5.
**Student Answer:** 2 + 3 = 6

<json>
{{
    "reasoning": "The student made an arithmetic error. The correct sum of 2 and 3 is 5, not 6. This is a fundamental calculation error that makes the answer incorrect.",
    "response": "incorrect"
}}
</json>

### Example 3: Partial Answer
**Problem:** Solve x² - 5x + 6 = 0 and find both roots.
**Official Solution:** The roots are x = 2 and x = 3.
**Student Answer:** x = 2

<json>
{{
    "reasoning": "The student found one correct root (x = 2) but failed to find the second root (x = 3). The answer is partially correct but incomplete, as the problem asked for both roots.",
    "response": "partial"
}}
</json>

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
4. Ensure the JSON is valid and properly formatted
5. Follow the format shown in the few-shot examples above"""
        
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
        
        self.log_fn(f"Processing response of length {len(last_text)} characters")
        
        # Stage 1: Try robust JSON extraction
        extracted = _extract_json_robust(last_text)
        if extracted:
            last_obj = extracted[-1]
            self.log_fn(f"Successfully extracted JSON with {len(last_obj)} fields")
            
            # Define field extractors in priority order: (field_name, value_processor)
            field_extractors = [
                ("response", lambda v: v.strip().lower() if isinstance(v, str) else str(v).lower()),
                ("grade", lambda v: v.strip() if isinstance(v, str) else str(v)),
                ("evaluation", lambda v: v.strip() if isinstance(v, str) else str(v)),
                ("answer", lambda v: v.strip() if isinstance(v, str) else str(v)),
                ("result", lambda v: v.strip() if isinstance(v, str) else str(v)),
                ("conclusion", lambda v: v.strip() if isinstance(v, str) else str(v)),
                ("score", lambda v: str(v)),
                ("correct", lambda v: "correct" if v else "incorrect" if isinstance(v, bool) else str(v)),
                ("points", lambda v: f"points:{v}"),
            ]
            
            for field, processor in field_extractors:
                if field in last_obj:
                    try:
                        return processor(last_obj[field])
                    except Exception:
                        return str(last_obj[field])
            
            # If no known field found, return the whole object
            return str(last_obj)
        
        # Stage 2: Regex-based extraction for common JSON patterns
        regex_patterns = [
            (r'"response"\s*:\s*"([^"]+)"', 1),
            (r"'response'\s*:\s*'([^']+)'", 1),
            (r'"grade"\s*:\s*"([^"]+)"', 1),
            (r'"evaluation"\s*:\s*"([^"]+)"', 1),
            (r'"answer"\s*:\s*"([^"]+)"', 1),
            (r'"result"\s*:\s*"([^"]+)"', 1),
            (r'"score"\s*:\s*"?([^"},\s]+)"?', 1),
            (r'"correct"\s*:\s*(true|false)', 1),
        ]
        for pattern, group in regex_patterns:
            match = re.search(pattern, last_text, re.IGNORECASE)
            if match:
                return match.group(group).lower()
        
        # Stage 3: Text-based keyword detection
        text_lower = last_text.lower()
        
        # Score patterns
        for pattern in [
            r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*/\s*\d+\s*(?:points?)?',
            r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?',
        ]:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        
        # Priority-based keyword detection (incorrect first to catch "not correct")
        keyword_checks = [
            (["incorrect", "not correct", "not right", "wrong answer", "false"], "incorrect"),
            (["partial", "partially correct", "some credit", "incomplete"], "partial"),
            (["correct"], "correct"),
        ]
        for indicators, result in keyword_checks:
            for indicator in indicators:
                if indicator in text_lower:
                    return result
        
        # Final fallback: return truncated text
        stripped = last_text.strip()
        return stripped[:200] if stripped else "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        Uses a priority-based approach to handle ambiguous cases.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        pred_clean = pred_lower.strip('"\'.,;: ')
        
        # Exact matches (highest priority)
        exact_matches = {
            "correct": "correct", "incorrect": "incorrect", "partial": "partial",
            "true": "correct", "false": "incorrect",
            "right": "correct", "wrong": "incorrect",
            "valid": "correct", "invalid": "incorrect",
            "accepted": "correct", "rejected": "incorrect",
            "incomplete": "partial", "partially correct": "partial",
            "partially incorrect": "partial",
        }
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Helper to classify score/ratio
        def classify_score(value: float, max_val: float = 7.0) -> str:
            ratio = value / max_val if max_val > 0 else 0
            if ratio >= 0.9 or value >= 6.5:
                return "correct"
            elif ratio <= 0.1 or value <= 0.5:
                return "incorrect"
            return "partial"
        
        # Handle score format (e.g., "score:7", "points:5")
        score_match = re.search(r'(?:score|points)[:\s]*(\d+(?:\.\d+)?)', pred_lower)
        if score_match:
            return classify_score(float(score_match.group(1)))
        
        # Handle fraction format (e.g., "7/7", "3/7")
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', pred_lower)
        if fraction_match:
            num, denom = float(fraction_match.group(1)), float(fraction_match.group(2))
            if denom > 0:
                return classify_score(num, denom)
        
        # Check for standalone numbers (assume IMO 0-7 scale)
        standalone_match = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', pred_clean)
        if standalone_match:
            return classify_score(float(standalone_match.group(1)))
        
        # Priority-based keyword detection (incorrect first to catch "not correct")
        keyword_checks = [
            (["incorrect", "wrong", "false", "invalid", "rejected", "error", "not correct", "not right", "not valid"], "incorrect"),
            (["partial", "partially", "incomplete", "some credit", "half", "partial credit", "partially correct"], "partial"),
            (["correct", "right", "true", "valid", "accepted", "full credit", "complete", "perfect"], "correct"),
        ]
        for indicators, result in keyword_checks:
            for indicator in indicators:
                if indicator in pred_lower:
                    return result
        
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
