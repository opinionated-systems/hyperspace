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
    1. <json>...</json> tags
    2. ```json code blocks (with various language specifiers)
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
    
    # Strategy 2: Markdown code blocks with various language specifiers
    if not results:
        # Match ```json, ```JSON, ```, or any language specifier
        pattern = r'```(?:json|JSON|javascript|js|text|plain)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            match = match.strip()
            if not match:
                continue
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                repaired = _repair_json(match)
                if repaired:
                    results.append(repaired)
                continue
    
    # Strategy 3: Look for JSON objects directly with proper brace matching
    if not results:
        # Find all potential JSON starting points
        start_indices = [m.start() for m in re.finditer(r'\{[\s]*"', text)]
        
        for start in start_indices:
            try:
                # Use proper brace counting with string awareness
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
            except Exception:
                continue
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing closing braces/brackets
    - Unescaped quotes within strings
    - Comments (// and /* */)
    """
    try:
        # First, try parsing as-is
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove comments first (// line comments and /* */ block comments)
    repaired = re.sub(r'//[^\n]*', '', text)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Try to fix single quotes (simple cases only)
    # Replace single quotes around keys/values with double quotes
    # This is a best-effort approach
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Escape unescaped newlines in strings (but not in already-escaped contexts)
    # Use a more careful approach: only escape newlines that are between unescaped quotes
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)
    
    # Escape unescaped tabs
    repaired = re.sub(r'(?<!\\)\t', r'\\t', repaired)
    
    # Try to balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    elif open_braces < 0:
        # Too many closing braces - try to find where they start
        repaired = repaired.rstrip()
        while open_braces < 0 and repaired.endswith('}'):
            repaired = repaired[:-1]
            open_braces += 1
    
    open_brackets = repaired.count('[') - repaired.count(']')
    if open_brackets > 0:
        repaired += ']' * open_brackets
    elif open_brackets < 0:
        repaired = repaired.rstrip()
        while open_brackets < 0 and repaired.endswith(']'):
            repaired = repaired[:-1]
            open_brackets += 1
    
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
            in_string = False
            escape_next = False
            
            for i, char in enumerate(repaired[start:]):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
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
                            try:
                                return json.loads(repaired[start:start+i+1])
                            except json.JSONDecodeError:
                                # Try repairing just this substring
                                sub_repaired = _repair_json(repaired[start:start+i+1])
                                if sub_repaired:
                                    return sub_repaired
                                return None
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
    
    # Quick check: does text contain any JSON-like structure?
    if '{' not in text or '"' not in text:
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
    # Try to find content between outermost braces with proper parsing
    try:
        # Find all potential starting points
        start_indices = [m.start() for m in re.finditer(r'\{', text)]
        
        for start in start_indices:
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
                            candidate = text[start:start+i+1]
                            repaired = _repair_json(candidate)
                            if repaired:
                                return [repaired]
                            break
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
        
        # Log the response length for debugging
        self.log_fn(f"Extracting prediction from response of {len(last_text)} characters")
        
        # Try robust extraction first (includes all strategies)
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                # Prefer response field, fallback to other common fields
                last_obj = extracted[-1]
                self.log_fn(f"Successfully extracted JSON with keys: {list(last_obj.keys())}")
                
                # Check for structured grading response first
                for key in ["grade", "score", "evaluation", "response", "answer", "result", "conclusion"]:
                    if key in last_obj:
                        value = last_obj[key]
                        self.log_fn(f"Found '{key}' field with value: {value}")
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
                self.log_fn(f"No recognized field found in JSON, returning full object")
                return str(last_obj)
            else:
                self.log_fn("No JSON found in response, falling back to text extraction")
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
        
        # Check for correctness indicators
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
            "mostly correct": "partial",
            "mostly incorrect": "partial",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format (e.g., "score:7", "7/7")
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Check for numeric patterns that might indicate scoring
        import re
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
        incorrect_indicators = ["incorrect", "wrong", "false", "invalid", "rejected", "error", "not correct", "not right", "not valid", "not accepted"]
        for indicator in incorrect_indicators:
            if indicator in pred_lower:
                return "incorrect"
        
        # Check for partial
        partial_indicators = ["partial", "partially", "incomplete", "some credit", "half", "partial credit", "mostly"]
        for indicator in partial_indicators:
            if indicator in pred_lower:
                return "partial"
        
        # Check for correct (after checking for "not correct")
        correct_indicators = ["correct", "right", "true", "valid", "accepted", "full credit", "complete", "perfect"]
        for indicator in correct_indicators:
            if indicator in pred_lower:
                return "correct"
        
        # Log that no normalization was applied
        self.log_fn(f"No normalization applied for prediction: '{prediction[:50]}...' if len > 50")
        
        # Return original if no normalization applied
        return prediction

    def _extract_reasoning(self, msg_history: list[dict]) -> str | None:
        """Extract reasoning field from message history for debugging.
        
        Returns the reasoning text if found, None otherwise.
        """
        if not msg_history:
            return None
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return None
        
        try:
            extracted = _extract_json_robust(last_text)
            if extracted:
                last_obj = extracted[-1]
                if "reasoning" in last_obj:
                    reasoning = last_obj["reasoning"]
                    if isinstance(reasoning, str):
                        return reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
        except Exception:
            pass
        
        return None

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
            # Log the raw response for debugging
            if msg_history:
                raw_text = msg_history[-1].get('text', '')
                self.log_fn(f"Raw response preview: {raw_text[:500]}...")
                # Try to extract any meaningful text as last resort
                prediction = self._last_resort_extraction(raw_text)
        else:
            # Log successful extraction
            preview = prediction[:100] if len(prediction) > 100 else prediction
            self.log_fn(f"Extracted prediction: {preview}")
            
            # Also log reasoning if available
            reasoning = self._extract_reasoning(msg_history)
            if reasoning:
                self.log_fn(f"Reasoning preview: {reasoning[:200]}...")

        return str(prediction), msg_history

    def _emergency_extract_prediction(self, msg_history: list[dict]) -> str:
        """Emergency extraction when standard methods fail.
        
        Uses aggressive text analysis to find any indication of grading.
        """
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        text_lower = last_text.lower()
        
        # Look for explicit statements about correctness
        if "the answer is correct" in text_lower or "this is correct" in text_lower:
            if "not correct" not in text_lower and "incorrect" not in text_lower:
                return "correct"
        
        if "the answer is incorrect" in text_lower or "this is incorrect" in text_lower:
            return "incorrect"
        
        if "partially correct" in text_lower or "partial credit" in text_lower:
            return "partial"
        
        # Look for conclusion/summary sections
        conclusion_patterns = [
            r'conclusion[\s:]*([^\n]{1,50})',
            r'evaluation[\s:]*([^\n]{1,50})',
            r'grade[\s:]*([^\n]{1,50})',
            r'verdict[\s:]*([^\n]{1,50})',
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                conclusion = match.group(1).strip()
                if any(word in conclusion for word in ["correct", "right", "true"]):
                    if "not" not in conclusion and "in" not in conclusion:
                        return "correct"
                if any(word in conclusion for word in ["incorrect", "wrong", "false"]):
                    return "incorrect"
                if "partial" in conclusion:
                    return "partial"
        
        return "None"

    def _last_resort_extraction(self, raw_text: str) -> str:
        """Last resort: extract any meaningful grading indicator from text.
        
        Returns the most likely prediction based on overall text sentiment.
        """
        if not raw_text:
            return "None"
        
        text_lower = raw_text.lower()
        
        # Count positive vs negative indicators
        positive_indicators = ["correct", "right", "valid", "properly", "accurate", "full credit"]
        negative_indicators = ["incorrect", "wrong", "invalid", "error", "mistake", "no credit"]
        partial_indicators = ["partial", "incomplete", "some credit", "partially"]
        
        pos_count = sum(1 for ind in positive_indicators if ind in text_lower)
        neg_count = sum(1 for ind in negative_indicators if ind in text_lower)
        part_count = sum(1 for ind in partial_indicators if ind in text_lower)
        
        # Check for negation of positive indicators
        negation_patterns = ["not correct", "not right", "not valid", "incorrect"]
        for pattern in negation_patterns:
            if pattern in text_lower:
                pos_count = max(0, pos_count - 2)  # Heavy penalty for negation
                neg_count += 1
        
        self.log_fn(f"Last resort extraction - pos:{pos_count} neg:{neg_count} part:{part_count}")
        
        # Determine based on counts
        if neg_count > pos_count and neg_count > part_count:
            return "incorrect"
        elif part_count > pos_count and part_count > neg_count:
            return "partial"
        elif pos_count > 0:
            return "correct"
        
        return "None"
