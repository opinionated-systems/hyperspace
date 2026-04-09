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
            # Try to repair common JSON errors
            repaired = _repair_json(inner)
            if repaired:
                results.append(repaired)
            continue
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability:
    1. ```json code blocks
    2. Raw JSON objects at start/end of text
    3. JSON-like structures with relaxed parsing
    """
    results = []
    
    # Strategy 1: ```json code blocks
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
    
    # Strategy 2: Look for JSON objects directly
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

5. **Assign Grade**: Based on your analysis, provide your evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here, including specific observations about the student's work",
    "response": "Your final evaluation here - should be one of: 'correct', 'incorrect', 'partial', or a specific score if applicable"
}}
</json>

The "response" field must contain a clear, concise final determination. Use:
- "correct" if the answer is fully correct with proper reasoning
- "incorrect" if the answer is wrong or has critical errors
- "partial" if the answer has some correct elements but is incomplete or has minor errors
- A specific score (e.g., "7" or "3/7") if the problem uses point-based scoring"""
        
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
                for key in ["grade", "score", "evaluation", "response", "answer", "result", "conclusion", "verdict"]:
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
        
        # Fallback: Enhanced text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores (e.g., "score: 7", "7/7", "7 points")
        score_patterns = [
            r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*/\s*\d+\s*(?:points?)?',
            r'(?:awarded|given|assigned)\s+(\d+)\s*(?:points?)?',
            r'(?:score|grade)\s+of\s+(\d+)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"score:{match.group(1)}"
        
        # Check for correctness indicators with improved logic
        # Look for explicit verdict patterns first
        verdict_patterns = [
            (r'\bthe\s+answer\s+is\s+(correct|incorrect|wrong|right)\b', 1),
            (r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', 1),
            (r'\b(correct|incorrect|partial)\s+answer\b', 1),
            (r'\banswer\s+is\s+(correct|incorrect|partial)\b', 1),
        ]
        for pattern, group in verdict_patterns:
            match = re.search(pattern, text_lower)
            if match:
                verdict = match.group(group).lower()
                if verdict in ["right", "correct"]:
                    return "correct"
                elif verdict in ["wrong", "incorrect"]:
                    return "incorrect"
                elif verdict == "partial":
                    return "partial"
        
        # Check for correctness indicators
        if "correct" in text_lower:
            # Distinguish between "correct" and "incorrect"
            if "incorrect" in text_lower or "not correct" in text_lower:
                return "incorrect"
            return "correct"
        
        # Check for partial credit indicators
        if any(term in text_lower for term in ["partial", "partially", "some credit", "incomplete", "partial credit"]):
            return "partial"
        
        # Fallback: return first 200 chars of text as prediction
        if last_text.strip():
            return last_text.strip()[:200]
        
        return "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Converts various forms of correct/incorrect/partial to standardized values.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score format (e.g., "score:7", "7/7")
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Normalize correct variations
        if any(term in pred_lower for term in ["correct", "right", "true", "valid", "accepted"]):
            if "incorrect" not in pred_lower and "not correct" not in pred_lower:
                return "correct"
        
        # Normalize incorrect variations
        if any(term in pred_lower for term in ["incorrect", "wrong", "false", "invalid", "rejected", "error"]):
            return "incorrect"
        
        # Normalize partial variations
        if any(term in pred_lower for term in ["partial", "partially", "incomplete", "some credit"]):
            return "partial"
        
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
