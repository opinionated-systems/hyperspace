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
    2. ```json code blocks
    3. Raw JSON objects with brace matching
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
            if repaired := _repair_json(inner):
                results.append(repaired)
    
    # Strategy 2: ```json code blocks
    if not results:
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                if repaired := _repair_json(match.strip()):
                    results.append(repaired)
    
    # Strategy 3: Look for JSON objects directly with proper brace matching
    if not results:
        start = 0
        while True:
            idx = text.find('{', start)
            if idx == -1:
                break
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(text[idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_str = text[idx:idx+i+1]
                        try:
                            results.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            if repaired := _repair_json(obj_str):
                                results.append(repaired)
                        start = idx + i + 1
                        break
            else:
                start = idx + 1
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes: trailing commas, single quotes, unescaped newlines, missing braces.
    """
    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Apply repairs
    repaired = text
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)  # Trailing commas
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)  # Single quote keys
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)  # Single quote values
    repaired = re.sub(r'(?<!\\)\n', r'\\n', repaired)  # Unescaped newlines
    
    # Balance braces/brackets
    if (diff := repaired.count('{') - repaired.count('}')) > 0:
        repaired += '}' * diff
    if (diff := repaired.count('[') - repaired.count(']')) > 0:
        repaired += ']' * diff
    
    # Try parsing repaired version
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Last resort: extract first complete JSON object
    if (start := repaired.find('{')) == -1:
        return None
    
    brace_count = 0
    for i, char in enumerate(repaired[start:]):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(repaired[start:start+i+1])
                except json.JSONDecodeError:
                    return None
    return None


def _safe_json_loads(text: str, max_depth: int = 10) -> dict | None:
    """Safely parse JSON with depth limiting."""
    if not text or not text.strip():
        return None
    
    # Quick depth check
    depth = max_found = 0
    for char in text:
        if char == '{':
            depth += 1
            max_found = max(max_found, depth)
        elif char == '}':
            depth -= 1
    
    if max_found > max_depth:
        logger.warning(f"JSON exceeds max depth {max_depth}")
        try:
            return json.loads(text, parse_constant=lambda x: None)
        except json.JSONDecodeError:
            return None
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies."""
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags
    if results := _extract_jsons(text):
        return results
    
    # Strategy 2: Flexible extraction
    if results := _extract_json_flexible(text):
        return results
    
    # Strategy 3: Outermost braces
    if (start := text.find('{')) != -1 and (end := text.rfind('}')) > start:
        if repaired := _repair_json(text[start:end+1]):
            return [repaired]
    
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
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history or not (last_text := msg_history[-1].get("text", "")):
            self.log_fn("Warning: Empty message history or text")
            return "None"
        
        self.log_fn(f"Processing response of length {len(last_text)} characters")
        
        # Stage 1: JSON extraction
        try:
            if extracted := _extract_json_robust(last_text):
                last_obj = extracted[-1]
                self.log_fn(f"Extracted JSON with {len(last_obj)} fields")
                
                # Check fields in priority order
                for key in ["response", "grade", "evaluation", "answer", "result", "conclusion", "score"]:
                    if key in last_obj:
                        value = last_obj[key]
                        if isinstance(value, str):
                            return value.strip().lower() if key == "response" else value.strip()
                        elif isinstance(value, bool):
                            return "correct" if value else "incorrect"
                        elif isinstance(value, (int, float)):
                            return str(value)
                
                if "correct" in last_obj:
                    val = last_obj["correct"]
                    return "correct" if isinstance(val, bool) and val else str(val)
                
                if "points" in last_obj:
                    return f"points:{last_obj['points']}"
                
                return str(last_obj)
        except Exception as e:
            self.log_fn(f"JSON extraction failed: {e}")
        
        # Stage 2: Regex patterns
        patterns = [
            (r'"response"\s*:\s*"([^"]+)"', 1),
            (r"'response'\s*:\s*'([^']+)'", 1),
            (r'"grade"\s*:\s*"([^"]+)"', 1),
            (r'"evaluation"\s*:\s*"([^"]+)"', 1),
            (r'"correct"\s*:\s*(true|false)', 1),
        ]
        for pattern, group in patterns:
            if match := re.search(pattern, last_text, re.IGNORECASE):
                return match.group(group).lower()
        
        # Stage 3: Text-based extraction
        text_lower = last_text.lower()
        
        # Check for scores
        if match := re.search(r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', text_lower):
            return f"score:{match.group(1)}"
        
        # Check indicators (incorrect first to catch "not correct")
        for indicator in ["incorrect", "not correct", "not right", "wrong answer", "false"]:
            if indicator in text_lower:
                return "incorrect"
        
        if "correct" in text_lower:
            return "correct"
        
        for indicator in ["partial", "partially correct", "some credit", "incomplete"]:
            if indicator in text_lower:
                return "partial"
        
        # Fallback: return truncated text
        return last_text.strip()[:200] if last_text.strip() else "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation."""
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        pred_clean = pred_lower.strip('"\'.,;: ')
        
        # Exact matches
        exact = {
            "correct": "correct", "true": "correct", "right": "correct",
            "valid": "correct", "accepted": "correct", "complete": "correct",
            "incorrect": "incorrect", "false": "incorrect", "wrong": "incorrect",
            "invalid": "incorrect", "rejected": "incorrect",
            "partial": "partial", "incomplete": "partial",
            "partially correct": "partial", "partial credit": "partial",
        }
        if pred_clean in exact:
            return exact[pred_clean]
        
        # Handle score/points format
        if pred_lower.startswith(("score:", "points:")) or "/" in pred_lower:
            if match := re.search(r'(?:score|points)[:\s]*(\d+(?:\.\d+)?)', pred_lower):
                score = float(match.group(1))
                return "correct" if score >= 6.5 else "incorrect" if score <= 0.5 else "partial"
            
            if match := re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', pred_lower):
                num, denom = float(match.group(1)), float(match.group(2))
                if denom > 0:
                    ratio = num / denom
                    return "correct" if ratio >= 0.9 else "incorrect" if ratio <= 0.1 else "partial"
        
        # Standalone numbers (IMO 0-7 scale)
        if match := re.search(r'^(\d+(?:\.\d+)?)$', pred_clean):
            num = float(match.group(1))
            return "correct" if num >= 6.5 else "incorrect" if num <= 0.5 else "partial"
        
        # Keyword detection (incorrect first to catch "not correct")
        for indicator in ["incorrect", "wrong", "false", "invalid", "rejected", "not correct"]:
            if indicator in pred_lower:
                return "incorrect"
        
        for indicator in ["partial", "incomplete", "some credit", "partial credit"]:
            if indicator in pred_lower:
                return "partial"
        
        for indicator in ["correct", "right", "true", "valid", "accepted", "full credit"]:
            if indicator in pred_lower:
                return "correct"
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Validate inputs
        required = ["problem", "solution", "student_answer"]
        if missing := [k for k in required if k not in inputs or not inputs[k]]:
            error = f"Error: Missing required inputs: {missing}"
            self.log_fn(error)
            return error, [{"role": "assistant", "text": error}]
        
        for key in required:
            if not isinstance(inputs[key], str):
                error = f"Error: Input '{key}' must be a string"
                self.log_fn(error)
                return error, [{"role": "assistant", "text": error}]
        
        instruction = self._build_prompt(inputs)
        self.log_fn(f"Processing: {inputs.get('problem', '')[:100]}...")
        self.log_fn(f"Student: {inputs.get('student_answer', '')[:50]}...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction, model=self.model, msg_history=[]
            )
        except Exception as e:
            error = f"Error: LLM call failed: {e}"
            self.log_fn(error)
            return error, [{"role": "assistant", "text": error}]

        raw = self._extract_prediction(msg_history)
        prediction = self._normalize_prediction(raw)
        
        if prediction == "None":
            self.log_fn("Warning: Could not extract valid prediction")
            if msg_history:
                self.log_fn(f"Raw preview: {msg_history[-1].get('text', '')[:500]}...")
        else:
            self.log_fn(f"Prediction: {prediction[:100]}")
            # Log reasoning if available
            if msg_history and (extracted := _extract_json_robust(msg_history[-1].get("text", ""))):
                if "reasoning" in extracted[-1]:
                    r = extracted[-1]["reasoning"]
                    self.log_fn(f"Reasoning: {r[:200]}...")

        return str(prediction), msg_history
