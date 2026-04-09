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
    """Extract JSON objects with fallback strategies.
    
    Tries: <json> tags, ```json blocks, raw JSON objects.
    """
    results = []
    
    # Strategy 1: <json> tags
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
        for match in re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                if repaired := _repair_json(match.strip()):
                    results.append(repaired)
    
    # Strategy 3: Raw JSON objects
    if not results:
        for match in re.finditer(r'\{[^{}]*"[^"]+"[^{}]*\}', text, re.DOTALL):
            start = match.start()
            brace_count = 0
            for i, char in enumerate(text[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_str = text[start:start+i+1]
                        try:
                            results.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            if repaired := _repair_json(obj_str):
                                results.append(repaired)
                        break
    
    return results or None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors.
    
    Fixes: trailing commas, single quotes, unescaped newlines, 
    missing braces, comments, control characters, BOM.
    """
    # Quick success path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove BOM and control characters
    repaired = text.lstrip('\ufeff')
    repaired = ''.join(c for c in repaired if c in '\t\n\r' or ord(c) >= 32)
    
    # Remove comments and trailing commas
    repaired = re.sub(r'//[^\n]*', '', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix single quotes to double quotes
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Balance braces/brackets
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    repaired += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
    
    # Try parsing
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Extract first complete JSON object
    start = repaired.find('{')
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    for i, char in enumerate(repaired[start:]):
        if char == '"' and (i == 0 or repaired[start+i-1] != '\\'):
            in_string = not in_string
        elif not in_string:
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


def _extract_json_robust(text: str) -> list[dict] | None:
    """Robust JSON extraction with multiple fallback strategies."""
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags
    if results := _extract_jsons(text):
        return results
    
    # Strategy 2: Flexible extraction
    if results := _extract_json_flexible(text):
        return results
    
    # Strategy 3: Outermost braces
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        if repaired := _repair_json(text[start:end+1]):
            return [repaired]
    
    # Strategy 4: Extract key fields directly
    extracted = {}
    if match := re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL):
        extracted["reasoning"] = match.group(1)
    if match := re.search(r'"response"\s*:\s*"([^"]*)"', text, re.IGNORECASE):
        extracted["response"] = match.group(1)
    elif extracted:
        text_lower = text.lower()
        if "incorrect" in text_lower:
            extracted["response"] = "incorrect"
        elif "correct" in text_lower:
            extracted["response"] = "correct"
        elif "partial" in text_lower:
            extracted["response"] = "partial"
    
    return [extracted] if extracted else None


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
        """Extract prediction from message history."""
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return "None"
        
        # Try JSON extraction
        if extracted := _extract_json_robust(last_text):
            last_obj = extracted[-1]
            for key in ["response", "grade", "score", "evaluation", "answer", "verdict"]:
                if key in last_obj:
                    return str(last_obj[key])
            if "correct" in last_obj:
                return "correct" if last_obj["correct"] else "incorrect"
            if "points" in last_obj:
                return f"points:{last_obj['points']}"
        
        # Regex patterns for common fields
        text_lower = last_text.lower()
        patterns = [
            (r'"response"\s*:\s*"([^"]+)"', 1),
            (r'"grade"\s*:\s*"([^"]+)"', 1),
            (r'"score"\s*:\s*"?([^"},\s]+)"?', 1),
            (r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', 1),
        ]
        for pattern, group in patterns:
            if match := re.search(pattern, last_text, re.IGNORECASE):
                return match.group(group).lower()
        
        # Verdict patterns
        if re.search(r'\bthe\s+answer\s+is\s+(incorrect|wrong)\b', text_lower):
            return "incorrect"
        if re.search(r'\bthe\s+answer\s+is\s+(correct|right)\b', text_lower):
            return "correct"
        if re.search(r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', text_lower):
            return re.search(r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', text_lower).group(1)
        
        # Simple keyword check
        if "incorrect" in text_lower or "not correct" in text_lower:
            return "incorrect"
        if "correct" in text_lower:
            return "correct"
        if "partial" in text_lower:
            return "partial"
        
        return last_text.strip()[:200] or "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format."""
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Score formats pass through
        if any(pred_lower.startswith(p) for p in ["score:", "points:"]) or "/" in pred_lower:
            return prediction
        
        # Check for incorrect first (to avoid "not correct" -> "correct")
        if any(t in pred_lower for t in ["incorrect", "wrong", "false", "invalid", "error"]):
            return "incorrect"
        
        # Check for correct
        if any(t in pred_lower for t in ["correct", "right", "true", "valid", "accepted"]):
            return "correct"
        
        # Check for partial
        if any(t in pred_lower for t in ["partial", "incomplete"]):
            return "partial"
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Validate required inputs
        required = ["problem", "solution", "student_answer"]
        if missing := [k for k in required if not inputs.get(k)]:
            error_msg = f"Error: Missing required inputs: {missing}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        self.log_fn(f"Processing: {inputs.get('problem', '')[:100]}...")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction, model=self.model, msg_history=[]
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            self.log_fn(error_msg)
            return error_msg, [{"role": "assistant", "text": error_msg}]

        prediction = self._normalize_prediction(self._extract_prediction(msg_history))
        
        if prediction == "None":
            self.log_fn("Warning: Could not extract valid prediction")
        else:
            self.log_fn(f"Prediction: {prediction[:100]}")

        return str(prediction), msg_history
