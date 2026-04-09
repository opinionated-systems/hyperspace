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
from agent.utils import validate_required_keys, truncate_string, format_dict_for_logging

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict | None:
    """Extract a single JSON object from text using multiple strategies.
    
    Tries strategies in order of reliability:
    1. <json>...</json> tags
    2. ```json code blocks
    3. Raw JSON between outermost braces
    4. Repair common JSON syntax errors
    
    Returns the first valid JSON dict found, or None.
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Strategy 1: <json> tags
    start = text.find("<json>")
    if start != -1:
        end = text.find("</json>", start)
        if end != -1:
            inner = text[start + 6:end].strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                repaired = _repair_json(inner)
                if repaired:
                    return repaired
    
    # Strategy 2: ```json code blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            repaired = _repair_json(match.group(1).strip())
            if repaired:
                return repaired
    
    # Strategy 3: Find JSON between outermost braces
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            if repaired:
                return repaired
    
    return None


def _repair_json(text: str) -> dict | None:
    """Attempt to repair common JSON syntax errors."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove control characters
    repaired = ''.join(char for char in text if char in '\t\n\r' or ord(char) >= 32)
    
    # Remove comments
    repaired = re.sub(r'//[^\n]*', '', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
    
    # Remove trailing commas
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # Fix single quotes (simple cases)
    repaired = re.sub(r"'([^']*?)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', repaired)
    
    # Balance braces
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
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
        """Extract prediction from message history.
        
        First tries to extract JSON and read the 'response' field,
        then falls back to text-based pattern matching.
        """
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return "None"
        
        # Try JSON extraction first
        data = _extract_json(last_text)
        if data:
            # Check for response field (primary)
            if "response" in data:
                return str(data["response"])
            # Check other common fields
            for key in ["grade", "score", "evaluation", "answer", "result", "verdict"]:
                if key in data:
                    return str(data[key])
            # Check for boolean correctness
            if "correct" in data:
                val = data["correct"]
                if isinstance(val, bool):
                    return "correct" if val else "incorrect"
                return str(val)
        
        # Text-based fallback
        text_lower = last_text.lower()
        
        # Check for numeric scores
        match = re.search(r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', text_lower)
        if match:
            return f"score:{match.group(1)}"
        
        # Check for explicit verdicts
        match = re.search(r'\bthe\s+answer\s+is\s+(correct|incorrect|wrong|right)\b', text_lower)
        if match:
            v = match.group(1).lower()
            return "correct" if v in ["correct", "right"] else "incorrect"
        
        match = re.search(r'\bverdict\s*[:=]\s*(correct|incorrect|partial)\b', text_lower)
        if match:
            return match.group(1).lower()
        
        # Check for correctness keywords
        if "correct" in text_lower:
            if "incorrect" in text_lower or "not correct" in text_lower:
                return "incorrect"
            return "correct"
        
        if "partial" in text_lower:
            return "partial"
        
        # Return truncated text as last resort
        return last_text.strip()[:200] if last_text.strip() else "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format."""
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Keep score format as-is
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Normalize to standard values
        if "incorrect" in pred_lower or "wrong" in pred_lower or "false" in pred_lower:
            return "incorrect"
        
        if "partial" in pred_lower:
            return "partial"
        
        if "correct" in pred_lower or "right" in pred_lower or "true" in pred_lower:
            return "correct"
        
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
        missing_keys = validate_required_keys(inputs, required_keys)
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            return error_msg, [{"role": "assistant", "text": error_msg}]

        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract and normalize prediction
        raw_prediction = self._extract_prediction(msg_history)
        prediction = self._normalize_prediction(raw_prediction)

        return str(prediction), msg_history
