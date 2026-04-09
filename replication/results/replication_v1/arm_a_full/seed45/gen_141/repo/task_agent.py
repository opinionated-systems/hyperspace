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


def _extract_json_robust(text: str) -> list[dict] | None:
    """Extract JSON objects with robust fallback strategies.
    
    Tries multiple strategies in order of reliability:
    1. <json> tags (original format)
    2. Markdown code blocks (```json)
    3. Raw JSON objects with repair
    4. Direct field extraction from malformed JSON
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (original format)
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
    if results:
        return results
    
    # Strategy 2: Markdown code blocks
    code_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_pattern, text, re.DOTALL):
        candidate = match.group(1).strip()
        try:
            return [json.loads(candidate)]
        except json.JSONDecodeError:
            # Try to repair common JSON errors
            repaired = _repair_json(candidate)
            if repaired:
                return [repaired]
    
    # Strategy 3: Raw JSON with repair - find outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return [json.loads(candidate)]
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            if repaired:
                return [repaired]
    
    # Strategy 4: Direct field extraction for common grading fields
    extracted = {}
    
    # Extract reasoning field
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', 
        text, re.DOTALL
    )
    if reasoning_match:
        extracted["reasoning"] = reasoning_match.group(1).replace('\\n', '\n').replace('\\"', '"')
    
    # Extract response/grade/score field
    for field in ["response", "grade", "score", "evaluation", "verdict"]:
        pattern = rf'"{field}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted["response"] = match.group(1)
            break
    
    # Try to infer response from text if not found
    if "response" not in extracted and "reasoning" in extracted:
        text_lower = text.lower()
        if "incorrect" in text_lower:
            extracted["response"] = "incorrect"
        elif "partial" in text_lower:
            extracted["response"] = "partial"
        elif "correct" in text_lower:
            extracted["response"] = "correct"
    
    return [extracted] if extracted else None


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
                        return json.loads(repaired[start:start+i+1])
            return None
        except Exception:
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
        """Extract prediction from message history with robust fallback strategies."""
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return "None"
        
        # Try JSON extraction first
        extracted = _extract_json_robust(last_text)
        if extracted:
            obj = extracted[-1]
            # Check common grading fields in priority order
            for key in ["response", "grade", "score", "evaluation", "verdict", "answer"]:
                if key in obj:
                    value = obj[key]
                    if isinstance(value, bool):
                        return "correct" if value else "incorrect"
                    return str(value)
            # Check for correctness boolean
            if "correct" in obj:
                val = obj["correct"]
                return "correct" if val else "incorrect"
            # Return first available value
            if obj:
                return str(next(iter(obj.values())))
        
        # Text-based extraction for grading scenarios
        text_lower = last_text.lower()
        
        # Check for numeric scores
        score_match = re.search(r'(?:score|grade|points?)\s*[:=]\s*(\d+(?:\.\d+)?)', text_lower)
        if score_match:
            return f"score:{score_match.group(1)}"
        
        # Check for explicit verdict patterns
        verdict_match = re.search(
            r'\b(?:the\s+answer\s+is|verdict|answer\s+is)\s+(correct|incorrect|partial|wrong|right)\b',
            text_lower
        )
        if verdict_match:
            v = verdict_match.group(1)
            return "correct" if v in ("correct", "right") else "incorrect" if v == "wrong" else v
        
        # Check for correctness indicators
        if "incorrect" in text_lower or "not correct" in text_lower:
            return "incorrect"
        if "partial" in text_lower:
            return "partial"
        if "correct" in text_lower:
            return "correct"
        
        # Return first 200 chars as fallback
        return last_text.strip()[:200] if last_text.strip() else "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation."""
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        
        # Handle score format
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Check for incorrect first (to handle "not correct")
        if any(t in pred_lower for t in ["incorrect", "wrong", "false", "invalid", "rejected"]):
            return "incorrect"
        
        # Check for correct
        if any(t in pred_lower for t in ["correct", "right", "true", "valid", "accepted"]):
            return "correct"
        
        # Check for partial
        if any(t in pred_lower for t in ["partial", "partially", "incomplete"]):
            return "partial"
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required = ["problem", "solution", "student_answer"]
        missing = [k for k in required if k not in inputs or not inputs[k]]
        if missing:
            error = f"Error: Missing required inputs: {missing}"
            self.log_fn(error)
            return error, [{"role": "assistant", "text": error}]
        
        instruction = self._build_prompt(inputs)
        self.log_fn(f"Processing: {inputs.get('problem', '')[:80]}...")

        # Retry loop with exponential backoff
        for attempt in range(3):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction, model=self.model, msg_history=[]
                )
                break
            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["invalid api key", "authentication", "context length", "too long"]):
                    self.log_fn(f"Non-retryable error: {e}")
                    return f"Error: {e}", [{"role": "assistant", "text": str(e)}]
                if attempt < 2:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    return f"Error: LLM failed after 3 attempts: {e}", [{"role": "assistant", "text": str(e)}]

        # Extract and normalize prediction
        raw = self._extract_prediction(msg_history)
        prediction = self._normalize_prediction(raw)
        
        if prediction == "None":
            self.log_fn("Warning: Could not extract prediction")
        else:
            self.log_fn(f"Prediction: {prediction[:80]}")

        return str(prediction), msg_history
