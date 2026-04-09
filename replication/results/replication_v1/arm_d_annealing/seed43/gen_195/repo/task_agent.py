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
from typing import Any

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
            # Try to fix common JSON issues before giving up
            try:
                fixed = _fix_common_json_issues(inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results or None


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON formatting issues that LLMs produce."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (simple cases)
    text = re.sub(r"'([^']*?)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*?)'([,}\]])", r': "\1"\2', text)
    # Remove comments
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    """
    results = []
    
    def try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON string with fixes."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed = _fix_common_json_issues(json_str)
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None
    
    # Try to find JSON objects in code blocks with proper brace balancing
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # Try to find balanced JSON objects within the code block
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_obj = try_parse_json(content[start_idx:i+1])
                    if json_obj and isinstance(json_obj, dict):
                        results.append(json_obj)
                    start_idx = -1
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_obj = try_parse_json(text[start_idx:i+1])
                    if json_obj and isinstance(json_obj, dict):
                        results.append(json_obj)
                    start_idx = -1
    
    # Final fallback: try to find any response-like pattern
    if not results:
        # Look for "response": "value" pattern with flexible whitespace
        response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', text)
        if response_match:
            results.append({"response": response_match.group(1)})
        # Look for reasoning pattern too
        reasoning_match = re.search(r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']', text, re.DOTALL)
        if reasoning_match:
            if results:
                results[0]["reasoning"] = reasoning_match.group(1)
            else:
                results.append({"reasoning": reasoning_match.group(1)})
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: Ensure your JSON is valid and properly formatted. The 'response' field should contain only the grade, not the reasoning."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses a multi-layer extraction strategy:
        1. Primary: Extract JSON from <json>...</json> tags
        2. Fallback: Extract JSON from code blocks or raw text with brace balancing
        3. Final fallback: Extract key-value patterns directly from text
        4. Semantic fallback: Look for grade keywords in the text
        
        Returns:
            (prediction, reasoning) tuple. prediction="None" if extraction fails.
        """
        prediction = "None"
        reasoning = ""
        
        # Layer 1: Try primary extraction method (exact <json> tags)
        extracted = _extract_jsons(text)
        
        # Layer 2: Fallback to regex extraction for malformed responses
        if extracted is None:
            extracted = _extract_json_with_regex(text)
        
        # Layer 3: Extract from the last valid JSON object found
        if extracted:
            last_json = extracted[-1]
            
            # Extract response field (handle various types)
            if "response" in last_json:
                response_val = last_json["response"]
                # Handle different response types gracefully
                if isinstance(response_val, (str, int, float, bool)):
                    prediction = str(response_val)
                elif isinstance(response_val, list) and response_val:
                    prediction = str(response_val[0])
                elif response_val is None:
                    prediction = "None"
                else:
                    prediction = str(response_val)
            
            # Extract reasoning field (handle various types)
            if "reasoning" in last_json:
                reasoning_val = last_json["reasoning"]
                if isinstance(reasoning_val, str):
                    reasoning = reasoning_val
                elif isinstance(reasoning_val, (int, float, bool)):
                    reasoning = str(reasoning_val)
                elif isinstance(reasoning_val, list):
                    reasoning = " ".join(str(r) for r in reasoning_val)
                else:
                    reasoning = str(reasoning_val)
        
        # Layer 4: Semantic fallback - look for grade keywords in text
        if prediction == "None":
            text_lower = text.lower()
            # Look for common grading keywords
            if any(word in text_lower for word in ["correct", "right", "accurate", "valid", "proper"]):
                # Check if negated
                if not any(neg in text_lower[:text_lower.find("correct")+20] if "correct" in text_lower else False 
                          for neg in ["not ", "incorrect", "wrong", "not correct"]):
                    prediction = "Correct"
            elif any(word in text_lower for word in ["partial", "partially", "incomplete", "some correct"]):
                prediction = "Partial"
            elif any(word in text_lower for word in ["incorrect", "wrong", "error", "invalid", "not correct", "not valid"]):
                prediction = "Incorrect"
            
            # Extract reasoning from the text itself
            if prediction != "None":
                # Use the last paragraph or sentence as reasoning
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                if paragraphs:
                    reasoning = paragraphs[-1][:500]
        
        # Validate prediction isn't empty
        if prediction == "":
            prediction = "None"
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if not inputs.get(k)]
        if missing_keys:
            self.log_fn(f"Warning: Missing required inputs: {missing_keys}")
        
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    # Store reasoning in msg_history for potential downstream use
                    msg_history.append({"role": "system", "text": f"[Extraction] Reasoning: {reasoning[:500]}"})
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Common mistakes to avoid:
- Trailing commas in JSON (e.g., {{"key": "value",}} is invalid)
- Single quotes instead of double quotes
- Comments inside JSON
- Text outside the <json> tags

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
