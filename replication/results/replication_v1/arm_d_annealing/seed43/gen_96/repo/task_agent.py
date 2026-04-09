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
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes for malformed JSON
            # Fix 1: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Fix 2: Handle unescaped newlines in strings
            fixed = re.sub(r'(?<!\\)\n', '\\n', fixed)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # If still failing, try to extract just the response field
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                if response_match:
                    results.append({"response": response_match.group(1)})
                continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks (with or without json label)
    code_block_pattern = re.findall(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    for match in code_block_pattern:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            # Try fixing common issues
            fixed = re.sub(r',(\s*[}\]])', r'\1', match)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON objects without code blocks
    if not results:
        # Look for JSON-like structures with balanced braces
        json_pattern = re.findall(r'\{[^{}]*"response"[^{}]*\}', text)
        for match in json_pattern:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                pass
    
    # Try to find any response field with more flexible matching
    if not results:
        response_patterns = [
            r'"response"\s*:\s*"([^"]+)"',
            r'"response"\s*:\s*\'([^\']+)\'',
            r'response["\']?\s*[:=]\s*["\']?([^"\'\n,}]+)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results.append({"response": match.group(1).strip()})
                break
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs.
        
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs or not inputs[f]]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
        return True, ""

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
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        extraction_method = "primary"
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
            extraction_method = "regex_fallback"
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
            
            # Log extraction details for debugging
            self.log_fn(f"Extraction method: {extraction_method}, JSON keys: {list(last_json.keys())}")
        else:
            self.log_fn("Warning: No JSON could be extracted from response")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
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
                    self.log_fn(f"[Call {self._call_count}] Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"[Call {self._call_count}] Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"[Call {self._call_count}] Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
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

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"[Call {self._call_count}] Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
