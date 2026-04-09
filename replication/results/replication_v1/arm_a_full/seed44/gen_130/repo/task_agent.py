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
import time

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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback when tags are missing/malformed.
    
    Tries to find JSON objects by looking for curly braces.
    Uses a stack-based approach to handle nested structures correctly.
    Includes additional heuristics to filter out false positives.
    """
    results = []
    i = 0
    n = len(text)
    
    while i < n:
        # Find the next opening brace
        if text[i] != '{':
            i += 1
            continue
            
        # Found a potential JSON start
        start = i
        brace_depth = 0
        in_string = False
        escape_next = False
        
        for j in range(i, n):
            char = text[j]
            
            # Handle string escaping
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            # Only count braces outside of strings
            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        # Found a complete JSON object
                        json_str = text[start:j+1]
                        
                        # Heuristic: skip if it looks like a code snippet (contains common programming patterns)
                        if _looks_like_code_snippet(json_str):
                            i = j + 1
                            break
                        
                        try:
                            obj = json.loads(json_str)
                            if isinstance(obj, dict) and len(obj) > 0:
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
        
        # If we didn't find a complete object, move forward
        i += 1
    
    return results if results else None


def _looks_like_code_snippet(json_str: str) -> bool:
    """Heuristic to detect if a string looks like a code snippet rather than JSON data.
    
    Returns True if the string contains patterns common in code but not in JSON data.
    """
    code_patterns = [
        'def ', 'class ', 'import ', 'from ', 'return ',
        'function', 'var ', 'let ', 'const ',
        'if (', 'for (', 'while (', 'try {', 'catch',
        '#include', 'printf', 'cout <<', 'System.out',
    ]
    json_str_lower = json_str.lower()
    for pattern in code_patterns:
        if pattern in json_str_lower:
            return True
    return False


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent takes a problem, solution, grading guidelines, and student answer,
    then uses an LLM to evaluate the student's answer and provide a score/feedback.
    
    Attributes:
        model: The LLM model to use for evaluation
        log_fn: Logging function for debug output
        max_retries: Maximum number of retry attempts for failed extractions
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2
        self._last_thinking = ""

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Based on the grading guidelines, determine the appropriate score or evaluation.
4. Provide your reasoning in the "thinking" field.
5. Provide your final evaluation in the "response" field.

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your step-by-step analysis and reasoning here...",
    "response": "Your final answer/evaluation here..."
}}
</json>"""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and thinking from response text.
        
        Uses a multi-stage extraction strategy:
        1. Try strict <json> tag extraction
        2. Fallback to fuzzy JSON extraction
        3. Look for markdown code blocks as last resort
        
        Returns:
            (prediction, thinking) tuple
        """
        prediction = "None"
        thinking = ""
        extraction_method = "none"
        
        # Stage 1: Try strict extraction with <json> tags
        extracted = _extract_jsons(text)
        if extracted:
            extraction_method = "strict"
        
        # Stage 2: Fallback to fuzzy extraction
        if not extracted:
            extracted = _extract_json_fuzzy(text)
            if extracted:
                extraction_method = "fuzzy"
        
        # Stage 3: Look for markdown code blocks with json
        if not extracted:
            extracted = self._extract_markdown_json(text)
            if extracted:
                extraction_method = "markdown"
        
        if extracted:
            # Use the last valid JSON object found (usually the final answer)
            last = extracted[-1]
            if isinstance(last, dict):
                # Extract response field with multiple fallback keys
                response_keys = ["response", "answer", "result", "evaluation", "output", "prediction"]
                for key in response_keys:
                    if key in last and last[key] is not None:
                        prediction = last[key]
                        break
                    
                # Extract thinking field with multiple fallback keys
                thinking_keys = ["thinking", "reasoning", "analysis", "explanation", "thoughts", "rationale"]
                for key in thinking_keys:
                    if key in last and last[key] is not None:
                        thinking = last[key]
                        break
        
        # Ensure we return strings
        if prediction is None:
            prediction = "None"
        if thinking is None:
            thinking = ""
        
        # Log extraction details for debugging
        logger.debug(f"Extraction method: {extraction_method}, found {len(extracted) if extracted else 0} JSON objects")
            
        return str(prediction), str(thinking)
    
    def _extract_markdown_json(self, text: str) -> list[dict] | None:
        """Extract JSON from markdown code blocks as last resort.
        
        Looks for ```json or ``` blocks and tries to parse them.
        """
        results = []
        
        # Pattern for markdown JSON blocks
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            match = match.strip()
            if not match:
                continue
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                # Try to find JSON objects within the block
                results.extend(_extract_json_fuzzy(match) or [])
        
        return results if results else None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or inputs[k] is None]
        if missing_keys:
            logger.warning(f"Missing required input keys: {missing_keys}")
        
        # Validate input types
        for key in required_keys:
            if key in inputs and inputs[key] is not None and not isinstance(inputs[key], str):
                logger.warning(f"Input key '{key}' is not a string, converting to string")
                inputs[key] = str(inputs[key])
        
        instruction = self._build_prompt(inputs)
        msg_history = []
        prediction = "None"
        thinking = ""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.log_fn(f"Attempt {attempt + 1}/{self.max_retries + 1}: Calling LLM (model={self.model})...")
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction from the last assistant message
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, thinking = self._extract_prediction(last_text)
                
                # Log extraction results with more detail
                pred_preview = str(prediction)[:100] if prediction != "None" else "None"
                self.log_fn(f"Attempt {attempt + 1}: prediction='{pred_preview}...', thinking_len={len(str(thinking))}")
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction on attempt {attempt + 1}")
                    break
                
                # If extraction failed and we have retries left, add feedback
                if attempt < self.max_retries:
                    feedback = self._build_retry_feedback(attempt, last_text)
                    msg_history.append({"role": "user", "text": feedback})
                    self.log_fn(f"Added feedback for retry {attempt + 2}")
                    
            except Exception as e:
                last_error = e
                self.log_fn(f"Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == self.max_retries:
                    break
                # Brief pause before retry on error
                time.sleep(0.5)
        
        if prediction == "None":
            if last_error:
                self.log_fn(f"Warning: Failed to extract valid prediction after all retries. Last error: {last_error}")
            else:
                self.log_fn("Warning: Failed to extract valid prediction after all retries")
        else:
            self.log_fn(f"Final prediction length: {len(str(prediction))} chars")
        
        # Store thinking for potential external access
        self._last_thinking = thinking
            
        return str(prediction), msg_history
    
    def _build_retry_feedback(self, attempt: int, last_response: str) -> str:
        """Build feedback message for retry attempts.
        
        Provides increasingly specific guidance based on the attempt number.
        """
        base_feedback = """Your previous response did not contain a valid JSON object with a "response" field. 

Please ensure you respond in the exact format:
<json>
{
    "thinking": "Your analysis here...",
    "response": "Your final answer here..."
}
</json>

The "response" field should contain your final evaluation/answer."""
        
        if attempt == 0:
            return base_feedback
        elif attempt == 1:
            # Second attempt - be more specific about common issues
            return base_feedback + """

Common issues to avoid:
- Make sure the JSON is properly formatted with double quotes
- Ensure there are no trailing commas in the JSON
- The response should be a complete evaluation, not just a number"""
        else:
            # Final attempt - simplify the request
            return """Please provide your answer in this exact format:

<json>
{
    "thinking": "Brief analysis of the student's answer",
    "response": "Your final evaluation/score here"
}
</json>"""
    
    def get_last_thinking(self) -> str:
        """Get the thinking/reasoning from the last forward call.
        
        Returns:
            The thinking string from the last evaluation, or empty string if none.
        """
        return self._last_thinking
