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
    Also handles markdown code blocks with json tag.
    Includes robust JSON repair for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        parsed = _parse_json_with_repair(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Also try markdown code blocks ```json ... ```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start = start + 7  # Skip past ```json
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            parsed = _parse_json_with_repair(inner)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _parse_json_with_repair(text: str) -> dict | None:
    """Parse JSON with multiple repair strategies for common LLM issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    """
    text = text.strip()
    if not text:
        return None
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 1: Remove trailing commas before } or ]
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 2: Replace single quotes with double quotes (carefully)
    try:
        # Only replace single quotes that appear to be string delimiters
        # This is a heuristic: single quotes followed by colon or comma or } are likely key delimiters
        fixed = re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*)'\s*(?=[,}])", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 3: Escape unescaped newlines in strings
    try:
        # Find strings and escape newlines within them
        fixed = re.sub(r'(?<!\\)\n', r'\\n', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Repair strategy 4: Combine strategies
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)  # trailing commas
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)  # newlines
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
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
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for JSON extraction failures

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required inputs are present and non-empty.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in inputs:
                missing_fields.append(field)
            elif not inputs[field] or (isinstance(inputs[field], str) and not inputs[field].strip()):
                empty_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        if empty_fields:
            return False, f"Empty required fields: {', '.join(empty_fields)}"
        
        return True, ""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []

        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required to solve this problem. Note any critical steps that must be present in a complete solution.

2. OFFICIAL SOLUTION REVIEW: Understand the expected reasoning path, key insights, and any alternative valid approaches mentioned in the official solution.

3. STUDENT ANSWER EVALUATION:
   - Check if the student's final answer is mathematically correct
   - Verify the logical flow of their reasoning
   - Identify any gaps, errors, or misconceptions
   - Note any creative or alternative valid approaches
   - Check for completeness: did they prove all necessary steps?

4. GRADING CRITERIA APPLICATION:
   - Apply the specific grading guidelines provided above
   - Look for partial credit criteria: partial progress, correct methods with calculation errors, etc.
   - Consider the rigor of mathematical proof required
   - Be fair but consistent with IMO standards

5. FINAL GRADE DETERMINATION:
   - Synthesize your analysis into a clear grade
   - Ensure the grade matches the format specified in the guidelines
   - Provide specific justification referencing the student's work

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Be thorough and specific about what the student did right or wrong. Reference specific parts of their answer.",
    "response": "The final grade/prediction. Use the exact format specified in the grading guidelines (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)."
}}
</json>"""

        # Attempt LLM call with retry for JSON extraction failures
        prediction = "None"
        all_msg_history = []
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=all_msg_history if attempt > 0 else [],
            )
            all_msg_history = msg_history
            
            # Extract prediction from JSON with fallback mechanisms
            extracted_prediction = self._extract_prediction(msg_history)
            
            if extracted_prediction is not None:
                prediction = extracted_prediction
                break
            
            if attempt < self.max_retries:
                self.log_fn(f"JSON extraction failed on attempt {attempt + 1}, retrying with reminder...")
                # Add a reminder to the conversation for the retry
                reminder = (
                    "Your previous response did not contain valid JSON in the required format. "
                    "Please respond with ONLY valid JSON wrapped in <json> tags, like this:\n"
                    "<json>\n{\"reasoning\": \"your analysis\", \"response\": \"your grade\"}\n</json>"
                )
                all_msg_history.append({"role": "user", "text": reminder})
        else:
            self.log_fn(f"Failed to extract valid JSON after {self.max_retries + 1} attempts")

        return str(prediction), all_msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str | None:
        """Extract prediction from message history.
        
        Returns:
            The extracted prediction string, or None if extraction failed.
        """
        try:
            if not msg_history:
                return None
                
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if not extracted:
                return None
            
            # Prefer response field, but accept other common field names
            last_json = extracted[-1]
            field_priority = [
                "response", "grade", "answer", "result", 
                "evaluation", "score", "verdict", "prediction"
            ]
            
            for field in field_priority:
                if field in last_json:
                    value = last_json[field]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, (int, float)):
                        return str(value)
            
            # If no known field, use the first string or numeric value found
            for key, value in last_json.items():
                if isinstance(value, str):
                    return value
                elif isinstance(value, (int, float)):
                    return str(value)
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return None
