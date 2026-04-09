"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery and nested JSON handling.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the content using brace matching
        try:
            # Find first { and use brace counting to find matching }
            json_start = inner.find("{")
            if json_start == -1:
                continue
            
            brace_count = 0
            json_end = -1
            in_string = False
            escape_next = False
            
            for i, char in enumerate(inner[json_start:], start=json_start):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i
                            break
            
            if json_end != -1:
                results.append(json.loads(inner[json_start:json_end + 1]))
        except json.JSONDecodeError:
            # Final fallback: try to fix common JSON issues
            try:
                # Remove trailing commas before } or ]
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner[json_start:json_end + 1] if json_end != -1 else inner)
                # Try to extract any valid JSON object
                match = re.search(r'\{.*\}', fixed, re.DOTALL)
                if match:
                    results.append(json.loads(match.group()))
            except (json.JSONDecodeError, AttributeError):
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try with brace extraction
                try:
                    match = re.search(r'\{.*\}', block, re.DOTALL)
                    if match:
                        results.append(json.loads(match.group()))
                except json.JSONDecodeError:
                    continue
        
        # Try bare JSON objects as fallback with improved pattern
        if not results:
            # Find JSON-like structures with nested support
            # Look for patterns starting with { and having key-value pairs
            potential_starts = [m.start() for m in re.finditer(r'\{\s*"', text)]
            for start_idx in potential_starts:
                # Use brace counting to find the end
                brace_count = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(text[start_idx:], start=start_idx):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == "\\":
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                try:
                                    results.append(json.loads(text[start_idx:i+1]))
                                except json.JSONDecodeError:
                                    pass
                                break
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Max retries for JSON extraction failures
        self._response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score", "verdict"]
        self._reasoning_keys = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking", "evaluation"]

    def _normalize_grade(self, grade: str) -> str:
        """Normalize grade to a standard format."""
        if not isinstance(grade, str):
            grade = str(grade)
        
        grade_lower = grade.lower().strip()
        
        # Map common variations to standard forms
        if any(x in grade_lower for x in ["correct", "right", "valid", "true", "yes"]):
            if "partial" in grade_lower or "part" in grade_lower:
                return "Partially Correct"
            return "Correct"
        elif any(x in grade_lower for x in ["incorrect", "wrong", "invalid", "false", "no", "error"]):
            return "Incorrect"
        elif any(x in grade_lower for x in ["partial", "incomplete", "half"]):
            return "Partially Correct"
        
        # Return original if no normalization applied
        return grade.strip()

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build the grading prompt from inputs."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] + "..." if len(problem) > max_len else problem
        solution = solution[:max_len] + "..." if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] + "..." if len(student_answer) > max_len else student_answer

        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history,
            )

            # Extract prediction from JSON
            extracted = None
            try:
                # Try the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                # If not found, try searching all messages
                if not extracted:
                    for msg in reversed(msg_history):
                        text = msg.get("text", "")
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
                
                if extracted:
                    result = extracted[-1]
                    
                    # Try multiple possible keys for the response
                    for key in self._response_keys:
                        if key in result:
                            prediction = result[key]
                            break
                    
                    # Extract reasoning if available
                    for key in self._reasoning_keys:
                        if key in result:
                            reasoning = result[key]
                            break
                    
                    # Normalize the grade
                    prediction = self._normalize_grade(prediction)
                    
                    # Log reasoning if available
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    
                    # Success - break out of retry loop
                    break
                elif attempt < self.max_retries:
                    # No JSON found - add feedback and retry
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = (
                        "Your previous response did not contain valid JSON in the required format. "
                        "Please respond with a JSON object wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback  # Update instruction for next iteration
                else:
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history
