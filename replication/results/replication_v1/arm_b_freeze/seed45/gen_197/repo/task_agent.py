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
    Includes improved handling for nested braces and escaped characters.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of extracted JSON objects, or None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        logger.debug("Invalid input to _extract_jsons: %s", type(text))
        return None
        
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug("Found opening <json> but no closing </json> tag")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug("JSON parse error in <json> block: %s", e)
            # Try to extract JSON from within the content using brace matching
            try:
                json_str = _extract_json_with_brace_matching(inner)
                if json_str:
                    results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError) as e2:
                logger.debug("Brace matching also failed: %s", e2)
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try brace matching extraction
                try:
                    json_str = _extract_json_with_brace_matching(block)
                    if json_str:
                        results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Try bare JSON objects as fallback with improved regex
        if not results:
            # Find JSON-like structures with nested brace support
            potential_jsons = _find_json_objects(text)
            for pj in potential_jsons:
                try:
                    results.append(json.loads(pj))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _extract_json_with_brace_matching(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces correctly by counting open/close braces.
    It properly handles strings (including escaped quotes) to avoid
    counting braces inside string literals.
    
    Args:
        text: The text to search for a JSON object.
        
    Returns:
        The extracted JSON string, or None if no valid JSON object found.
    """
    if not text or not isinstance(text, str):
        return None
        
    start = text.find("{")
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Validate the extracted JSON before returning
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        # Continue searching for a valid JSON object
                        continue
    
    return None


def _find_json_objects(text: str) -> list[str]:
    """Find all potential JSON objects in text using brace matching.
    
    Args:
        text: The text to search for JSON objects.
        
    Returns:
        A list of valid JSON strings found in the text.
    """
    if not text or not isinstance(text, str):
        return []
        
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            json_str = _extract_json_with_brace_matching(text[i:])
            if json_str and '"' in json_str:  # Must contain at least one quoted string
                results.append(json_str)
                i += len(json_str)
            else:
                i += 1
        else:
            i += 1
    return results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    This agent uses an LLM to evaluate student answers against official solutions
    and grading guidelines, with robust JSON extraction and retry logic.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased max retries for better robustness
        self._response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "verdict", "conclusion"]
        self._reasoning_keys = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]
        self._log_file = log_file
    
    def _log_interaction(self, stage: str, content: str, attempt: int = 0) -> None:
        """Log an interaction for debugging purposes.
        
        Args:
            stage: The stage of interaction (e.g., 'input', 'output', 'error')
            content: The content to log
            attempt: The retry attempt number (if applicable)
        """
        attempt_str = f" (attempt {attempt})" if attempt > 0 else ""
        truncated = content[:500] + "..." if len(content) > 500 else content
        self.log_fn(f"[{stage}{attempt_str}] {truncated}")

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs with comprehensive error handling
        if not isinstance(inputs, dict):
            logger.error(f"Invalid inputs type: {type(inputs)}")
            return "Error: Invalid inputs - expected dict", []
        
        # Extract fields for better prompting with type validation
        try:
            domain = str(inputs.get("domain", "Mathematics")) if inputs.get("domain") is not None else "Mathematics"
            problem = str(inputs.get("problem", "")) if inputs.get("problem") is not None else ""
            solution = str(inputs.get("solution", "")) if inputs.get("solution") is not None else ""
            grading_guidelines = str(inputs.get("grading_guidelines", "")) if inputs.get("grading_guidelines") is not None else ""
            student_answer = str(inputs.get("student_answer", "")) if inputs.get("student_answer") is not None else ""
        except Exception as e:
            logger.error(f"Error converting input fields to strings: {e}")
            return f"Error: Failed to process inputs - {e}", []
        
        # Validate required fields with detailed logging
        missing_fields = []
        if not problem:
            missing_fields.append("problem")
        if not student_answer:
            missing_fields.append("student_answer")
        if not solution:
            missing_fields.append("solution")
        
        if missing_fields:
            logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            # Continue anyway as the LLM might still be able to provide useful feedback

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

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

Important: 
- The JSON must be valid and properly formatted.
- Wrap the JSON in <json>...</json> tags.
- The 'response' field should contain only the final grade/assessment.
- The 'reasoning' field should contain your detailed analysis.

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Log the initial instruction (truncated for privacy)
        self._log_interaction("input", instruction[:200])
        
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    continue
                return f"Error: LLM call failed after {self.max_retries + 1} attempts", msg_history

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
                    # Validate result is a dict
                    if not isinstance(result, dict):
                        self.log_fn(f"Warning: Extracted result is not a dict: {type(result)}")
                        if attempt < self.max_retries:
                            feedback = (
                                f"Your response was extracted but is not a valid JSON object. "
                                f"Got type: {type(result).__name__}. "
                                "Please respond with a JSON object wrapped in <json>...</json> tags."
                            )
                            msg_history.append({"role": "user", "text": feedback})
                            instruction = feedback
                            continue
                        break
                    
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
                    
                    # Log reasoning if available
                    if reasoning:
                        self._log_interaction("reasoning", reasoning)
                    
                    # Log the final prediction
                    self._log_interaction("prediction", str(prediction), attempt)
                    
                    # Success - break out of retry loop
                    break
                elif attempt < self.max_retries:
                    # No JSON found - add feedback and retry
                    self.log_fn(f"No JSON found in attempt {attempt + 1}, retrying with feedback...")
                    feedback = (
                        "Your previous response did not contain valid JSON in the required format. "
                        "Please respond with a JSON object wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields. "
                        "Example format:\n"
                        "<json>\n"
                        '{\n  "reasoning": "The student correctly identified...",\n  "response": "Correct"\n}'
                        "\n</json>"
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback  # Update instruction for next iteration
                else:
                    self.log_fn(f"No JSON found after {self.max_retries + 1} attempts")
                    
            except Exception as e:
                self._log_interaction("error", str(e), attempt)
                if attempt < self.max_retries:
                    feedback = (
                        f"Error parsing your response: {e}. "
                        "Please ensure your response contains valid JSON wrapped in <json>...</json> tags. "
                        "The JSON must have 'reasoning' and 'response' fields."
                    )
                    msg_history.append({"role": "user", "text": feedback})
                    instruction = feedback

        return str(prediction), msg_history

    def _build_instruction(self, inputs: dict) -> str:
        """Build the instruction prompt for the LLM.
        
        Args:
            inputs: Dictionary containing problem data.
            
        Returns:
            The formatted instruction string.
        """
        domain = str(inputs.get("domain", "Mathematics")) if inputs.get("domain") is not None else "Mathematics"
        problem = str(inputs.get("problem", "")) if inputs.get("problem") is not None else ""
        solution = str(inputs.get("solution", "")) if inputs.get("solution") is not None else ""
        grading_guidelines = str(inputs.get("grading_guidelines", "")) if inputs.get("grading_guidelines") is not None else ""
        student_answer = str(inputs.get("student_answer", "")) if inputs.get("student_answer") is not None else ""
        
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

Important: 
- The JSON must be valid and properly formatted.
- Wrap the JSON in <json>...</json> tags.
- The 'response' field should contain only the final grade/assessment.
- The 'reasoning' field should contain your detailed analysis.

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""
