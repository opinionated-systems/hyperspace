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
    Includes robust error recovery for malformed JSON.
    """
    if not text:
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
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse the JSON content
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            parsed = _try_parse_json(block.strip())
            if parsed:
                results.append(parsed)
        
        # Try bare ``` ... ``` blocks that might contain JSON
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') or block.startswith('['):
                    parsed = _try_parse_json(block)
                    if parsed:
                        results.append(parsed)
        
        # Try bare JSON objects as fallback
        if not results:
            results = _extract_bare_json(text)
    
    return results if results else None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with various fixes for common issues."""
    if not text:
        return None
        
    # First, try direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from within the content (in case there's extra text)
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        candidate = text[json_start:json_end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        # Try fixing common JSON issues
        try:
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', candidate)
            # Fix single quotes (convert to double quotes)
            fixed = fixed.replace("'", '"')
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    
    return None


def _extract_bare_json(text: str) -> list[dict]:
    """Extract bare JSON objects from text using brace matching."""
    results = []
    
    # Find JSON-like structures with nested support
    brace_positions = []
    for i, char in enumerate(text):
        if char == '{':
            brace_positions.append((i, 'open'))
        elif char == '}':
            brace_positions.append((i, 'close'))
    
    # Try to find valid JSON by matching braces
    if brace_positions:
        stack = []
        for pos, kind in brace_positions:
            if kind == 'open':
                stack.append(pos)
            elif kind == 'close' and stack:
                start_pos = stack.pop()
                candidate = text[start_pos:pos+1]
                
                # Only consider reasonably-sized objects (not too small, not too large)
                if len(candidate) < 10:
                    continue
                    
                parsed = _try_parse_json(candidate)
                if parsed and isinstance(parsed, dict):
                    # Check if it has expected grading keys
                    if any(k in parsed for k in ["response", "grade", "answer", "result", "assessment", "evaluation", "score", "reasoning"]):
                        results.append(parsed)
    
    return results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for failed JSON extraction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions with precision and consistency.

Your task is to grade a student's answer by systematically comparing it against the official solution and applying the grading guidelines rigorously.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Instructions:
Follow this structured approach for consistent evaluation:

1. **Understanding Check**: Verify you understand the problem requirements and what constitutes a correct solution.

2. **Step-by-Step Analysis**: Break down the student's answer and compare each step with the official solution:
   - Identify all correct steps and valid reasoning
   - Identify any errors, omissions, or misconceptions
   - Note any alternative valid approaches

3. **Guideline Application**: Map the grading guidelines to the student's work:
   - Check for required elements mentioned in guidelines
   - Apply partial credit rules if applicable
   - Consider the severity of errors (minor vs. major)

4. **Consistency Check**: Ensure your grading aligns with the intended difficulty and standards.

5. **Final Assessment**: Provide a clear, justified grade.

## Response Format (REQUIRED):
You MUST respond with valid JSON in this exact format:

<json>
{{
    "reasoning": "Detailed analysis covering: (1) key aspects of the problem, (2) what the student did correctly, (3) any errors or gaps, (4) comparison to official solution, (5) application of grading guidelines",
    "response": "The final grade - use exact values from guidelines when specified (e.g., 'Correct', 'Partially Correct', 'Incorrect', or numeric scores like '7/10')"
}}
</json>

Important: Ensure your JSON is valid with no trailing commas and proper escaping of quotes within strings."""

        # System message for consistent behavior
        system_msg = "You are a precise grading assistant. Always respond with valid JSON in the specified format. Be thorough in your analysis and consistent in your grading."
        
        # Try with retries for better reliability
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
                system_msg=system_msg,
            )

            # Extract prediction from JSON
            prediction, reasoning = self._extract_prediction(msg_history)
            
            if prediction != "None":
                break
            
            if attempt < self.max_retries:
                self.log_fn(f"Retry {attempt + 1}: No valid JSON found, retrying with stronger prompt...")
                # Add a reminder to the instruction for the retry
                instruction += "\n\nCRITICAL: Your previous response did not contain valid JSON. You MUST respond ONLY with valid JSON in the exact <json>...</json> format shown above. No other text outside the JSON block."

        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Try the last assistant message first (most likely to contain the answer)
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            # If not found, try searching all messages from most recent to oldest
            if not extracted:
                for msg in reversed(msg_history):
                    text = msg.get("text", "")
                    if text:
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
            
            if extracted:
                # Use the last JSON object found (most likely the final answer)
                result = extracted[-1]
                
                # Try multiple possible keys for the response/grade
                response_keys = ["response", "grade", "answer", "result", "assessment", 
                                "evaluation", "score", "verdict", "final_grade", "mark"]
                for key in response_keys:
                    if key in result and result[key] is not None:
                        prediction = str(result[key]).strip()
                        break
                
                # Extract reasoning if available
                reasoning_keys = ["reasoning", "analysis", "thought", "explanation", 
                                 "rationale", "thinking", "evaluation", "comments"]
                for key in reasoning_keys:
                    if key in result and result[key] is not None:
                        reasoning = str(result[key]).strip()
                        break
                
                # Log successful extraction for debugging
                if prediction != "None":
                    self.log_fn(f"Extracted grade: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning
