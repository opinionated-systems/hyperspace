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
    Also attempts to parse raw JSON objects if no <json> tags are found.
    Includes additional heuristics for common LLM output patterns.
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Fallback 1: try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError:
                    # Try fixing common issues
                    fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                    results.append(json.loads(fixed))
        except json.JSONDecodeError:
            pass
    
    # Fallback 2: look for code blocks with json
    if not results:
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM with clear section ordering.
    """
    # Define preferred order for grading context
    key_order = [
        "domain",
        "problem", 
        "solution",
        "grading_guidelines",
        "student_answer"
    ]
    
    parts = []
    # First add keys in preferred order if they exist
    for key in key_order:
        if key in inputs:
            value = inputs[key]
            parts.append(f"{'='*60}")
            parts.append(f"{key.upper().replace('_', ' ')}")
            parts.append(f"{'='*60}")
            parts.append(f"{value}")
            parts.append("")
    
    # Then add any remaining keys
    for key, value in inputs.items():
        if key not in key_order:
            parts.append(f"{'='*60}")
            parts.append(f"{key.upper().replace('_', ' ')}")
            parts.append(f"{'='*60}")
            parts.append(f"{value}")
            parts.append("")
    
    return "\n".join(parts)


def _extract_grade(output: str) -> int | None:
    """Extract grade 0-3 from LLM output, trying multiple patterns.
    
    Designed to be robust to verbose LLM output that wraps the grade
    in reasoning. Prioritises the LAST occurrence of grade indicators
    (the final verdict, not intermediate reasoning).
    """
    if not output or not output.strip():
        return None

    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]

    # Priority 1: last line is a bare digit 0-3
    if lines:
        last = lines[-1].strip("*_`#$\\boxed{}() .:").strip()
        if last in ("0", "1", "2", "3"):
            return int(last)

    # Priority 2: LAST "Grade: X" or "grade is X" in the output
    matches = list(re.finditer(r'[Gg]rade[:\s]+([0-3])\b', output))
    if matches:
        return int(matches[-1].group(1))

    # Priority 3: keyword match — find the LAST grade keyword in the output
    # to capture the final verdict, not intermediate reasoning
    text_lower = output.lower()
    last_pos = -1
    last_grade = None
    for name, grade in [("incorrect", 0), ("partial", 1), ("almost", 2)]:
        pos = text_lower.rfind(name)
        if pos > last_pos:
            last_pos = pos
            last_grade = grade
    # "correct" must not match "incorrect"
    for m in re.finditer(r'(?<!in)correct', text_lower):
        if m.start() > last_pos:
            last_pos = m.start()
            last_grade = 3
    if last_grade is not None:
        return last_grade

    # Priority 4: last digit 0-3 on the last line (but NOT negative numbers)
    if lines:
        # Only match standalone digits, not part of negative numbers or larger ints
        m_all = list(re.finditer(r'(?<!\d)(?<!-)([0-3])(?!\d)', lines[-1]))
        if m_all:
            return int(m_all[-1].group(1))

    return None


def _validate_response_schema(response: dict) -> tuple[bool, str]:
    """Validate that the response follows the expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Response is not a dict, got {type(response).__name__}"
    
    if "response" not in response:
        # Check for common alternative keys
        alternatives = ["answer", "result", "evaluation", "grade", "output"]
        for alt in alternatives:
            if alt in response:
                return False, f"Response uses '{alt}' instead of required 'response' key. Please use 'response' as the key name."
        return False, "Response missing required 'response' key"
    
    if not isinstance(response["response"], str):
        return False, f"'response' value is not a string, got {type(response['response']).__name__}"
    
    if len(response["response"].strip()) == 0:
        return False, "'response' value is empty"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", max_retries: int = 2) -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.max_retries = max_retries

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        # Build comprehensive grading instruction
        instruction = f"""You are an expert grading agent for mathematical olympiad problems. Your task is to evaluate student answers with precision and consistency.

{formatted_inputs}

GRADE DEFINITIONS (apply strictly, err on the side of lower grades):
  0 = Incorrect: contains fatal logical errors, false claims, or circular reasoning. No mathematically valid progress toward the solution.
  1 = Partial: contains genuine, mathematically valid progress (proves a useful lemma, handles critical special cases, or sets up correct framework) but fails to complete the main proof due to major gaps.
  2 = Almost: the main proof strategy and all key ideas are correct and valid. Only minor gaps, omitted trivial details, or small non-critical errors prevent full rigor. Would receive 6/7 points in IMO.
  3 = Correct: complete, rigorous, and correct. All non-trivial claims justified. Would receive full marks (7/7) in IMO.

GRADING INSTRUCTIONS:
1. Carefully read the problem and official solution first
2. Review the grading guidelines to understand the scoring criteria
3. Analyze the student's answer step by step
4. Compare the student's approach with the official solution
5. Determine the appropriate grade (0-3) based on the definitions above

IMPORTANT RULES:
- Be conservative and objective. Most solutions have errors.
- Grade 3 requires every step justified with no gaps.
- Award partial credit where appropriate based on the guidelines
- If there are errors, identify what is wrong and what credit should be given

OUTPUT FORMAT:
You MUST respond in valid JSON format wrapped in <json> tags:
<json>
{{
    "response": "Your detailed evaluation here. Include: (1) whether the answer is correct/partially correct/incorrect, (2) specific points awarded if applicable, (3) brief reasoning for your decision. Then on the very last line, write ONLY the integer grade (0, 1, 2, or 3)."
}}
</json>

The "response" field must contain your complete evaluation as a single string, ending with the numeric grade on the last line."""

        # Retry loop with validation
        for attempt in range(self.max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                self.log_fn(f"LLM call completed (attempt {attempt + 1}), response length: {len(response)}")
            except Exception as e:
                self.log_fn(f"Error in LLM call (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return "Error: LLM call failed", []
                continue

            # Extract prediction from JSON with better error handling
            prediction = "None"
            validation_passed = False
            
            try:
                if msg_history and len(msg_history) > 0:
                    last_message = msg_history[-1]
                    text_content = last_message.get("text", "")
                    extracted = _extract_jsons(text_content)
                    if extracted:
                        last_extracted = extracted[-1]
                        is_valid, error_msg = _validate_response_schema(last_extracted)
                        if is_valid:
                            response_text = last_extracted["response"]
                            # Try to extract numeric grade from the response
                            grade = _extract_grade(response_text)
                            if grade is not None:
                                prediction = str(grade)
                                validation_passed = True
                                self.log_fn(f"Successfully extracted grade: {grade}")
                            else:
                                # Fallback: use the full response text
                                prediction = response_text
                                self.log_fn(f"Could not extract grade, using full response: {str(prediction)[:100]}")
                                # Still consider this valid - let the evaluator handle extraction
                                validation_passed = True
                        else:
                            self.log_fn(f"Response validation failed: {error_msg}")
                            if attempt < self.max_retries:
                                # Add validation feedback to the instruction for retry
                                instruction += f"\n\n[VALIDATION ERROR - ATTEMPT {attempt + 1}]: {error_msg}\n\nPlease fix this issue and respond with valid JSON using the exact schema specified above."
                    else:
                        self.log_fn("No JSON found in response")
                        # Try to extract grade directly from text
                        grade = _extract_grade(text_content)
                        if grade is not None:
                            prediction = str(grade)
                            validation_passed = True
                            self.log_fn(f"Extracted grade from raw text: {grade}")
                        elif attempt < self.max_retries:
                            instruction += f"\n\n[VALIDATION ERROR - ATTEMPT {attempt + 1}]: No valid JSON found. You MUST wrap your response in <json>...</json> tags with a valid JSON object containing a 'response' key."
                else:
                    self.log_fn("Empty message history")
            except Exception as e:
                self.log_fn(f"Error extracting prediction (attempt {attempt + 1}): {e}")
            
            if validation_passed:
                return str(prediction), msg_history
            
            if attempt == self.max_retries:
                self.log_fn(f"Max retries ({self.max_retries}) reached, returning best effort result")
                return str(prediction), msg_history
        
        return "Error: Unexpected end of forward method", []
