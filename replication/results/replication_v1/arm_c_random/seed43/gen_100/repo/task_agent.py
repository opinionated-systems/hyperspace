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
    Also handles markdown code blocks and raw JSON objects.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
            # Try to clean up common issues
            try:
                # Remove trailing commas before closing braces
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(1).strip()))
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key
        json_pattern = r'\{[^{}]*"response"[^{}]*\}'
        for match in re.finditer(json_pattern, text):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

1. **Analyze the approach**: Identify the student's overall strategy and key mathematical techniques used.
2. **Step-by-step verification**: Check each claim and calculation against the official solution.
3. **Error classification**: Categorize any errors as:
   - Conceptual (misunderstanding of the problem)
   - Computational (arithmetic/algebraic mistakes)
   - Logical (flawed reasoning)
   - Completeness (missing cases or steps)
4. **Partial credit assessment**: If the solution is incomplete, identify which parts are correct and assign partial credit based on the grading guidelines.
5. **Final grade determination**: Use the standard IMO 0-7 scale where:
   - 7: Complete, correct solution
   - 6: Minor flaw or omission
   - 5-3: Significant progress with gaps
   - 2-1: Some meaningful progress
   - 0: No significant progress or irrelevant

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis including: (1) approach summary, (2) step-by-step verification, (3) error classification if any, (4) partial credit justification...",
    "response": "Your final grade as a number 0-7 (e.g., '7', '5', '2', '0')"
}}
</json>

The "response" field MUST contain ONLY a single integer from 0 to 7. The "reasoning" field contains your full analysis."""

    def _validate_grade(self, grade: str) -> str | None:
        """Validate and normalize the grade to IMO 0-7 scale.
        
        Returns normalized grade string or None if invalid.
        """
        if not grade:
            return None
        
        # Clean up the grade string
        grade = str(grade).strip()
        
        # Try to extract a number from the grade
        import re
        numbers = re.findall(r'\d+', grade)
        if numbers:
            num = int(numbers[0])
            # Clamp to valid IMO range
            if 0 <= num <= 7:
                return str(num)
        
        # Check for text-based grades
        grade_lower = grade.lower()
        if any(x in grade_lower for x in ['full', 'complete', 'correct', '7']):
            return '7'
        if any(x in grade_lower for x in ['none', 'zero', '0', 'no progress', 'irrelevant']):
            return '0'
        if any(x in grade_lower for x in ['partial', 'incomplete']):
            # Try to extract partial credit
            if '3' in grade:
                return '3'
            if '2' in grade:
                return '2'
            if '1' in grade:
                return '1'
            return '3'  # Default partial credit
        
        return None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        
        all_msg_history = []
        prediction = "None"
        
        # Retry loop for better grade extraction
        for attempt in range(self.max_retries):
            response, msg_history, info = get_response_from_llm(
                msg=instruction if attempt == 0 else f"Previous response had an invalid grade format. Please respond with ONLY a number 0-7 in the 'response' field.\n\n{instruction}",
                model=self.model,
                msg_history=[] if attempt == 0 else msg_history,
            )
            
            all_msg_history.extend(msg_history)
            
            # Extract prediction from JSON with better error handling
            reasoning = ""
            try:
                # Try to extract from the last assistant message
                last_msg = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_jsons(last_msg)
                
                if extracted:
                    last_json = extracted[-1]
                    if "response" in last_json:
                        raw_prediction = last_json["response"]
                        validated = self._validate_grade(raw_prediction)
                        if validated:
                            prediction = validated
                            if "reasoning" in last_json:
                                reasoning = last_json["reasoning"]
                            # Log the reasoning for debugging
                            if reasoning:
                                self.log_fn(f"Reasoning: {reasoning[:200]}...")
                            break  # Valid grade found, exit retry loop
                        else:
                            self.log_fn(f"Invalid grade format on attempt {attempt + 1}: {raw_prediction}")
                    else:
                        self.log_fn(f"No 'response' field found on attempt {attempt + 1}")
                else:
                    # Fallback: try to find any JSON-like structure
                    json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                    if json_match:
                        try:
                            fallback = json.loads(json_match.group())
                            raw_prediction = fallback.get("response", "None")
                            validated = self._validate_grade(raw_prediction)
                            if validated:
                                prediction = validated
                                break
                        except json.JSONDecodeError:
                            pass
                    
                    # Last resort: try to find any digit in the response
                    digits = re.findall(r'\b[0-7]\b', last_msg)
                    if digits:
                        prediction = digits[-1]  # Use last digit found
                        self.log_fn(f"Extracted grade via digit search on attempt {attempt + 1}: {prediction}")
                        break
                            
            except Exception as e:
                self.log_fn(f"Error extracting prediction on attempt {attempt + 1}: {e}")
        
        # Log final result
        self.log_fn(f"Final grade: {prediction} (after {min(attempt + 1, self.max_retries)} attempt(s))")

        return str(prediction), all_msg_history
