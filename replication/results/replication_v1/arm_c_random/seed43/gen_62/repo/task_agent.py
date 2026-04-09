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
    Also handles markdown code blocks, raw JSON objects, and nested structures.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks with proper nesting support
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the matching </json> tag, accounting for nested content
        end = start + 6
        json_depth = 0
        in_string = False
        escape_next = False
        
        while end < len(text):
            if end + 7 <= len(text) and text[end:end+7] == "</json>":
                if json_depth == 0:
                    break
            
            char = text[end]
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    json_depth += 1
                elif char == '}':
                    json_depth -= 1
            end += 1
        
        if end >= len(text) or end + 7 > len(text):
            break
            
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse with multiple cleanup strategies
        for cleanup_fn in [
            lambda x: x,  # No cleanup
            lambda x: re.sub(r',(\s*[}\]])', r'\1', x),  # Remove trailing commas
            lambda x: re.sub(r'\n\s*', ' ', x),  # Remove newlines
            lambda x: re.sub(r'\s+', ' ', x),  # Normalize whitespace
            lambda x: re.sub(r'"\s*:\s*"', '": "', x),  # Normalize colons
            lambda x: re.sub(r'"\s*:\s*(\d)', '": \1', x),  # Normalize number values
        ]:
            try:
                cleaned = cleanup_fn(inner)
                results.append(json.loads(cleaned))
                break
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            for cleanup_fn in [
                lambda x: x,
                lambda x: re.sub(r',(\s*[}\]])', r'\1', x),
                lambda x: re.sub(r'\n\s*', ' ', x),
                lambda x: re.sub(r'\s+', ' ', x),
            ]:
                try:
                    results.append(json.loads(cleanup_fn(content)))
                    break
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - improved pattern for nested braces
        json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*"response"(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        for match in re.finditer(json_pattern, text):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    
    # Extra last resort: try to find any JSON object with "reasoning" field
    if not results:
        json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*"reasoning"(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        for match in re.finditer(json_pattern, text):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs.

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

## Evaluation Framework

Follow this structured approach:

1. **Understanding Check**: Verify you understand the problem and official solution completely. Identify the key insights required.

2. **Student's Approach Analysis**:
   - What approach did the student take?
   - Is it the same as the official solution or a valid alternative?
   - Are there any creative or novel elements?

3. **Correctness Verification**:
   - Check each claim and step in the student's proof
   - Identify any logical gaps, errors, or unjustified assertions
   - Note any missing cases or incomplete arguments

4. **Partial Credit Assessment**:
   - What correct progress did the student make?
   - How close did they get to a complete solution?
   - What fraction of the problem did they solve correctly?

5. **Final Grade Determination**:
   - IMO problems are graded on a 0-7 point scale
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5-3: Significant progress with varying degrees of completeness
   - 2-1: Some meaningful progress
   - 0: No significant progress or completely wrong

## Response Format (CRITICAL - FOLLOW EXACTLY)

You MUST respond with valid JSON in this exact format. Do not add any text before or after the JSON:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering: (1) approach taken, (2) correctness of each step, (3) errors found, (4) partial credit justification, (5) final grade rationale",
    "response": "X"
}}
</json>

IMPORTANT:
- The "response" field must contain ONLY a single digit from 0-7, OR text like "Partial credit: 3"
- Do not include any explanation in the response field
- Do not include any text outside the <json>...</json> tags
- Ensure the JSON is valid (no trailing commas, proper quotes, etc.)"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with comprehensive error handling
        prediction = "None"
        reasoning = ""
        confidence = 0.0
        
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            # First attempt: use the enhanced _extract_jsons function
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                    confidence = 1.0
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                    # Log the reasoning for debugging
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            
            # Second attempt: look for grade patterns in the raw text
            if prediction == "None" or not prediction:
                # Look for explicit grade mentions with more comprehensive patterns
                grade_patterns = [
                    r'grade[\s:]+(\d+)',
                    r'score[\s:]+(\d+)',
                    r'points?[\s:]+(\d+)',
                    r'final grade[\s:]+(\d+)',
                    r'response[\s:]+(\d+)',
                    r'"response"\s*:\s*"(\d+)"',
                    r'"response"\s*:\s*(\d+)',
                    r'^(\d)$',  # Single digit on its own line
                    r'\bgrade\s+(\d)\b',
                    r'\bscore\s+(\d)\b',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_msg, re.IGNORECASE | re.MULTILINE)
                    if match:
                        potential_grade = match.group(1)
                        if potential_grade.isdigit() and 0 <= int(potential_grade) <= 7:
                            prediction = potential_grade
                            confidence = 0.7
                            self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                            break
            
            # Third attempt: look for text-based grades
            if prediction == "None" or not prediction:
                text_grades = {
                    r'\bzero\b': '0',
                    r'\bone\b': '1',
                    r'\btwo\b': '2',
                    r'\bthree\b': '3',
                    r'\bfour\b': '4',
                    r'\bfive\b': '5',
                    r'\bsix\b': '6',
                    r'\bseven\b': '7',
                    r'\bfull marks?\b|\bcomplete solution\b': '7',
                    r'\bpartial credit[\s:]+(\d+)': None,  # Special handling
                }
                for pattern, grade in text_grades.items():
                    match = re.search(pattern, last_msg, re.IGNORECASE)
                    if match:
                        if grade is None:  # Partial credit pattern
                            prediction = f"Partial credit: {match.group(1)}"
                        else:
                            prediction = grade
                        confidence = 0.6
                        self.log_fn(f"Extracted grade via text matching: {prediction}")
                        break
            
            # Fourth attempt: look for any standalone digit 0-7 in the response
            if prediction == "None" or not prediction:
                # Find all standalone digits 0-7
                digits = re.findall(r'\b([0-7])\b', last_msg)
                if digits:
                    # Take the last one (likely the final grade)
                    prediction = digits[-1]
                    confidence = 0.5
                    self.log_fn(f"Extracted grade via digit search: {prediction}")
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Even on error, try one more fallback
            try:
                if response:
                    # Try to find any digit that could be a grade
                    digits = re.findall(r'\b([0-7])\b', response)
                    if digits:
                        prediction = digits[-1]  # Take the last one (likely the final grade)
                        confidence = 0.5
                        self.log_fn(f"Fallback extraction: {prediction}")
            except Exception:
                pass

        # Log confidence level for monitoring
        self.log_fn(f"Prediction: {prediction} (confidence: {confidence:.2f})")
        
        return str(prediction), msg_history
