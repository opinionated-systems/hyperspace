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

1. **Understanding Check**: Verify you understand the problem and official solution completely.

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
   - IMO problems are typically graded 0-7 points
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5-3: Significant progress with varying degrees of completeness
   - 2-1: Some meaningful progress
   - 0: No significant progress or completely wrong

## Response Format

You MUST respond with valid JSON in this exact format:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering: (1) approach taken, (2) correctness of each step, (3) errors found, (4) partial credit justification, (5) final grade rationale",
    "response": "X"
}}
</json>

Where "response" is ONLY the final grade (a number 0-7, or text like 'Partial credit: 3'). Do not include any explanation in the response field."""

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
        
        def _validate_grade(grade_str: str) -> tuple[str, float]:
            """Validate and normalize a grade string. Returns (normalized_grade, confidence)."""
            if not grade_str:
                return "None", 0.0
            
            grade_str = str(grade_str).strip()
            
            # Check for partial credit format
            partial_match = re.match(r'partial\s*credit\s*[:\s]*(\d+)', grade_str, re.IGNORECASE)
            if partial_match:
                val = int(partial_match.group(1))
                if 0 <= val <= 7:
                    return f"Partial credit: {val}", 0.9
            
            # Check for numeric grade
            if grade_str.isdigit():
                val = int(grade_str)
                if 0 <= val <= 7:
                    return str(val), 1.0
            
            # Check for decimal grades (round to nearest int)
            try:
                val = float(grade_str)
                if 0 <= val <= 7:
                    return str(int(round(val))), 0.8
            except ValueError:
                pass
            
            return "None", 0.0
        
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            # First attempt: use the enhanced _extract_jsons function
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    raw_prediction = str(last_json["response"]).strip()
                    prediction, conf = _validate_grade(raw_prediction)
                    confidence = conf
                    if confidence > 0:
                        self.log_fn(f"Extracted grade from JSON: {prediction}")
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                    # Log the reasoning for debugging
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            
            # Second attempt: look for grade patterns in the raw text
            if confidence < 0.5:
                # Look for explicit grade mentions with more specific patterns
                grade_patterns = [
                    (r'final\s*grade[\s:]+(\d+(?:\.\d+)?)', 0.75),
                    (r'grade[\s:]+(\d+(?:\.\d+)?)', 0.7),
                    (r'score[\s:]+(\d+(?:\.\d+)?)', 0.7),
                    (r'points?[\s:]+(\d+(?:\.\d+)?)', 0.65),
                    (r'response[\s:]+(\d+(?:\.\d+)?)', 0.65),
                    (r'^(\d(?:\.\d+)?)$', 0.6),  # Single number on its own line
                ]
                for pattern, conf in grade_patterns:
                    match = re.search(pattern, last_msg, re.IGNORECASE | re.MULTILINE)
                    if match:
                        potential_grade = match.group(1)
                        pred, val_conf = _validate_grade(potential_grade)
                        if val_conf > 0:
                            prediction = pred
                            confidence = min(conf, val_conf)  # Take lower of pattern and validation confidence
                            self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                            break
            
            # Third attempt: look for text-based grades
            if confidence < 0.5:
                text_grades = [
                    (r'\bfull\s*marks?\b|\bcomplete\s*solution\b', '7', 0.6),
                    (r'\bseven\b', '7', 0.55),
                    (r'\bsix\b', '6', 0.55),
                    (r'\bfive\b', '5', 0.55),
                    (r'\bfour\b', '4', 0.55),
                    (r'\bthree\b', '3', 0.55),
                    (r'\btwo\b', '2', 0.55),
                    (r'\bone\b', '1', 0.55),
                    (r'\bzero\b|\bno\s*(?:credit|points|marks?)\b', '0', 0.55),
                ]
                for pattern, grade, conf in text_grades:
                    match = re.search(pattern, last_msg, re.IGNORECASE)
                    if match:
                        prediction = grade
                        confidence = conf
                        self.log_fn(f"Extracted grade via text matching: {prediction}")
                        break
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Even on error, try one more fallback
            try:
                if response:
                    # Try to find any digit that could be a grade (prefer later occurrences)
                    digits = re.findall(r'\b([0-7])\b', response)
                    if digits:
                        prediction = digits[-1]  # Take the last one (likely the final grade)
                        confidence = 0.4
                        self.log_fn(f"Fallback extraction: {prediction}")
            except Exception:
                pass

        # Final validation and logging
        final_pred, final_conf = _validate_grade(prediction)
        if final_conf > 0:
            prediction = final_pred
            confidence = max(confidence, final_conf)
        
        self.log_fn(f"Prediction: {prediction} (confidence: {confidence:.2f})")
        
        return str(prediction), msg_history
