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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json tag.
    """
    results = []
    search_from = 0
    
    # First try to find explicit <json> tags
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
                cleaned = re.sub(r',\s*}', '}', inner)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                try:
                    cleaned = re.sub(r',\s*}', '}', inner)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    return results or None


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


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', text)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    # Fix single quotes to double quotes (common LLM mistake)
    cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    return cleaned


def _extract_grade_with_retry(
    text: str, 
    max_retries: int = 3
) -> tuple[str, list[dict] | None]:
    """Extract grade from LLM response with multiple fallback strategies.
    
    Args:
        text: The LLM response text
        max_retries: Number of extraction attempts with different strategies
        
    Returns:
        Tuple of (prediction, extracted_json_list)
    """
    prediction = "None"
    extracted = None
    
    for attempt in range(max_retries):
        try:
            # Try primary extraction method
            if attempt == 0:
                extracted = _extract_jsons(text)
            # Try with aggressive cleaning
            elif attempt == 1:
                cleaned = _clean_json_string(text)
                extracted = _extract_jsons(cleaned)
                if extracted is None:
                    extracted = _extract_any_json(cleaned)
            # Last resort: look for any JSON-like structure
            else:
                extracted = _extract_any_json(text)
            
            if extracted:
                last_json = extracted[-1]
                
                # Priority order for grade fields
                grade_fields = [
                    "response", "grade", "answer", "result", "evaluation", 
                    "prediction", "score", "verdict", "assessment", "grading"
                ]
                
                for field in grade_fields:
                    if field in last_json:
                        field_value = last_json[field]
                        if isinstance(field_value, (str, int, float, bool)):
                            prediction = str(field_value)
                            return prediction, extracted
                
                # If no known field, use first string/numeric value
                for key, value in last_json.items():
                    if isinstance(value, (str, int, float)) and key != "reasoning":
                        prediction = str(value)
                        return prediction, extracted
                        
                # Check for grade-like values
                for key, value in last_json.items():
                    if isinstance(value, str):
                        val_lower = value.lower()
                        if val_lower in ["correct", "incorrect", "partial", "true", "false"]:
                            prediction = value
                            return prediction, extracted
                            
        except Exception as e:
            logger.debug(f"Extraction attempt {attempt + 1} failed: {e}")
            continue
    
    return prediction, extracted


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "extraction_retries": 0,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        start_time = time.time()
        
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log problem info for debugging
        self.log_fn(f"Processing problem in domain: {domain}")
        self.log_fn(f"Student answer length: {len(student_answer)} chars")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis following this structure:

1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required to solve this problem.

2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format. Note what constitutes a complete and correct solution.

3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps, valid insights, or correct intermediate results
   - Identify errors, gaps, misconceptions, or missing steps
   - Check if the final answer matches the expected form

4. GRADING CRITERIA CHECK:
   - Systematically verify if the student met each criterion in the grading guidelines
   - Award partial credit for incomplete but valid reasoning
   - Note any specific point deductions for errors or omissions

5. FINAL DETERMINATION: Assign a clear grade based on:
   - Completeness of the solution
   - Mathematical correctness
   - Adherence to grading guidelines
   - Quality of reasoning shown

Respond ONLY in JSON format with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be specific about what the student did right and wrong.",
    "response": "The final grade/prediction. Use exactly one of: 'Correct', 'Incorrect', 'Partial', or a numeric score if specified in guidelines."
}}
</json>

Important guidelines:
- Be objective, consistent, and thorough in your analysis
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- If the student answer is empty or completely irrelevant, grade as 'Incorrect'
- If the student shows significant correct work but has minor errors, consider 'Partial'
- Only output the JSON block, no additional text before or after"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error", [{"role": "system", "text": f"LLM call failed: {e}"}]

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Use the new retry-based extraction function
            prediction, extracted = _extract_grade_with_retry(last_message, max_retries=3)
            
            if extracted:
                self.stats["successful_extractions"] += 1
                self.log_fn(f"Successfully extracted grade: {prediction}")
            else:
                self.stats["failed_extractions"] += 1
                self.log_fn(f"Failed to extract JSON from response. Raw response preview: {last_message[:200]}...")
                
        except Exception as e:
            self.stats["failed_extractions"] += 1
            self.log_fn(f"Error extracting prediction: {e}")

        elapsed = time.time() - start_time
        self.log_fn(f"Task completed in {elapsed:.2f}s with prediction: {prediction}")
        
        return str(prediction), msg_history
    
    def get_stats(self) -> dict[str, Any]:
        """Return agent statistics for monitoring."""
        return dict(self.stats)
