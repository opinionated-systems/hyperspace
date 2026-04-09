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
    Also handles nested JSON objects and malformed tags gracefully.
    """
    results = []
    search_from = 0
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            # Try to find partial/closing tag without exact match
            # Look for the next closing angle bracket after <json>
            next_tag = text.find(">", start)
            if next_tag != -1 and next_tag < start + 20:
                # Malformed tag like <json >, try to recover
                start = next_tag + 1
            else:
                break
        else:
            inner = text[start + 6:end].strip()
            search_from = end + 7
            
            # Try to parse the JSON, handling common LLM formatting issues
            try:
                # Remove any markdown code block markers that might be inside
                inner_clean = re.sub(r'^```json\s*', '', inner)
                inner_clean = re.sub(r'\s*```$', '', inner_clean)
                results.append(json.loads(inner_clean))
            except json.JSONDecodeError:
                # Try to extract just the first valid JSON object if multiple exist
                try:
                    # Find the first { and last } to extract a JSON object
                    first_brace = inner.find('{')
                    last_brace = inner.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        results.append(json.loads(inner[first_brace:last_brace+1]))
                except json.JSONDecodeError:
                    continue
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with enhanced chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with extensive experience in mathematical problem evaluation.

Your task is to evaluate a student's solution to a mathematical problem with precision and fairness.

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

## Structured Evaluation Process

Please follow this systematic approach:

### 1. Problem Understanding
- Restate the key requirements and constraints of the problem
- Identify the core mathematical concepts being tested

### 2. Solution Analysis
- Break down the official solution into key logical steps
- Identify the critical insights required for a complete solution

### 3. Student Answer Evaluation
- Map the student's answer against each key step of the official solution
- Identify what the student got right (correct methods, valid insights, proper notation)
- Identify errors, gaps, or misconceptions (calculation errors, missing cases, logical flaws)
- Note any creative alternative approaches that are mathematically valid

### 4. Partial Credit Assessment
- Award credit for each correct step or insight, even if the final answer is wrong
- Consider the IMO's partial credit philosophy: reward valid mathematical reasoning
- Deduct for: missing cases, calculation errors, logical gaps, incorrect claims

### 5. Final Grade Determination
- Synthesize your analysis into a fair grade based on the guidelines
- IMO problems are typically graded 0-7 points
- Be generous with partial credit for genuine mathematical progress

## Response Format

Respond ONLY in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering: (1) problem understanding, (2) comparison with official solution, (3) specific strengths and weaknesses in the student's answer, (4) justification for partial credit awarded",
    "response": "Your final grade as a number (0-7) or descriptive text like 'Partial credit: 3'",
    "confidence": "high|medium|low",
    "breakdown": {{
        "correct_steps": "List the key steps the student got right",
        "errors_identified": "List specific errors or gaps",
        "partial_credit_justification": "Explain why this grade was awarded"
    }}
}}
</json>

Important:
- The "response" field must contain ONLY the final grade (e.g., "5", "Partial credit: 2", "0", "7")
- Be thorough in your reasoning - it helps verify the grade is fair
- Confidence should reflect how clear-cut the grading decision is
- The breakdown helps validate your assessment"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        # Retry mechanism with exponential backoff for transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                break
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "Error: LLM call failed after retries", []
                import time
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        reasoning = ""
        confidence = "unknown"
        breakdown = None
        
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                if "confidence" in last_json:
                    confidence = last_json["confidence"]
                if "breakdown" in last_json:
                    breakdown = last_json["breakdown"]
                
                # Log the reasoning, confidence, and breakdown for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
                self.log_fn(f"Confidence: {confidence}")
                if breakdown:
                    self.log_fn(f"Breakdown: {str(breakdown)[:200]}...")
            else:
                # Fallback 1: try to find JSON with response field using more flexible pattern
                json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                
                # Fallback 2: look for grade patterns in plain text
                if prediction == "None":
                    # Look for common grade patterns like "Grade: 7", "Score: 6", "Final grade: 5"
                    grade_patterns = [
                        r'(?:final\s+)?(?:grade|score|mark)s?[\s:]+([0-7])(?:\s*points?)?',
                        r'(?:partial\s+credit\s*[:\s]+)?([0-7])\s*(?:points?)?',
                        r'["\']([0-7])["\']',
                    ]
                    for pattern in grade_patterns:
                        grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                        if grade_match:
                            prediction = grade_match.group(1)
                            self.log_fn(f"Extracted grade from text pattern: {prediction}")
                            break
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction format with enhanced checks
        prediction_str = str(prediction).strip()
        if prediction_str.lower() in ["none", "null", "", "error"]:
            self.log_fn(f"Warning: Invalid prediction format: {prediction_str}")
            # Try to extract any numeric value as last resort, prioritizing single digits 0-7
            numeric_match = re.search(r'\b([0-7])\b', last_msg)
            if numeric_match:
                prediction_str = numeric_match.group(1)
                self.log_fn(f"Fallback to single digit extraction: {prediction_str}")
            else:
                # Try any numeric value
                numeric_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', last_msg)
                if numeric_match:
                    prediction_str = numeric_match.group(1)
                    self.log_fn(f"Fallback to numeric extraction: {prediction_str}")

        # Final validation: ensure grade is within reasonable bounds for IMO (0-7)
        try:
            grade_val = float(prediction_str)
            if grade_val < 0 or grade_val > 7:
                self.log_fn(f"Warning: Grade {grade_val} outside IMO range [0,7]")
        except ValueError:
            pass  # Non-numeric grades like "Partial credit: 3" are valid

        return prediction_str, msg_history
