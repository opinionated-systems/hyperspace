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
            continue
    return results or None


def _normalize_grade(raw_response: str, valid_grades: set[str]) -> str:
    """Normalize and validate a grade response.
    
    Handles various edge cases like extra whitespace, punctuation,
    and common variations of grade labels.
    """
    if not raw_response:
        return "None"
    
    # Normalize the response
    normalized = raw_response.strip().lower()
    
    # Remove common punctuation and quotes
    normalized = normalized.strip('"\'.,;:!?')
    
    # Check for exact matches first
    if normalized in valid_grades:
        return normalized
    
    # Check for partial matches (e.g., "mostly correct" -> "almost")
    grade_mappings = {
        "correct": ["correct", "fully correct", "complete", "right", "true", "valid", "accurate"],
        "almost": ["almost", "nearly correct", "mostly correct", "minor gaps", "minor errors", "close"],
        "partial": ["partial", "partially correct", "some correct", "incomplete", "significant gaps"],
        "incorrect": ["incorrect", "wrong", "false", "invalid", "inaccurate", "does not address"],
    }
    
    for grade, keywords in grade_mappings.items():
        for keyword in keywords:
            if keyword in normalized:
                return grade
    
    # If no match found, return the normalized response for debugging
    return raw_response


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Evaluation Framework

Follow this structured approach to evaluate the student's answer:

### Step 1: Understanding Check
- What is the problem asking for?
- What are the key concepts or techniques needed?
- What is the expected final answer or conclusion?

### Step 2: Solution Analysis
- Break down the correct solution into key steps or components
- Identify the critical insights or methods required
- Note any alternative valid approaches

### Step 3: Student Answer Evaluation
- Compare each part of the student's answer to the correct solution
- Identify what the student did correctly (concepts, calculations, reasoning)
- Identify errors, gaps, or misconceptions (missing steps, wrong calculations, incomplete reasoning)
- Check if the final answer matches the expected result

### Step 4: Grade Determination
Use these criteria to assign the appropriate grade:

**"correct"**: The answer is fully correct and complete
- All required steps are present and correct
- The reasoning is sound and logical
- The final answer matches the solution
- No significant errors or omissions

**"almost"**: The answer is nearly correct with only minor gaps
- The main approach is correct
- Minor calculation errors or notation issues
- Small gaps in explanation that don't affect correctness
- The student demonstrates understanding of the key concepts

**"partial"**: The answer has significant gaps but some correct elements
- Some correct steps or partial progress
- Major gaps in reasoning or missing critical components
- Incorrect final answer but some valid intermediate work
- The student shows partial understanding but misses key insights

**"incorrect"**: The answer is wrong or does not address the problem
- Fundamentally wrong approach or method
- No meaningful progress toward the solution
- Answer is irrelevant or completely off-track
- Serious misconceptions or errors throughout

### Step 5: Final Assessment
Provide your detailed reasoning and final grade in the JSON format below.

## Response Format
Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the framework above...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

IMPORTANT: The "response" field must contain ONLY one of these exact grade labels: "correct", "almost", "partial", or "incorrect"."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Prefer "response" field, but fallback to other common fields
                last_json = extracted[-1]
                raw_response = None
                if "response" in last_json:
                    raw_response = last_json["response"]
                elif "grade" in last_json:
                    raw_response = last_json["grade"]
                elif "answer" in last_json:
                    raw_response = last_json["answer"]
                elif "result" in last_json:
                    raw_response = last_json["result"]
                
                if raw_response:
                    prediction = _normalize_grade(str(raw_response), valid_grades)
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # No JSON found, try to extract grade from raw text
                text = msg_history[-1]["text"]
                prediction = _normalize_grade(text, valid_grades)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract from raw text as fallback
            try:
                text = msg_history[-1]["text"]
                prediction = _normalize_grade(text, valid_grades)
            except:
                pass

        return str(prediction), msg_history
