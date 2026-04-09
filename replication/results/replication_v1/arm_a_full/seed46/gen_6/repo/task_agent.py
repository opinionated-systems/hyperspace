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


def _normalize_grade(grade: str) -> str:
    """Normalize grade to standard format.
    
    Converts various grade formats to a consistent representation.
    """
    if not grade or not isinstance(grade, str):
        return str(grade) if grade else "None"
    
    grade = grade.strip().lower()
    
    # Map common variations to standard grades
    grade_map = {
        "correct": "Correct",
        "right": "Correct",
        "true": "Correct",
        "yes": "Correct",
        "partial": "Partial",
        "partially correct": "Partial",
        "partial credit": "Partial",
        "incorrect": "Incorrect",
        "wrong": "Incorrect",
        "false": "Incorrect",
        "no": "Incorrect",
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
    }
    
    return grade_map.get(grade, grade.capitalize())


def _calculate_confidence(analysis: str, partial_credit_reasoning: str, understanding: str) -> str:
    """Calculate confidence level based on analysis completeness.
    
    Returns: "High", "Medium", or "Low" based on the depth of analysis.
    """
    if not analysis or not isinstance(analysis, str):
        return "Low"
    
    # Count key indicators of thorough analysis
    indicators = 0
    
    # Check for mathematical reasoning indicators
    math_indicators = ["therefore", "because", "since", "hence", "thus", "implies", 
                       "contradiction", "valid", "invalid", "error", "correct"]
    for indicator in math_indicators:
        if indicator in analysis.lower():
            indicators += 1
    
    # Check for step-by-step analysis
    if len(analysis) > 200:
        indicators += 1
    if len(analysis) > 500:
        indicators += 1
    
    # Check for partial credit reasoning quality
    if partial_credit_reasoning and len(partial_credit_reasoning) > 100:
        indicators += 1
    
    # Check for understanding quality
    if understanding and len(understanding) > 50:
        indicators += 1
    
    # Determine confidence level
    if indicators >= 6:
        return "High"
    elif indicators >= 3:
        return "Medium"
    else:
        return "Low"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer by comparing it against the official solution and grading guidelines.

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
Follow this structured evaluation process:

1. **Understanding Check**: Briefly summarize what the problem is asking and what the correct approach should be.

2. **Step-by-Step Analysis**: Go through the student's answer carefully:
   - Identify each key step or claim they make
   - Check if each step is mathematically valid
   - Note any errors, gaps, or incorrect assumptions
   - Compare their approach to the official solution

3. **Partial Credit Assessment**: Based on the grading guidelines, determine:
   - Which parts of the solution they completed correctly
   - What partial credit they deserve for incomplete or partially correct work
   - Whether they demonstrated understanding of key concepts

4. **Final Grade Decision**: Assign a grade that reflects:
   - Full correctness (Correct/7) if completely solved with valid reasoning
   - Partial credit (Partial/1-6) for incomplete or partially correct solutions
   - Incorrect (Incorrect/0) for fundamentally wrong or empty answers

Respond in JSON format with the following schema:
<json>
{{
    "understanding": "Brief summary of the problem and correct approach",
    "analysis": "Detailed step-by-step analysis of the student's answer, including what they got right and wrong",
    "partial_credit_reasoning": "Explanation of partial credit based on grading guidelines",
    "response": "Your final grade/score (use: 'Correct', 'Partial', 'Incorrect', or numeric 0-7)"
}}
</json>"""

        # Retry mechanism for LLM calls with exponential backoff
        max_retries = 3
        base_delay = 2
        msg_history = []
        
        for attempt in range(max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        confidence = "Low"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                
                # Priority order for grade extraction
                grade_fields = ["response", "grade", "score", "result", "final_grade"]
                for field in grade_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        break
                
                # Normalize the grade for consistency
                prediction = _normalize_grade(prediction)
                
                # Calculate confidence based on analysis quality
                analysis = last_extract.get("analysis", "")
                partial_credit_reasoning = last_extract.get("partial_credit_reasoning", "")
                understanding = last_extract.get("understanding", "")
                confidence = _calculate_confidence(analysis, partial_credit_reasoning, understanding)
                
                # Log detailed analysis for debugging
                if analysis:
                    self.log_fn(f"Analysis: {analysis[:200]}...")
                if partial_credit_reasoning:
                    self.log_fn(f"Partial Credit: {partial_credit_reasoning[:200]}...")
                if understanding:
                    self.log_fn(f"Understanding: {understanding[:200]}...")
                self.log_fn(f"Confidence: {confidence}")
            else:
                self.log_fn("Warning: No JSON blocks found in response")
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
        except KeyError as e:
            self.log_fn(f"Missing key in response: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
