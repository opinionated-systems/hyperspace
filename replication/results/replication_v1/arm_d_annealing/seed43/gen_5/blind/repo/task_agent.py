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


def _extract_json_flexible(text: str) -> dict | None:
    """Flexible JSON extraction with multiple fallback strategies.
    
    Tries multiple approaches to extract valid JSON from LLM output.
    """
    # Strategy 1: Look for <json> tags
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: Look for JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON between curly braces
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and provide a grade.

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

1. **Analyze the problem**: Understand what is being asked and the key mathematical concepts involved.

2. **Review the official solution**: Note the correct approach and expected answer.

3. **Evaluate the student's answer**: 
   - Check if the approach is correct
   - Verify calculations and reasoning
   - Identify any errors or gaps
   - Compare against the grading guidelines

4. **Determine the grade**: Based on your analysis, assign an appropriate grade.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your evaluation process.",
    "grade_assigned": "The final grade you assign (e.g., '0', '1', '2', '7', etc.)",
    "confidence": "high|medium|low",
    "response": "The final grade (same as grade_assigned, for compatibility)"
}}
</json>

Be thorough in your reasoning. The grade should be a single value that matches the grading scale specified in the guidelines."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with flexible parsing
        prediction = "None"
        reasoning = ""
        confidence = "unknown"
        
        try:
            response_text = msg_history[-1]["text"]
            extracted = _extract_json_flexible(response_text)
            
            if extracted:
                # Try to get response from various possible keys
                if "response" in extracted:
                    prediction = extracted["response"]
                elif "grade_assigned" in extracted:
                    prediction = extracted["grade_assigned"]
                elif "grade" in extracted:
                    prediction = extracted["grade"]
                
                # Extract additional metadata if available
                reasoning = extracted.get("reasoning", "")
                confidence = extracted.get("confidence", "unknown")
                
                self.log_fn(f"Extracted grade: {prediction}, confidence: {confidence}")
            else:
                self.log_fn("No valid JSON found in response, attempting text extraction")
                # Fallback: try to extract a number that looks like a grade
                grade_pattern = r'(?:grade|score|assigned)["\']?\s*[:=]\s*["\']?(\d+)["\']?'
                match = re.search(grade_pattern, response_text, re.IGNORECASE)
                if match:
                    prediction = match.group(1)
                    self.log_fn(f"Extracted grade via regex: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
