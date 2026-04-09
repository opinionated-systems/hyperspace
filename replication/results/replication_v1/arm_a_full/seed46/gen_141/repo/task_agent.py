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
    Also handles markdown code blocks (```json...```) as fallback.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to extract JSON from within the content if it's wrapped in other text
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
            else:
                end = text.find("```", start + 7)
            if end == -1:
                break
            
            if text[start:start+7] == "```json":
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    return results or None


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

## Grading Categories
You must classify the student's answer into exactly one of these four categories:

1. **correct** - The answer is completely correct with valid reasoning that matches the official solution.
   - All steps are logically sound
   - The final answer matches the official solution
   - No significant errors or gaps in reasoning

2. **almost** - The answer is nearly correct but has a minor flaw or missing small detail.
   - The core approach is correct
   - The final answer is correct or very close
   - Has a minor error, oversight, or missing justification that doesn't invalidate the main result
   - Example: Correct answer but missing a small case check, or correct approach with a minor calculation error

3. **partial** - The answer shows some correct progress but has significant gaps or errors.
   - Some correct steps or partial progress toward the solution
   - Significant errors, missing major steps, or incomplete reasoning
   - The final answer is incorrect or missing
   - Example: Started correctly but made a critical error, or solved a simpler case but not the full problem

4. **incorrect** - The answer is fundamentally wrong or shows no meaningful progress.
   - The approach is completely wrong or irrelevant
   - No valid mathematical reasoning
   - The answer is nonsense, blank, or completely unrelated to the problem
   - Example: Wrong formula applied, random numbers, or no attempt at a solution

## Instructions
1. First, analyze the student's answer step by step. Identify what they got right and what they got wrong.
2. Compare their reasoning against the official solution.
3. Check if they followed the grading guidelines.
4. Determine which of the four categories (correct, almost, partial, incorrect) best describes the answer.
5. Provide your detailed reasoning in the "analysis" field.
6. Provide exactly one of the four category labels in the "response" field: "correct", "almost", "partial", or "incorrect".

Respond in JSON format with the following schema:
<json>
{{
    "analysis": "Your detailed step-by-step analysis of the student's answer...",
    "response": "One of: correct, almost, partial, incorrect"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries multiple extraction strategies and field names for robustness.
        Normalizes output to one of the four valid categories.
        """
        if not msg_history:
            return "incorrect"
        
        # Get the last assistant message
        last_msg = msg_history[-1]
        text = last_msg.get("text", "")
        
        if not text:
            return "incorrect"
        
        # Try to extract JSON blocks
        extracted = _extract_jsons(text)
        
        raw_value = None
        
        if not extracted:
            # Fallback: try to find any JSON-like structure in the text
            self.log_fn("No JSON blocks found, trying fallback extraction")
            try:
                # Look for patterns like "response": "..." or "grade": "..."
                response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', text)
                if response_match:
                    raw_value = response_match.group(1)
                else:
                    grade_match = re.search(r'["\']grade["\']\s*:\s*["\']([^"\']+)["\']', text)
                    if grade_match:
                        raw_value = grade_match.group(1)
                    else:
                        score_match = re.search(r'["\']score["\']\s*:\s*["\']?([^"\'\s,}]+)', text)
                        if score_match:
                            raw_value = score_match.group(1)
            except Exception as e:
                self.log_fn(f"Fallback extraction failed: {e}")
        else:
            # Try to get response from extracted JSON
            last_extract = extracted[-1]
            
            # Priority order for field names
            field_priority = ["response", "grade", "score", "result", "evaluation", "verdict", "label"]
            
            for field in field_priority:
                if field in last_extract:
                    raw_value = last_extract[field]
                    # Log the analysis if available for debugging
                    if "analysis" in last_extract:
                        self.log_fn(f"Analysis: {str(last_extract['analysis'])[:200]}...")
                    break
            
            # If no known field found, try the first string value
            if raw_value is None:
                for key, value in last_extract.items():
                    if isinstance(value, str) and value:
                        raw_value = value
                        break
        
        # Normalize the value to one of the four valid categories
        return self._normalize_category(raw_value)
    
    def _normalize_category(self, value: str | None) -> str:
        """Normalize a value to one of the four valid grading categories.
        
        Args:
            value: The raw value extracted from the response
            
        Returns:
            One of: "correct", "almost", "partial", "incorrect"
        """
        if value is None:
            return "incorrect"
        
        value = str(value).strip().lower()
        
        # Direct matches
        if value in ("correct", "almost", "partial", "incorrect"):
            return value
        
        # Handle variations and synonyms
        if value in ("right", "true", "yes", "valid", "perfect", "complete", "7", "full"):
            return "correct"
        
        if value in ("almost correct", "nearly correct", "mostly correct", "minor error", 
                     "small error", "tiny mistake", "6"):
            return "almost"
        
        if value in ("partially correct", "some correct", "incomplete", "half", "partial credit",
                     "some progress", "3", "4", "5"):
            return "partial"
        
        if value in ("wrong", "false", "no", "invalid", "error", "none", "0", "1", "2",
                     "bad", "fail", "failed", "missing", "empty", "blank"):
            return "incorrect"
        
        # Check for substrings
        if "almost" in value or "nearly" in value:
            return "almost"
        
        if "partial" in value or "incomplete" in value or "some" in value:
            return "partial"
        
        if "correct" in value and "incorrect" not in value:
            return "correct"
        
        if "wrong" in value or "error" in value or "invalid" in value:
            return "incorrect"
        
        # Default to incorrect if we can't determine
        self.log_fn(f"Could not normalize value '{value}', defaulting to 'incorrect'")
        return "incorrect"
