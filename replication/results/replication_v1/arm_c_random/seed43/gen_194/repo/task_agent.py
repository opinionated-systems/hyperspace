"""
Task agent: solves a given task with enhanced reasoning for IMO grading.

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
    Also handles markdown code blocks (```json) as a fallback.
    Includes robust JSON repair for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse, with repair attempts
        parsed = _try_parse_json_with_repair(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try to find ```json code blocks directly
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
            inner = text[start:end].strip()
            # Remove the ```json or ``` prefix
            if inner.startswith("```json"):
                inner = inner[7:].strip()
            elif inner.startswith("```"):
                inner = inner[3:].strip()
            search_from = end + 3
            
            parsed = _try_parse_json_with_repair(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with braces
    if not results:
        try:
            # Find outermost braces
            brace_start = text.find('{')
            brace_end = text.rfind('}')
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                inner = text[brace_start:brace_end + 1]
                parsed = _try_parse_json_with_repair(inner)
                if parsed is not None:
                    results.append(parsed)
        except Exception:
            pass
    
    return results or None


def _try_parse_json_with_repair(text: str) -> dict | None:
    """Attempt to parse JSON with automatic repair for common issues.
    
    Repairs:
    - Trailing commas in objects/arrays
    - Unescaped newlines in strings
    - Missing quotes around keys
    - Single quotes instead of double quotes
    """
    text = text.strip()
    if not text:
        return None
    
    # First, try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try common repairs
    repairs = [
        # Remove trailing commas before } or ]
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Fix unescaped newlines in strings (simple heuristic)
        lambda t: re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', t),
        # Convert single quotes to double quotes (carefully)
        lambda t: re.sub(r"(?<!\\)'", '"', t),
    ]
    
    for repair in repairs:
        try:
            repaired = repair(text)
            return json.loads(repaired)
        except (json.JSONDecodeError, re.error):
            continue
    
    return None


def _extract_score_from_text(text: str) -> str | None:
    """Extract a numeric score from free-form text as a fallback.
    
    Looks for patterns like:
    - "Score: 7"
    - "Grade: 3/7"
    - "Final score: 0"
    - "The student receives 5 points"
    - Just a number at the end of the response
    """
    # Pattern: Score/Grade/Final/Points followed by number
    patterns = [
        r'[Ss]core[:\s]+(\d+)',
        r'[Gg]rade[:\s]+(\d+)',
        r'[Ff]inal\s+(?:score|grade)[:\s]+(\d+)',
        r'(?:receives?|gets?|awarded?)[:\s]+(\d+)\s*(?:points?)?',
        r'(?:points?|score)[:\s]+(\d+)',
        r'(?:^|\n)\s*(\d+)\s*(?:points?)?\s*$',  # Number at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # Look for fraction patterns like "3/7" and extract numerator
    fraction_match = re.search(r'(\d+)\s*/\s*\d+', text)
    if fraction_match:
        return fraction_match.group(1)
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

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

## Your Task

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Identify what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Understand the expected approach and the key steps required for a correct solution.

3. **Review Grading Guidelines**: Note the specific criteria for awarding points (partial or full).

4. **Evaluate Student's Answer**: 
   - Check if the student correctly identified the approach
   - Verify each step of their reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or alternative valid approaches

5. **Assign Score**: Based on the grading guidelines, assign an appropriate score. IMO problems are typically scored 0-7 points. Be precise and justify your decision.

## Response Format (IMPORTANT)

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must be properly formatted with double quotes around all keys and string values.

<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Explain your thinking process.",
    "evaluation": "Summary of what the student did correctly or incorrectly.",
    "response": "7"
}}
</json>

CRITICAL: 
- The "response" field MUST contain ONLY a numeric score (e.g., "7", "3", "0", "6", etc.)
- Do NOT include any text, explanation, or units in the response field
- The score should be a single number representing the points awarded
- IMO problems are typically scored from 0 to 7 points
- Ensure the JSON is valid: use double quotes, no trailing commas"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response from the last valid JSON block
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "evaluation" in last_json:
                    # Fallback to evaluation if response not present
                    prediction = last_json["evaluation"]
                elif "score" in last_json:
                    # Another common field name
                    prediction = last_json["score"]
            else:
                # Fallback: try to extract score from free-form text
                text_score = _extract_score_from_text(msg_history[-1]["text"])
                if text_score:
                    prediction = text_score
                    self.log_fn(f"Extracted score from text fallback: {text_score}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: try text extraction on the raw response
            try:
                text_score = _extract_score_from_text(msg_history[-1]["text"])
                if text_score:
                    prediction = text_score
            except Exception:
                pass

        return str(prediction), msg_history
