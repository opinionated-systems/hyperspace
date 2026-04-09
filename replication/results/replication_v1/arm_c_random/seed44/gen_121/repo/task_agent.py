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
    Includes robust handling for nested JSON and common formatting issues.
    """
    results = []
    search_from = 0
    
    def _try_parse_json(content: str) -> dict | None:
        """Try to parse JSON with multiple fallback strategies."""
        content = content.strip()
        if not content:
            return None
            
        # Try direct parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object boundaries with brace matching
        # This handles nested braces correctly
        def find_json_bounds(s: str) -> tuple[int, int] | None:
            """Find the outermost JSON object boundaries using brace counting."""
            start = -1
            depth = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(s):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and start != -1:
                            return (start, i + 1)
            return None
        
        bounds = find_json_bounds(content)
        if bounds:
            try:
                return json.loads(content[bounds[0]:bounds[1]])
            except json.JSONDecodeError:
                pass
        
        # Fallback: simple find first { and last }
        json_start = content.find("{")
        json_end = content.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end + 1])
            except json.JSONDecodeError:
                pass
        
        return None
    
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
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Fallback: try markdown code blocks if no results yet
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without "json" specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
                search_from = end + 3
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
                search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed:
                results.append(parsed)
    
    # Last resort: try to find any JSON object in the text
    if not results:
        parsed = _try_parse_json(text)
        if parsed:
            results.append(parsed)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract structured fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Validate inputs
        if not problem:
            return "Error: No problem statement provided.", []
        if not student_answer:
            return "Error: No student answer provided.", []

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
Carefully evaluate the student's answer against the official solution and grading guidelines. Consider:
1. Mathematical correctness and rigor
2. Completeness of the solution
3. Logical reasoning and proof structure
4. Whether the student addressed all parts of the problem
5. Partial credit for incomplete but correct approaches

## Output Format
You MUST provide your evaluation as a JSON object wrapped in <json>...</json> tags. The JSON must be valid and properly formatted.

<json>
{{
    "response": "Your detailed grading feedback here. Include: (1) whether the solution is correct/partially correct/incorrect, (2) specific points where the student succeeded or failed, (3) a numerical score if applicable based on the grading guidelines."
}}
</json>

IMPORTANT RULES:
- Your response MUST start with <json> and end with </json>
- The content inside must be valid JSON (no trailing commas, proper quotes)
- Escape any double quotes inside the response value with backslash
- Do not include any text outside the <json> tags"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}", []

        # Extract prediction from JSON with multiple fallback strategies
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
                
                extracted = _extract_jsons(text)
                if extracted:
                    if "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                    else:
                        # Use first available key if "response" not found
                        prediction = str(extracted[-1])
                else:
                    # Fallback: use raw text if no JSON found
                    prediction = text[:2000] if len(text) > 2000 else text
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort fallback
            try:
                if msg_history and len(msg_history) > 0:
                    prediction = str(msg_history[-1])[:2000]
            except:
                prediction = "Error: Failed to extract prediction"

        return str(prediction), msg_history
