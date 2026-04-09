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


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues.
    
    Handles trailing commas, unescaped quotes, and other common
    formatting problems that LLMs often introduce.
    """
    cleaned = json_str.strip()
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix unescaped newlines in string values (common LLM issue)
    # Match content between quotes and escape newlines
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape unescaped newlines and tabs
        content = content.replace('\n', '\\n').replace('\t', '\\t')
        content = content.replace('\r', '\\r')
        return f'"{content}"'
    
    # Only process if it looks like a JSON object
    if cleaned.startswith('{'):
        cleaned = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_newlines_in_strings, cleaned)
    
    return cleaned


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and raw JSON objects.
    
    Enhanced with better error logging and multiple fallback strategies.
    """
    results = []
    search_from = 0
    extraction_errors = []
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("Found unclosed <json> tag, skipping")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try parsing with progressive fallback strategies
        for attempt, parser in enumerate([
            lambda x: json.loads(x),  # Raw parse
            lambda x: json.loads(_clean_json_string(x)),  # Cleaned parse
        ]):
            try:
                parsed = parser(inner)
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON on attempt {attempt + 1}")
                break
            except json.JSONDecodeError as e:
                if attempt == 0:
                    extraction_errors.append(f"Raw parse failed: {e}")
                elif attempt == 1:
                    extraction_errors.append(f"Cleaned parse failed: {e}")
                    logger.debug(f"Failed to parse JSON block: {inner[:100]}...")
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            try:
                results.append(json.loads(content))
            except json.JSONDecodeError:
                try:
                    results.append(json.loads(_clean_json_string(content)))
                except json.JSONDecodeError as e:
                    extraction_errors.append(f"Markdown block parse failed: {e}")
                    continue
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - use non-greedy matching
        json_pattern = r'\{[^{}]*"response"[^{}]*\}'
        for match in re.finditer(json_pattern, text):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                try:
                    results.append(json.loads(_clean_json_string(match.group())))
                except json.JSONDecodeError as e:
                    extraction_errors.append(f"Fallback pattern parse failed: {e}")
                    continue
    
    # Log summary if we had errors but still got results
    if extraction_errors and results:
        logger.debug(f"Had {len(extraction_errors)} extraction errors but recovered {len(results)} valid JSON(s)")
    elif extraction_errors and not results:
        logger.warning(f"All JSON extraction attempts failed: {extraction_errors[:3]}")
    
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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader.

Your task is to evaluate a student's solution to a mathematical problem.

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

1. First, analyze the student's answer step by step. Compare it against the official solution.
2. Identify any errors, missing steps, or creative alternative approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

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

        # Extract prediction from JSON with better error handling
        prediction = "None"
        reasoning = ""
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
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
