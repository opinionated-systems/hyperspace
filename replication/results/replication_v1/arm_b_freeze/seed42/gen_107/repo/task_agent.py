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
    Also attempts to extract JSON from markdown code blocks as fallback.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find any JSON object in the text
    if not results:
        # Look for content between first { and last }
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json = text[start_idx:end_idx + 1]
                results.append(json.loads(potential_json))
        except json.JSONDecodeError:
            pass
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Build a more structured prompt for better results
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Domain: {domain}

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Please evaluate the student's answer and provide your assessment in the following JSON format:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Be thorough and fair in your evaluation."""

        self.log_fn(f"Processing task with model: {self.model}")
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call completed. Usage: {info.get('usage', {})}")
        except Exception as e:
            self.log_fn(f"Error during LLM call: {e}")
            return f"Error: {e}", []

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    if "response" in last_extracted:
                        prediction = last_extracted["response"]
                        self.log_fn(f"Successfully extracted prediction")
                    else:
                        self.log_fn(f"No 'response' key in extracted JSON: {last_extracted.keys()}")
                else:
                    self.log_fn(f"No JSON found in response. Raw text preview: {text_content[:200]}...")
            else:
                self.log_fn("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
