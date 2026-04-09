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

## Instructions
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade assessment in the JSON format below.

CRITICAL: Your response MUST be ONLY a valid JSON object wrapped in <json>...</json> tags. 
The "response" field MUST contain EXACTLY ONE of these four strings (no extra text, no explanations, no punctuation):
- "correct"
- "almost" 
- "partial"
- "incorrect"

Do NOT add any text before or after the JSON. Do NOT explain your grade in the response field.

Example of correct output format:
<json>
{{
    "reasoning": "The student correctly proved the theorem using induction. All steps are valid and the conclusion follows logically from the premises.",
    "response": "correct"
}}
</json>

Another example:
<json>
{{
    "reasoning": "The student had the right approach but made a calculation error in step 3 and missed one case in the proof.",
    "response": "partial"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            temperature=0.0,  # Use deterministic output for grading
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                raw_response = None
                
                # Try to get response from various possible field names
                for field in ["response", "grade", "answer", "result", "evaluation", "verdict"]:
                    if field in last_json:
                        raw_response = last_json[field]
                        break
                
                if raw_response:
                    # Clean up the response - remove quotes, extra whitespace, punctuation
                    if isinstance(raw_response, str):
                        cleaned = raw_response.strip().lower()
                        # Remove surrounding quotes if present
                        cleaned = cleaned.strip('"').strip("'")
                        # Remove trailing punctuation
                        cleaned = cleaned.rstrip('.!?,:;')
                        cleaned = cleaned.strip()
                        
                        # Check for exact match first
                        if cleaned in valid_grades:
                            prediction = cleaned
                        else:
                            # Check if any valid grade appears in the text
                            for grade in valid_grades:
                                if grade in cleaned:
                                    prediction = grade
                                    break
                            else:
                                # No valid grade found - check for common synonyms
                                if cleaned in ["full", "complete", "right", "true", "yes"]:
                                    prediction = "correct"
                                elif cleaned in ["mostly", "nearly", "close"]:
                                    prediction = "almost"
                                elif cleaned in ["some", "part", "half", "incomplete"]:
                                    prediction = "partial"
                                elif cleaned in ["wrong", "false", "no", "bad", "error"]:
                                    prediction = "incorrect"
                                else:
                                    # Default to the cleaned response
                                    prediction = cleaned[:100]  # Limit length
                    else:
                        # Non-string response
                        prediction = str(raw_response)[:100]
                else:
                    # No recognized field - try to infer from reasoning text
                    reasoning = last_json.get("reasoning", "")
                    if reasoning and isinstance(reasoning, str):
                        reasoning_lower = reasoning.lower()
                        # Look for grade indicators in reasoning
                        if any(x in reasoning_lower for x in ["fully correct", "complete solution", "correct answer"]):
                            prediction = "correct"
                        elif any(x in reasoning_lower for x in ["almost correct", "nearly complete", "minor gaps"]):
                            prediction = "almost"
                        elif any(x in reasoning_lower for x in ["partial credit", "some correct", "significant gaps"]):
                            prediction = "partial"
                        elif any(x in reasoning_lower for x in ["incorrect", "wrong answer", "does not address"]):
                            prediction = "incorrect"
                        else:
                            prediction = json.dumps(last_json)[:200]
                    else:
                        prediction = json.dumps(last_json)[:200]
            else:
                # No JSON found - try to extract grade from raw text
                text = msg_history[-1]["text"].lower()
                for grade in valid_grades:
                    if grade in text:
                        prediction = grade
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
