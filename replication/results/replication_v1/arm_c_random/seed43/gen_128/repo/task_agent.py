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
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)",
    "confidence": "high|medium|low - indicate how confident you are in this grade"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis.
Include a confidence assessment to help identify uncertain gradings that may need human review."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON with better error handling
        prediction = "None"
        reasoning = ""
        confidence = "unknown"
        
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
                
                # Log the reasoning and confidence for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
                self.log_fn(f"Confidence: {confidence}")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                
                # Second fallback: look for grade patterns in plain text
                if prediction == "None":
                    # Look for common grade patterns like "Grade: 7" or "Score: 6"
                    grade_match = re.search(r'(?:grade|score|mark)s?[:\s]+([0-9]+(?:\.[0-9]+)?)', last_msg, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
                        self.log_fn(f"Extracted grade from text: {prediction}")
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction format
        prediction_str = str(prediction).strip()
        if prediction_str.lower() in ["none", "null", "", "error"]:
            self.log_fn(f"Warning: Invalid prediction format: {prediction_str}")
            # Try to extract any numeric value as last resort
            numeric_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', last_msg)
            if numeric_match:
                prediction_str = numeric_match.group(1)
                self.log_fn(f"Fallback to numeric extraction: {prediction_str}")

        # Additional validation: ensure prediction is a valid IMO grade (0-7)
        try:
            grade_val = float(prediction_str)
            if not (0 <= grade_val <= 7):
                self.log_fn(f"Warning: Grade {grade_val} outside IMO range [0,7]")
        except ValueError:
            # Non-numeric predictions are allowed (e.g., "Partial credit", "Incorrect")
            pass

        return prediction_str, msg_history
