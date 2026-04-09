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


def _extract_score_from_text(text: str) -> str | None:
    """Fallback extraction: try to find a numeric score in the text."""
    # Look for patterns like "score: 7", "score is 3", "final score: 0"
    patterns = [
        r'["\']?score["\']?\s*[:=]\s*["\']?(\d+)["\']?',
        r'["\']?response["\']?\s*[:=]\s*["\']?(\d+)["\']?',
        r'final\s+score\s*[:=]\s*(\d+)',
        r'assigned\s+score\s*[:=]\s*(\d+)',
        r'\bscore\s+of\s+(\d+)',
        r'\b(\d+)\s*points?\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        prompt = f"""You are an expert {domain} grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a competition problem with precision and consistency.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Evaluation Framework

Follow this systematic evaluation process:

### 1. Problem Decomposition
- Identify the core mathematical concepts and techniques required
- Break down the official solution into key logical steps
- Note the critical insights needed for a complete solution

### 2. Solution Mapping
- Map the student's answer against the official solution structure
- Identify which steps the student completed correctly
- Note any alternative valid approaches the student may have used

### 3. Error Analysis
- Categorize any errors: conceptual, computational, or logical gaps
- Determine if errors are minor (slips) or major (misunderstandings)
- Assess whether partial credit conditions from the guidelines are met

### 4. Scoring Decision
- Apply the grading guidelines strictly but fairly
- Consider partial credit for significant progress toward solution
- Ensure consistency with IMO grading standards

## Response Format (REQUIRED)

You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Detailed step-by-step analysis covering: (1) problem understanding, (2) comparison with official solution, (3) error analysis, (4) scoring rationale",
    "evaluation": {{
        "answer_correct": true/false,
        "proof_complete": true/false,
        "key_strengths": ["specific strength 1", "specific strength 2"],
        "key_weaknesses": ["specific weakness 1", "specific weakness 2"],
        "partial_credit_justification": "Explanation if partial credit is awarded"
    }},
    "score": <numerical_score>,
    "response": "<numerical_score>"
}}
</json>

IMPORTANT:
- The "score" field must be a number (e.g., 7, 3, 0, 1)
- The "response" field must contain the same numerical score as a string
- Both fields are required for proper evaluation
- IMO problems are typically scored 0-7 points"""

        return prompt

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        self.log_fn(f"TaskAgent: Processing problem with domain: {inputs.get('domain', 'Unknown')}")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        evaluation_details = {}

        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                data = extracted[-1]
                # Try response field first, then score field
                if "response" in data and data["response"] is not None:
                    prediction = str(data["response"]).strip()
                elif "score" in data and data["score"] is not None:
                    prediction = str(data["score"]).strip()
                
                # Validate that prediction is numeric
                try:
                    float(prediction)
                except (ValueError, TypeError):
                    # Try fallback extraction
                    fallback = _extract_score_from_text(msg_history[-1]["text"])
                    if fallback:
                        prediction = fallback
                        self.log_fn(f"TaskAgent: Used fallback extraction, score: {prediction}")
                
                # Extract additional metadata if available
                reasoning = data.get("reasoning", "")
                evaluation_details = data.get("evaluation", {})
                
                self.log_fn(f"TaskAgent: Extracted prediction: {prediction}")
                if reasoning:
                    self.log_fn(f"TaskAgent: Reasoning length: {len(reasoning)} chars")
            else:
                # No JSON found, try fallback extraction
                fallback = _extract_score_from_text(msg_history[-1]["text"])
                if fallback:
                    prediction = fallback
                    self.log_fn(f"TaskAgent: No JSON found, used fallback extraction, score: {prediction}")
        except json.JSONDecodeError as e:
            self.log_fn(f"TaskAgent: JSON decode error: {e}")
            # Try fallback extraction on error
            fallback = _extract_score_from_text(msg_history[-1]["text"])
            if fallback:
                prediction = fallback
                self.log_fn(f"TaskAgent: Used fallback extraction after JSON error, score: {prediction}")
        except Exception as e:
            self.log_fn(f"TaskAgent: Error extracting prediction: {e}")
            # Try fallback extraction on any error
            try:
                fallback = _extract_score_from_text(msg_history[-1]["text"])
                if fallback:
                    prediction = fallback
                    self.log_fn(f"TaskAgent: Used fallback extraction after error, score: {prediction}")
            except Exception:
                pass

        return str(prediction), msg_history
