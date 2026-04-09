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


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Enhanced with partial credit scoring and detailed error analysis for more
    accurate and consistent grading of mathematical solutions.
    """

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking and identify key requirements
2. Review the official solution approach and identify critical steps
3. Compare the student's answer to the official solution:
   - Check for correct final answer
   - Verify logical reasoning and proof structure
   - Identify any missing steps or errors
   - Note any creative alternative approaches
4. Evaluate against grading guidelines for partial credit
5. Determine the appropriate grade with justification

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Include: (a) key problem requirements, (b) comparison of student approach to official solution, (c) specific errors or gaps found, (d) strengths of the solution, (e) justification for the grade assigned",
    "response": "The final grade/prediction. Use: 'Correct' (fully correct), 'Incorrect' (fundamentally wrong or empty), 'Partial-N' where N is 1-9 (partial credit level), or numeric score if specified in guidelines",
    "confidence": "High/Medium/Low - your confidence in this grading decision",
    "key_issues": ["List any specific mathematical errors, logical gaps, or missing components"],
    "positive_aspects": ["List any correct steps, good insights, or creative approaches"]
}}
</json>

Grading rubric:
- Correct: Complete, rigorous solution matching official solution quality
- Partial-9: Minor flaw (e.g., small calculation error, slightly incomplete justification)
- Partial-7: Significant progress but missing some key elements
- Partial-5: Some correct ideas but major gaps in reasoning
- Partial-3: Minimal progress, mostly incorrect but shows some understanding
- Partial-1: Almost no correct work, but not completely blank
- Incorrect: Completely wrong, blank, or nonsensical answer"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                
                # Log additional grading metadata if available
                confidence = last_json.get("confidence", "N/A")
                key_issues = last_json.get("key_issues", [])
                positive_aspects = last_json.get("positive_aspects", [])
                
                if confidence != "N/A" or key_issues or positive_aspects:
                    self.log_fn(f"Grading metadata - Confidence: {confidence}, "
                              f"Issues: {len(key_issues)}, Positives: {len(positive_aspects)}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
