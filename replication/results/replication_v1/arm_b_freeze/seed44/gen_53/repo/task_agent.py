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
        except json.JSONDecodeError:
            # Try to clean up common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                try:
                    cleaned = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON-like object in the text
    if not results:
        # Look for patterns like {"key": "value"}
        json_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate score based on the official solution and grading guidelines.

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

Follow this structured evaluation process:

### Step 1: Problem Understanding
- Identify the key mathematical concepts and techniques required
- Note the critical steps in the official solution
- Understand the scoring rubric from the grading guidelines
- Determine the maximum possible score for this problem

### Step 2: Student Answer Analysis
- Check if the student stated the final answer correctly
- Identify which solution steps the student completed
- Note any missing or incorrect steps
- Evaluate the logical flow and mathematical rigor
- Look for alternative valid approaches that differ from the official solution

### Step 3: Partial Credit Assessment
- Award points for each correct step completed
- Deduct points for logical gaps or errors
- Consider alternative valid approaches
- Be generous with partial credit when reasoning is sound
- Note: Even incomplete solutions may earn significant partial credit

### Step 4: Final Score Determination
- Sum the points earned across all steps
- Verify against the grading guidelines
- Ensure consistency with the official scoring rubric
- Double-check that your score reflects the student's actual work

## Important Grading Principles
1. **Correct final answer** with valid reasoning → Full marks
2. **Correct approach** with minor errors → Deduct 1-2 points
3. **Partial progress** → Award proportional points
4. **Alternative valid solutions** → Award full credit if mathematically sound
5. **No valid work shown** → Score of 0

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis following the steps above. Include: (1) key concepts identified, (2) steps completed by student, (3) partial credit breakdown with specific point allocations, (4) justification for final score",
    "response": "The final score (e.g., '0', '1', '2', '7', etc.)"
}}
</json>

Be thorough in your reasoning, generous with partial credit for correct reasoning, and precise in your final scoring. Always verify your score against the grading guidelines before finalizing."""

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
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                if "response" in last_extract:
                    prediction = last_extract["response"]
                elif "score" in last_extract:
                    prediction = last_extract["score"]
                elif "answer" in last_extract:
                    prediction = last_extract["answer"]
                elif "grade" in last_extract:
                    prediction = last_extract["grade"]
                elif "mark" in last_extract:
                    prediction = last_extract["mark"]
                elif "points" in last_extract:
                    prediction = last_extract["points"]
                else:
                    # If no recognized field, use the first string value found
                    for key, value in last_extract.items():
                        if isinstance(value, (str, int, float)):
                            # Check if it looks like a score (numeric or simple string)
                            str_val = str(value).strip()
                            if str_val.replace('.', '').replace('-', '').isdigit():
                                prediction = str_val
                                break
                    else:
                        prediction = list(last_extract.values())[0] if last_extract else "None"
            else:
                # Fallback: try to extract a score directly from the text
                # Look for patterns like "Score: 7" or "Final score: 2"
                score_patterns = [
                    r'[Ff]inal\s+[Ss]core[:\s]+(\d+)',
                    r'[Ss]core[:\s]+(\d+)',
                    r'[Gg]rade[:\s]+(\d+)',
                    r'[Mm]ark[:\s]+(\d+)',
                    r'[Pp]oints[:\s]+(\d+)',
                    r'[Rr]esponse[:\s]+["\']?(\d+)["\']?',
                    r'["\']?(\d+)["\']?\s*points?',
                ]
                text = msg_history[-1]["text"]
                for pattern in score_patterns:
                    match = re.search(pattern, text)
                    if match:
                        prediction = match.group(1)
                        self.log_fn(f"Extracted score via pattern: {prediction}")
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Clean up the prediction - ensure it's a valid string representation
        prediction = str(prediction).strip()
        if prediction.lower() in ["none", "null", "undefined", ""]:
            prediction = "None"

        return prediction, msg_history
