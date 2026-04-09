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
    Also handles markdown code blocks and raw JSON objects.
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
            # Try to clean up common issues
            try:
                # Remove trailing commas before closing braces
                cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        json_code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        for block in json_code_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                continue
    
    # If still no results, try to find raw JSON objects
    if not results:
        # Look for JSON objects starting with { and ending with }
        potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for potential in potential_jsons:
            try:
                results.append(json.loads(potential))
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
        # Extract key fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer based on the problem, official solution, and grading guidelines.

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

### Step 1: Understanding Check
- Restate the key mathematical concepts in the problem
- Identify the core claims/theorems that need to be proven
- Note the expected difficulty level

### Step 2: Student's Approach Analysis
- Identify the student's overall strategy (if any)
- Track their logical progression step-by-step
- Note any alternative valid approaches they may have taken

### Step 3: Error and Gap Identification
- List each error or gap in the student's reasoning
- Classify errors as: conceptual, computational, logical, or incomplete
- Note any unjustified claims or missing proof steps

### Step 4: Partial Credit Assessment
- Map each correct element to the grading guidelines
- Identify which sub-problems or claims were correctly addressed
- Calculate partial credit based on demonstrated understanding

### Step 5: Final Score Determination
- Synthesize the analysis into a numerical score
- IMO problems are typically scored 0-7 points
- Ensure the score reflects both correct work and penalizes errors appropriately

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the 5-step process above",
    "score": "The numerical score (0-7 or as specified in guidelines)",
    "response": "The final score as a number or string"
}}
</json>

Important: 
- The "response" field should contain only the final score that will be used for evaluation.
- Be conservative: only award points for clearly demonstrated understanding.
- If the student makes an arithmetic error but the approach is correct, consider partial credit.
- If the student states a correct answer without proof, they typically receive 0-1 points."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Prefer "response" field, fallback to "score" if available
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                else:
                    # If no recognized field, look for numeric values
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)):
                            prediction = str(value)
                            break
                        elif isinstance(value, str):
                            # Try to extract a number from string values
                            num_match = re.search(r'\d+', value)
                            if num_match:
                                prediction = num_match.group(0)
                                break
                    else:
                        prediction = list(last_json.values())[0] if last_json else "None"
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn("No JSON found in response, attempting fallback extraction")
                # Fallback: try to extract any number that looks like a score
                text = msg_history[-1]["text"]
                # Look for patterns like "score": 5 or "response": "7"
                score_match = re.search(r'["\']?(?:score|response)["\']?\s*[:=]\s*["\']?(\d+)["\']?', text, re.IGNORECASE)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Fallback extraction found: {prediction}")
                else:
                    # Try to find any standalone number that could be a score (0-7)
                    score_match = re.search(r'\b([0-7])\b', text)
                    if score_match:
                        prediction = score_match.group(1)
                        self.log_fn(f"Number extraction found: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Clean up prediction - ensure it's a valid string representation
        prediction = str(prediction).strip()
        
        return prediction, msg_history
