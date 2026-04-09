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


def _extract_label_direct(text: str) -> str | None:
    """Extract label directly from text using multiple strategies."""
    text_lower = text.lower()
    valid_labels = ["correct", "partial", "almost", "incorrect"]
    
    # Strategy 1: Look for JSON-like patterns with "response" field
    json_pattern = r'"response"\s*:\s*"([^"]+)"'
    matches = re.findall(json_pattern, text_lower)
    for match in matches:
        match_clean = match.lower().strip()
        if match_clean in valid_labels:
            return match_clean
    
    # Strategy 2: Look for labels in code blocks
    code_block_pattern = r'```(?:json)?\s*\{[^}]*"response"[^}]*\}```'
    code_matches = re.findall(code_block_pattern, text_lower, re.DOTALL)
    for match in code_matches:
        for label in valid_labels:
            if label in match:
                return label
    
    # Strategy 3: Look for standalone labels with word boundaries
    for label in valid_labels:
        # Match label as whole word, possibly surrounded by quotes or punctuation
        pattern = r'(?:^|[\s"\'\[\({\:,])' + label + r'(?:[\s"\'\]\)}\:,]|$)'
        if re.search(pattern, text_lower):
            return label
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's solution and classify it into exactly one of four categories.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Criteria:

**"correct"** - The solution is complete and correct:
- All key steps of the proof are present and logically sound
- The solution follows the intended approach or a valid alternative
- No significant gaps or errors in reasoning
- Would receive full marks in competition

**"partial"** - The solution has significant progress but is incomplete:
- Contains valid and useful observations toward the solution
- Has made substantial progress but missing key components
- Has minor gaps that could likely be filled
- Would receive partial credit (roughly 30-70%)

**"almost"** - The solution is nearly complete but has minor mistakes:
- The main structure of the proof is correct
- Contains non-negligible errors that affect the conclusion
- Significant progress made but fundamental flaw exists
- Would receive some credit but not full marks

**"incorrect"** - The solution is fundamentally wrong:
- Makes no significant progress toward the solution
- Contains major conceptual errors
- Approach is completely off-track
- Would receive minimal or no credit

## Few-Shot Examples:

Example 1 - Correct:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: We can factor n^3 - n = n(n-1)(n+1). These are three consecutive integers, so one is divisible by 2 and one by 3. Thus the product is divisible by 6.
Classification: <json>{{"response": "correct"}}</json>

Example 2 - Partial:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: We can factor n^3 - n = n(n-1)(n+1). These are three consecutive integers.
Classification: <json>{{"response": "partial"}}</json>

Example 3 - Incorrect:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Student: Let's test n=2: 8-2=6 which is divisible by 6. n=3: 27-3=24 divisible by 6. So it's true.
Classification: <json>{{"response": "incorrect"}}</json>

## Your Task:
Analyze the student's answer carefully. Compare it to the official solution and grading guidelines. Provide your classification in the exact JSON format below.

IMPORTANT: Your response MUST be ONLY the JSON object in <json> tags. Do not include any other text, explanation, or analysis outside the JSON.

<json>
{{
    "response": "correct" | "partial" | "almost" | "incorrect"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response using multiple strategies
        prediction = "incorrect"  # Default fallback
        valid_labels = ["correct", "partial", "almost", "incorrect"]
        
        try:
            # Get the assistant's response text
            assistant_text = ""
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    assistant_text = msg.get("text", "")
                    break
            
            # Strategy 1: Try to extract from <json> tags
            extracted = _extract_jsons(assistant_text)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        resp = item["response"].lower().strip()
                        if resp in valid_labels:
                            prediction = resp
                            break
            
            # Strategy 2: Try direct extraction if JSON extraction failed
            if prediction == "incorrect":
                direct_label = _extract_label_direct(assistant_text)
                if direct_label:
                    prediction = direct_label
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"  # Default fallback on error

        return str(prediction), msg_history
