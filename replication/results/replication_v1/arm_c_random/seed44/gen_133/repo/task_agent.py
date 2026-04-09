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
    Also handles markdown code blocks and raw JSON as fallback.
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
            # Try to clean up common formatting issues
            try:
                # Remove trailing commas before closing braces/brackets
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
                # Try to extract JSON from the block
                try:
                    json_match = re.search(r'\{.*\}', block, re.DOTALL)
                    if json_match:
                        results.append(json.loads(json_match.group()))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON object in the text
    if not results:
        try:
            json_match = re.search(r'\{\s*"reasoning".*\}', text, re.DOTALL)
            if json_match:
                results.append(json.loads(json_match.group()))
        except json.JSONDecodeError:
            pass
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

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

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the critical steps, theorems, and reasoning required for a correct solution.

3. **Evaluate the Student's Answer**: 
   - Check if the student correctly identified the approach
   - Verify each step against the official solution
   - Note any errors, omissions, or creative valid alternatives
   - Consider partial credit according to the grading guidelines

4. **Determine the Score**: Based on your analysis, assign a numerical score that reflects the student's performance.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here...",
    "score_breakdown": {{
        "correctness": "analysis of mathematical correctness",
        "completeness": "analysis of solution completeness", 
        "clarity": "analysis of presentation clarity"
    }},
    "response": <numerical_score>
}}
</json>

IMPORTANT: 
- The "response" field must contain ONLY a numerical score (e.g., 7, 3.5, 0, etc.)
- Do NOT include quotes around the number in the "response" field
- Ensure the JSON is valid with no trailing commas
- The score should reflect the student's performance based on the grading guidelines"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        try:
            # Get the last assistant message
            assistant_msgs = [m for m in msg_history if m.get("role") == "assistant"]
            if assistant_msgs:
                last_text = assistant_msgs[-1].get("text", "")
                extracted = _extract_jsons(last_text)
                if extracted:
                    last_json = extracted[-1]
                    
                    # Extract response/score with type validation
                    if "response" in last_json:
                        resp_val = last_json["response"]
                        # Handle various numeric formats
                        if isinstance(resp_val, (int, float)):
                            prediction = resp_val
                        elif isinstance(resp_val, str):
                            # Try to parse numeric string
                            try:
                                prediction = float(resp_val)
                                # Convert to int if it's a whole number
                                if prediction == int(prediction):
                                    prediction = int(prediction)
                            except ValueError:
                                prediction = resp_val
                        else:
                            prediction = str(resp_val)
                    
                    # Extract reasoning for debugging
                    if "reasoning" in last_json:
                        reasoning = last_json["reasoning"]
                        self.log_fn(f"Agent reasoning: {reasoning[:200]}...")
                    
                    # Log score breakdown if available
                    if "score_breakdown" in last_json:
                        self.log_fn(f"Score breakdown: {last_json['score_breakdown']}")
                else:
                    self.log_fn("No JSON found in response, attempting text extraction")
                    # Fallback: try to extract a number from the text
                    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', last_text)
                    if numbers:
                        # Use the last number as it's likely the score
                        prediction = float(numbers[-1])
                        if prediction == int(prediction):
                            prediction = int(prediction)
            else:
                self.log_fn("No assistant messages found in history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
