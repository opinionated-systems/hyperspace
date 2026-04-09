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

__version__ = "1.5.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles malformed tags and extra whitespace.
    Falls back to extracting raw JSON objects if no tags are found.
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
            # Try to fix common JSON issues (trailing commas, single quotes)
            try:
                fixed = inner.replace("'", '"').rstrip(",").rstrip()
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try stripping markdown code fences that sometimes leak in
                try:
                    cleaned = re.sub(r'^```(?:json)?\s*', '', inner).rstrip('`').strip()
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    # If no tagged JSON found, try to extract raw JSON objects as fallback
    if not results:
        raw_jsons = _extract_raw_jsons(text)
        if raw_jsons:
            return raw_jsons
    return results or None


def _extract_raw_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from text without <json> tags.

    Uses brace counting to find balanced JSON objects.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape_next = False
            while i < len(text):
                ch = text[i]
                if escape_next:
                    escape_next = False
                elif ch == '\\' and in_string:
                    escape_next = True
                elif ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1].strip()
                            try:
                                results.append(json.loads(candidate))
                            except json.JSONDecodeError:
                                pass
                            break
                i += 1
        i += 1
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract label directly from text using pattern matching as fallback.

    Improved version: checks JSON-like key-value patterns first, then
    looks for explicit label mentions with word boundaries.
    """
    text_lower = text.lower()

    # Look for explicit label mentions in JSON-like patterns first
    # Check for patterns like "response": "partial" or "label": "correct"
    json_pattern = re.search(r'["\'](?:response|label|classification|answer|grade|result)["\']\s*:\s*["\'](\w+)["\']', text_lower)
    if json_pattern:
        label = json_pattern.group(1)
        if label in ("correct", "incorrect", "partial"):
            return label

    # Look for explicit label mentions
    # Check for "partial" first since it contains "correct"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    if re.search(r'\balmost\b', text_lower):
        return "partial"
    if re.search(r'\bincorrect\b|\bwrong\b', text_lower):
        return "incorrect"
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"

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
        # Extract fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and classify it as EXACTLY ONE of:
- "correct": The answer is fully correct and complete. All key steps are present, logically sound, and the conclusion is right.
- "partial": The answer has some valid progress but is incomplete or has significant gaps. This includes: correct setup but incomplete proof, correct key insight but missing details, partially correct reasoning with minor errors, or a correct answer with insufficient justification.
- "incorrect": The answer is wrong or fundamentally flawed. This includes: wrong approach, major logical errors, incorrect conclusion, or essentially no valid mathematical content.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Analyze the student's answer carefully using the following steps:
1. Identify the key insights and approach required by the problem
2. Check if the student's approach aligns with the official solution
3. Verify each step of the student's reasoning for logical soundness
4. Look for any gaps, errors, missing steps, or unjustified claims
5. Compare the completeness of the student's work against the grading guidelines
6. Determine if the student made meaningful progress even if the solution is incomplete

IMPORTANT: First provide your reasoning and analysis, then output your final classification as a JSON object inside <json> tags.

Your response should follow this format:
[Your detailed analysis and reasoning here]

<json>
{{"response": "correct"}}
</json>

OR

<json>
{{"response": "partial"}}
</json>

OR

<json>
{{"response": "incorrect"}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = None
        try:
            # Get the last assistant message
            last_message = msg_history[-1]["text"] if msg_history else ""

            # Try JSON extraction first
            extracted = _extract_jsons(last_message)
            if extracted:
                for obj in extracted:
                    if "response" in obj:
                        prediction = str(obj["response"]).lower().strip()
                        break

            # If JSON extraction failed, try text extraction
            if prediction is None:
                prediction = _extract_label_from_text(last_message)

            # Normalize the prediction
            if prediction:
                # Clean up the prediction
                prediction = prediction.strip('"\'')

                # Map to valid labels
                if prediction in ("partial", "almost"):
                    prediction = "partial"
                elif prediction in ["incorrect", "wrong", "false", "error"]:
                    prediction = "incorrect"
                elif prediction == "correct":
                    prediction = "correct"
                else:
                    # Try to infer from content
                    if "partial" in prediction or "almost" in prediction:
                        prediction = "partial"
                    elif "incorrect" in prediction or "wrong" in prediction:
                        prediction = "incorrect"
                    elif "correct" in prediction:
                        prediction = "correct"
                    else:
                        prediction = None

        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None

        # Default to "incorrect" if we couldn't extract a valid prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            prediction = "incorrect"

        return prediction, msg_history
