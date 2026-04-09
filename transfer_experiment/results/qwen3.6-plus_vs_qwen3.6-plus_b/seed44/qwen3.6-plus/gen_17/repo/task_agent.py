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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

__version__ = "1.8.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also falls back to scanning for bare JSON objects if no tags are found.
    Handles nested braces within JSON values by tracking brace depth.
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
            # Try to fix common JSON issues (trailing commas, unescaped quotes)
            try:
                fixed = inner.rstrip(",").strip()
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue

    # Fallback: if no tagged JSON found, try to extract bare JSON objects
    if not results:
        try:
            # Try parsing the entire text as JSON first
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
            elif isinstance(parsed, list):
                results.extend(item for item in parsed if isinstance(item, dict))
        except json.JSONDecodeError:
            # Try to find JSON-like objects using manual brace tracking
            results = _extract_json_objects_manual(text)

    return results or None


def _extract_json_objects_manual(text: str) -> list[dict]:
    """Manually extract JSON objects by tracking brace depth."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find the matching closing brace
            depth = 0
            start = i
            in_string = False
            escape_next = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    j += 1
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
            else:
                # No matching brace found
                pass
        i += 1
    return results


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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", inputs.get("response", ""))
        domain = inputs.get("domain", "")

        domain_context = f"\nDomain: {domain}" if domain else ""

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's answer to a math problem.{domain_context}

**Problem:**
{problem}

**Official Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}

**Student's Answer:**
{student_answer}

Carefully analyze the student's answer step by step:
1. First, understand the problem and what is being asked
2. Compare the student's reasoning against the official solution
3. Check if the student's approach is mathematically valid and logically sound
4. Identify any errors, gaps, or missing steps in the student's reasoning
5. Determine if the final answer/conclusion is correct
6. Assign a grade based on the grading guidelines

Grading categories (choose exactly one):
- "correct": The student's answer is fully correct with valid reasoning and correct conclusion. All key steps are present and mathematically sound.
- "almost": The student's answer is nearly correct — the approach and reasoning are sound, but there is a minor calculation error, a small omission, or a slight gap that does not undermine the overall correctness. Use this when the student has essentially solved the problem but has a small flaw (e.g., minor arithmetic mistake, forgot to check an edge case, slight notation issue, or one small step missing). The student's core logic and final answer direction are correct.
- "partial": The student's answer has some correct elements but is significantly incomplete, has notable errors in reasoning, or only solves part of the problem. The student has made genuine mathematical progress (e.g., proved useful lemmas, identified key patterns, set up correct framework) but the solution is substantially incomplete.
- "incorrect": The student's answer has fundamental errors, uses a wrong approach, or reaches an incorrect conclusion. The core reasoning is flawed.

Important distinctions:
- "almost" vs "partial": Use "almost" when the student's reasoning is fundamentally correct and they are very close to the full solution — only a minor issue separates them from "correct". Use "partial" when the student has made meaningful progress but the solution is substantially incomplete or has significant errors.
- "almost" vs "correct": Use "almost" when there is a genuine flaw or omission that prevents the solution from being fully correct, even though the approach is right.

Decision tree for grading:
1. Is everything correct and complete? → "correct"
2. Does the student have the right approach and reasoning but only a minor error/omission? → "almost"
3. Does the student have some correct ideas but the solution is substantially incomplete or has significant errors? → "partial"
4. Is the student's approach fundamentally wrong or does it reach an incorrect conclusion? → "incorrect"

Before giving your final grade, briefly consider:
- Does the student's answer contain all the key ideas from the official solution?
- Are there any logical gaps or unjustified claims?
- Is the final answer correct?
- If there are errors, are they minor (arithmetic, notation, small omission) or fundamental (wrong approach, flawed logic)?

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis of the student's answer",
    "response": "correct" or "incorrect" or "partial" or "almost",
    "confidence": "A float between 0.0 and 1.0 indicating your confidence in this grade"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        confidence = None
        try:
            if msg_history:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    # Extract confidence if available
                    if "confidence" in extracted[-1]:
                        try:
                            confidence = float(extracted[-1]["confidence"])
                            # Clamp confidence to valid range
                            confidence = max(0.0, min(1.0, confidence))
                        except (ValueError, TypeError):
                            confidence = None
            else:
                self.log_fn("Warning: msg_history is empty, cannot extract prediction")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def get_metadata(self) -> dict:
        """Return metadata about this task agent instance.

        Returns:
            dict with model name, version, and configuration
        """
        return {
            "model": self.model,
            "version": __version__,
            "type": "task_agent",
        }
