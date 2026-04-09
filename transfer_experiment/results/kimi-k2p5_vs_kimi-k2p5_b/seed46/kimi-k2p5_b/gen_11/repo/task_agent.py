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
    """Extract JSON objects from <json>...</json> blocks or raw JSON."""
    results = []
    
    # Strategy 1: Look for explicit <json> tags
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
            # Try to find JSON object boundaries within the content
            try:
                # Find the first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Look for raw JSON objects with response/label/grade fields
    if not results:
        # Find JSON objects that might not be in tags
        json_pattern = re.findall(r'\{[^{}]*"(?:response|label|grade|evaluation)"[^{}]*\}', text, re.IGNORECASE)
        for match in json_pattern:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies."""
    text_lower = text.lower()
    
    # Look for explicit patterns like "grade: correct" or "evaluation: partial"
    # These patterns capture the label value directly
    patterns = [
        r'grade[d]?\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'evaluation\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'label\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'response\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'classification\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"label"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"',
        r"'response'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'label'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'grade'\s*:\s*'(correct|incorrect|partial|almost)'",
        r"'evaluation'\s*:\s*'(correct|incorrect|partial|almost)'",
        # Additional patterns for markdown code blocks
        r'```\s*\n?\s*\{[^}]*"response"\s*:\s*"(correct|incorrect|partial|almost)"',
        r'```json\s*\n?\s*\{[^}]*"response"\s*:\s*"(correct|incorrect|partial|almost)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Check for standalone labels with word boundaries in priority order
    # Priority: almost > incorrect > partial > correct (most specific first)
    # This avoids misclassifying "incorrect" as "correct"
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    
    if re.search(r'\bincorrect\b', text_lower):
        return "incorrect"
    
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
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
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into exactly one of these four categories: "correct", "almost", "partial", or "incorrect".

LABEL DEFINITIONS (STRICT INTERPRETATION):
- "correct": The solution is fully correct and complete with rigorous proof. All logical steps are valid, all cases are covered, and the conclusion is correct. The student demonstrates complete mastery of the problem.

- "almost": The solution is nearly correct (70-95% complete). The student has the right approach and main ideas correct, with only minor gaps or errors that don't affect the core conclusion. The proof structure is sound but may have small technical issues or missing details that don't invalidate the main result.

- "partial": The solution shows some correct insights but has significant gaps (30-70% complete). The student demonstrates understanding of key concepts and has made meaningful progress, but major parts of the proof are missing, or there are significant errors that affect the conclusion. The approach may be partially correct but incomplete.

- "incorrect": The solution is fundamentally wrong, has no meaningful progress, or uses completely invalid reasoning (0-30% complete). The approach is wrong, or the student has only written trivial/irrelevant observations without making real progress toward the solution.

IMPORTANT GRADING NOTES:
1. Be conservative with "correct" - only award if the proof is essentially complete and valid.
2. "Almost" is for solutions that would receive 7/7 or 6/7 marks - minor issues only.
3. "Partial" is for solutions with good ideas but significant gaps - typically 2-5/7 marks.
4. "Incorrect" is for solutions that show little to no valid progress - 0-1/7 marks.

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Analyze the student's answer carefully:
1. What approach did the student take? Is it aligned with the official solution?
2. Which parts are correct? Which parts have errors or gaps?
3. Does the student prove all necessary claims or leave gaps?
4. Are there logical errors, calculation mistakes, or missing cases?
5. Estimate the completeness percentage and assign the appropriate label.

You MUST respond with ONLY a JSON object in this EXACT format (no other text before or after):

<json>
{{
    "reasoning": "Your detailed analysis here explaining the strengths and weaknesses of the solution",
    "response": "correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed analysis here explaining the strengths and weaknesses of the solution",
    "response": "almost"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed analysis here explaining the strengths and weaknesses of the solution",
    "response": "partial"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed analysis here explaining the strengths and weaknesses of the solution",
    "response": "incorrect"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase, no quotes around the value in the field)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction
        prediction = "None"
        raw_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Strategy 1: Try to extract from <json> tags or JSON objects
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    if isinstance(json_obj, dict):
                        # Check all possible fields in priority order
                        for key in ["response", "label", "grade", "evaluation", "classification"]:
                            val = json_obj.get(key, "")
                            if isinstance(val, str):
                                val = val.strip().lower()
                                if val in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val
                                    break
                            elif isinstance(val, (int, float)):
                                # Handle numeric grades if present
                                val_str = str(val).lower()
                                if val_str in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val_str
                                    break
                        if prediction != "None":
                            break
            
            # Strategy 2: Direct text extraction as fallback
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
            
            # Strategy 3: Look for the label in the last line of the response
            if prediction == "None":
                lines = raw_text.strip().split('\n')
                for line in reversed(lines):
                    line_lower = line.lower().strip()
                    for label in ["almost", "incorrect", "partial", "correct"]:
                        if label in line_lower:
                            # Make sure it's not part of another word
                            if re.search(rf'\b{label}\b', line_lower):
                                prediction = label
                                break
                    if prediction != "None":
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
