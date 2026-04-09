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
    
    # Strategy 2: Look for raw JSON objects with response/label/grade fields (multiline)
    if not results:
        # More permissive pattern for JSON objects spanning multiple lines
        json_pattern = re.findall(r'\{[^{}]*"(?:response|label|grade|evaluation)"[\s\S]*?\}', text, re.IGNORECASE | re.DOTALL)
        for match in json_pattern:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for JSON in markdown code blocks
    if not results:
        code_block_pattern = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        for block in code_block_pattern:
            try:
                # Try to find JSON objects in the code block
                json_obj = json.loads(block.strip())
                if isinstance(json_obj, dict):
                    results.append(json_obj)
            except json.JSONDecodeError:
                # Try to extract just the JSON object part
                json_start = block.find("{")
                json_end = block.rfind("}")
                if json_start != -1 and json_end != -1:
                    try:
                        results.append(json.loads(block[json_start:json_end+1]))
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


def _extract_label_from_final_answer(text: str) -> str | None:
    """Extract label from the final answer section of the response."""
    text_lower = text.lower()
    
    # Look for common final answer patterns
    final_patterns = [
        r'final\s*(?:answer|grade|evaluation|label)[\s:]*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:answer|grade|evaluation|label)\s*(?:is|[:=])\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:therefore|thus|hence|so)\s*[,:]?\s*(?:the\s+)?(?:answer|grade|evaluation|label)\s*(?:is|[:=])\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:i\s+)?(?:conclude|determine|assess|grade)\s*(?:that\s+)?(?:it\s+is\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
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
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Evaluate the student's answer and assign one of four grades.

GRADE DEFINITIONS:
- "correct": Fully correct and complete solution (90-100%). Valid proof with all necessary steps.
- "almost": Nearly correct with minor gaps only (70-89%). Right approach, small technical issues.
- "partial": Some correct insights but significant gaps (30-69%). Meaningful progress but incomplete.
- "incorrect": Fundamentally wrong or no valid progress (0-29%). Invalid reasoning.

GRADING NOTES:
- Be conservative with "correct" - only for essentially complete proofs.
- "Almost" = 6-7/7 marks, "Partial" = 2-5/7 marks, "Incorrect" = 0-1/7 marks.

Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Analyze the student's answer and respond with ONLY a JSON object in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis of strengths and weaknesses",
    "response": "correct"
}}
</json>

The "response" field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (all lowercase)."""

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
            
            # Strategy 3: Look for final answer patterns
            if prediction == "None":
                final_pred = _extract_label_from_final_answer(raw_text)
                if final_pred:
                    prediction = final_pred
            
            # Strategy 4: Look for the label in the last line of the response
            if prediction == "None":
                lines = raw_text.strip().split('\n')
                for line in reversed(lines):
                    line_lower = line.lower().strip()
                    # Skip empty lines and common non-label lines
                    if not line_lower or line_lower in ['</json>', '```', 'json']:
                        continue
                    for label in ["almost", "incorrect", "partial", "correct"]:
                        if label in line_lower:
                            # Make sure it's not part of another word
                            if re.search(rf'\b{label}\b', line_lower):
                                prediction = label
                                break
                    if prediction != "None":
                        break
            
            # Strategy 5: Look for any label anywhere in the text (last resort)
            if prediction == "None":
                text_lower = raw_text.lower()
                for label in ["almost", "incorrect", "partial", "correct"]:
                    if re.search(rf'\b{label}\b', text_lower):
                        prediction = label
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        self.log_fn(f"Final prediction: {prediction}")
        return str(prediction), msg_history
