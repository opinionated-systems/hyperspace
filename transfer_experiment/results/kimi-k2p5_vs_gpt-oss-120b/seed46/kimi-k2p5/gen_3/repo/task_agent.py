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


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Looks for exact matches of the three valid labels, prioritizing
    quoted JSON-style values over plain text mentions.
    """
    text_lower = text.lower()
    
    # Priority 1: Look for quoted JSON-style values first
    # Use regex to find "response": "value" patterns
    json_pattern = r'"response"\s*:\s*"([^"]+)"'
    matches = re.findall(json_pattern, text_lower)
    for match in matches:
        clean_match = match.strip().lower()
        if clean_match in ["correct", "incorrect", "partial"]:
            return clean_match
    
    # Priority 2: Look for standalone quoted labels
    for label in ["correct", "incorrect", "partial"]:
        # Look for the label as a complete quoted value
        pattern = rf'"{label}"'
        if re.search(pattern, text_lower):
            # Make sure it's not part of a larger word by checking context
            # Count how many times each label appears in quotes
            pass
    
    # Priority 3: Check for explicit label declarations
    # Look for patterns like "The answer is: correct" or "Classification: partial"
    declaration_patterns = [
        rf'\b(is|are)\s*[:\s]+\s*"?({"correct|incorrect|partial"})"?\b',
        rf'\b(classification|grade|label|verdict|result)\s*[:\s]+\s*"?({"correct|incorrect|partial"})"?\b',
        rf'\b(answer|response)\s*[:\s]+\s*"?({"correct|incorrect|partial"})"?\b',
    ]
    
    for pattern in declaration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Extract the label from the last capturing group
            for group in match.groups():
                if group in ["correct", "incorrect", "partial"]:
                    return group
    
    # Priority 4: Count occurrences of each label as whole words
    label_counts = {}
    for label in ["correct", "incorrect", "partial"]:
        # Count whole word occurrences
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    # Return the label with highest count if any found
    if any(label_counts.values()):
        # Filter to only labels that appear
        valid_labels = {k: v for k, v in label_counts.items() if v > 0}
        if valid_labels:
            return max(valid_labels, key=valid_labels.get)
    
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
        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) style problems.

Your task is to evaluate a student's answer and classify it into EXACTLY ONE of these three categories:
- "correct": The student's answer is fully correct, complete, and matches the expected solution.
- "incorrect": The student's answer is wrong, incomplete in a critical way, or demonstrates fundamental misunderstanding.
- "partial": The student's answer shows significant progress, correct insights, or partial correctness but is not fully complete or has minor errors.

PROBLEM:
```
{problem}
```

OFFICIAL SOLUTION:
```
{solution}
```

GRADING GUIDELINES:
```
{grading_guidelines}
```

STUDENT'S ANSWER TO EVALUATE:
```
{student_answer}
```

Carefully analyze the student's answer against the official solution and grading guidelines. Consider:
1. Does the student understand the core concepts?
2. Are the key steps correct?
3. Is the final answer correct?
4. How does this compare to the grading guidelines?

IMPORTANT: You must respond with EXACTLY ONE of these three labels: "correct", "incorrect", or "partial".

Respond in the following JSON format:
<json>
{{
    "response": "correct" | "incorrect" | "partial"
}}
</json>

The value of "response" MUST be exactly one of: "correct", "incorrect", or "partial"."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "incorrect"  # Default fallback
        text_to_parse = response or ""
        
        try:
            # First try to extract from JSON blocks
            extracted = _extract_jsons(text_to_parse)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        raw_pred = str(item["response"]).strip().lower()
                        if raw_pred in ["correct", "incorrect", "partial"]:
                            prediction = raw_pred
                            break
                        # Handle cases where the value might have extra punctuation
                        clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                        if clean_pred in ["correct", "incorrect", "partial"]:
                            prediction = clean_pred
                            break
            
            # If JSON extraction failed, try text extraction
            if prediction == "incorrect":  # Still default, try text extraction
                text_pred = _extract_label_from_text(text_to_parse)
                if text_pred:
                    prediction = text_pred
            
            # If still no prediction, check msg_history as last resort
            if prediction == "incorrect" and msg_history:
                last_msg = msg_history[-1]
                if isinstance(last_msg, dict):
                    last_content = last_msg.get("text") or last_msg.get("content") or str(last_msg)
                else:
                    last_content = str(last_msg)
                
                if last_content != text_to_parse:
                    extracted = _extract_jsons(last_content)
                    if extracted:
                        for item in extracted:
                            if isinstance(item, dict) and "response" in item:
                                raw_pred = str(item["response"]).strip().lower()
                                if raw_pred in ["correct", "incorrect", "partial"]:
                                    prediction = raw_pred
                                    break
                    
                    if prediction == "incorrect":
                        text_pred = _extract_label_from_text(last_content)
                        if text_pred:
                            prediction = text_pred
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"  # Safe default

        return str(prediction), msg_history
