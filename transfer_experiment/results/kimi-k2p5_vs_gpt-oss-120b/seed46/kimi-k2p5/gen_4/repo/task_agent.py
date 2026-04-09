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
    valid_labels = ["correct", "incorrect", "partial"]
    
    # Priority 1: Look for quoted JSON-style values first
    # Use regex to find "response": "value" patterns
    json_pattern = r'"response"\s*:\s*"([^"]+)"'
    matches = re.findall(json_pattern, text_lower)
    for match in matches:
        clean_match = match.strip().lower()
        clean_match = re.sub(r'[^a-z]', '', clean_match)
        if clean_match in valid_labels:
            return clean_match
    
    # Priority 2: Look for single-quoted JSON-style values
    json_pattern_single = r"'response'\s*:\s*'([^']+)'"
    matches = re.findall(json_pattern_single, text_lower)
    for match in matches:
        clean_match = match.strip().lower()
        clean_match = re.sub(r'[^a-z]', '', clean_match)
        if clean_match in valid_labels:
            return clean_match
    
    # Priority 3: Check for explicit label declarations with colons
    # Look for patterns like "The answer is: correct" or "Classification: partial"
    declaration_patterns = [
        rf'\b(is|are|be)\s*[:=]\s*["\']?({"correct|incorrect|partial"})["\']?\b',
        rf'\b(classification|grade|label|verdict|result|evaluation|assessment)\s*[:=]\s*["\']?({"correct|incorrect|partial"})["\']?\b',
        rf'\b(answer|response|prediction|output)\s*[:=]\s*["\']?({"correct|incorrect|partial"})["\']?\b',
        rf'\b(the\s+student\s+(answer|response)\s+(is|should\s+be))\s*["\']?({"correct|incorrect|partial"})["\']?\b',
    ]
    
    for pattern in declaration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Extract the label from the capturing groups
            for group in match.groups():
                if group in valid_labels:
                    return group
    
    # Priority 4: Look for labels in code blocks or backticks
    for label in valid_labels:
        # Match backtick-quoted labels
        pattern = rf'`{label}`'
        if re.search(pattern, text_lower):
            return label
        # Match bold/italic markdown labels
        pattern = rf'\*\*{label}\*\*|\*{label}\*'
        if re.search(pattern, text_lower):
            return label
    
    # Priority 5: Count occurrences of each label as whole words
    # But exclude common false positives (e.g., "correct" in "correctly")
    label_counts = {}
    for label in valid_labels:
        # Count whole word occurrences with word boundaries
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    # Return the label with highest count if any found
    if any(label_counts.values()):
        # Filter to only labels that appear
        valid_counts = {k: v for k, v in label_counts.items() if v > 0}
        if valid_counts:
            return max(valid_counts, key=valid_counts.get)
    
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
- "correct": The student's answer is fully correct, complete, and matches the expected solution. The logic is sound, all key steps are present, and the final answer is right.
- "incorrect": The student's answer is wrong, incomplete in a critical way, demonstrates fundamental misunderstanding, or contains major logical errors that invalidate the solution.
- "partial": The student's answer shows significant progress with correct insights, valid intermediate steps, or partial correctness, but is not fully complete, missing some details, or has minor errors that don't invalidate the main approach.

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

Carefully analyze the student's answer against the official solution and grading guidelines:

1. **Understanding**: Does the student grasp the core concepts and problem structure?
2. **Key Steps**: Are the critical logical steps correct and well-reasoned?
3. **Final Answer**: Is the final answer mathematically correct?
4. **Completeness**: Does the answer address all parts of the problem?
5. **Grading Guidelines**: How does the answer align with the specific grading criteria provided?

**Decision Framework**:
- Choose "correct" ONLY if the answer is essentially complete and correct, matching the official solution's key points.
- Choose "incorrect" if there are fundamental flaws, missing critical components, or the main argument is wrong.
- Choose "partial" if the student shows genuine understanding and progress but the solution has gaps, minor errors, or incomplete reasoning that could be fixed with small corrections.

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
        text_to_parse = (response or "").strip()
        valid_labels = ["correct", "incorrect", "partial"]
        
        try:
            # First try to extract from JSON blocks
            extracted = _extract_jsons(text_to_parse)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        raw_pred = str(item["response"]).strip().lower()
                        # Direct match
                        if raw_pred in valid_labels:
                            prediction = raw_pred
                            break
                        # Handle cases where the value might have extra punctuation
                        clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                        if clean_pred in valid_labels:
                            prediction = clean_pred
                            break
            
            # If JSON extraction failed, try text extraction
            if prediction == "incorrect":  # Still default, try text extraction
                text_pred = _extract_label_from_text(text_to_parse)
                if text_pred:
                    prediction = text_pred
            
            # If still no prediction, check msg_history as last resort
            if prediction == "incorrect" and msg_history:
                for msg in reversed(msg_history):
                    if isinstance(msg, dict):
                        last_content = msg.get("text") or msg.get("content") or str(msg)
                    else:
                        last_content = str(msg)
                    
                    last_content = last_content.strip()
                    if last_content and last_content != text_to_parse:
                        # Try JSON extraction from history
                        extracted = _extract_jsons(last_content)
                        if extracted:
                            for item in extracted:
                                if isinstance(item, dict) and "response" in item:
                                    raw_pred = str(item["response"]).strip().lower()
                                    if raw_pred in valid_labels:
                                        prediction = raw_pred
                                        break
                                    clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                                    if clean_pred in valid_labels:
                                        prediction = clean_pred
                                        break
                        
                        if prediction == "incorrect":
                            text_pred = _extract_label_from_text(last_content)
                            if text_pred:
                                prediction = text_pred
                                break
                        else:
                            break
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"  # Safe default
        
        # Final validation - ensure prediction is valid
        if prediction not in valid_labels:
            prediction = "incorrect"

        return str(prediction), msg_history
