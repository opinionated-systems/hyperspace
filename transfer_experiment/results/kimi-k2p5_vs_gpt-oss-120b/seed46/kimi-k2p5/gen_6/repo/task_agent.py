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

# Valid grading labels
VALID_LABELS = ["correct", "incorrect", "partial"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also tries to find JSON in code blocks and raw JSON objects.
    """
    results = []
    search_from = 0
    
    # Try to find <json>...</json> blocks
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
    
    # Also try to find JSON in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            json_str = match.group(1).strip()
            if json_str:
                results.append(json.loads(json_str))
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON objects with "response" field
    json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    for match in re.finditer(json_pattern, text):
        try:
            results.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with single quotes
    json_pattern_single = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
    for match in re.finditer(json_pattern_single, text):
        try:
            results.append(json.loads(match.group(0).replace("'", '"')))
        except json.JSONDecodeError:
            continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract the grading label from raw text using pattern matching.
    
    Looks for exact matches of the three valid labels with priority-based
    matching to find the most likely intended label.
    """
    text_lower = text.lower()
    
    # Priority 1: Look for quoted JSON-style values with "response" key
    json_pattern = r'"response"\s*[:=]\s*"([^"]+)"'
    for match in re.finditer(json_pattern, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return clean_match
    
    # Priority 2: Look for single-quoted JSON-style values
    json_pattern_single = r"'response'\s*[:=]\s*'([^']+)'"
    for match in re.finditer(json_pattern_single, text_lower):
        clean_match = re.sub(r'[^a-z]', '', match.group(1).lower())
        if clean_match in VALID_LABELS:
            return clean_match
    
    # Priority 3: Look for explicit label declarations with colons or equals
    label_alternatives = "correct|incorrect|partial"
    declaration_patterns = [
        rf'(?:is|are|be)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:classification|grade|label|verdict|result|evaluation|assessment)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:answer|response|prediction|output)\s*[:=]\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:the\s+student\s+(?:answer|response)\s+(?:is|should\s+be))\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:i\s+(?:would|will)\s+(?:classify|grade|label|mark))\s*["\']?({label_alternatives})["\']?\b',
        rf'(?:this\s+(?:is|should\s+be))\s*["\']?({label_alternatives})["\']?\b',
    ]
    
    for pattern in declaration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            for group in match.groups():
                if group in VALID_LABELS:
                    return group
    
    # Priority 4: Look for labels in code blocks or backticks
    for label in VALID_LABELS:
        # Match backtick-quoted labels
        if re.search(rf'`{label}`', text_lower):
            return label
        # Match bold/italic markdown labels
        if re.search(rf'\*\*{label}\*\*|\*{label}\*', text_lower):
            return label
    
    # Priority 5: Look for labels at the end of sentences or standalone
    for label in VALID_LABELS:
        # Match at end of sentence
        if re.search(rf'\b{label}\b[.!?]*\s*$', text_lower):
            return label
        # Match after "therefore", "thus", "so", "hence"
        if re.search(rf'(?:therefore|thus|so|hence)[,:]?\s+\b{label}\b', text_lower):
            return label
    
    # Priority 6: Count occurrences of each label as whole words
    label_counts = {}
    for label in VALID_LABELS:
        count = len(re.findall(rf'\b{label}\b', text_lower))
        label_counts[label] = count
    
    if any(label_counts.values()):
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

## FEW-SHOT EXAMPLES

### Example 1: Correct Answer
Problem: Find the sum of 2 + 3.
Official Solution: The sum is 5.
Student Answer: 2 + 3 = 5
Analysis: The student correctly computed the sum. The answer is complete and accurate.
Verdict: correct

### Example 2: Incorrect Answer
Problem: Prove that for all positive integers n, n² + n + 41 is prime.
Official Solution: Counterexample at n=40: 40² + 40 + 41 = 1681 = 41², which is not prime.
Student Answer: This formula always produces primes because 41 is prime and we're adding multiples.
Analysis: The student made a fundamental error - they didn't verify their claim and missed the counterexample. The reasoning is flawed.
Verdict: incorrect

### Example 3: Partial Answer
Problem: Solve x² - 5x + 6 = 0
Official Solution: Factoring gives (x-2)(x-3)=0, so x=2 or x=3.
Student Answer: x² - 5x + 6 = (x-2)(x-3), so the roots are 2 and 3. Wait, let me check: 2² - 5(2) + 6 = 4 - 10 + 6 = 0. And 3² - 5(3) + 6 = 9 - 15 + 6 = 0. So x = 2, 3.
Analysis: The student correctly factored and found the roots, but the presentation is a bit messy with the self-correction. The answer is essentially correct but could be cleaner.
Verdict: correct

### Example 4: Partial Answer
Problem: Prove the Pythagorean theorem.
Official Solution: Full geometric proof with diagram and algebraic steps.
Student Answer: For a right triangle with sides a, b and hypotenuse c, we have a² + b² = c². This can be shown by drawing squares on each side.
Analysis: The student states the theorem correctly and mentions the geometric approach, but doesn't complete the actual proof. They show understanding but the solution is incomplete.
Verdict: partial

## CURRENT PROBLEM

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

## CHAIN-OF-THOUGHT ANALYSIS

Think through this step-by-step:

1. **Understanding**: Does the student grasp the core concepts and problem structure? What key ideas do they demonstrate?

2. **Key Steps**: Are the critical logical steps correct and well-reasoned? What specific mathematical reasoning is present?

3. **Final Answer**: Is the final answer mathematically correct? Does it match the official solution?

4. **Completeness**: Does the answer address all parts of the problem? Are there gaps or missing components?

5. **Grading Guidelines**: How does the answer align with the specific grading criteria provided? What would an IMO grader say?

6. **Decision**: Based on the above analysis, which category best fits this answer?
   - "correct" ONLY if the answer is essentially complete and correct, matching the official solution's key points.
   - "incorrect" if there are fundamental flaws, missing critical components, or the main argument is wrong.
   - "partial" if the student shows genuine understanding and progress but the solution has gaps, minor errors, or incomplete reasoning.

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
        prediction = None
        text_to_parse = (response or "").strip()
        
        try:
            # First try to extract from JSON blocks
            extracted = _extract_jsons(text_to_parse)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict) and "response" in item:
                        raw_pred = str(item["response"]).strip().lower()
                        # Direct match
                        if raw_pred in VALID_LABELS:
                            prediction = raw_pred
                            break
                        # Handle cases where the value might have extra punctuation
                        clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                        if clean_pred in VALID_LABELS:
                            prediction = clean_pred
                            break
            
            # If JSON extraction failed, try text extraction
            if prediction is None:
                text_pred = _extract_label_from_text(text_to_parse)
                if text_pred:
                    prediction = text_pred
            
            # If still no prediction, check msg_history as last resort
            if prediction is None and msg_history:
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
                                    if raw_pred in VALID_LABELS:
                                        prediction = raw_pred
                                        break
                                    clean_pred = re.sub(r'[^a-z]', '', raw_pred)
                                    if clean_pred in VALID_LABELS:
                                        prediction = clean_pred
                                        break
                        
                        if prediction is None:
                            text_pred = _extract_label_from_text(last_content)
                            if text_pred:
                                prediction = text_pred
                                break
                        else:
                            break
            
            # Last resort: look for any valid label in the text
            if prediction is None:
                text_lower = text_to_parse.lower()
                for label in VALID_LABELS:
                    if label in text_lower:
                        prediction = label
                        break
                            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None
        
        # Final validation - ensure prediction is valid
        if prediction not in VALID_LABELS:
            # Default to "incorrect" as the safest fallback
            # This is better than random guessing when we can't determine the label
            prediction = "incorrect"

        return str(prediction), msg_history
