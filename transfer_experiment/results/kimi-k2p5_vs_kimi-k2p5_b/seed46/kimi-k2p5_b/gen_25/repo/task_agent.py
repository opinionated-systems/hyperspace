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

# Valid grading labels for IMO evaluation - 3 class system
VALID_LABELS = ["correct", "partial", "incorrect"]


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks or raw JSON.
    
    Uses multiple strategies to find JSON content with improved robustness.
    """
    results = []
    
    # Strategy 1: Look for <json>...</json> tags (most reliable)
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Look for JSON objects in code blocks
    json_block_pattern = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    for block in json_block_pattern:
        try:
            results.append(json.loads(block.strip()))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', block.strip())
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for raw JSON objects with grade/response/label fields
    # Use a more robust pattern that handles nested braces
    raw_json_pattern = re.findall(r'\{[^{}]*"(?:grade|response|label|evaluation)"[^{}]*\}', text, re.DOTALL)
    for match in raw_json_pattern:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = match.replace("'", '"')
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Look for JSON-like structures with relaxed parsing
    # Find content between curly braces that might be JSON
    brace_pattern = re.findall(r'\{([^{}]*)\}', text, re.DOTALL)
    for content in brace_pattern:
        # Check if it contains grade-related keys
        if any(key in content.lower() for key in ['grade', 'response', 'label', 'evaluation']):
            try:
                json_str = '{' + content + '}'
                results.append(json.loads(json_str))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_label_from_text(text: str) -> str | None:
    """Extract grading label from raw text using multiple strategies with priority ordering."""
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit grade assignments with high confidence patterns
    # Pattern: "grade is X", "grade: X", "grade = X", "assigned grade: X"
    grade_patterns = [
        r'grade\s+is\s+["\']?(\w+)["\']?',
        r'grade\s*[:=]\s*["\']?(\w+)["\']?',
        r'assigned\s+grade\s*[:=]\s*["\']?(\w+)["\']?',
        r'final\s+grade\s*[:=]\s*["\']?(\w+)["\']?',
        r'evaluation\s*[:=]\s*["\']?(\w+)["\']?',
        r'response\s*[:=]\s*["\']?(\w+)["\']?',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1).lower().strip()
            if val in VALID_LABELS:
                return val
    
    # Strategy 2: Look for explicit label mentions with word boundaries
    # Check for "incorrect" first (most specific - contains "correct")
    if re.search(r'\bin\s*correct\b|\bincorrect\b', text_lower):
        return "incorrect"
    
    # Check for "partial"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    
    # Check for "correct" last (least specific)
    if re.search(r'\bcorrect\b', text_lower):
        return "correct"
    
    return None


def _normalize_grade(value: str) -> str | None:
    """Normalize a grade value to one of the valid labels."""
    if not isinstance(value, str):
        value = str(value)
    value = value.lower().strip()
    
    # Direct match
    if value in VALID_LABELS:
        return value
    
    # Check for partial matches with priority
    if "incorrect" in value:
        return "incorrect"
    if "partial" in value:
        return "partial"
    if "correct" in value:
        return "correct"
    
    return None


def _validate_student_answer(student_answer: str) -> tuple[bool, str]:
    """Validate if the student answer contains meaningful content."""
    if not student_answer or not student_answer.strip():
        return False, "empty"
    
    # Check if it's just whitespace or very short
    stripped = student_answer.strip()
    if len(stripped) < 3:
        return False, "too_short"
    
    # Check for common "no answer" patterns
    no_answer_patterns = [
        r'^\s*no\s+answer\s*$',
        r'^\s*none\s*$',
        r'^\s*n/a\s*$',
        r'^\s*not\s+applicable\s*$',
        r'^\s*skip\s*$',
        r'^\s*skipped\s*$',
        r'^\s*blank\s*$',
        r'^\s*empty\s*$',
    ]
    
    for pattern in no_answer_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "no_answer"
    
    return True, "valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced accuracy."""

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
        
        # Validate student answer first
        is_valid, validation_reason = _validate_student_answer(student_answer)
        if not is_valid:
            # Return incorrect for empty/invalid answers
            return "incorrect", [{"role": "system", "text": f"Invalid answer detected: {validation_reason}"}]
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate student answers and assign exactly one of three grades.

## GRADE DEFINITIONS:

**"correct"**: The answer is fully correct and complete. It contains a valid proof or solution with all necessary steps. Minor typos or notation differences are acceptable, but there must be no logical gaps or significant errors. The student demonstrates full understanding of the problem.

**"partial"**: The answer has some correct elements and shows meaningful progress toward the solution, but is incomplete or has significant gaps. The student understood part of the problem and made non-trivial progress, but did not reach a complete solution. Partial credit applies when there's genuine mathematical insight but incomplete execution.

**"incorrect"**: The answer is wrong or fundamentally flawed. No valid mathematical progress toward the solution. The approach is completely wrong, trivial, or the answer is blank/empty. No meaningful understanding demonstrated.

## GRADING GUIDELINES CONTEXT:
{grading_guidelines}

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

## YOUR TASK:

Analyze the student's answer carefully following these steps:

1. **Understand the Problem**: Verify you understand what the problem is asking and what constitutes a correct solution.

2. **Review the Official Solution**: Note the key steps, techniques, and final answer required.

3. **Analyze the Student's Answer**: 
   - Check if the student addressed the actual problem asked
   - Identify any correct mathematical insights or techniques
   - Note any errors, gaps, or logical flaws
   - Assess the completeness of the solution

4. **Compare to Guidelines**: Use the grading guidelines to determine partial vs incorrect distinctions.

5. **Assign Grade**: Choose exactly one of: "correct", "partial", or "incorrect"

IMPORTANT: You must respond with ONLY a JSON object in <json> tags. Do NOT include any other text before or after the JSON.

<json>
{{
    "reasoning": "Detailed explanation of your evaluation including specific strengths and weaknesses found",
    "grade": "correct" | "partial" | "incorrect"
}}
</json>

The "grade" field MUST be exactly one of: "correct", "partial", or "incorrect" (all lowercase). Be strict in your evaluation - only award "correct" for complete, rigorous solutions."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using multiple strategies with priority
        prediction = "None"
        try:
            raw_text = msg_history[-1]["text"] if msg_history else ""
            
            # Strategy 1: Try to extract from JSON tags (highest priority)
            extracted = _extract_jsons(raw_text)
            if extracted:
                for json_obj in extracted:
                    # Try multiple possible keys in order of preference
                    for key in ["grade", "response", "label", "evaluation"]:
                        if key in json_obj:
                            val = json_obj[key]
                            normalized = _normalize_grade(val)
                            if normalized:
                                prediction = normalized
                                break
                    if prediction != "None":
                        break
            
            # Strategy 2: If JSON extraction failed, try text extraction
            if prediction == "None":
                text_pred = _extract_label_from_text(raw_text)
                if text_pred:
                    prediction = text_pred
            
            # Strategy 3: Last resort - look for any valid label in the text
            if prediction == "None":
                text_lower = raw_text.lower()
                # Check in priority order
                if "incorrect" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "correct" in text_lower:
                    prediction = "correct"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
