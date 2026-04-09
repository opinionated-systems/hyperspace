"""
Task agent: solves a given task with chain-of-thought reasoning and self-consistency.

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
from collections import Counter

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Few-shot examples for better grading accuracy
_FEW_SHOT_EXAMPLES = """
## Example 1: Correct Answer
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 2 + 3 = 5
Grade: correct
Reasoning: The student provided the correct answer with the correct reasoning.

## Example 2: Almost Correct Answer
Problem: Solve x^2 - 4 = 0.
Solution: x^2 = 4, so x = 2 or x = -2.
Student Answer: x = 2 (forgot the negative solution)
Grade: almost
Reasoning: The student found one correct solution but missed the other valid solution.

## Example 3: Partial Answer
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with diagram]
Student Answer: I think it's 180 because my teacher said so.
Grade: partial
Reasoning: The student knows the correct fact but provides no mathematical reasoning or proof.

## Example 4: Incorrect Answer
Problem: Find the derivative of x^2.
Solution: d/dx(x^2) = 2x
Student Answer: The derivative is x^3/3
Grade: incorrect
Reasoning: The student confused differentiation with integration, showing a fundamental misunderstanding.
"""


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


def _extract_json_with_fallback(text: str) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies."""
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Second attempt: try to find JSON without tags
    try:
        # Look for JSON-like structures
        json_pattern = re.search(r'\{[^{}]*"[^"]+"[^{}]*\}', text, re.DOTALL)
        if json_pattern:
            try:
                parsed = json.loads(json_pattern.group())
                return [parsed]
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    
    # Third attempt: fix common JSON errors (trailing commas)
    try:
        fixed_text = re.sub(r',(\s*[}\]])', r'\1', text)
        result = _extract_jsons(fixed_text)
        if result:
            return result
    except Exception:
        pass
    
    return None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from plain text if JSON parsing fails."""
    valid_grades = {"correct", "almost", "partial", "incorrect"}
    text_lower = text.lower()
    
    # Look for explicit grade mentions with priority order
    # More specific patterns first
    patterns = [
        (r'grade[\s]*[:=][\s]*["\']?(correct|almost|partial|incorrect)["\']?', 1),
        (r'response[\s]*[:=][\s]*["\']?(correct|almost|partial|incorrect)["\']?', 1),
        (r'["\'](correct|almost|partial|incorrect)["\']', 1),
        (r'\b(correct|almost|partial|incorrect)\b', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(group)
            if grade in valid_grades:
                return grade
    
    # Check for grade indicators in reasoning
    if "fully correct" in text_lower or "completely correct" in text_lower:
        return "correct"
    if "nearly correct" in text_lower or "minor error" in text_lower:
        return "almost"
    if "partially correct" in text_lower or "some correct" in text_lower:
        return "partial"
    if "fundamentally wrong" in text_lower or "completely wrong" in text_lower:
        return "incorrect"
    
    return None


def _normalize_grade(raw_response: str) -> str | None:
    """Normalize a raw response to a valid grade."""
    if not raw_response or not isinstance(raw_response, str):
        return None
    
    valid_grades = {"correct", "almost", "partial", "incorrect"}
    normalized = raw_response.strip().lower()
    
    # Exact match
    if normalized in valid_grades:
        return normalized
    
    # Handle common variations
    variations = {
        "correct": ["correct", "right", "true", "valid", "accurate", "full marks"],
        "almost": ["almost", "nearly", "minor errors", "small mistakes", "close"],
        "partial": ["partial", "incomplete", "some correct", "partial credit"],
        "incorrect": ["incorrect", "wrong", "false", "invalid", "error", "no credit"],
    }
    
    for grade, variants in variations.items():
        for variant in variants:
            if variant in normalized:
                return grade
    
    # Substring match as last resort
    for grade in valid_grades:
        if grade in normalized:
            return grade
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning and self-consistency."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.num_samples = 3  # Number of samples for self-consistency

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Few-Shot Examples
{_FEW_SHOT_EXAMPLES}

## Instructions
1. First, analyze the student's answer step by step. Compare it to the correct solution.
2. Identify what the student did correctly and what errors they made.
3. Consider the grading guidelines carefully - these define the specific criteria for each grade level.
4. Refer to the few-shot examples above to understand the grading standards.
5. Provide your reasoning for the grade you will assign, citing specific evidence from the student's answer.
6. Finally, provide your grade assessment in the JSON format below.

## Grade Definitions
- "correct": The student's answer is fully correct, complete, and matches the solution. All key steps and reasoning are present and accurate.
- "almost": The student's answer is nearly correct with only minor gaps or small errors. The core approach is right but missing minor details.
- "partial": The student's answer has significant gaps or errors but contains some correct elements or partial progress toward the solution.
- "incorrect": The student's answer is fundamentally wrong, does not address the problem, or shows no meaningful understanding.

CRITICAL: The "response" field MUST contain EXACTLY one of these four values: "correct", "almost", "partial", or "incorrect". No other text, no explanations, just the exact grade label.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>"""

        # Self-consistency: sample multiple times and vote
        samples = []
        all_histories = []
        
        for i in range(self.num_samples):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                all_histories.extend(msg_history)
                
                prediction = self._extract_prediction(msg_history)
                if prediction and prediction != "None":
                    samples.append(prediction)
                    self.log_fn(f"Sample {i+1}: {prediction}")
            except Exception as e:
                self.log_fn(f"Sample {i+1} failed: {e}")
                continue
        
        # If no valid samples, try once more with clearer prompt
        if not samples:
            self.log_fn("All samples failed, retrying with clearer prompt...")
            retry_instruction = f"""Your task is to grade a student's answer. Provide ONLY a valid JSON response.

Problem: {problem[:200]}...

Student Answer: {student_answer[:200]}...

You MUST respond with EXACTLY this JSON format (no other text):
<json>
{{
    "reasoning": "Brief analysis of why the answer deserves this grade...",
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

Remember: the "response" field must be EXACTLY one of: "correct", "almost", "partial", or "incorrect"."""
            
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=retry_instruction,
                    model=self.model,
                    msg_history=[],
                )
                all_histories.extend(msg_history)
                prediction = self._extract_prediction(msg_history)
                if prediction and prediction != "None":
                    samples.append(prediction)
            except Exception as e:
                self.log_fn(f"Retry failed: {e}")
        
        # Aggregate samples using majority voting
        final_prediction = self._aggregate_samples(samples)
        self.log_fn(f"Final prediction (from {len(samples)} samples): {final_prediction}")
        
        return str(final_prediction), all_histories

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract and validate the grade prediction from message history."""
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        
        try:
            # Get the last assistant message
            last_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_msg = msg
                    break
            
            if not last_msg:
                return "None"
            
            text = last_msg.get("text", "")
            
            # Try to extract JSON with fallback strategies
            extracted = _extract_json_with_fallback(text)
            if extracted:
                last_json = extracted[-1]
                raw_response = None
                
                # Try multiple possible field names in priority order
                for field in ["response", "grade", "answer", "result", "evaluation", "score"]:
                    if field in last_json:
                        raw_response = last_json[field]
                        break
                
                if raw_response is not None:
                    normalized = _normalize_grade(str(raw_response))
                    if normalized:
                        return normalized
                    # If normalization failed but we have a string, return it for logging
                    if isinstance(raw_response, str):
                        return raw_response.strip()
            
            # Fallback: try to extract grade from plain text
            grade_from_text = _extract_grade_from_text(text)
            if grade_from_text:
                return grade_from_text
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
    
    def _aggregate_samples(self, samples: list[str]) -> str:
        """Aggregate multiple samples using majority voting with confidence."""
        if not samples:
            return "None"
        
        if len(samples) == 1:
            return samples[0]
        
        valid_grades = {"correct", "almost", "partial", "incorrect"}
        
        # Filter to valid grades only
        valid_samples = [s for s in samples if s in valid_grades]
        
        if not valid_samples:
            # If no valid grades, return the most common raw response
            counter = Counter(samples)
            return counter.most_common(1)[0][0]
        
        # Count valid grades
        counter = Counter(valid_samples)
        most_common = counter.most_common()
        
        # Return the majority grade
        best_grade, count = most_common[0]
        confidence = count / len(valid_samples)
        
        self.log_fn(f"Grade distribution: {dict(counter)}, confidence: {confidence:.2f}")
        
        # If confidence is low (< 0.5), consider the middle ground
        if confidence < 0.5 and len(most_common) > 1:
            # Return "partial" as a conservative choice when uncertain
            if best_grade == "correct" and most_common[1][0] in ["almost", "partial"]:
                return "almost"
            if best_grade == "incorrect" and most_common[1][0] in ["partial", "almost"]:
                return "partial"
        
        return best_grade
