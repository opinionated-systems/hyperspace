"""
Task agent: solves a given task with chain-of-thought reasoning.

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

# Compile regex patterns once for efficiency
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)

# Additional patterns for grade extraction from free-form text
_NUMERIC_GRADE_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+)', re.IGNORECASE)
_PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)%', re.IGNORECASE)
_WORD_GRADE_PATTERN = re.compile(r'\b(Correct|Incorrect|Partially Correct|Partial|Full|Zero|None|Pass|Fail|Excellent|Good|Fair|Poor)\b', re.IGNORECASE)

# Extended patterns for more robust extraction
_FRACTION_PATTERN = re.compile(r'\b(\d+)\s*out\s*of\s*(\d+)\b', re.IGNORECASE)
_DECIMAL_GRADE_PATTERN = re.compile(r'\b(0\.\d+|1\.0?)\b', re.IGNORECASE)
_LETTER_GRADE_PATTERN = re.compile(r'\b([A-F][+-]?)\b', re.IGNORECASE)

logger = logging.getLogger(__name__)


def _normalize_grade(grade: str) -> str:
    """Normalize extracted grade to a standard format.
    
    Handles various grade formats and converts them to a consistent representation.
    """
    grade = grade.strip()
    
    # Normalize word-based grades
    grade_lower = grade.lower()
    if grade_lower in ("partial", "partially"):
        return "Partially Correct"
    if grade_lower in ("full", "complete"):
        return "Correct"
    if grade_lower in ("zero", "none", "0"):
        return "Incorrect"
    
    # Check for numeric patterns already in grade
    if _NUMERIC_GRADE_PATTERN.match(grade):
        return grade
    if _PERCENTAGE_PATTERN.match(grade):
        return grade
    
    return grade


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade from free-form text using multiple patterns.
    
    Returns the first successfully extracted grade or None.
    """
    # Try explicit grade patterns first
    grade_match = _GRADE_PATTERN.search(text)
    if grade_match:
        return _normalize_grade(grade_match.group(1).strip())
    
    # Try numeric fraction pattern (e.g., "7/10")
    numeric_match = _NUMERIC_GRADE_PATTERN.search(text)
    if numeric_match:
        score, total = numeric_match.groups()
        return f"{score}/{total}"
    
    # Try "X out of Y" pattern
    fraction_match = _FRACTION_PATTERN.search(text)
    if fraction_match:
        score, total = fraction_match.groups()
        return f"{score}/{total}"
    
    # Try percentage pattern
    percent_match = _PERCENTAGE_PATTERN.search(text)
    if percent_match:
        return f"{percent_match.group(1)}%"
    
    # Try decimal grade (0-1 scale)
    decimal_match = _DECIMAL_GRADE_PATTERN.search(text)
    if decimal_match:
        return decimal_match.group(1)
    
    # Try letter grade
    letter_match = _LETTER_GRADE_PATTERN.search(text)
    if letter_match:
        return letter_match.group(1)
    
    # Try word-based grade
    word_match = _WORD_GRADE_PATTERN.search(text)
    if word_match:
        return _normalize_grade(word_match.group(1))
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
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
            # Try to extract JSON from within the text if it's wrapped in other content
            try:
                # Look for JSON-like content with braces
                brace_start = inner.find("{")
                brace_end = inner.rfind("}")
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    results.append(json.loads(inner[brace_start:brace_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

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

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond ONLY in JSON format wrapped in <json>...</json> tags with the following exact schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

## Example Response 1 (Partial Credit):
<json>
{{
    "reasoning": "The student correctly identified the key theorem but made an error in the algebraic manipulation at step 3. The final answer is incorrect due to this calculation error.",
    "response": "Partially Correct"
}}
</json>

## Example Response 2 (Full Credit):
<json>
{{
    "reasoning": "The student's solution is complete and correct. They correctly applied the quadratic formula, showed all work, and arrived at the right answer. The reasoning is clear and follows the official solution approach.",
    "response": "Correct"
}}
</json>

## Example Response 3 (Numeric Score):
<json>
{{
    "reasoning": "The student solved the first two parts correctly but missed the third part entirely. They showed good understanding of the core concepts but the incomplete solution warrants partial credit.",
    "response": "7/10"
}}
</json>

## Important Notes:
- The "response" field should contain ONLY the grade/assessment, not the reasoning
- Use standard grading terms like "Correct", "Partially Correct", "Incorrect" or provide numeric scores
- Be consistent with the grading guidelines provided above
- Your response MUST be valid JSON inside <json> tags

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Try multiple possible keys for the response
                for key in ["response", "grade", "answer", "result", "assessment", "evaluation"]:
                    if key in last_json:
                        prediction = last_json[key]
                        break
                # Log reasoning if available
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
                # Also check for alternative reasoning keys
                elif "analysis" in last_json:
                    reasoning = last_json["analysis"]
                    self.log_fn(f"Analysis: {reasoning[:200]}...")
                elif "explanation" in last_json:
                    reasoning = last_json["explanation"]
                    self.log_fn(f"Explanation: {reasoning[:200]}...")
            else:
                # Fallback: use the comprehensive grade extraction function
                response_text = msg_history[-1]["text"]
                extracted_grade = _extract_grade_from_text(response_text)
                if extracted_grade:
                    prediction = extracted_grade
                    self.log_fn(f"Extracted grade via pattern matching: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Calculate confidence based on extraction method
        confidence = self._calculate_confidence(prediction, msg_history)
        if confidence < 1.0:
            self.log_fn(f"Low confidence extraction: {confidence:.2f}")
        
        return str(prediction), msg_history
    
    def _calculate_confidence(self, prediction: str, msg_history: list[dict]) -> float:
        """Calculate confidence score for the extracted prediction.
        
        Returns a value between 0 and 1, where 1 indicates high confidence.
        """
        if not msg_history:
            return 0.0
        
        last_msg = msg_history[-1].get("text", "")
        
        # High confidence: JSON extraction succeeded
        if _extract_jsons(last_msg) is not None:
            return 1.0
        
        # Medium-high confidence: explicit grade pattern found
        if _GRADE_PATTERN.search(last_msg):
            return 0.9
        
        # Medium confidence: numeric or percentage grade
        if _NUMERIC_GRADE_PATTERN.search(last_msg) or _PERCENTAGE_PATTERN.search(last_msg):
            return 0.8
        
        # Lower confidence: word-based or other patterns
        if _WORD_GRADE_PATTERN.search(last_msg) or _FRACTION_PATTERN.search(last_msg):
            return 0.6
        
        # Very low confidence: no recognizable pattern
        if prediction and prediction != "None":
            return 0.3
        
        return 0.0
