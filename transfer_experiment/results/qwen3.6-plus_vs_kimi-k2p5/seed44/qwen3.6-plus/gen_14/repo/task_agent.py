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

__version__ = "1.8.0"

# Valid grade labels for normalization
_VALID_GRADES = {"correct", "almost", "partial", "incorrect"}

# Order matters: check "incorrect" before "correct" to avoid substring match
_GRADE_CHECK_ORDER = ["incorrect", "almost", "partial", "correct"]

# Additional synonym mappings for common LLM output variations
_GRADE_SYNONYMS = {
    "mostly correct": "almost",
    "nearly correct": "almost",
    "almost correct": "almost",
    "partially correct": "partial",
    "partly correct": "partial",
    "somewhat correct": "partial",
    "completely correct": "correct",
    "fully correct": "correct",
    "totally correct": "correct",
    "completely wrong": "incorrect",
    "totally wrong": "incorrect",
    "entirely wrong": "incorrect",
    "wrong": "incorrect",
}

# Detailed grading rubric with specific criteria for each grade
_GRADING_RUBRIC = {
    "Correct": [
        "All required steps are present and mathematically sound",
        "The final answer/conclusion matches the reference solution",
        "All key claims are properly justified or proven",
        "No logical gaps, errors, or unjustified leaps",
        "Notation and terminology are used correctly throughout",
    ],
    "Almost": [
        "The overall approach and final answer are correct",
        "Minor issues present: small arithmetic error, missing minor justification, slight notation issue",
        "The core argument/proof structure is sound",
        "The student clearly understands the problem and its solution",
        "Issues do not affect the validity of the main result",
    ],
    "Partial": [
        "Shows genuine understanding of some aspects of the problem",
        "Correct setup or approach but incomplete execution",
        "Valid reasoning for part of a multi-part problem",
        "Contains a significant error but demonstrates relevant knowledge",
        "Missing key steps but the direction of reasoning is correct",
    ],
    "Incorrect": [
        "Fundamentally wrong approach or conclusion",
        "Fails to meet key criteria from the grading guidelines",
        "Shows misunderstanding of core concepts",
        "The answer is essentially unrelated to the problem",
        "Critical logical errors that invalidate the entire argument",
    ],
}


def _normalize_prediction(raw: str) -> str:
    """Normalize a raw prediction string to one of the valid grade labels.

    Returns the capitalized grade if recognized, otherwise "None".
    Checks "incorrect" before "correct" to avoid substring false matches.
    Also handles common synonym phrases like "mostly correct", "partially correct", etc.
    """
    cleaned = raw.strip().strip('"').strip("'").lower()
    # Strip common prefixes like "grade:", "prediction:", etc.
    cleaned = re.sub(r'^(grade|prediction|result|answer|verdict|evaluation)\s*[:=]\s*', '', cleaned).strip()
    
    # First check for multi-word synonym phrases (before single-word checks)
    for phrase, grade in _GRADE_SYNONYMS.items():
        if phrase in cleaned:
            return grade.capitalize()
    
    # Use ordered list to check "incorrect" before "correct" (substring safety)
    for grade in _GRADE_CHECK_ORDER:
        if re.search(r'\b' + re.escape(grade) + r'\b', cleaned):
            return grade.capitalize()
    return "None"


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
            # Try to clean up common issues
            try:
                cleaned = re.sub(r',\s*}', '}', inner)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks as a fallback."""
    results = []
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r',\s*}', '}', match.strip())
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects by finding matching braces as a last resort."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            count = 1
            j = i + 1
            while j < len(text) and count > 0:
                if text[j] == '{':
                    count += 1
                elif text[j] == '}':
                    count -= 1
                j += 1
            if count == 0:
                try:
                    results.append(json.loads(text[i:j]))
                except json.JSONDecodeError:
                    try:
                        cleaned = re.sub(r',\s*}', '}', text[i:j])
                        cleaned = re.sub(r',\s*]', ']', cleaned)
                        cleaned = cleaned.replace("'", '"')
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        pass
            i = j
        else:
            i += 1
    return results or None


# Error-indicating phrases that suggest the answer is NOT fully correct
_ERROR_INDICATORS = [
    r'\b(missing|omitted|incomplete|incorrect|wrong|error|flawed|gap|lacks|fails)\b',
    r'\b(should have|should be|needs to|unclear|ambiguous|insufficient|inadequate)\b',
    r'\b(not fully|not entirely|not quite|not exactly|not sufficient|not complete)\b',
    r'\b(not correct|not right|not accurate|not valid|not rigorous|not justified)\b',
    r'\b(not explained|not shown|not proven|not demonstrated|not derived)\b',
    r'\b(not stated|not mentioned|not addressed|not considered|not accounted for)\b',
    r'\b(not included|not provided|not given|not specified|not defined)\b',
]

# Positive-indicating phrases that suggest the answer IS correct
_POSITIVE_INDICATORS = [
    r'\b(fully correct|completely correct|entirely correct|fully satisfies)\b',
    r'\b(no errors|no issues|no problems|no gaps|no omissions)\b',
    r'\b(meets all|satisfies all|fulfills all|matches the reference)\b',
    r'\b(properly justified|properly proven|rigorously proven)\b',
    r'\b(sound reasoning|valid approach|correct approach|correct method)\b',
]

# Phrases that indicate the answer has significant problems (stronger than general error indicators)
_CRITICAL_ERRORS = [
    r'\b(fundamentally wrong|completely wrong|entirely wrong|wholly incorrect)\b',
    r'\b(fails to|does not address|does not consider|ignores the key)\b',
    r'\b(misunderstands|misapplies|misinterprets|confuses)\b',
    r'\b(logical fallacy|logical error|fatal flaw|critical error)\b',
    r'\b(unrelated to|irrelevant to|does not solve)\b',
]

# Grade downgrade mapping based on error severity
_GRADE_DOWNGRADE = {
    "Correct": "Almost",
    "Almost": "Partial",
    "Partial": "Incorrect",
}

# Grade upgrade mapping for when reasoning is more positive than the grade suggests
_GRADE_UPGRADE = {
    "Incorrect": "Partial",
    "Partial": "Almost",
    "Almost": "Correct",
}


def _check_response_for_errors(response_text: str) -> int:
    """Count error-indicating phrases in the response text.
    
    Returns the number of error indicators found.
    """
    count = 0
    for pattern in _ERROR_INDICATORS:
        count += len(re.findall(pattern, response_text, re.IGNORECASE))
    return count


def _check_response_for_positive_indicators(response_text: str) -> int:
    """Count positive-indicating phrases in the response text.
    
    Returns the number of positive indicators found.
    """
    count = 0
    for pattern in _POSITIVE_INDICATORS:
        count += len(re.findall(pattern, response_text, re.IGNORECASE))
    return count


def _check_response_for_critical_errors(response_text: str) -> int:
    """Count critical error phrases in the response text.
    
    Returns the number of critical error indicators found.
    """
    count = 0
    for pattern in _CRITICAL_ERRORS:
        count += len(re.findall(pattern, response_text, re.IGNORECASE))
    return count


def _compute_confidence(prediction: str, response_text: str) -> float:
    """Compute a confidence score (0.0 to 1.0) for the prediction.
    
    Higher confidence means the reasoning text strongly supports the grade.
    Lower confidence means the reasoning is ambiguous or contradictory.
    """
    if prediction == "None":
        return 0.0
    
    error_count = _check_response_for_errors(response_text)
    positive_count = _check_response_for_positive_indicators(response_text)
    critical_count = _check_response_for_critical_errors(response_text)
    
    # Start with a base confidence
    confidence = 0.5
    
    # Adjust based on consistency between grade and reasoning
    if prediction == "Correct":
        # High confidence if many positive indicators and no errors
        confidence += 0.1 * min(positive_count, 5)
        confidence -= 0.15 * min(error_count, 5)
        confidence -= 0.3 * min(critical_count, 2)
    elif prediction == "Almost":
        # High confidence if some errors but also positive indicators
        if 1 <= error_count <= 3 and positive_count >= 1:
            confidence += 0.2
        confidence -= 0.2 * min(critical_count, 2)
    elif prediction == "Partial":
        # High confidence if moderate errors and few positive indicators
        if error_count >= 2 and positive_count <= 2:
            confidence += 0.2
    elif prediction == "Incorrect":
        # High confidence if many errors or critical errors
        confidence += 0.1 * min(error_count, 5)
        confidence += 0.2 * min(critical_count, 3)
        confidence -= 0.15 * min(positive_count, 3)
    
    return max(0.0, min(1.0, confidence))


def _adjust_prediction_based_on_reasoning(prediction: str, response_text: str) -> str:
    """Adjust prediction if the reasoning text contradicts the grade.
    
    Uses a balanced scoring system that weighs error indicators against positive
    indicators to determine if the grade should be adjusted up or down.
    """
    if prediction not in _GRADE_DOWNGRADE and prediction not in _GRADE_UPGRADE:
        return prediction
    
    error_count = _check_response_for_errors(response_text)
    positive_count = _check_response_for_positive_indicators(response_text)
    critical_count = _check_response_for_critical_errors(response_text)
    
    # Critical errors strongly suggest the answer should be "Incorrect"
    if critical_count >= 2:
        if prediction != "Incorrect":
            return "Incorrect"
    
    # If there are significant error indicators and few positive ones, downgrade
    if error_count >= 3 and positive_count <= 1:
        new_prediction = _GRADE_DOWNGRADE.get(prediction, prediction)
        return new_prediction
    
    # If there are many positive indicators and very few errors, consider upgrading
    if positive_count >= 3 and error_count <= 1 and prediction in _GRADE_UPGRADE:
        new_prediction = _GRADE_UPGRADE.get(prediction, prediction)
        return new_prediction
    
    return prediction


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
        # Build a structured prompt with clearly labeled fields
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer and classify it into one of four categories: "Correct", "Almost", "Partial", or "Incorrect".

Domain: {domain}

Problem:
{problem}

Reference Solution:
{solution}

Grading Guidelines:
{grading_guidelines}

Student's Answer:
{student_answer}

Carefully analyze the student's answer against the reference solution and grading guidelines. Check each criterion in the grading guidelines to determine if the student's answer meets the requirements.

IMPORTANT: Be critical and thorough in your evaluation. Do NOT default to "Correct" unless the student's answer truly meets ALL criteria. Many student answers have subtle errors, missing steps, or incomplete reasoning that should result in lower grades.

CRITICAL: Before deciding on a grade, you MUST first identify any issues, errors, gaps, or omissions in the student's answer. Use the following decision tree:
1. Does the student's answer contain ANY error, gap, missing step, or omission? If YES → it is NOT "Correct".
2. Does the student reach the correct final answer/conclusion? If NO → it is "Incorrect" or "Partial".
3. Does the student's reasoning match the reference solution's approach? If NO → consider "Partial" or "Incorrect".
4. If the answer is correct but has minor issues (small arithmetic error, missing justification for one step, minor notation issue) → use "Almost".
5. If the answer shows genuine understanding but is incomplete or has significant issues → use "Partial".
6. Only use "Correct" if the answer is essentially equivalent to the reference solution with ZERO errors, gaps, or omissions.

Respond in JSON format with the following schema:
<json>
{{
    "response": "your detailed evaluation here. First list all issues found, then explain your grade decision.",
    "prediction": "Correct" or "Almost" or "Partial" or "Incorrect"
}}
</json>

Rules for classification:
- "Correct": The student's answer fully satisfies ALL criteria in the grading guidelines, reaches the correct conclusion, and shows complete reasoning with no significant errors or omissions. Every required step is present and correct.
- "Almost": The student's answer reaches the correct conclusion and has the right overall approach, but has minor issues such as: a small arithmetic error, a missing justification for one step, a minor notation issue, or a small gap that doesn't affect the core argument. The student clearly understands the problem.
- "Partial": The student's answer shows genuine understanding of some aspects but is incomplete or has significant issues. Examples: correct setup but incomplete execution, correct reasoning for part of a multi-part problem, a valid approach that contains a significant error, or missing key steps in the argument.
- "Incorrect": The student's answer is fundamentally wrong, reaches an incorrect conclusion, fails to meet the key criteria, or shows a misunderstanding of the core concepts. The approach is flawed from the start or the answer is essentially unrelated to the problem."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple methods
        prediction = "None"
        response_text = ""
        try:
            if msg_history and len(msg_history) > 0:
                text = msg_history[-1].get("text", "")
                response_text = text

                # Try multiple extraction methods in order of reliability
                extracted = _extract_jsons(text)
                if not extracted:
                    extracted = _extract_json_from_markdown(text)
                if not extracted:
                    extracted = _extract_json_braces(text)

                if extracted:
                    last_extract = extracted[-1]

                    # Try multiple field names in order of preference
                    pred_value = None
                    for field in ["prediction", "response", "grade", "evaluation", "result", "answer", "verdict"]:
                        if field in last_extract:
                            pred_value = last_extract[field]
                            break

                    if pred_value is not None:
                        prediction = _normalize_prediction(str(pred_value))
                        self.log_fn(f"Extracted prediction: {prediction}")
                    else:
                        # Try to find any value that looks like a valid grade
                        for key, value in last_extract.items():
                            if isinstance(value, str):
                                normalized = _normalize_prediction(value)
                                if normalized != "None":
                                    prediction = normalized
                                    self.log_fn(f"Found valid grade in field '{key}': {prediction}")
                                    break
                else:
                    # No JSON found, try to extract grade from raw text
                    prediction = _normalize_prediction(text)
                    self.log_fn(f"No JSON found, normalized from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Post-process: adjust prediction if reasoning contradicts the grade
        if prediction != "None" and response_text:
            adjusted = _adjust_prediction_based_on_reasoning(prediction, response_text)
            if adjusted != prediction:
                self.log_fn(f"Downgraded prediction from {prediction} to {adjusted} based on error indicators in reasoning")
                prediction = adjusted

        return str(prediction), msg_history
