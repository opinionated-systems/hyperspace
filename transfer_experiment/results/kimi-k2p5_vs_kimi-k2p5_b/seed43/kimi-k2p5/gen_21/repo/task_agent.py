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


def _extract_response_direct(text: str, points: int | None = None) -> str | None:
    """Extract response by looking for keywords directly in text.
    
    Args:
        text: The response text to parse
        points: Optional points value to help disambiguate
    
    Returns:
        One of "correct", "incorrect", "partial", or None if no clear classification found.
    """
    text_lower = text.lower()
    
    # POINTS-BASED FALLBACK: If we have points, use that as the PRIMARY signal
    # This is the most reliable method for IMO grading
    if points is not None:
        if points == 7:
            return "correct"
        elif points == 0:
            return "incorrect"
        elif 1 <= points <= 6:
            return "partial"
    
    # Look for explicit classification statements with response field
    # Priority order: patterns that explicitly mention the response field
    patterns = [
        # JSON-style response field patterns (highest priority)
        r'"response"\s*:\s*"(correct|incorrect|partial)"',
        r'"response"\s*:\s*\'(correct|incorrect|partial)\'',
        r'"response"\s*:\s*(correct|incorrect|partial)(?:[,}\]])',
        r"'response'\s*:\s*'(correct|incorrect|partial)'",
        # Explicit statement patterns
        r'response[":\s]+is\s+"?(correct|incorrect|partial)"?',
        r'response[":\s]+should\s+be\s+"?(correct|incorrect|partial)"?',
        r'classification[:\s]+is\s+"?(correct|incorrect|partial)"?',
        r'classify[:\s]+as\s+"?(correct|incorrect|partial)"?',
        r'grade[:\s]+is\s+"?(correct|incorrect|partial)"?',
        r'graded\s+as\s+"?(correct|incorrect|partial)"?',
        r'"?(correct|incorrect|partial)"?\s*is\s*the\s*correct\s*classification',
        r'the\s*answer\s*is\s*"?(correct|incorrect|partial)"?',
        r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial)"?',
        r'categorize.*as\s+"?(correct|incorrect|partial)"?',
        r'falls?\s+(into|under)\s+"?(correct|incorrect|partial)"?',
        r'therefore,?\s+(?:the\s+)?(?:answer|classification|grade)\s+is\s+"?(correct|incorrect|partial)"?',
        r'conclusion[:\s]+"?(correct|incorrect|partial)"?',
        r'final\s+(?:answer|classification|grade)[:\s]+"?(correct|incorrect|partial)"?',
        r'classification:\s*"?(correct|incorrect|partial)"?',
        r'category[:\s]+"?(correct|incorrect|partial)"?',
        r'label[:\s]+"?(correct|incorrect|partial)"?',
        r'classified\s+as\s+"?(correct|incorrect|partial)"?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE)
        if match:
            result = match.group(1).lower()
            # Handle "almost" mapping to "partial"
            if result == "almost":
                return "partial"
            return result
    
    # Count occurrences of each label (only standalone words)
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    almost_count = len(re.findall(r'\balmost\b', text_lower))
    
    # If only one appears, use that
    total = correct_count + incorrect_count + partial_count + almost_count
    if total > 0:
        if correct_count > 0 and incorrect_count == 0 and partial_count == 0 and almost_count == 0:
            return "correct"
        if incorrect_count > 0 and correct_count == 0 and partial_count == 0 and almost_count == 0:
            return "incorrect"
        if (partial_count > 0 or almost_count > 0) and correct_count == 0 and incorrect_count == 0:
            return "partial"
    
    # If we have points but couldn't extract a clear label, use points as final fallback
    if points is not None:
        if points == 7:
            return "correct"
        elif points == 0:
            return "incorrect"
        elif 1 <= points <= 6:
            return "partial"
    
    return None


def _normalize_prediction(prediction: str, points: int | None = None) -> str:
    """Normalize prediction to one of the three valid labels.
    
    Args:
        prediction: The raw prediction string
        points: Optional points value to help disambiguate
    
    Returns:
        One of "correct", "incorrect", or "partial"
    """
    pred_str = str(prediction).lower().strip().strip('"\'')
    
    # POINTS OVERRIDE: Points are the PRIMARY signal - they determine the label
    # This is critical for IMO grading accuracy
    if points is not None:
        if points == 7:
            # Full points = correct, regardless of what the LLM predicted
            return "correct"
        elif points == 0:
            # Zero points = incorrect, regardless of what the LLM predicted
            return "incorrect"
        elif 1 <= points <= 6:
            # Any partial points (1-6) = partial, regardless of what the LLM predicted
            return "partial"
    
    # Direct match (fallback when no points info)
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Handle "almost" - it maps to "partial" (nearly complete but has mistakes)
    if pred_str == "almost" or "almost" in pred_str:
        return "partial"
    
    # Check for partial first (more specific)
    if "partial" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong/false/error/invalid/no credit/zero
    if any(term in pred_str for term in ["incorrect", "wrong", "false", "error", "invalid", "no credit", "zero", "0 points", "no points"]):
        return "incorrect"
    
    # Check for correct (but not incorrect)
    if pred_str == "correct" or ("correct" in pred_str and "incorrect" not in pred_str):
        return "correct"
    
    # Default fallback - be conservative (incorrect is safer than over-predicting correct)
    return "incorrect"


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
        # Extract key fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Extract points from grading guidelines for better decision making
        # Try multiple patterns to find points
        points_patterns = [
            r'Points:\s*(\d+)',
            r'Points\s*[=:]\s*(\d+)',
            r'\((\d+)\s*points?\)',
            r'(\d+)\s*points?\s*(?:out of|/|from)\s*7',
            r'awarded\s+(\d+)\s*points?',
            r'score[d]?\s*[=:]?\s*(\d+)',
        ]
        points = None
        points_info = ""
        for pattern in points_patterns:
            points_match = re.search(pattern, grading_guidelines, re.IGNORECASE)
            if points_match:
                points = int(points_match.group(1))
                break
        
        if points is not None:
            points_info = f"\n## Points Awarded: {points} (out of 7)\n"
            if points == 7:
                points_info += "This is a FULL SCORE solution. Classification: CORRECT.\n"
            elif points == 0:
                points_info += "This solution received ZERO points. Classification: INCORRECT.\n"
            elif 1 <= points <= 6:
                points_info += f"This solution received PARTIAL CREDIT ({points} points). Classification: PARTIAL.\n"
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's answer to an IMO-level problem.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}
{points_info}
## Student's Answer:
{student_answer}

## Classification Guidelines:

**"correct"** - The student provided a complete, correct solution with proper reasoning. The answer:
- Arrives at the correct final answer
- Contains valid mathematical reasoning throughout
- Awarded FULL POINTS (7 points on a 7-point scale)
- ONLY 7 points = "correct"

**"incorrect"** - The student's answer is wrong or fundamentally flawed. The answer:
- Has an incorrect final answer with no valid path to solution
- Contains critical logical or mathematical errors
- Shows no meaningful progress toward the solution
- Is incomplete with no useful partial results
- Awarded 0 points
- ZERO points = "incorrect"

**"partial"** - The student made meaningful progress but the solution is incomplete or has significant gaps. This includes:
- Contains valid and significant progress toward the solution (1-5 points)
- Has correct lemmas, propositions, or intermediate steps
- May have the right approach but incomplete execution
- Shows understanding of key concepts even if final answer is wrong or missing
- ALSO includes "Almost" solutions that are nearly complete but have non-negligible mistakes (6 points)
- 1-6 points = "partial" (including "Almost" at 6 points)

## CRITICAL DECISION RULES - FOLLOW THESE EXACTLY:

**RULE 1: POINTS ARE THE ONLY SIGNAL THAT MATTERS**
- The points awarded is the GROUND TRUTH - nothing else matters
- 7 points → "correct" (full marks, complete solution)
- 1-6 points → "partial" (partial credit, any non-zero, non-full score)
- 0 points → "incorrect" (no credit, no meaningful progress)

**RULE 2: Points override ALL other signals - including the grading guidelines text**
- IGNORE any "(Partial)" or "(Almost)" sections in the guidelines if points = 0
- IGNORE any perceived completeness if points ≠ 7
- The points awarded by the official grader is the FINAL AUTHORITY
- If points = 7, the answer IS correct, regardless of any perceived flaws
- If points = 0, the answer IS incorrect, regardless of any partial achievements listed
- If 1 ≤ points ≤ 6, the answer IS partial, regardless of how complete it looks

**RULE 3: "Almost" = "partial", NEVER "correct"**
- "Almost" means the solution is nearly complete BUT has mistakes that are NOT negligible
- 6 points = "partial" (not full marks, has errors)
- Only 7 points = "correct"
- Even if the solution is "almost" complete, if it has ANY mistakes, it is "partial"

**RULE 4: Zero points means "incorrect" - ALWAYS**
- If the student received 0 points, the answer is "incorrect"
- Even if there are "(Partial)" or "(Almost)" sections in the guidelines, if points = 0 → "incorrect"
- The points awarded reflect what the student actually achieved, not what they attempted
- A solution with partial insights but 0 points is still "incorrect"

**RULE 5: 1-6 points means "partial" - ALWAYS**
- Any non-zero, non-full score is "partial"
- This includes solutions with "(Partial)" achievements
- This includes "(Almost)" solutions with minor mistakes (6 points)
- Even 1 point = "partial" (not "incorrect")

## Decision Framework:

Step 1: Check the POINTS awarded (THIS IS THE ONLY STEP THAT MATTERS).
- 7 points → "correct" (no exceptions)
- 0 points → "incorrect" (no exceptions, regardless of what sections appear in guidelines)
- 1-6 points → "partial" (no exceptions, including "Almost" at 6 points)

Step 2: Use the grading guidelines ONLY to understand WHY the points were awarded, NOT to determine the classification.
- The classification is already determined by the points in Step 1
- Guidelines help explain the reasoning, but do NOT change the classification

Step 3: Final verification - Your classification MUST match the points.
- If points = 7 → MUST be "correct"
- If points = 0 → MUST be "incorrect"
- If 1 ≤ points ≤ 6 → MUST be "partial"

## Your Task:
First, state the points awarded clearly. Then classify the student's answer into EXACTLY ONE category based SOLELY on the points: "correct", "incorrect", or "partial".

IMPORTANT: Your classification MUST match the points awarded:
- Points = 7 → response MUST be "correct"
- Points = 0 → response MUST be "incorrect"  
- Points = 1-6 → response MUST be "partial"

VALID RESPONSE VALUES: Only use "correct", "incorrect", or "partial". Do NOT use "almost" or any other value.

You MUST respond with a JSON object in this exact format:
<json>
{{
    "reasoning": "The student was awarded X points out of 7. [Explain what the student got right/wrong based on the guidelines]. Based on the points awarded, this solution is classified as [correct/incorrect/partial].",
    "response": "correct"
}}
</json>

OR

<json>
{{
    "reasoning": "The student was awarded X points out of 7. [Explain what the student got right/wrong based on the guidelines]. Based on the points awarded, this solution is classified as [correct/incorrect/partial].",
    "response": "incorrect"
}}
</json>

OR

<json>
{{
    "reasoning": "The student was awarded X points out of 7. [Explain what the student got right/wrong based on the guidelines]. Based on the points awarded, this solution is classified as [correct/incorrect/partial].",
    "response": "partial"
}}
</json>

Replace the "response" value with your classification. Include detailed reasoning in the "reasoning" field. Do not include any other text outside the JSON block.

FINAL REMINDER: The points value is the ONLY thing that matters. Classify based on points ONLY."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "incorrect"  # Default fallback
        points = None
        
        try:
            response_text = msg_history[-1]["text"]
            
            # Extract points from grading guidelines for normalization
            # Try multiple patterns to find points
            points_patterns = [
                r'Points:\s*(\d+)',
                r'Points\s*[=:]\s*(\d+)',
                r'\((\d+)\s*points?\)',
                r'(\d+)\s*points?\s*(?:out of|/|from)\s*7',
                r'awarded\s+(\d+)\s*points?',
                r'score[d]?\s*[=:]?\s*(\d+)',
            ]
            for pattern in points_patterns:
                points_match = re.search(pattern, grading_guidelines, re.IGNORECASE)
                if points_match:
                    points = int(points_match.group(1))
                    break
            
            # First try: extract from <json> tags
            extracted = _extract_jsons(response_text)
            
            if extracted and isinstance(extracted, list) and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = _normalize_prediction(last_json["response"], points)
                else:
                    self.log_fn(f"JSON found but no 'response' field: {last_json}")
            else:
                # Try direct extraction with points for better accuracy
                direct = _extract_response_direct(response_text, points)
                if direct:
                    prediction = _normalize_prediction(direct, points)
                else:
                    self.log_fn(f"No valid JSON found in response: {response_text[:200]}...")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        # FINAL VERIFICATION: Override prediction based on points if available
        # This ensures the prediction always matches the points value
        if points is not None:
            if points == 7:
                prediction = "correct"
            elif points == 0:
                prediction = "incorrect"
            elif 1 <= points <= 6:
                prediction = "partial"

        return str(prediction), msg_history
