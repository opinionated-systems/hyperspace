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


def _extract_score_from_text(text: str) -> int | None:
    """Extract numerical score from text when JSON parsing fails.
    
    Uses multiple pattern strategies to find IMO scores (0-7).
    Returns the most likely valid score, or None if no score detected.
    """
    text_lower = text.lower()
    valid_scores = []
    
    # Pattern 1: Explicit score declarations (highest confidence)
    explicit_patterns = [
        r'score[\s]*[:=][\s]*(\d)',
        r'score[\s]+of[\s]+(\d)',
        r'(?:assign|give|award)[\s]+(?:a[\s]+)?score[\s]+(?:of[\s]+)?(\d)',
        r'grade[\s]*[:=][\s]*(\d)',
        r'final[\s]+score[\s]*[:=][\s]*(\d)',
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                score = int(match)
                if 0 <= score <= 7:
                    valid_scores.append((score, 3))  # High confidence
            except (ValueError, TypeError):
                continue
    
    # Pattern 2: Score in context like "X/7" (medium confidence)
    fraction_matches = re.findall(r'(?:^|\s)(\d)[\s]*[/\-][\s]*7', text_lower)
    for match in fraction_matches:
        try:
            score = int(match)
            if 0 <= score <= 7:
                valid_scores.append((score, 2))  # Medium confidence
        except (ValueError, TypeError):
            continue
    
    # Pattern 3: Standalone numbers 0-7 near keywords (lower confidence)
    # Look for numbers near grading-related words
    keyword_context = re.findall(
        r'(?:score|grade|point|mark|evaluation|assessment)[^.]*?(\d)[^.]*?',
        text_lower
    )
    for match in keyword_context:
        try:
            score = int(match)
            if 0 <= score <= 7:
                valid_scores.append((score, 1))  # Lower confidence
        except (ValueError, TypeError):
            continue
    
    # Return the score with highest confidence, preferring higher scores on ties
    if valid_scores:
        # Sort by confidence (desc), then by score value (desc)
        valid_scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return valid_scores[0][0]
    
    return None


def _validate_score(score: int, evaluation_text: str) -> tuple[int, str]:
    """Validate and potentially correct a score based on evaluation context.
    
    Returns (validated_score, validation_note)
    """
    if score is None:
        return None, "No score found"
    
    if not isinstance(score, int):
        try:
            score = int(score)
        except (ValueError, TypeError):
            return None, f"Invalid score type: {type(score)}"
    
    # Clamp to valid range
    if score < 0:
        return 0, f"Score {score} clamped to 0 (minimum)"
    if score > 7:
        return 7, f"Score {score} clamped to 7 (maximum)"
    
    return score, "Valid"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced evaluation."""

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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematics problem with the highest accuracy and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task
Carefully evaluate the student's answer against the official solution and grading guidelines. Follow this structured evaluation process:

### Step 1: Problem Understanding
- Identify the key mathematical concepts and techniques required
- Note the critical steps in the official solution
- Understand what constitutes a complete and correct solution

### Step 2: Student Solution Analysis
- Check mathematical correctness: Are all calculations valid? Is the reasoning sound?
- Check completeness: Does the answer address all parts of the problem? Are there gaps?
- Check approach: Does the student use valid methods, even if different from the official solution?
- Check clarity: Is the reasoning well-structured and easy to follow?

### Step 3: Error Identification (if any)
- Categorize any errors: conceptual, computational, logical, or completeness gaps
- Assess the severity of each error
- Determine if partial credit is warranted based on IMO standards

### Step 4: Score Assignment
- IMO scores range from 0 to 7 (7 = complete correct solution, 0 = no progress or completely wrong)
- Consider partial credit for significant progress, even if incomplete
- Be consistent with IMO grading standards: award points for valid approaches and correct partial results

### Step 5: Final Verification
- Double-check that the score reflects the actual quality of the solution
- Ensure the score is justified by your detailed analysis

Provide your evaluation in JSON format with the following schema:
<json>
{{
    "score": <integer 0-7>,
    "evaluation": "Your detailed evaluation: (1) summary of solution quality, (2) specific errors or gaps found, (3) strengths and valid approaches used, (4) justification for partial credit if applicable",
    "reasoning": "Step-by-step reasoning: problem analysis, student approach assessment, error categorization, score justification"
}}
</json>

IMPORTANT: 
- The score MUST be an integer between 0 and 7 (inclusive)
- Be fair but rigorous - IMO grading rewards correct mathematics
- Award partial credit generously for meaningful progress toward a solution
- If the student uses a different but valid approach, evaluate based on mathematical correctness, not similarity to the official solution"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced validation and fallback mechanisms
        prediction = "None"
        validation_note = ""
        
        try:
            response_text = msg_history[-1]["text"] if msg_history else ""
            
            # Try to extract from JSON blocks first
            extracted = _extract_jsons(response_text)
            if extracted:
                last_json = extracted[-1]
                
                # Build comprehensive evaluation string
                parts = []
                
                # Extract and validate score from JSON
                raw_score = None
                if "score" in last_json:
                    raw_score = last_json["score"]
                
                # Validate the score
                score, validation_note = _validate_score(raw_score, response_text)
                
                # If no valid score in JSON, try to extract from text
                if score is None:
                    score = _extract_score_from_text(response_text)
                    if score is not None:
                        validation_note = "Extracted from text (not JSON)"
                
                if score is not None:
                    parts.append(f"Score: {score}/7")
                    if validation_note and validation_note != "Valid":
                        parts.append(f"Note: {validation_note}")
                
                # Add evaluation text
                if "evaluation" in last_json:
                    parts.append(f"Evaluation: {last_json['evaluation']}")
                elif "response" in last_json:
                    parts.append(f"Evaluation: {last_json['response']}")
                
                # Add reasoning if available
                if "reasoning" in last_json:
                    parts.append(f"Reasoning: {last_json['reasoning']}")
                
                if parts:
                    prediction = " | ".join(parts)
                else:
                    # Fallback: use the entire JSON content
                    prediction = json.dumps(last_json)
            else:
                # No JSON found, try to extract score from raw text
                score = _extract_score_from_text(response_text)
                if score is not None:
                    prediction = f"Score: {score}/7 | Raw text evaluation (no JSON found)"
                else:
                    prediction = "Error: No valid JSON or score found in response"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback extraction
            try:
                if msg_history:
                    score = _extract_score_from_text(msg_history[-1]["text"])
                    if score is not None:
                        prediction = f"Score: {score}/7 | Extracted via fallback (error: {e})"
            except Exception:
                pass

        return str(prediction), msg_history
