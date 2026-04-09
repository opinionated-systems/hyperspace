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


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text with improved patterns."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with quotes - prioritize these
    patterns = [
        (r'"response"\s*:\s*"(correct|incorrect|partial)"', 1),  # Exact JSON format
        (r'"response"\s*:\s*\'(correct|incorrect|partial)\'', 1),  # Single quotes
        (r'classification[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'classify[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'response[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'grade[\s:]+"?(correct|incorrect|partial)"?', 2),
        (r'"?(correct|incorrect|partial)"?\s*is\s*the\s*correct\s*classification', 2),
        (r'the\s*answer\s*is\s*"?(correct|incorrect|partial)"?', 2),
        (r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial)"?', 2),
        (r'\b(correct|incorrect|partial)\b[.!]?\s*$', 3),  # At end of text
        (r'\b(correct|incorrect|partial)\b\s*\n', 3),
    ]
    
    results = []
    for pattern, priority in patterns:
        match = re.search(pattern, text_lower)
        if match:
            results.append((match.group(1).lower(), priority, match.start()))
    
    # Sort by priority (lower is better), then by position (earlier is better)
    if results:
        results.sort(key=lambda x: (x[1], x[2]))
        return results[0][0]
    
    # Count occurrences of each label (excluding "not correct" etc)
    # Remove common negation patterns first
    cleaned_text = re.sub(r'\bnot\s+correct\b', '', text_lower)
    cleaned_text = re.sub(r'\bnot\s+partial\b', '', cleaned_text)
    cleaned_text = re.sub(r'\bnot\s+incorrect\b', '', cleaned_text)
    
    correct_count = len(re.findall(r'\bcorrect\b', cleaned_text))
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
        if partial_count > 0 and correct_count == 0 and incorrect_count == 0 and almost_count == 0:
            return "partial"
        if almost_count > 0 and correct_count == 0 and incorrect_count == 0 and partial_count == 0:
            return "partial"  # Treat "almost" as "partial"
    
    # Check for majority
    counts = [("correct", correct_count), ("incorrect", incorrect_count), 
              ("partial", partial_count), ("almost", almost_count)]
    counts.sort(key=lambda x: x[1], reverse=True)
    if counts[0][1] > counts[1][1]:
        result = counts[0][0]
        if result == "almost":
            return "partial"
        return result
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the three valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Handle "almost" as "partial"
    if pred_str == "almost":
        return "partial"
    
    # Check for partial first (more specific)
    if "partial" in pred_str or "almost" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong/false (but not "not incorrect")
    if "incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str:
        # Make sure it's not a negation like "not wrong"
        if "not wrong" not in pred_str and "not incorrect" not in pred_str:
            return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Default fallback - be more lenient, default to partial if there's any valid content
    return "partial"


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
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's answer to an IMO-level problem.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines (CRITICAL - FOLLOW THESE EXACTLY):
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Guidelines:

**"correct"** - The student provided a complete, correct solution with proper reasoning. The answer:
- Arrives at the correct final answer through valid mathematical reasoning
- Contains a complete proof with all necessary steps
- May have minor notation issues that don't affect mathematical correctness
- Awarded full points (6-7 points)

**"incorrect"** - The student's answer is wrong or fundamentally flawed. The answer:
- Has an incorrect final answer with no valid path to solution
- Contains critical logical or mathematical errors that invalidate the proof
- Shows no meaningful progress toward the solution (no correct lemmas, no valid intermediate results)
- Is essentially empty or completely off-track
- Awarded 0 points

**"partial"** - The student made MEANINGFUL PROGRESS but the solution is incomplete. The answer:
- Contains valid and significant progress toward the solution
- Has correct lemmas, propositions, or intermediate steps (even if final answer is wrong)
- Shows understanding of key concepts and correct approach
- May have gaps in the proof or incomplete execution
- Awarded 1-5 points (partial credit)

## Decision Framework (FOLLOW IN ORDER):

Step 1: READ THE GRADING GUIDELINES CAREFULLY
- If guidelines mention "Partial" criteria and the student meets ANY of them → "partial"
- If guidelines mention "Almost" criteria and the student meets them → "partial"
- The grading guidelines are the PRIMARY source for determining credit

Step 2: Check for CORRECT classification (ALL must be true):
- Final answer is mathematically correct
- Proof is complete with valid reasoning throughout
- No significant gaps in logic
If ALL true → "correct"

Step 3: Check for PARTIAL classification (ANY of these indicates partial):
- Student proved any correct lemma or intermediate result
- Student identified the right approach or key insight
- Student made significant progress even if final answer is wrong
- Student has correct setup and valid initial steps
- Student demonstrated understanding of key concepts
If ANY true → "partial"

Step 4: Only classify as INCORRECT if:
- The answer is fundamentally wrong AND shows no meaningful progress
- No correct lemmas, no valid intermediate results
- The approach is completely off-track

## IMPORTANT REMINDERS:
- "Partial" is for solutions with MEANINGFUL PROGRESS - be generous in recognizing valid steps
- An incomplete proof with correct insights deserves "partial", not "incorrect"
- A complete-looking proof with critical errors may still be "incorrect"
- When in doubt between "partial" and "incorrect", choose "partial" if any valid mathematical work is present

## Your Task:
Analyze the student's answer and classify into EXACTLY ONE category: "correct", "incorrect", or "partial".

You MUST respond with a JSON object in this exact format:
<json>
{{
    "response": "correct"
}}
</json>

OR

<json>
{{
    "response": "incorrect"
}}
</json>

OR

<json>
{{
    "response": "partial"
}}
</json>

Replace the value with your classification. Do not include any other text outside the JSON block."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "partial"  # Default fallback - be more lenient
        try:
            response_text = msg_history[-1]["text"]
            
            # First try: extract from <json> tags
            extracted = _extract_jsons(response_text)
            
            if extracted and isinstance(extracted, list) and len(extracted) > 0:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    prediction = _normalize_prediction(last_json["response"])
                else:
                    self.log_fn(f"JSON found but no 'response' field: {last_json}")
                    # Try direct extraction as fallback
                    direct = _extract_response_direct(response_text)
                    if direct:
                        prediction = _normalize_prediction(direct)
            else:
                # Try direct extraction
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = _normalize_prediction(direct)
                else:
                    self.log_fn(f"No valid JSON found in response: {response_text[:200]}...")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
