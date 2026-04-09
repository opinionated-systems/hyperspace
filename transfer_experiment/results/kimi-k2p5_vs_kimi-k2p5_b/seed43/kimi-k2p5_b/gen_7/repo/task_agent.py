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


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using multiple strategies.
    
    Tries multiple extraction methods in order of reliability:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects with "response" field
    4. Relaxed JSON parsing
    """
    text = text.strip()
    
    # Strategy 1: <json>...</json> tags
    json_tag_pattern = r'<json>\s*(.*?)\s*</json>'
    matches = re.findall(json_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: ```json...``` code blocks
    code_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: ```...``` code blocks (without json label)
    code_block_pattern2 = r'```\s*(\{.*?\})\s*```'
    matches = re.findall(code_block_pattern2, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Raw JSON objects with "response" field
    # Use a more careful approach to find JSON objects
    # Look for patterns like {"response": ...}
    json_pattern = r'\{\s*"[^"]+"\s*:[^}]+\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "response" in data:
                return data
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Try to find any JSON-like structure with response field
    # More aggressive pattern matching
    response_pattern = r'"response"\s*:\s*"([^"]+)"'
    match = re.search(response_pattern, text, re.IGNORECASE)
    if match:
        return {"response": match.group(1)}
    
    return None


def _extract_response_direct(text: str) -> str | None:
    """Extract response by looking for keywords directly in text."""
    text_lower = text.lower()
    
    # Look for explicit classification statements with response/classification keywords
    patterns = [
        r'"response"\s*:\s*"(correct|incorrect|partial)"',
        r'"classification"\s*:\s*"(correct|incorrect|partial)"',
        r'"grade"\s*:\s*"(correct|incorrect|partial)"',
        r'"result"\s*:\s*"(correct|incorrect|partial)"',
        r'"answer"\s*:\s*"(correct|incorrect|partial)"',
        r'"outcome"\s*:\s*"(correct|incorrect|partial)"',
        r'"evaluation"\s*:\s*"(correct|incorrect|partial)"',
        r'"verdict"\s*:\s*"(correct|incorrect|partial)"',
        r'classification[\s:]+"?(correct|incorrect|partial)"?',
        r'classify[\s:]+"?(correct|incorrect|partial)"?',
        r'response[\s:]+"?(correct|incorrect|partial)"?',
        r'grade[\s:]+"?(correct|incorrect|partial)"?',
        r'result[\s:]+"?(correct|incorrect|partial)"?',
        r'outcome[\s:]+"?(correct|incorrect|partial)"?',
        r'evaluation[\s:]+"?(correct|incorrect|partial)"?',
        r'verdict[\s:]+"?(correct|incorrect|partial)"?',
        r'"?(correct|incorrect|partial)"?\s*is\s*the\s*correct\s*classification',
        r'the\s*answer\s*is\s*"?(correct|incorrect|partial)"?',
        r'should\s*be\s*classified\s*as\s*"?(correct|incorrect|partial)"?',
        r'classify\s*as\s*"?(correct|incorrect|partial)"?',
        r'classified\s*as\s*"?(correct|incorrect|partial)"?',
        r'is\s*"?(correct|incorrect|partial)"?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = match.group(1).lower().strip()
            if result in ["correct", "incorrect", "partial"]:
                return result
    
    # Count occurrences of each label (only count standalone words)
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    
    # If only one appears, use that
    if correct_count > 0 and incorrect_count == 0 and partial_count == 0:
        return "correct"
    if incorrect_count > 0 and correct_count == 0 and partial_count == 0:
        return "incorrect"
    if partial_count > 0 and correct_count == 0 and incorrect_count == 0:
        return "partial"
    
    # If multiple appear, use the one that appears most
    counts = [("correct", correct_count), ("incorrect", incorrect_count), ("partial", partial_count)]
    counts.sort(key=lambda x: x[1], reverse=True)
    if counts[0][1] > 0:
        return counts[0][0]
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the three valid labels."""
    pred_str = str(prediction).lower().strip()
    
    # Direct match
    if pred_str in ["correct", "incorrect", "partial"]:
        return pred_str
    
    # Check for partial first (more specific)
    if "partial" in pred_str:
        return "partial"
    
    # Check for incorrect/wrong/false
    if "incorrect" in pred_str or "wrong" in pred_str or "false" in pred_str or "error" in pred_str:
        return "incorrect"
    
    # Check for correct (but not incorrect)
    if "correct" in pred_str and "incorrect" not in pred_str:
        return "correct"
    
    # Check for complete/full
    if "complete" in pred_str or "full" in pred_str:
        return "correct"
    
    # Default fallback
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
        
        instruction = f"""You are an expert mathematical olympiad grader. Your task is to evaluate a student's answer to an IMO-level problem.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Guidelines:

**"correct"** - The student provided a complete, correct solution with proper reasoning. The answer:
- Arrives at the correct final answer
- Contains valid mathematical reasoning throughout
- May have minor notation issues or small gaps that don't affect correctness
- Awarded full points (6-7 points)

**"incorrect"** - The student's answer is wrong or fundamentally flawed. The answer:
- Has an incorrect final answer with no valid path to solution
- Contains critical logical or mathematical errors
- Shows no meaningful progress toward the solution
- Is incomplete with no useful partial results
- Awarded 0 points

**"partial"** - The student made meaningful progress but the solution is incomplete. The answer:
- Contains valid and significant progress toward the solution
- Has correct lemmas, propositions, or intermediate steps
- May have the right approach but incomplete execution
- Shows understanding of key concepts even if final answer is wrong or missing
- Awarded 1-5 points (partial credit)

## CRITICAL DECISION FRAMEWORK - FOLLOW THESE STEPS IN ORDER:

**STEP 1: Check Grading Guidelines for Keywords**
The grading guidelines are the MOST IMPORTANT signal. Look for:
- "Partial" mentioned → classify as "partial"
- "Almost" mentioned → classify as "partial" (this means near-complete with minor issues)
- "Full credit" or "Complete" → classify as "correct"
- "No credit" or "0 points" → classify as "incorrect"

**STEP 2: Evaluate for "partial" - BE GENEROUS WITH PARTIAL CREDIT**
Classify as "partial" if ANY of the following are true:
- Student proved at least one correct lemma or intermediate result
- Student identified the right approach or key insight (even if execution failed)
- Student made significant progress toward solution (even if final answer is wrong)
- Student showed understanding of key concepts
- Student has correct equations/setup but failed to complete
- Student's answer is incomplete but contains valid mathematical content

**STEP 3: Distinguish "partial" from "incorrect"**
- "partial": Student made MEANINGFUL progress (valid lemmas, correct approach, significant work)
- "incorrect": No meaningful progress, completely wrong approach, or trivial/incorrect statements only

**STEP 4: Check for "correct"**
Only classify as "correct" if:
- Final answer is completely correct
- Reasoning is valid throughout
- Solution is essentially complete (minor notation issues OK)

## Your Task:
First, think through your analysis step by step. Then classify the student's answer into EXACTLY ONE category: "correct", "incorrect", or "partial".

**IMPORTANT**: When in doubt between "partial" and "incorrect", choose "partial". Partial credit should be awarded for any meaningful mathematical progress.

You MUST respond with a JSON object in this exact format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "correct"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "incorrect"
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain what the student got right, what they got wrong, and why you chose this classification.",
    "response": "partial"
}}
</json>

Replace the "response" value with your classification. Include detailed reasoning in the "reasoning" field. Do not include any other text outside the JSON block."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "incorrect"  # Default fallback
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                if response_text:
                    # First try: extract from JSON
                    json_data = _extract_json_from_text(response_text)
                    
                    if json_data and isinstance(json_data, dict):
                        if "response" in json_data:
                            prediction = _normalize_prediction(json_data["response"])
                        elif "classification" in json_data:
                            prediction = _normalize_prediction(json_data["classification"])
                        elif "grade" in json_data:
                            prediction = _normalize_prediction(json_data["grade"])
                        elif "result" in json_data:
                            prediction = _normalize_prediction(json_data["result"])
                        else:
                            # Try to find any value that looks like our labels
                            for key, value in json_data.items():
                                if isinstance(value, str):
                                    val_lower = value.lower()
                                    if val_lower in ["correct", "incorrect", "partial"]:
                                        prediction = val_lower
                                        break
                                    elif "correct" in val_lower or "partial" in val_lower or "incorrect" in val_lower:
                                        prediction = _normalize_prediction(val_lower)
                                        break
                    else:
                        # Try direct extraction from text
                        direct = _extract_response_direct(response_text)
                        if direct:
                            prediction = direct
                        else:
                            self.log_fn(f"No valid response found in: {response_text[:200]}...")
                else:
                    self.log_fn("Empty response text in msg_history")
            else:
                self.log_fn("Empty msg_history")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(traceback.format_exc())

        return str(prediction), msg_history
