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

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles nested JSON and common formatting issues.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try cleaning common issues
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
        # Fix unescaped newlines in strings
        cleaned = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', cleaned)
        
        try:
            results.append(json.loads(cleaned))
        except json.JSONDecodeError:
            # Try extracting just the outermost JSON object
            try:
                brace_start = cleaned.find('{')
                brace_end = cleaned.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    results.append(json.loads(cleaned[brace_start:brace_end+1]))
            except (json.JSONDecodeError, ValueError):
                continue
    return results or None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks or raw JSON objects.
    """
    patterns = [
        r'```json\s*(.*?)\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',  # Generic code blocks with JSON
        r'\{\s*"reasoning".*?\}',  # Raw JSON with reasoning field
        r'\{\s*"response".*?\}',   # Raw JSON with response field
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return None


def _validate_and_normalize_prediction(prediction: str, grading_guidelines: str) -> str:
    """Validate and normalize the prediction based on grading guidelines.
    
    Ensures the prediction matches expected format from grading guidelines.
    Uses multi-layer validation to extract the most accurate grade.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    
    # Layer 1: Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring - look for explicit score mentions
        # First try to find standalone digits
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        
        # Look for patterns like "score: 5", "grade: 3", "points: 7"
        score_patterns = [
            r'(?:score|grade|points|mark)[\s:]*([0-7])',
            r'(?:worth|value)[\s:]*([0-7])',
            r'(?:\band\s+)?(?:score|grade)[\s:]*(?:of\s+)?([0-7])',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1)
    
    # Layer 2: Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        pred_lower = prediction.lower()
        # Check for explicit "correct" without "incorrect" nearby
        if re.search(r'\bcorrect\b', pred_lower) and not re.search(r'\bincorrect\b', pred_lower):
            return "Correct"
        elif re.search(r'\bincorrect\b|\bwrong\b|\berror\b', pred_lower):
            return "Incorrect"
    
    # Layer 3: Check for boolean-like responses
    pred_lower = prediction.lower()
    if pred_lower in ("true", "yes", "pass", "passed", "full marks"):
        return "Correct"
    if pred_lower in ("false", "no", "fail", "failed", "no marks"):
        return "Incorrect"
    
    # Layer 4: Check for fraction patterns (e.g., "1/7", "3/7")
    fraction_match = re.search(r'\b([0-7])\s*/\s*7\b', prediction)
    if fraction_match:
        return fraction_match.group(1)
    
    return prediction


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer to a mathematical problem with precision and consistency.

## Domain
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

Please evaluate the student's answer following this structured approach:

### Step 1: Problem Analysis
- Summarize what the problem is asking
- Identify key mathematical concepts and required techniques
- Note any critical assumptions or constraints

### Step 2: Solution Mapping
- Break down the official solution into key milestones
- Identify which milestones are essential vs. optional
- Note common alternative valid approaches

### Step 3: Student Answer Evaluation
- Map the student's work to solution milestones
- Identify correct steps with clear justification
- Flag any errors, gaps, or logical flaws
- Check for partial credit opportunities per guidelines

### Step 4: Grade Determination
- Apply grading guidelines systematically
- Consider: correctness, completeness, and clarity
- Assign the most appropriate grade/score

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "response" field must contain ONLY the grade/score (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines.
2. Do NOT add explanations, quotes, or extra text in the "response" field.
3. For IMO problems with 0-7 scoring, the response should be a single digit from 0-7.
4. For binary grading, use exactly "Correct" or "Incorrect" (case-sensitive).
5. Ensure your JSON is valid - no trailing commas, properly escaped quotes and newlines."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"]
        try:
            extracted = _extract_jsons(last_text)
            if extracted:
                # Try to get response field, fall back to other common fields
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                else:
                    # If no recognized field, use the whole JSON as string
                    prediction = json.dumps(last_json)
            else:
                # Try fallback extraction for non-tagged JSON
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    if "response" in fallback:
                        prediction = fallback["response"]
                    elif "grade" in fallback:
                        prediction = fallback["grade"]
                    elif "score" in fallback:
                        prediction = fallback["score"]
                    elif "answer" in fallback:
                        prediction = fallback["answer"]
                    else:
                        prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
            
            # Validate and normalize the prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
