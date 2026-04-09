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
    Also handles nested JSON objects within the content.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks within the content
            if "```json" in inner or "```" in inner:
                code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', inner, re.DOTALL)
                if code_block_match:
                    try:
                        results.append(json.loads(code_block_match.group(1).strip()))
                        continue
                    except json.JSONDecodeError:
                        pass
            # Try to find JSON object boundaries by brace counting
            try:
                # Find the first '{' and last '}'
                first_brace = inner.find('{')
                last_brace = inner.rfind('}')
                if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                    results.append(json.loads(inner[first_brace:last_brace+1]))
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
    Handles various edge cases and normalizes common variations.
    """
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    
    # Handle common variations of "None" or empty responses
    if prediction.lower() in ["none", "null", "n/a", "na", "-", "--", "..."]:
        return "None"
    
    # Extract expected score patterns from grading guidelines
    # Common IMO patterns: "7" (full score), "0" (no score), "1-6" (partial)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # IMO-style 0-7 scoring
        # Look for standalone digits first
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Look for digits in quotes or parentheses
        match = re.search(r'["\'\(]([0-7])["\'\)]', prediction)
        if match:
            return match.group(1)
    
    # Check for "Correct"/"Incorrect" format
    if "correct" in grading_guidelines.lower() or "incorrect" in grading_guidelines.lower():
        pred_lower = prediction.lower()
        # Check for "Correct" variations
        if any(word in pred_lower for word in ["correct", "right", "true", "valid", "yes"]):
            # Make sure it's not negated
            if not any(neg in pred_lower for neg in ["not correct", "incorrect", "not right", "not valid"]):
                return "Correct"
        # Check for "Incorrect" variations
        if any(word in pred_lower for word in ["incorrect", "wrong", "false", "invalid", "no", "error"]):
            return "Incorrect"
    
    # Handle percentage-based scoring (0-100%)
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', prediction)
    if pct_match:
        pct = float(pct_match.group(1))
        # Convert percentage to IMO 0-7 scale if guidelines suggest it
        if re.search(r'\b[0-7]\b', grading_guidelines):
            # Map 0-100% to 0-7
            score = round(pct / 100 * 7)
            return str(min(7, max(0, score)))
        return f"{pct}%"
    
    # Handle fraction-based scoring (e.g., "3/7", "1/2")
    frac_match = re.search(r'(\d+)\s*/\s*(\d+)', prediction)
    if frac_match:
        numerator = int(frac_match.group(1))
        denominator = int(frac_match.group(2))
        if denominator > 0:
            # If guidelines use 0-7 scale, convert the fraction
            if re.search(r'\b[0-7]\b', grading_guidelines) and denominator == 7:
                return str(min(7, max(0, numerator)))
            return f"{numerator}/{denominator}"
    
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

IMPORTANT: The "response" field must contain ONLY the grade/score (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines. Do not add explanations or extra text in this field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"]
        extraction_method = "none"
        
        try:
            # First try: extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                extraction_method = "json_tags"
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
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                else:
                    # If no recognized field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str):
                            prediction = value
                            break
                    else:
                        # If no string values, use the whole JSON as string
                        prediction = json.dumps(last_json)
            
            # Second try: fallback extraction for non-tagged JSON
            if prediction == "None":
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    extraction_method = "fallback"
                    if "response" in fallback:
                        prediction = fallback["response"]
                    elif "grade" in fallback:
                        prediction = fallback["grade"]
                    elif "score" in fallback:
                        prediction = fallback["score"]
                    elif "answer" in fallback:
                        prediction = fallback["answer"]
                    elif "evaluation" in fallback:
                        prediction = fallback["evaluation"]
                    elif "result" in fallback:
                        prediction = fallback["result"]
                    else:
                        # If no recognized field, use the first string value found
                        for key, value in fallback.items():
                            if isinstance(value, str):
                                prediction = value
                                break
                        else:
                            prediction = json.dumps(fallback)
                    self.log_fn(f"Used fallback JSON extraction: {prediction}")
            
            # Third try: direct pattern matching in the text
            if prediction == "None":
                # Look for common patterns like "Score: 5" or "Grade: 7"
                score_patterns = [
                    r'[Ss]core\s*[:=]\s*["\']?([0-7])["\']?',
                    r'[Gg]rade\s*[:=]\s*["\']?([0-7])["\']?',
                    r'[Ff]inal\s+(?:score|grade)\s*[:=]\s*["\']?([0-7])["\']?',
                    r'[Rr]esponse\s*[:=]\s*["\']?([0-7])["\']?',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, last_text)
                    if match:
                        prediction = match.group(1)
                        extraction_method = "pattern_match"
                        self.log_fn(f"Used pattern matching extraction: {prediction}")
                        break
            
            # Validate and normalize the prediction
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            
            if extraction_method != "none":
                self.log_fn(f"Extraction method: {extraction_method}, prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
