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
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.DOTALL)
            if code_block_match:
                try:
                    results.append(json.loads(code_block_match.group(1)))
                except json.JSONDecodeError:
                    pass
            # Try to find JSON object directly with brace balancing
            else:
                json_obj = _extract_json_with_brace_balancing(inner)
                if json_obj:
                    results.append(json_obj)
    return results or None


def _extract_json_with_brace_balancing(text: str) -> dict | None:
    """Extract a JSON object from text using brace balancing.
    
    This handles cases where JSON might be embedded in other text without
    proper code block delimiters.
    """
    # Find the first opening brace
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Use brace counting to find the matching closing brace
    brace_count = 0
    end_idx = start_idx
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
    
    if brace_count == 0:
        try:
            return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError:
            pass
    
    return None


def _extract_json_fallback(text: str) -> dict | None:
    """Fallback JSON extraction for non-tagged JSON or markdown code blocks.

    Tries to find JSON in markdown code blocks.
    """
    # Try markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
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
    """Validate and normalize the prediction based on grading guidelines."""
    if not prediction or prediction.strip() == "":
        return "None"
    
    prediction = prediction.strip()
    pred_lower = prediction.lower()
    
    # IMO-style 0-7 scoring (most common for IMO problems)
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # Look for standalone digit 0-7
        match = re.search(r'\b([0-7])\b', prediction)
        if match:
            return match.group(1)
        # Look for patterns like "score: 5" or "grade: 3"
        score_match = re.search(r'(?:score|grade|points?)\s*[:=]?\s*([0-7])\b', pred_lower)
        if score_match:
            return score_match.group(1)
    
    # Correct/Incorrect format
    if "correct" in grading_guidelines.lower():
        # Check for explicit incorrect/wrong first (more specific)
        if re.search(r'\b(incorrect|wrong|false|error)\b', pred_lower):
            return "Incorrect"
        elif re.search(r'\b(correct|right|true|valid)\b', pred_lower):
            return "Correct"
    
    # Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\byes\b', pred_lower):
            return "Yes"
        elif re.search(r'\bno\b', pred_lower):
            return "No"
    
    # Partial credit patterns (e.g., "partial", "half", "some")
    if re.search(r'\bpartial\b', pred_lower):
        # Try to extract a numeric score if present
        num_match = re.search(r'\b([0-9]+)\b', prediction)
        if num_match:
            return num_match.group(1)
    
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

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis following the 4 steps above...",
    "response": "The final grade/score as specified in the grading guidelines"
}}
</json>

CRITICAL INSTRUCTIONS:
1. The "response" field must contain ONLY the grade/score value (e.g., "7", "2", "0", "Correct", "Incorrect", etc.) exactly as specified in the grading guidelines.
2. Do NOT add explanations, reasoning, or extra text in the "response" field.
3. Do NOT use markdown formatting, code blocks, or any other formatting inside the JSON.
4. The JSON must be valid and properly escaped.
5. Wrap your entire JSON response in <json>...</json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        last_text = msg_history[-1]["text"] if msg_history else ""
        
        try:
            # Try to extract from <json> tags
            extracted = _extract_jsons(last_text)
            if extracted:
                last_json = extracted[-1]
                # Try multiple possible keys for the prediction
                for key in ["response", "grade", "score", "answer", "prediction", "result"]:
                    if key in last_json:
                        prediction = str(last_json[key])
                        break
            else:
                # Fallback: try markdown code blocks
                fallback = _extract_json_fallback(last_text)
                if fallback:
                    for key in ["response", "grade", "score", "answer", "prediction", "result"]:
                        if key in fallback:
                            prediction = str(fallback[key])
                            break
            
            # If still no prediction, try direct extraction from text
            if prediction == "None":
                prediction = _extract_prediction_from_text(last_text, grading_guidelines)
            
            # Validate and normalize
            prediction = _validate_and_normalize_prediction(prediction, grading_guidelines)
            self.log_fn(f"Final prediction: {prediction}")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history


def _extract_prediction_from_text(text: str, grading_guidelines: str) -> str:
    """Extract prediction directly from text when JSON parsing fails.
    
    This is a last-resort fallback that looks for common patterns.
    """
    text_lower = text.lower()
    
    # IMO-style 0-7 scoring
    if re.search(r'\b[0-7]\b', grading_guidelines):
        # Look for patterns like "the score is 5" or "grade: 3"
        patterns = [
            r'(?:score|grade|points?|mark)s?\s*(?:is|of|=|:)\s*([0-7])\b',
            r'(?:student|answer)\s+(?:gets?|receives?|earns?)\s+([0-7])\b',
            r'(?:assign|give|award)\s+(?:a?\s*)?(?:score|grade|points?)?\s*(?:of\s*)?([0-7])\b',
            r'\bfinal\s+(?:score|grade|points?)\s*(?:is|:|=)?\s*([0-7])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
    
    # Correct/Incorrect format
    if "correct" in grading_guidelines.lower():
        if re.search(r'\b(incorrect|wrong|false|error)\b', text_lower):
            return "Incorrect"
        elif re.search(r'\b(correct|right|true|valid)\b', text_lower):
            return "Correct"
    
    # Yes/No format
    if re.search(r'\b(yes|no)\b', grading_guidelines, re.IGNORECASE):
        if re.search(r'\byes\b', text_lower):
            return "Yes"
        elif re.search(r'\bno\b', text_lower):
            return "No"
    
    return "None"
