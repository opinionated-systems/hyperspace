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
    Also handles markdown code blocks (```json) as a fallback.
    Includes advanced JSON repair for common LLM output issues.
    """
    results = []
    search_from = 0
    
    def _try_parse_json(inner: str) -> dict | None:
        """Try to parse JSON with multiple repair strategies."""
        inner = inner.strip()
        
        # Strategy 1: Direct parse
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove trailing commas before closing braces/brackets
        try:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', inner)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix unescaped newlines in strings
        try:
            # Replace newlines within string values with escaped newlines
            cleaned = re.sub(r'(?<=")([^"\n]*\n[^"]*)', lambda m: m.group(1).replace('\n', '\\n'), inner)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Extract just the first valid JSON object
        try:
            # Find the first { and last }
            start_idx = inner.find('{')
            end_idx = inner.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return json.loads(inner[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass
        
        return None
    
    # First try <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try markdown code blocks if no results yet
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start += 7
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            
            parsed = _try_parse_json(inner)
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        # Look for patterns like {"score": ...} or {"response": ...}
        json_pattern = re.search(r'\{[^{}]*"(?:score|response|reasoning)"[^{}]*\}', text, re.DOTALL)
        if json_pattern:
            parsed = _try_parse_json(json_pattern.group(0))
            if parsed is not None:
                results.append(parsed)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO). Your task is to evaluate a student's answer based on the problem, official solution, and grading guidelines.

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

Follow this structured evaluation process carefully:

### Step 1: Understanding Check
- Identify the key mathematical concepts required
- Note the critical steps in the official solution
- Understand what constitutes a complete proof

### Step 2: Student's Approach Analysis
- Summarize the student's overall strategy
- Identify which key steps they attempted
- Note any creative or alternative approaches

### Step 3: Error and Gap Identification
- List any mathematical errors (logical, computational, conceptual)
- Identify missing steps or incomplete proofs
- Check if assumptions are properly justified

### Step 4: Partial Credit Assessment
- Map each successful step to the grading guidelines
- Calculate partial credit for incomplete progress
- Note any "nontrivial progress" as per IMO standards

### Step 5: Final Score Determination
- IMO problems are scored 0-7 points
- 7: Complete, correct proof
- 6: Minor flaw in otherwise correct proof
- 5-3: Significant progress with gaps
- 2-1: Nontrivial but insufficient progress
- 0: No meaningful progress or completely wrong

## Response Format

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must be parseable and contain these exact fields:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the 5-step process above. Include specific evidence from the student's answer.",
    "score": 5,
    "response": "5",
    "confidence": "high"
}}
</json>

Field requirements:
- "reasoning": Detailed analysis with specific citations from the student's work
- "score": Integer 0-7 (the numerical score)
- "response": String representation of the score (e.g., "5") - this is the final answer
- "confidence": One of "high", "medium", or "low" indicating your certainty

Important grading principles:
1. Be conservative - only award points for clearly demonstrated progress
2. A complete proof with minor errors gets 6, not 7
3. Significant progress toward solution gets 3-5 points
4. Nontrivial insights without full solution get 1-2 points
5. No meaningful progress or completely wrong gets 0"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved error handling
        prediction = "None"
        confidence = "unknown"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                # Extract confidence if available
                if "confidence" in last_json:
                    confidence = str(last_json["confidence"]).lower()
                
                # Prefer "response" field, fallback to "score" if available
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                else:
                    # If no recognized field, look for numeric values
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)) and 0 <= value <= 7:
                            prediction = str(value)
                            self.log_fn(f"Found numeric value in field '{key}': {prediction}")
                            break
                    else:
                        # If no numeric value found, use the first value
                        prediction = list(last_json.values())[0] if last_json else "None"
                self.log_fn(f"Extracted prediction: {prediction} (confidence: {confidence})")
            else:
                self.log_fn("No JSON found in response, attempting fallback extraction")
                # Fallback: try to extract any number that looks like a score
                text = msg_history[-1]["text"]
                # Look for patterns like "score": 5 or "response": "7"
                score_match = re.search(r'["\']?(?:score|response)["\']?\s*[:=]\s*["\']?(\d+)["\']?', text, re.IGNORECASE)
                if score_match:
                    prediction = score_match.group(1)
                    self.log_fn(f"Fallback extraction found: {prediction}")
                else:
                    # Last resort: find any standalone digit 0-7 that might be a score
                    # Look for patterns like "Score: 5" or "The score is 7"
                    standalone_match = re.search(r'(?:score|grade|points?|mark)(?:\s*(?:is|:|=)\s*)([0-7])', text, re.IGNORECASE)
                    if standalone_match:
                        prediction = standalone_match.group(1)
                        self.log_fn(f"Standalone extraction found: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction is a reasonable score (0-7 for IMO)
        try:
            pred_val = int(str(prediction))
            if pred_val < 0 or pred_val > 7:
                self.log_fn(f"Warning: extracted score {pred_val} outside IMO range [0,7], clamping")
                pred_val = max(0, min(7, pred_val))
                prediction = str(pred_val)
        except (ValueError, TypeError):
            self.log_fn(f"Warning: could not convert prediction '{prediction}' to integer, defaulting to 0")
            prediction = "0"

        return str(prediction), msg_history
