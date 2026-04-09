"""
Task agent: solves a given task with enhanced reasoning for IMO grading.

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
    Also handles markdown code blocks (```json) as fallback.
    Includes additional heuristics for malformed JSON.
    """
    results = []
    search_from = 0
    
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the content if it's wrapped
        try:
            # Look for JSON object boundaries
            json_start = inner.find("{")
            json_end = inner.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON issues
        try:
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Fix single quotes to double quotes (common LLM error)
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
    
    # Fallback: try markdown code blocks if no results yet
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = "```"
            else:
                end_marker = "```"
            start += len("```json") if text[start:start+7] == "```json" else 3
            end = text.find(end_marker, start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            
            # Try direct parsing
            try:
                results.append(json.loads(inner))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try with common fixes
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try extracting just the object
            try:
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

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

Please evaluate the student's answer following this structured approach:

### Phase 1: Problem Analysis
- Identify the key mathematical concepts and techniques required
- Note the critical steps that must be present for a complete solution
- Understand what constitutes a correct proof/answer

### Phase 2: Solution Mapping
- Map the student's answer against the official solution
- Identify which steps the student completed correctly
- Note any missing, incorrect, or incomplete steps
- Recognize alternative valid approaches (not just the official one)

### Phase 3: Error Analysis
- Categorize any errors: conceptual, computational, logical, or presentation
- Determine if errors are minor (deduct 0-1 points) or major (deduct 2+ points)
- Check for partial credit situations where student made progress but didn't complete

### Phase 4: Score Assignment
- Apply the grading guidelines strictly but fairly
- Award partial credit for meaningful progress toward solution
- Deduct points proportionally to the severity of errors/omissions
- Ensure consistency with IMO grading standards (0-7 scale)

### Scoring Reference:
- 7: Complete, correct solution
- 6: Minor error or omission (1 point deduction)
- 5: Significant progress with notable gaps (2 point deduction)
- 3-4: Partial solution with some correct elements
- 1-2: Minimal progress or significant errors
- 0: No meaningful progress or completely wrong approach

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all phases above",
    "evaluation": "Concise summary: what was correct, what was wrong, key issues identified",
    "response": "The final score as a single number (0-7)"
}}
</json>

IMPORTANT: The "response" field must contain ONLY a single integer from 0 to 7 representing the final score. Do not include any explanation in this field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        
        try:
            # Try to extract from the assistant's response
            assistant_text = ""
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    assistant_text = msg.get("text", "")
                    break
            
            if not assistant_text:
                self.log_fn("Warning: No assistant response found in message history")
            
            extracted = _extract_jsons(assistant_text)
            if extracted:
                # Try to get response from the last valid JSON block
                last_json = extracted[-1]
                self.log_fn(f"Extracted JSON with keys: {list(last_json.keys())}")
                
                # Priority order for score extraction
                score_fields = ["response", "score", "grade", "evaluation", "result", "answer"]
                for field in score_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        extraction_method = f"json_{field}"
                        break
                else:
                    # No recognized field found - try to find numeric value in any field
                    self.log_fn(f"Warning: JSON extracted but no recognized field found. Keys: {list(last_json.keys())}")
                    for key, value in last_json.items():
                        if isinstance(value, (int, float)):
                            prediction = str(value)
                            extraction_method = f"json_numeric_{key}"
                            break
                        elif isinstance(value, str):
                            # Try to extract number from string
                            nums = re.findall(r'\b([0-7])\b', value)
                            if nums:
                                prediction = nums[0]
                                extraction_method = f"json_extracted_{key}"
                                break
                    else:
                        # Use first value as last resort
                        if last_json:
                            first_value = list(last_json.values())[0]
                            if isinstance(first_value, (str, int, float)):
                                prediction = str(first_value)
                                extraction_method = "json_first_value"
            else:
                # Fallback: try to extract any numeric value that looks like a score
                # Look for patterns like "score: 5", "grade: 7", "result: 3", etc.
                score_patterns = [
                    r'(?:score|grade|result|answer)[\s:=]+([0-7])\b',
                    r'(?:score|grade|result|answer) is[\s:]+([0-7])\b',
                    r'\bfinal[\s:]+([0-7])\b',
                    r'\b([0-7])\s*(?:points?|/\s*7)\b',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, assistant_text, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        extraction_method = "regex_pattern_fallback"
                        self.log_fn(f"Used regex pattern fallback to extract score: {prediction}")
                        break
                else:
                    # Last resort: any standalone 0-7
                    numbers = re.findall(r'\b([0-7])\b', assistant_text)
                    if numbers:
                        prediction = numbers[-1]  # Last number 0-7 is likely the score
                        extraction_method = "regex_fallback"
                        self.log_fn(f"Used regex fallback to extract score: {prediction}")
                    else:
                        self.log_fn("Warning: Could not extract any valid JSON or numeric score")
                    
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error during prediction extraction: {e}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")

        # Validate and normalize the prediction
        prediction_str = str(prediction).strip()
        # Try to extract just the numeric part if there's extra text
        numeric_match = re.search(r'\b([0-7])\b', prediction_str)
        if numeric_match:
            prediction_str = numeric_match.group(1)
        
        # Log the extraction result for debugging
        self.log_fn(f"Prediction extracted via {extraction_method}: {prediction_str}")
        
        return prediction_str, msg_history
