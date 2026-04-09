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
    Also attempts to extract raw JSON objects as a fallback.
    Includes advanced cleanup for common LLM JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First try to find JSON in <json> tags
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple cleanup strategies
        candidates = [inner]
        
        # Remove markdown code block markers if present
        if inner.startswith("```json"):
            inner_clean = inner[7:].strip()
            if inner_clean.endswith("```"):
                inner_clean = inner_clean[:-3].strip()
            candidates.append(inner_clean)
        elif inner.startswith("```"):
            inner_clean = inner[3:].strip()
            if inner_clean.endswith("```"):
                inner_clean = inner_clean[:-3].strip()
            candidates.append(inner_clean)
        
        for candidate in candidates:
            try:
                results.append(json.loads(candidate))
                break  # Success, move to next block
            except json.JSONDecodeError:
                # Try progressive cleanup strategies
                cleaned = candidate
                
                # Strategy 1: Remove trailing commas before closing braces/brackets
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                try:
                    results.append(json.loads(cleaned))
                    break
                except json.JSONDecodeError:
                    pass
                
                # Strategy 2: Fix single quotes to double quotes (common LLM error)
                # Only replace quotes that are not inside already-quoted strings
                cleaned = re.sub(r"(?<!\\)'", '"', candidate)
                try:
                    results.append(json.loads(cleaned))
                    break
                except json.JSONDecodeError:
                    pass
                
                # Strategy 3: Extract just the JSON object boundaries
                try:
                    # Find the first { and last }
                    first_brace = cleaned.find('{')
                    last_brace = cleaned.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        core_json = cleaned[first_brace:last_brace+1]
                        results.append(json.loads(core_json))
                        break
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Strategy 4: Handle unescaped newlines in strings
                cleaned = re.sub(r'\n(?![^"]*")', '\\n', candidate)
                try:
                    results.append(json.loads(cleaned))
                    break
                except json.JSONDecodeError:
                    continue
    
    # Fallback: try to find raw JSON objects (for cases without tags)
    if not results:
        # Look for JSON-like structures with proper brace matching
        brace_pattern = re.compile(r'\{[\s\S]*?\}', re.DOTALL)
        for match in brace_pattern.finditer(text):
            candidate = match.group()
            # Ensure it has required fields
            if '"reasoning"' in candidate or '"score"' in candidate or '"response"' in candidate:
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    # Try cleanup on raw JSON too
                    try:
                        cleaned = re.sub(r',(\s*[}\]])', r'\1', candidate)
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        continue
    
    return results or None


def _validate_score(score: any) -> str:
    """Validate and normalize the score value.
    
    Args:
        score: The score value from JSON extraction
        
    Returns:
        Validated score as string, or "None" if invalid
    """
    if score is None:
        return "None"
    
    # Handle numeric types
    if isinstance(score, (int, float)):
        # Validate that it's a reasonable score (0-7 for IMO problems)
        if 0 <= score <= 7:
            return str(score)
        return str(score)  # Still return but don't validate range
    
    # Handle string scores
    if isinstance(score, str):
        score = score.strip()
        
        # Check for common non-numeric responses first
        score_lower = score.lower()
        if score_lower in ['none', 'null', 'nan', '', 'n/a', 'undefined']:
            return "None"
        
        # Try to extract numeric portion with more precision
        # Match integers, decimals, and fractions
        numeric_match = re.search(r'[-+]?\d+(?:\.\d+)?(?:/\d+)?', score)
        if numeric_match:
            numeric_str = numeric_match.group()
            # Handle fractions like "3/7"
            if '/' in numeric_str:
                try:
                    parts = numeric_str.split('/')
                    if len(parts) == 2:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        if denominator != 0:
                            return str(numerator / denominator)
                except (ValueError, ZeroDivisionError):
                    pass
            return numeric_str
        
        # Check for spelled-out numbers
        spelled_numbers = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        for word, num in spelled_numbers.items():
            if word in score_lower:
                return num
        
        return score  # Return original if no numeric extraction possible
    
    # Handle boolean (rare but possible)
    if isinstance(score, bool):
        return "1" if score else "0"
    
    return "None"


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
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to evaluate a student's solution to a mathematical problem.

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

Please evaluate the student's answer following these steps:

1. **Understand the Problem**: Briefly summarize what the problem is asking and the key mathematical concepts involved.

2. **Analyze the Official Solution**: Identify the critical steps, key insights, and the final answer in the official solution.

3. **Review Grading Guidelines**: Note the specific criteria and point allocation scheme.

4. **Evaluate Student's Answer**: 
   - Check if the final answer matches the official solution
   - Assess the reasoning and proof structure
   - Identify any gaps, errors, or creative valid approaches
   - Compare against the grading guidelines

5. **Determine Score**: Based on your analysis, assign a score that reflects the student's work.

## Important Scoring Rules

- IMO problems are typically scored from 0 to 7 points
- A score of 7 means a complete, correct solution
- Partial credit (1-6) is given for significant progress with gaps
- Score 0 means no meaningful progress or completely wrong
- Be precise: if the student has the right idea but incomplete proof, assign partial credit
- If the student uses a valid alternative approach not in the official solution, give full credit if correct

Respond ONLY in JSON format with the following schema (no markdown code blocks, just raw JSON inside the tags):
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering steps 1-4 above. Be specific about what the student got right and wrong.",
    "score": "The numerical score assigned to the student's answer (0-7 for IMO problems)",
    "response": "The final score as a number (same as score field, for compatibility)"
}}
</json>

Be thorough in your reasoning and fair in your grading. Ensure your JSON is valid and properly formatted."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        try:
            # Try to extract from the last assistant message
            last_message = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_message)
            
            if extracted:
                result = extracted[-1]
                # Prefer "response" field, fallback to "score" field
                if "response" in result:
                    prediction = _validate_score(result["response"])
                elif "score" in result:
                    prediction = _validate_score(result["score"])
                else:
                    # If neither field exists, log the available fields
                    self.log_fn(f"Warning: No 'response' or 'score' field found. Available fields: {list(result.keys())}")
                    # Try to find any numeric-looking field
                    for key, value in result.items():
                        if isinstance(value, (int, float)) or (isinstance(value, str) and any(c.isdigit() for c in value)):
                            prediction = _validate_score(value)
                            self.log_fn(f"Using field '{key}' with value '{value}' as score")
                            break
            else:
                self.log_fn("Warning: No JSON found in response, attempting fallback extraction")
                # Fallback: try to find any number in the response that looks like a score
                score_patterns = [
                    r'["\']score["\']\s*:\s*([0-7])',
                    r'["\']response["\']\s*:\s*([0-7])',
                    r'score\s*[=:]\s*([0-7])',
                    r'(?:score|grade|points?)\s*(?:is|of|=|:)\s*([0-7])',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, last_message, re.IGNORECASE)
                    if match:
                        prediction = match.group(1)
                        self.log_fn(f"Fallback extraction found score: {prediction}")
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
