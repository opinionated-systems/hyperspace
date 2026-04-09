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
    Includes enhanced cleaning for common LLM JSON errors and improved
    handling of nested structures and markdown code blocks.
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
        
        # Try to parse with progressive cleaning
        for attempt in range(5):
            try:
                results.append(json.loads(inner))
                break
            except json.JSONDecodeError:
                if attempt == 0:
                    # Remove trailing commas before closing braces/brackets
                    inner = re.sub(r',(\s*[}\]])', r'\1', inner)
                elif attempt == 1:
                    # Fix unescaped newlines in strings
                    inner = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', inner)
                elif attempt == 2:
                    # Remove comments and fix quotes
                    inner = re.sub(r'//.*?\n', '\n', inner)
                    inner = re.sub(r'/\*.*?\*/', '', inner, flags=re.DOTALL)
                elif attempt == 3:
                    # Fix common quote escaping issues
                    inner = re.sub(r'(?<!\\)"', r'\"', inner)
                    inner = inner.replace('\\"', '"')
                else:
                    # Final attempt: try to extract just the object structure
                    try:
                        # Find the first { and last }
                        first_brace = inner.find('{')
                        last_brace = inner.rfind('}')
                        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                            inner = inner[first_brace:last_brace+1]
                            results.append(json.loads(inner))
                    except json.JSONDecodeError:
                        continue
    
    # Second: try to find JSON in markdown code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            candidate = match.strip()
            if candidate.startswith('{') and candidate.endswith('}'):
                for attempt in range(3):
                    try:
                        parsed = json.loads(candidate)
                        if any(key in parsed for key in ['reasoning', 'score', 'response']):
                            results.append(parsed)
                            break
                    except json.JSONDecodeError:
                        if attempt == 0:
                            candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
                        elif attempt == 1:
                            candidate = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', candidate)
                        else:
                            continue
    
    # Fallback: try to find raw JSON objects (for cases without tags)
    if not results:
        # Look for JSON-like structures with nested brace handling
        # Use a more sophisticated approach to handle nested braces
        depth = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx != -1:
                    candidate = text[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        # Ensure it has required fields for our use case
                        if any(key in parsed for key in ['reasoning', 'score', 'response']):
                            results.append(parsed)
                    except json.JSONDecodeError:
                        # Try cleaning the candidate with progressive attempts
                        cleaned = candidate
                        for _ in range(3):
                            try:
                                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                                cleaned = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', cleaned)
                                parsed = json.loads(cleaned)
                                if any(key in parsed for key in ['reasoning', 'score', 'response']):
                                    results.append(parsed)
                                    break
                            except json.JSONDecodeError:
                                continue
                    start_idx = -1
    
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
        # Handle special float values
        if isinstance(score, float):
            if score != score:  # NaN check
                return "None"
            if score == float('inf') or score == float('-inf'):
                return "None"
        return str(int(score)) if isinstance(score, int) or score == int(score) else str(score)
    
    # Handle string scores
    if isinstance(score, str):
        score = score.strip()
        # Check for common non-numeric responses first
        lower_score = score.lower()
        if lower_score in ['none', 'null', 'nan', '', 'undefined', 'n/a', 'na']:
            return "None"
        
        # Try to extract numeric portion with better pattern
        # Handle fractions like "7/7" or "3/5"
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', score)
        if fraction_match:
            numerator = int(fraction_match.group(1))
            return str(numerator)
        
        # Handle numeric extraction
        numeric_match = re.search(r'[-+]?\d+\.?\d*', score)
        if numeric_match:
            num_str = numeric_match.group()
            try:
                # Try as int first
                if '.' not in num_str:
                    return str(int(num_str))
                else:
                    val = float(num_str)
                    return str(int(val)) if val == int(val) else str(val)
            except ValueError:
                return num_str
        
        # If no numeric portion found but string isn't empty, return as-is
        if score:
            return score
    
    # Handle boolean (shouldn't happen but be safe)
    if isinstance(score, bool):
        return "1" if score else "0"
    
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {"calls": 0, "json_extracted": 0, "json_failed": 0, "score_valid": 0, "score_invalid": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with structured reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["calls"] += 1
        
        # Extract fields for structured prompting with validation
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log input sizes for debugging
        self.log_fn(f"Input sizes - problem: {len(problem)}, solution: {len(solution)}, "
                   f"guidelines: {len(grading_guidelines)}, answer: {len(student_answer)}")

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

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering steps 1-4 above",
    "score": "The numerical score assigned to the student's answer",
    "response": "The final score (same as score field, for compatibility)"
}}
</json>

Be thorough in your reasoning and fair in your grading."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error in LLM call: {e}")
            return "None", [{"role": "error", "text": str(e)}]

        # Extract prediction from JSON
        prediction = "None"
        try:
            if not msg_history:
                self.log_fn("Warning: Empty message history from LLM")
                return "None", msg_history
                
            last_msg = msg_history[-1]
            if "text" not in last_msg:
                self.log_fn(f"Warning: Last message has no 'text' field: {last_msg.keys()}")
                return "None", msg_history
                
            extracted = _extract_jsons(last_msg["text"])
            if extracted:
                self.stats["json_extracted"] += 1
                result = extracted[-1]
                # Prefer "response" field, fallback to "score" field
                if "response" in result:
                    prediction = _validate_score(result["response"])
                elif "score" in result:
                    prediction = _validate_score(result["score"])
                else:
                    # If neither field exists, log the available fields
                    self.log_fn(f"Warning: No 'response' or 'score' field found. Available fields: {list(result.keys())}")
                    self.stats["score_invalid"] += 1
                
                # Track score validation
                if prediction != "None":
                    self.stats["score_valid"] += 1
                else:
                    self.stats["score_invalid"] += 1
            else:
                self.stats["json_failed"] += 1
                self.log_fn("Warning: No JSON found in response")
                # Try to extract any numeric value from the raw text as last resort
                numbers = re.findall(r'\b\d+\b', last_msg.get("text", ""))
                if numbers:
                    self.log_fn(f"Fallback: Found numbers in response: {numbers}")
        except Exception as e:
            self.stats["json_failed"] += 1
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
    
    def get_stats(self) -> dict:
        """Return current statistics about agent performance."""
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self.stats = {"calls": 0, "json_extracted": 0, "json_failed": 0, "score_valid": 0, "score_invalid": 0}
