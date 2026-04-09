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
    Also handles markdown code blocks and bare JSON objects.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        json_code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        for block in json_code_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', block.strip())
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # If still no results, try to find bare JSON objects
    if not results:
        # Look for JSON-like structures with score field
        json_pattern = re.search(r'\{\s*"[^"]+":\s*[^}]+\}', text, re.DOTALL)
        if json_pattern:
            try:
                results.append(json.loads(json_pattern.group()))
            except json.JSONDecodeError:
                pass
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert {domain} grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a problem and assign a score from 0 to 7 points (IMO scoring).

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Evaluation Instructions:

1. **Analyze the Problem**: Understand what the problem is asking and what constitutes a complete solution.

2. **Review the Official Solution**: Note the key steps, techniques, and insights required.

3. **Evaluate the Student's Answer**: 
   - Check if the student understood the problem correctly
   - Identify which key steps the student completed
   - Note any errors, gaps, or incorrect reasoning
   - Check for partial progress that deserves partial credit

4. **Assign Score (0-7)**:
   - 7: Complete, correct solution
   - 6: Minor flaw in an otherwise correct solution
   - 5-3: Partial progress with varying degrees of completeness
   - 2-1: Significant progress but major gaps
   - 0: No meaningful progress or completely wrong

5. **Provide Reasoning**: Explain your scoring decision with specific references to the student's work.

## IMPORTANT: Response Format

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must contain exactly these three fields:

<json>
{{
    "thinking": "Your detailed chain-of-thought analysis here...",
    "score": 7,
    "reasoning": "Explanation of why this score was assigned, referencing specific parts of the student's solution"
}}
</json>

Requirements:
- The "score" field MUST be an integer from 0 to 7 (inclusive)
- Do not include any text outside the <json> tags
- Ensure the JSON is valid (no trailing commas, proper quotes)
- The score must reflect the IMO grading standards described above"""

    def _extract_score(self, data: dict) -> str:
        """Extract the score from the JSON response, with validation.
        
        Handles various field names and formats for maximum compatibility.
        """
        # Priority order for score fields
        score_fields = ["score", "points", "grade", "mark", "rating", "value", "result"]
        
        for field in score_fields:
            if field in data:
                score = data[field]
                # Handle different types
                if isinstance(score, (int, float)):
                    score_int = int(score)
                    if 0 <= score_int <= 7:
                        return str(score_int)
                elif isinstance(score, str):
                    # Try to parse numeric string
                    try:
                        score_int = int(float(score))
                        if 0 <= score_int <= 7:
                            return str(score_int)
                    except (ValueError, TypeError):
                        # Try to find a number 0-7 in the string
                        numbers = re.findall(r'\b([0-7])\b', score)
                        if numbers:
                            return numbers[0]
        
        # Fallback: try to extract from response field for backward compatibility
        if "response" in data:
            resp = data["response"]
            if isinstance(resp, (int, float)):
                score_int = int(resp)
                if 0 <= score_int <= 7:
                    return str(score_int)
            elif isinstance(resp, str):
                # Try to find a number 0-7 in the response
                numbers = re.findall(r'\b([0-7])\b', resp)
                if numbers:
                    return numbers[-1]
        
        # Last resort: search entire JSON for any number 0-7 in likely score contexts
        json_str = json.dumps(data)
        # Look for patterns like "score": 5 or "points": 3
        score_patterns = re.findall(r'"(?:score|points|grade|mark|rating)":\s*(\d)', json_str, re.IGNORECASE)
        if score_patterns:
            for p in score_patterns:
                if 0 <= int(p) <= 7:
                    return p
        
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            # First try JSON extraction
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                prediction = self._extract_score(extracted[-1])
                self.log_fn(f"Extracted score from JSON: {prediction}")
            
            # If JSON extraction failed, try direct pattern matching
            if prediction == "None":
                # Look for explicit score mentions
                score_patterns = [
                    r'["\']score["\']\s*:\s*(\d)',
                    r'["\']points["\']\s*:\s*(\d)',
                    r'["\']grade["\']\s*:\s*(\d)',
                    r'score\s*(?:of|is|=|:)\s*(\d)',
                    r'(?:assign|give|award)\s+(\d)\s*(?:points?)?',
                    r'(?:score|grade|rating)\s+(?:of\s+)?(\d)',
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, last_msg, re.IGNORECASE)
                    if match:
                        score_val = int(match.group(1))
                        if 0 <= score_val <= 7:
                            prediction = str(score_val)
                            self.log_fn(f"Extracted score from pattern: {prediction}")
                            break
            
            # Final fallback: find any standalone number 0-7
            if prediction == "None":
                numbers = re.findall(r'\b([0-7])\b', last_msg)
                if numbers:
                    prediction = numbers[-1]  # Take the last number found (usually the conclusion)
                    self.log_fn(f"Fallback extraction: {prediction}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
