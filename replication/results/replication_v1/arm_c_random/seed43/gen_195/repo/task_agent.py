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
    Also handles markdown code blocks (```json...```) as fallback.
    Includes additional heuristics for common LLM output patterns.
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
            # Try to extract JSON from within the content if it's wrapped
            try:
                # Look for JSON object boundaries
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # Second: try markdown code blocks with json specifier
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to find JSON object boundaries
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    try:
                        results.append(json.loads(inner[json_start:json_end + 1]))
                    except json.JSONDecodeError:
                        continue
    
    # Third: try plain markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```", search_from)
            if start == -1:
                break
            end = text.find("```", start + 3)
            if end == -1:
                break
            inner = text[start + 3:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to find JSON object boundaries
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    try:
                        results.append(json.loads(inner[json_start:json_end + 1]))
                    except json.JSONDecodeError:
                        continue
    
    # Fourth: try to find JSON objects with specific IMO grading keys
    if not results:
        # Look for patterns that indicate IMO grading JSON
        imo_patterns = ['"reasoning"', '"evaluation"', '"response"', '"score"', '"grade"']
        for pattern in imo_patterns:
            if pattern in text:
                # Find the containing JSON object
                pattern_idx = text.find(pattern)
                if pattern_idx == -1:
                    continue
                # Find the start of the object
                obj_start = text.rfind("{", 0, pattern_idx)
                if obj_start == -1:
                    continue
                # Find the matching end brace
                brace_count = 1
                pos = obj_start + 1
                while pos < len(text) and brace_count > 0:
                    if text[pos] == '{':
                        brace_count += 1
                    elif text[pos] == '}':
                        brace_count -= 1
                    pos += 1
                
                if brace_count == 0:
                    try:
                        candidate = json.loads(text[obj_start:pos])
                        # Verify it has at least one expected key
                        if any(k in candidate for k in ["reasoning", "evaluation", "response", "score", "grade"]):
                            results.append(candidate)
                            break  # Found a valid IMO grading JSON
                    except json.JSONDecodeError:
                        continue
    
    # Final fallback: try to find any JSON object in the text
    if not results:
        # Look for JSON-like patterns: {...}
        json_start = text.find("{")
        while json_start != -1:
            # Find the matching closing brace by counting
            brace_count = 1
            pos = json_start + 1
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                try:
                    results.append(json.loads(text[json_start:pos]))
                    break  # Found valid JSON, stop searching
                except json.JSONDecodeError:
                    pass
            
            json_start = text.find("{", json_start + 1)
    
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

Please evaluate the student's answer following these rigorous steps:

1. **Understand the Problem**: 
   - Identify what the problem is asking
   - List the key mathematical concepts and theorems involved
   - Note any constraints or special conditions

2. **Analyze the Official Solution**: 
   - Break down the solution into key logical steps
   - Identify critical insights or proof techniques required
   - Note the point distribution if provided in guidelines

3. **Review Grading Guidelines Carefully**: 
   - Understand the specific criteria for partial credit
   - Note common errors that result in point deductions
   - Identify what constitutes a complete vs. incomplete solution

4. **Evaluate Student's Answer Systematically**: 
   - Check if the student correctly identified the approach
   - Verify each step of their reasoning for logical validity
   - Identify any errors, gaps, or incorrect assumptions with specific line references
   - Note any creative or alternative valid approaches (alternative solutions can be valid)
   - Check for "proof by example" errors in proof-based problems
   - Verify that all conditions of the problem are addressed

5. **Assign Score with Justification**: 
   - Base the score strictly on the grading guidelines
   - Award partial credit for correct intermediate steps even if final answer is wrong
   - Deduct points for logical gaps, missing cases, or incorrect claims
   - Be precise: justify every point awarded or deducted

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Include: (1) Problem understanding, (2) Key steps in official solution, (3) Step-by-step evaluation of student's work with specific references, (4) Point-by-point justification for the score",
    "evaluation": "Summary of what the student did correctly and incorrectly. Be specific about errors and strengths.",
    "response": "The final score/grade as a number or string (e.g., 7, 3, 0, 1/7, etc.)"
}}
</json>

IMPORTANT: The "response" field must contain ONLY the final score that will be used for evaluation. Use the format specified in the grading guidelines (e.g., "7" for full points, "0" for no points, or partial scores like "3" or "2/7")."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        extraction_log = []
        try:
            # Try the last assistant message first
            last_msg = msg_history[-1]
            extracted = _extract_jsons(last_msg["text"])
            extraction_log.append(f"Last message extraction: {len(extracted) if extracted else 0} JSONs found")
            
            if not extracted and len(msg_history) >= 2:
                # Try previous messages if last one doesn't have JSON
                for i in range(len(msg_history) - 2, -1, -1):
                    if msg_history[i]["role"] == "assistant":
                        extracted = _extract_jsons(msg_history[i]["text"])
                        if extracted:
                            extraction_log.append(f"Found JSON in message {i}")
                            break
            
            if extracted:
                # Try to get response from the last valid JSON block
                last_json = extracted[-1]
                extraction_log.append(f"Using JSON with keys: {list(last_json.keys())}")
                
                # Priority order for score fields - "response" is the primary field per our schema
                score_fields = ["response", "score", "grade", "evaluation", "result", "answer"]
                for field in score_fields:
                    if field in last_json:
                        prediction = last_json[field]
                        extraction_log.append(f"Extracted '{field}': {prediction}")
                        break
                else:
                    # If no known field, use the first string/numeric value found
                    for key, value in last_json.items():
                        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                            prediction = str(value)
                            extraction_log.append(f"Using first valid value from '{key}': {prediction}")
                            break
                
                # Validate the prediction - clean up common formatting issues
                if isinstance(prediction, str):
                    # Remove extra whitespace and newlines
                    prediction = prediction.strip()
                    # Handle cases like "Score: 7" or "The answer is 3"
                    if prediction.lower().startswith("score:"):
                        prediction = prediction[6:].strip()
                    # Extract just the number if there's extra text
                    number_match = re.search(r'\b(\d+(?:/\d+)?)\b', prediction)
                    if number_match and len(prediction) > len(number_match.group(1)) + 5:
                        # If there's a clear number embedded in text, prefer it
                        prediction = number_match.group(1)
            else:
                extraction_log.append("No JSON found in any message")
                
        except Exception as e:
            extraction_log.append(f"Error during extraction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")

        # Log extraction details for debugging
        self.log_fn(f"Prediction extraction: {'; '.join(extraction_log)}")
        
        return str(prediction), msg_history
