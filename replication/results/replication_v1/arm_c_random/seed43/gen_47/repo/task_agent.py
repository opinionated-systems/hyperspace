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
    """Extract JSON objects from <json>...</json> blocks with enhanced robustness.

    This function uses a state-machine parser to find outermost tag pairs,
    avoiding the lazy .*? regex bug that truncates content with nested braces.
    It also handles markdown code blocks, raw JSON objects, and nested structures.

    Args:
        text: The input text containing potential JSON blocks

    Returns:
        List of parsed JSON objects, or None if no valid JSON found
    """
    results = []
    search_from = 0
    max_iterations = 100  # Safety limit to prevent infinite loops
    iterations = 0
    
    # Phase 1: Extract from <json>...</json> blocks with proper nesting support
    while iterations < max_iterations:
        iterations += 1
        start = text.find("<json>", search_from)
        if start == -1:
            break
        
        # Find the matching </json> tag using state machine for brace counting
        content_start = start + 6
        end = content_start
        brace_count = 0
        in_string = False
        escape_next = False
        
        while end < len(text):
            char = text[end]
            
            # Handle escape sequences in strings
            if escape_next:
                escape_next = False
            elif char == '\\' and in_string:
                escape_next = True
            # Handle string boundaries
            elif char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            # Handle braces (only when not in string)
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            # Check for closing tag when braces are balanced
            if brace_count == 0 and text[end:end+7] == "</json>":
                break
            end += 1
        
        # Validate we found a proper closing tag
        if end >= len(text) or text[end:end+7] != "</json>":
            # Fall back to simple search if nesting detection fails
            end = text.find("</json>", start)
            if end == -1:
                logger.warning("Found <json> tag but no matching </json>")
                break
        
        inner = text[content_start:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
            logger.debug(f"Successfully parsed JSON from <json> block: {list(parsed.keys())}")
    
    # Phase 2: If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group(1).strip())
            if parsed:
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON from markdown block: {list(parsed.keys())}")
    
    # Phase 3: Last resort - find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key using a more robust pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON from fallback pattern: {list(parsed.keys())}")
    
    if not results:
        logger.warning("No valid JSON objects found in text")
        
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple cleanup strategies."""
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes (common LLM mistake)
    try:
        # Replace single quotes with double quotes, but be careful with apostrophes
        cleaned = re.sub(r"(?<!\\)'", '"', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Fix unquoted keys
    try:
        # Add quotes to unquoted keys (simple pattern for common cases)
        cleaned = re.sub(r'(\{|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Extract just the response field if present
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
        if response_match:
            return {"response": response_match.group(1)}
    except Exception:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

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

## Evaluation Framework

Follow this systematic approach:

1. **Understanding Check**: Verify you understand the problem and official solution completely.

2. **Student's Approach Analysis**: 
   - Identify the student's key ideas and strategy
   - Note any creative or alternative approaches
   - Check if the approach is valid even if different from official solution

3. **Correctness Verification**:
   - Check each claim and step for logical validity
   - Identify any gaps, errors, or unjustified assertions
   - Verify calculations and algebraic manipulations

4. **Completeness Assessment**:
   - Does the solution cover all cases?
   - Are all conditions from the problem statement addressed?
   - Is the conclusion properly justified?

5. **Grading Decision**:
   - Apply the grading guidelines strictly
   - Consider partial credit for correct ideas with minor gaps
   - Be consistent with IMO standards

## Response Format

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis following the framework above...",
    "response": "Your final grade/evaluation (e.g., '7', '6', 'Partial credit: 3', 'Incorrect', etc.)"
}}
</json>

The "response" field should contain only the final grade/evaluation, while "reasoning" contains your full analysis."""

    def _validate_prediction(self, prediction: str, grading_guidelines: str) -> str:
        """Validate and normalize the prediction against grading guidelines.
        
        Args:
            prediction: The raw prediction string from the model
            grading_guidelines: The grading guidelines to validate against
            
        Returns:
            Normalized prediction string
        """
        if not prediction or prediction == "None":
            return "None"
        
        prediction = str(prediction).strip()
        
        # Check if it's a numeric grade (0-7 for IMO problems)
        if prediction.isdigit():
            score = int(prediction)
            if 0 <= score <= 7:
                return str(score)
            return "Invalid score"
        
        # Check for partial credit patterns
        partial_patterns = ["partial", "partial credit", "partially correct"]
        if any(p in prediction.lower() for p in partial_patterns):
            # Try to extract numeric value
            import re
            numbers = re.findall(r'\d+', prediction)
            if numbers:
                return f"Partial credit: {numbers[0]}"
        
        # Check for common non-numeric responses
        if prediction.lower() in ["correct", "full credit", "complete"]:
            return "7"
        if prediction.lower() in ["incorrect", "wrong", "no credit", "zero"]:
            return "0"
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction = "None"
        reasoning = ""
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{[^}]*"response"[^}]*\}', last_msg)
                if json_match:
                    try:
                        fallback = json.loads(json_match.group())
                        prediction = fallback.get("response", "None")
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate and normalize the prediction
        prediction = self._validate_prediction(prediction, grading_guidelines)
        self.log_fn(f"Final prediction: {prediction}")

        return str(prediction), msg_history
