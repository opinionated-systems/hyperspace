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
    Also handles markdown code blocks and raw JSON objects.
    Includes enhanced error recovery for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json(match.group(1).strip())
            if parsed:
                results.append(parsed)
    
    # Try to find raw JSON objects with "response" or "reasoning" fields
    if not results:
        # Look for complete JSON objects - more robust pattern for nested braces
        # This pattern matches balanced braces
        json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed and ("response" in parsed or "reasoning" in parsed):
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - use a more flexible pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\}[^{}]*\})?'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json(match.group())
            if parsed:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON with multiple recovery strategies.
    
    Tries raw parsing first, then applies common fixes for LLM-generated JSON.
    """
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
    
    # Strategy 3: Fix single quotes to double quotes
    try:
        # Replace single quotes around keys and string values
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the fields we need with regex
    try:
        response_match = re.search(r'"response"\s*:\s*"([^"]*)"', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        
        if response_match or reasoning_match:
            result = {}
            if response_match:
                result["response"] = response_match.group(1)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1)
            return result
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
   - IMO problems are typically graded 0-7 points

## Response Format (CRITICAL - FOLLOW EXACTLY)

You MUST respond with a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed analysis following the framework above. Write your complete step-by-step evaluation here.",
    "response": "Your final grade as a number (0-7) or text (e.g., '7', '6', '5', 'Partial credit: 3', '0', 'Incorrect')"
}}
</json>

IMPORTANT:
- The JSON must be valid with double quotes around all keys and string values
- Do not use single quotes
- Do not include trailing commas
- The "response" field must contain ONLY the final grade/evaluation
- The "reasoning" field contains your full analysis
- IMO grades are typically integers from 0 to 7
- If the solution is completely correct, the grade should be 7
- If the solution is completely wrong or empty, the grade should be 0
- Partial credit should be given for partial progress toward a solution"""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with better error handling
        prediction, reasoning = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Uses multiple extraction strategies for robustness.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"])
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
                        prediction = str(fallback.get("response", "None"))
                    except json.JSONDecodeError:
                        pass
                
                # Last resort: look for grade patterns in plain text
                if prediction == "None":
                    # Look for patterns like "Grade: 7" or "Final grade: 6"
                    grade_patterns = [
                        r'[Gg]rade[:\s]+(\d+)',
                        r'[Ff]inal[^:]*[:\s]+(\d+)',
                        r'[Ss]core[:\s]+(\d+)',
                        r'\b([0-7])\s*/\s*7\b',
                        r'"response"\s*:\s*"(\d+)"',
                        r'"response"\s*:\s*(\d+)',
                        r'[Rr]esponse[:\s]+(\d+)',
                        r'[Ee]valuation[:\s]+(\d+)',
                        r'[Aa]ssigned[^:]*[:\s]+(\d+)',
                        r'[Vv]alue[:\s]+(\d+)',
                        r'\bgrade\s+(\d+)\b',
                        r'\bis\s+(\d+)\b',
                        r'\b(\d+)\s*points?\b',
                        r'\bpoints?\s*[:\s]+(\d+)\b',
                    ]
                    for pattern in grade_patterns:
                        match = re.search(pattern, last_msg)
                        if match:
                            prediction = match.group(1)
                            self.log_fn(f"Extracted grade from text: {prediction}")
                            break
                    
                    # If still no grade found, look for numbers 0-7 in context of grading
                    if prediction == "None":
                        # Look for standalone numbers 0-7 near grading keywords
                        context_pattern = r'(?:grade|score|evaluation|mark|points?).{0,30}(\b[0-7]\b)'
                        match = re.search(context_pattern, last_msg, re.IGNORECASE | re.DOTALL)
                        if match:
                            prediction = match.group(1)
                            self.log_fn(f"Extracted grade from context: {prediction}")
            
            # Validate the prediction - IMO grades should be 0-7
            prediction = self._validate_grade(prediction)
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return prediction, reasoning
    
    def _validate_grade(self, prediction: str) -> str:
        """Validate and normalize the extracted grade.
        
        IMO grades are typically 0-7. This method attempts to extract
        a valid numeric grade from the prediction string.
        
        Args:
            prediction: The raw prediction string
            
        Returns:
            A validated grade string
        """
        if prediction == "None" or not prediction:
            return "None"
        
        # Try to extract a number from the prediction
        # First, check if it's already a clean number
        prediction = prediction.strip()
        
        # Try to find a number in the string
        number_match = re.search(r'\b(\d+)\b', prediction)
        if number_match:
            grade = int(number_match.group(1))
            # Clamp to valid IMO range
            if grade > 7:
                grade = 7
            self.log_fn(f"Validated grade: {grade}")
            return str(grade)
        
        # Check for text-based grades
        lower_pred = prediction.lower()
        if any(word in lower_pred for word in ['incorrect', 'wrong', 'error', 'invalid', 'none', 'fail']):
            return "0"
        if any(word in lower_pred for word in ['correct', 'perfect', 'complete', 'full']):
            return "7"
        if any(word in lower_pred for word in ['partial', 'incomplete']):
            # Try to extract partial credit value
            partial_match = re.search(r'(\d+)', prediction)
            if partial_match:
                return partial_match.group(1)
            return "3"  # Default partial credit
        
        return prediction
