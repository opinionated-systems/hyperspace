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

# Compile regex patterns once for efficiency
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer|assessment|evaluation)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
# Pattern to find JSON-like objects with nested braces (improved for nested structures)
_JSON_OBJECT_PATTERN = re.compile(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', re.DOTALL)
# Pattern to extract content between <json> tags
_JSON_TAG_PATTERN = re.compile(r'<json>\s*(.*?)\s*</json>', re.DOTALL | re.IGNORECASE)

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks with improved robustness.

    Uses multiple extraction strategies:
    1. Direct <json> tag extraction using regex
    2. Markdown code block extraction
    3. Raw JSON object detection in text
    
    Handles nested braces and common LLM output formatting issues.
    """
    results = []
    
    # Strategy 1: Extract from <json> tags using regex
    json_tag_matches = _JSON_TAG_PATTERN.findall(text)
    for inner in json_tag_matches:
        inner = inner.strip()
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object within the tag content
        try:
            brace_start = inner.find("{")
            brace_end = inner.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                results.append(json.loads(inner[brace_start:brace_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try regex-based extraction for nested braces
        json_matches = _JSON_OBJECT_PATTERN.findall(inner)
        for match in json_matches:
            try:
                results.append(json.loads(match))
                break  # Take first valid match
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Fallback to markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            block = block.strip()
            try:
                results.append(json.loads(block))
                continue
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON objects from within the block
            json_matches = _JSON_OBJECT_PATTERN.findall(block)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        break
                except json.JSONDecodeError:
                    continue
    
    # Strategy 3: Search entire text for JSON objects
    if not results:
        json_matches = _JSON_OBJECT_PATTERN.findall(text)
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and ("response" in parsed or "reasoning" in parsed):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results if results else None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions with precision and fairness.

Your task is to grade a student's answer by systematically comparing it against the official solution and applying the grading guidelines rigorously.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Instructions:
Follow this structured approach:

1. **Understanding Check**: Verify you understand the problem, official solution, and what constitutes a correct answer.

2. **Step-by-Step Analysis**: 
   - Break down the official solution into key steps/concepts
   - Map the student's answer to these steps
   - Identify what the student got right (concepts, methods, calculations)
   - Identify errors (conceptual mistakes, calculation errors, missing steps)

3. **Error Classification**:
   - Minor error: Small calculation mistake but correct approach
   - Major error: Wrong method or missing critical concept
   - Complete: All steps correct with proper justification

4. **Grade Determination**: Based on the guidelines, assign an appropriate grade.

## Response Format (STRICT):
You MUST respond ONLY with a JSON object wrapped in <json>...</json> tags. Use this exact schema:

<json>
{{
    "reasoning": "Detailed analysis covering: (1) understanding of problem, (2) comparison with official solution, (3) specific errors found, (4) justification for the grade",
    "response": "The final grade - use exact terms from guidelines or standard grades like 'Correct', 'Partially Correct', 'Incorrect'"
}}
</json>

## Example Responses:

Example 1 - Correct:
<json>
{{
    "reasoning": "The student demonstrated complete understanding of the problem. All steps match the official solution: correctly applied the quadratic formula, showed all work, and arrived at the correct roots. The reasoning is clear and mathematically sound.",
    "response": "Correct"
}}
</json>

Example 2 - Partially Correct:
<json>
{{
    "reasoning": "The student used the correct method (integration by parts) and set up the problem correctly. However, there is a sign error in the final calculation (should be -1/2, not +1/2). This is a minor calculation error with correct conceptual understanding.",
    "response": "Partially Correct"
}}
</json>

Example 3 - Incorrect:
<json>
{{
    "reasoning": "The student fundamentally misunderstood the problem. They attempted to use the Pythagorean theorem on a non-right triangle and made no attempt to apply the law of cosines as required. The approach is incorrect for this problem type.",
    "response": "Incorrect"
}}
</json>

## Critical Requirements:
- Response MUST start with <json> and end with </json>
- JSON must be valid with no syntax errors
- Both "reasoning" and "response" fields are REQUIRED
- Do not include ANY text outside the JSON tags
- Be objective and consistent with the official solution
- Consider partial credit when guidelines allow it

Provide your assessment now."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        response_text = msg_history[-1]["text"]
        
        try:
            extracted = _extract_jsons(response_text)
            if extracted:
                # Use the last valid JSON object (most likely to be the final answer)
                last_json = extracted[-1]
                
                # Try multiple possible keys for the response (ordered by priority)
                response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", 
                               "score", "verdict", "conclusion", "decision", "output", "final_grade"]
                for key in response_keys:
                    if key in last_json and last_json[key] is not None:
                        prediction = str(last_json[key]).strip()
                        break
                
                # Log reasoning if available
                reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking", 
                               "rationale", "justification", "notes", "commentary", "evaluation_details"]
                for key in reasoning_keys:
                    if key in last_json and last_json[key]:
                        reasoning = last_json[key]
                        self.log_fn(f"Reasoning ({key}): {str(reasoning)[:200]}...")
                        break
                
                self.log_fn(f"Extracted prediction from JSON: {prediction}")
            else:
                # Fallback: try to extract any meaningful text from the response
                # Look for common patterns like "Grade: X" or "Answer: X"
                grade_match = _GRADE_PATTERN.search(response_text)
                if grade_match:
                    prediction = grade_match.group(1).strip()
                    self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                else:
                    # Try to find standalone grades like "Correct", "Incorrect", "Partially Correct"
                    grade_keywords = ["Correct", "Incorrect", "Partially Correct", "Partial", 
                                    "Full Credit", "No Credit", "Pass", "Fail", "Excellent", "Good", 
                                    "Fair", "Poor", "Zero", "Full Marks"]
                    for keyword in grade_keywords:
                        if keyword.lower() in response_text.lower():
                            prediction = keyword
                            self.log_fn(f"Extracted grade via keyword: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Last resort: check if response contains any recognizable grade
            try:
                text_lower = response_text.lower()
                if "correct" in text_lower and "incorrect" not in text_lower:
                    prediction = "Correct"
                elif "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "Incorrect"
                elif "partial" in text_lower:
                    prediction = "Partially Correct"
            except:
                pass

        return str(prediction), msg_history
