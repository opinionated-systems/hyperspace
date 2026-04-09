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
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes additional heuristics for malformed JSON and nested structures.
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
        
        # Try direct JSON parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the text if it's wrapped in other content
        try:
            # Look for JSON-like content with braces (handle nested braces)
            brace_start = inner.find("{")
            brace_end = inner.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                results.append(json.loads(inner[brace_start:brace_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON formatting issues
        try:
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Fix single quotes to double quotes (common LLM mistake)
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            block = block.strip()
            try:
                results.append(json.loads(block))
            except json.JSONDecodeError:
                # Try fixing common issues in markdown blocks too
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', block)
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        # Look for patterns like {"key": "value"} or {"key": value}
        json_like_pattern = re.compile(r'\{[^{}]*"[^"]+"[^{}]*\}')
        matches = json_like_pattern.findall(text)
        for match in matches:
            try:
                results.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    
    return results or None


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

        instruction = f"""You are an expert {domain} grader evaluating student solutions with precision and consistency.

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
Follow this systematic approach:

1. **Understanding Check**: First, verify you understand what the problem is asking and what the official solution demonstrates.

2. **Step-by-Step Analysis**: Break down the student's answer into logical steps. For each step:
   - Identify what the student attempted
   - Check if the step is mathematically/logically valid
   - Compare against the corresponding step in the official solution
   - Note any errors, omissions, or alternative valid approaches

3. **Key Elements Verification**: Check if the student addressed all critical components:
   - Did they state the answer clearly?
   - Did they show sufficient work/explanation?
   - Did they use correct notation and terminology?
   - Are there any conceptual misunderstandings?

4. **Partial Credit Assessment**: If the answer is not fully correct, determine what partial credit is warranted based on:
   - Correct approach with minor errors
   - Correct intermediate steps but wrong final answer
   - Partial understanding demonstrated

5. **Final Grade Decision**: Synthesize your analysis into a clear grade that reflects:
   - Accuracy of the final answer
   - Validity of the reasoning process
   - Adherence to the grading guidelines

## Response Format:
Respond ONLY in JSON format wrapped in <json>...</json> tags with this exact schema:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering: (1) understanding of the problem, (2) step-by-step evaluation of student's work, (3) identification of correct/incorrect elements, (4) comparison to official solution, (5) justification for the grade assigned",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score as specified in guidelines)"
}}
</json>

## Example Response:
<json>
{{
    "reasoning": "Step 1 - Understanding: The problem requires finding the derivative of f(x) = x^2 * sin(x). Step 2 - Student's approach: The student applied the product rule correctly, identifying u=x^2 and v=sin(x). Step 3 - Evaluation: The derivatives du/dx=2x and dv/dx=cos(x) were correctly computed. The application of the product rule formula du*v + u*dv was executed correctly, resulting in 2x*sin(x) + x^2*cos(x). Step 4 - Comparison: This matches the official solution exactly. Step 5 - Conclusion: The student demonstrated complete understanding of the product rule and executed all steps correctly.",
    "response": "Correct"
}}
</json>

## Important Notes:
- Be objective and consistent in your grading
- Acknowledge alternative valid approaches if they differ from the official solution but are mathematically correct
- Distinguish between conceptual errors and calculation errors
- Your response MUST be valid JSON inside <json> tags
- Ensure your reasoning is detailed enough to justify the grade assigned"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        confidence = 0.0
        
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                extracted = _extract_jsons(response_text)
                
                if extracted:
                    last_json = extracted[-1]
                    # Try multiple possible keys for the response (ordered by priority)
                    response_keys = ["response", "grade", "result", "answer", "assessment", "evaluation", "score"]
                    for key in response_keys:
                        if key in last_json:
                            prediction = last_json[key]
                            confidence = 1.0
                            break
                    
                    # Log reasoning if available
                    reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking"]
                    for key in reasoning_keys:
                        if key in last_json:
                            reasoning = last_json[key]
                            self.log_fn(f"Reasoning ({key}): {reasoning[:200]}...")
                            break
                    
                    # Check for confidence score if available
                    if "confidence" in last_json:
                        try:
                            confidence = float(last_json["confidence"])
                        except (ValueError, TypeError):
                            pass
                else:
                    # Fallback: try to extract any meaningful text from the response
                    # Look for common patterns like "Grade: X" or "Answer: X"
                    grade_match = _GRADE_PATTERN.search(response_text)
                    if grade_match:
                        prediction = grade_match.group(1).strip()
                        confidence = 0.5  # Lower confidence for pattern-matched extraction
                        self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                    else:
                        # Last resort: use the raw response (truncated)
                        prediction = response_text[:500].strip()
                        confidence = 0.3
                        self.log_fn(f"Using raw response (no JSON found): {prediction[:100]}...")
            else:
                self.log_fn("Warning: Empty message history")
                prediction = "Error: No response"
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = f"Error: {str(e)[:100]}"

        # Log final prediction with confidence
        self.log_fn(f"Final prediction (confidence={confidence:.2f}): {str(prediction)[:100]}")

        return str(prediction), msg_history
