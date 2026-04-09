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
    if not text or not isinstance(text, str):
        logger.debug("_extract_jsons: text is empty or not a string")
        return None
    
    logger.debug(f"_extract_jsons: processing text of length {len(text)}")
    results = []
    search_from = 0
    json_blocks_found = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning("_extract_jsons: found <json> but no closing </json>")
            break
        
        json_blocks_found += 1
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try direct JSON parsing first
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug(f"_extract_jsons: successfully parsed JSON block #{json_blocks_found}")
            continue
        except json.JSONDecodeError as e:
            logger.debug(f"_extract_jsons: direct JSON parse failed for block #{json_blocks_found}: {e}")
        
        # Try to extract JSON from within the text if it's wrapped in other content
        try:
            # Look for JSON-like content with braces (handle nested braces)
            brace_start = inner.find("{")
            brace_end = inner.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                extracted = inner[brace_start:brace_end + 1]
                parsed = json.loads(extracted)
                results.append(parsed)
                logger.debug(f"_extract_jsons: extracted JSON from braces for block #{json_blocks_found}")
                continue
        except json.JSONDecodeError as e:
            logger.debug(f"_extract_jsons: brace extraction failed for block #{json_blocks_found}: {e}")
        
        # Try to fix common JSON formatting issues
        try:
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            # Fix single quotes to double quotes (common LLM mistake)
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            # Fix unescaped newlines in strings
            fixed = re.sub(r'(?<=")\n(?=")', '\\n', fixed)
            parsed = json.loads(fixed)
            results.append(parsed)
            logger.debug(f"_extract_jsons: fixed and parsed JSON for block #{json_blocks_found}")
            continue
        except json.JSONDecodeError as e:
            logger.debug(f"_extract_jsons: JSON fixing failed for block #{json_blocks_found}: {e}")
    
    logger.debug(f"_extract_jsons: found {json_blocks_found} <json> blocks, successfully parsed {len(results)}")
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        logger.debug(f"_extract_jsons: trying markdown fallback, found {len(json_blocks)} code blocks")
        for i, block in enumerate(json_blocks):
            block = block.strip()
            try:
                results.append(json.loads(block))
                logger.debug(f"_extract_jsons: parsed markdown block #{i+1}")
            except json.JSONDecodeError:
                # Try fixing common issues in markdown blocks too
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', block)
                    fixed = re.sub(r"(?<!\\)'", '"', fixed)
                    fixed = re.sub(r'(?<=")\n(?=")', '\\n', fixed)
                    results.append(json.loads(fixed))
                    logger.debug(f"_extract_jsons: fixed and parsed markdown block #{i+1}")
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        # Look for patterns like {"key": "value"} or {"key": value}
        json_like_pattern = re.compile(r'\{[^{}]*"[^"]+"[^{}]*\}')
        matches = json_like_pattern.findall(text)
        logger.debug(f"_extract_jsons: trying last resort pattern, found {len(matches)} matches")
        for i, match in enumerate(matches):
            try:
                results.append(json.loads(match))
                logger.debug(f"_extract_jsons: parsed pattern match #{i+1}")
            except json.JSONDecodeError:
                continue
    
    if results:
        logger.info(f"_extract_jsons: successfully extracted {len(results)} JSON object(s)")
    else:
        logger.warning("_extract_jsons: no valid JSON found in response")
    
    return results or None


def _validate_grading_response(data: dict) -> tuple[bool, str, str]:
    """Validate that the grading response has the expected format.
    
    Args:
        data: The parsed JSON data from the LLM response
        
    Returns:
        (is_valid, prediction, reasoning) tuple where:
        - is_valid: True if the response has valid grading format
        - prediction: The extracted grade/response value
        - reasoning: The extracted reasoning text
    """
    if not isinstance(data, dict):
        return False, "", ""
    
    # Try multiple possible keys for the response (ordered by priority)
    response_keys = ["response", "grade", "result", "answer", "assessment", "evaluation", "score"]
    prediction = ""
    for key in response_keys:
        if key in data:
            prediction = str(data[key]).strip()
            break
    
    # Try multiple possible keys for reasoning
    reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking"]
    reasoning = ""
    for key in reasoning_keys:
        if key in data:
            reasoning = str(data[key]).strip()
            break
    
    # A valid response should have at least a prediction
    is_valid = bool(prediction) and prediction.lower() not in ["none", "null", ""]
    
    return is_valid, prediction, reasoning


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
        # Validate inputs
        if not isinstance(inputs, dict):
            logger.error(f"Invalid inputs type: {type(inputs)}")
            return "Error: Invalid inputs", []
        
        # Extract fields for better prompting with validation
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Log warning if critical fields are missing
        if not problem:
            logger.warning("Missing 'problem' field in inputs")
        if not solution:
            logger.warning("Missing 'solution' field in inputs")
        if not student_answer:
            logger.warning("Missing 'student_answer' field in inputs")

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
        extraction_method = "unknown"
        
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                # Log the raw response for debugging
                self.log_fn(f"Raw response length: {len(response_text)} chars")
                
                extracted = _extract_jsons(response_text)
                
                if extracted:
                    # Use the last JSON block (most likely to be the final answer)
                    last_json = extracted[-1]
                    self.log_fn(f"Extracted {len(extracted)} JSON block(s), using last one with keys: {list(last_json.keys())}")
                    
                    # Validate and extract from the JSON
                    is_valid, pred, reason = _validate_grading_response(last_json)
                    
                    if is_valid:
                        prediction = pred
                        reasoning = reason
                        confidence = 1.0
                        extraction_method = "json_validated"
                        
                        # Log reasoning if available
                        if reasoning:
                            self.log_fn(f"Reasoning extracted ({len(reasoning)} chars): {reasoning[:200]}...")
                        
                        # Check for explicit confidence score if available
                        if "confidence" in last_json:
                            try:
                                confidence = float(last_json["confidence"])
                            except (ValueError, TypeError):
                                pass
                    else:
                        # JSON found but didn't have expected format
                        prediction = str(last_json)[:200]
                        confidence = 0.4
                        extraction_method = "json_unvalidated"
                        self.log_fn(f"JSON found but missing expected keys. Keys found: {list(last_json.keys())}")
                else:
                    # Fallback: try to extract any meaningful text from the response
                    # Look for common patterns like "Grade: X" or "Answer: X"
                    grade_match = _GRADE_PATTERN.search(response_text)
                    if grade_match:
                        prediction = grade_match.group(1).strip()
                        confidence = 0.5  # Lower confidence for pattern-matched extraction
                        extraction_method = "pattern_match"
                        self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                    else:
                        # Last resort: use the raw response (truncated)
                        prediction = response_text[:500].strip()
                        confidence = 0.3
                        extraction_method = "raw_fallback"
                        self.log_fn(f"Using raw response (no JSON found): {prediction[:100]}...")
            else:
                self.log_fn("Warning: Empty message history")
                prediction = "Error: No response"
                extraction_method = "error"
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = f"Error: {str(e)[:100]}"
            confidence = 0.0
            extraction_method = "exception"

        # Log final prediction with confidence and extraction method
        self.log_fn(f"Final prediction (method={extraction_method}, confidence={confidence:.2f}): {str(prediction)[:100]}")

        return str(prediction), msg_history
