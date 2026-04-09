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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _format_task_inputs(inputs: dict) -> str:
    """Format task inputs in a structured, readable way for the LLM.
    
    This improves clarity by separating different sections with clear headers.
    """
    sections = []
    
    # Domain
    if "domain" in inputs:
        sections.append(f"## Domain\n{inputs['domain']}")
    
    # Problem statement
    if "problem" in inputs:
        sections.append(f"## Problem\n{inputs['problem']}")
    
    # Solution
    if "solution" in inputs:
        sections.append(f"## Reference Solution\n{inputs['solution']}")
    
    # Grading guidelines
    if "grading_guidelines" in inputs:
        sections.append(f"## Grading Guidelines\n{inputs['grading_guidelines']}")
    
    # Student answer to evaluate
    if "student_answer" in inputs:
        sections.append(f"## Student Answer (to evaluate)\n{inputs['student_answer']}")
    
    return "\n\n".join(sections)


def _find_json_objects(text: str) -> list[str]:
    """Find all JSON objects in text using brace counting.
    
    Returns a list of JSON object strings found in the text.
    """
    results = []
    i = 0
    while i < len(text):
        # Find opening brace
        brace_start = text.find('{', i)
        if brace_start == -1:
            break
        
        # Find matching closing brace using brace counting
        brace_count = 0
        brace_end = -1
        in_string = False
        escape_next = False
        
        for j in range(brace_start, len(text)):
            char = text[j]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = j + 1
                        break
        
        if brace_end > brace_start:
            results.append(text[brace_start:brace_end])
            i = brace_end
        else:
            i = brace_start + 1
    
    return results


def _fix_json_string(json_str: str) -> str:
    """Fix common JSON formatting issues.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes converted to double quotes
    - Unescaped newlines in strings
    """
    # Fix trailing commas before closing braces
    fixed = re.sub(r',\s*}', '}', json_str)
    # Fix trailing commas before closing brackets
    fixed = re.sub(r',\s*]', ']', fixed)
    # Fix single quotes (convert to double) - but be careful with apostrophes
    # Only convert single quotes that appear to be JSON string delimiters
    fixed = re.sub(r"(?<=[\{\,\[])\s*'([^']*)'\s*(?=[\,\}\]])", r'"\1"', fixed)
    # Fix unescaped newlines in strings (replace with \n)
    fixed = re.sub(r'(?<=")([^"]*)\n([^"]*)"', lambda m: '"' + m.group(1) + '\\n' + m.group(2) + '"', fixed)
    return fixed


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and common formatting issues.
    """
    results = []
    
    # Clean up the text first - remove markdown code block markers if present
    cleaned_text = text
    if "<json>" not in cleaned_text and "```" in cleaned_text:
        # Try to extract JSON from markdown code blocks
        # Use brace-counting approach for robust nested brace handling
        code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_text, re.DOTALL)
        for block in code_blocks:
            # Try to find JSON objects within the block using brace matching
            json_objects = _find_json_objects(block)
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and "response" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    # Try fixing common issues
                    try:
                        fixed = _fix_json_string(obj_str)
                        obj = json.loads(fixed)
                        if isinstance(obj, dict) and "response" in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
        if results:
            return results
    
    search_from = 0
    while True:
        start = cleaned_text.find("<json>", search_from)
        if start == -1:
            break
        end = cleaned_text.find("</json>", start)
        if end == -1:
            break
        inner = cleaned_text[start + 6:end].strip()
        search_from = end + 7
        
        # Clean up common issues in the JSON content
        # Remove leading/trailing whitespace and newlines
        inner = inner.strip()
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                fixed = _fix_json_string(inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Log the error for debugging but continue
                logger.debug(f"Failed to parse JSON: {e}, content: {inner[:200]}")
                continue
    
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key.
    Uses a robust brace-matching approach to handle nested structures.
    """
    results = []
    
    # Clean up common formatting issues
    cleaned_text = text.strip()
    # Remove markdown code blocks if present
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    # Use the helper function to find all JSON objects
    json_objects = _find_json_objects(cleaned_text)
    
    for json_str in json_objects:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try fixing common issues
            try:
                fixed = _fix_json_string(json_str)
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                pass

    # If brace matching fails, try to parse the entire text as JSON
    if not results:
        try:
            obj = json.loads(cleaned_text)
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Last resort: try to find any JSON-like structure with response field
    if not results:
        # Look for patterns like {"response": "..."} or {"response": "...", ...}
        # Match JSON objects with response field
        pattern = r'\{\s*"response"\s*:\s*"([^"]*)"[^}]*\}'
        matches = re.findall(pattern, cleaned_text, re.DOTALL)
        for match in matches:
            try:
                # Reconstruct a minimal valid JSON
                obj = {"response": match}
                results.append(obj)
            except Exception:
                pass

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction.
    
    Enhanced with structured prompting and confidence scoring for better grading accuracy.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured, detailed prompt for the grading task."""
        formatted_inputs = _format_task_inputs(inputs)
        
        return f"""You are an expert mathematics grader evaluating student answers for mathematical olympiad problems.

Your task is to carefully evaluate the student's answer against the reference solution and grading guidelines.

{formatted_inputs}

## Instructions

1. Carefully read the problem, reference solution, and grading guidelines.
2. Analyze the student's answer step by step, checking:
   - Mathematical correctness of each step
   - Logical flow and reasoning
   - Whether the final answer matches the reference solution
   - Partial credit for correct intermediate steps even if final answer is wrong
3. Compare the student's approach with the reference solution - alternative valid approaches should be recognized.
4. Apply the grading guidelines strictly but fairly.
5. Provide your evaluation in the exact JSON format below.

## Grading Principles

- Award full credit if the student's answer is mathematically correct, even if the format differs from the reference solution.
- Award partial credit for correct reasoning even if the final answer is incorrect.
- Check for common errors: calculation mistakes, missing cases, incorrect assumptions.
- Consider the problem's difficulty level when evaluating.
- Be generous with partial credit - students often have good ideas even if execution is imperfect.

## Response Format

You MUST respond with a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "response": "Your grade/evaluation here. Be specific about what was correct or incorrect. Include the numerical score if applicable.",
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of your grading decision, referencing specific aspects of the student's work"
}}
</json>

The "response" field should contain the final grade or evaluation (e.g., "Correct - 7/7", "Partial credit - 3/7", "Incorrect - 0/7").
The "confidence" field indicates how certain you are (high/medium/low).
The "reasoning" field explains your decision with specific references to the student's work.

## Examples

Example 1 - Correct answer:
<json>
{{
    "response": "Correct - 7/7",
    "confidence": "high",
    "reasoning": "The student provided a complete and correct solution. All steps are mathematically sound and the final answer matches the reference solution."
}}
</json>

Example 2 - Partial credit:
<json>
{{
    "response": "Partial credit - 3/7",
    "confidence": "medium",
    "reasoning": "The student correctly identified the approach and made progress, but made a calculation error in step 3. The final answer is incorrect but the reasoning up to that point was valid."
}}
</json>

Example 3 - Incorrect:
<json>
{{
    "response": "Incorrect - 0/7",
    "confidence": "high",
    "reasoning": "The student's approach is fundamentally flawed. They misunderstood the problem statement and applied an incorrect method."
}}
</json>

IMPORTANT: Ensure your JSON is valid - use proper escaping for quotes and newlines within strings."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            import traceback
            error_details = f"Error calling LLM: {e}\n{traceback.format_exc()}"
            self.log_fn(error_details)
            return "Error: LLM call failed", [{"role": "system", "text": error_details}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        confidence = "unknown"
        reasoning = ""
        
        # Get the assistant's response text
        assistant_text = ""
        for msg in reversed(msg_history):
            if msg.get("role") == "assistant":
                assistant_text = msg.get("text", "")
                break
        
        if not assistant_text:
            self.log_fn("No assistant response found in message history")
            return "Error: No response from model", msg_history
        
        try:
            extracted = _extract_jsons(assistant_text)
            if extracted:
                result = extracted[-1]
                if "response" in result:
                    prediction = result["response"]
                if "confidence" in result:
                    confidence = result["confidence"]
                if "reasoning" in result:
                    reasoning = result["reasoning"]
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(assistant_text)
                if extracted:
                    result = extracted[-1]
                    if "response" in result:
                        prediction = result["response"]
                    if "confidence" in result:
                        confidence = result["confidence"]
                    if "reasoning" in result:
                        reasoning = result["reasoning"]
                    extraction_method = "fallback"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(assistant_text)
                if extracted:
                    result = extracted[-1]
                    if "response" in result:
                        prediction = result["response"]
                    if "confidence" in result:
                        confidence = result["confidence"]
                    if "reasoning" in result:
                        reasoning = result["reasoning"]
                    extraction_method = "fallback"
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        # Validate the prediction
        if prediction == "None" or not prediction or prediction.strip() == "":
            self.log_fn(f"Warning: Empty or invalid prediction. Raw response: {assistant_text[:500]}")
            # Try one more time with a simpler extraction
            prediction = self._extract_simple_response(assistant_text)
            extraction_method = "simple_fallback"

        self.log_fn(f"Extraction method: {extraction_method}, Confidence: {confidence}")
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
            
        return str(prediction), msg_history
    
    def _extract_simple_response(self, text: str) -> str:
        """Last resort extraction - try to find any meaningful response text."""
        # Look for common patterns in grading responses
        import re
        
        # First, try to extract just the response value from any JSON-like structure
        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
        if response_match:
            return response_match.group(1)
        
        # Look for grade patterns like "7/7", "3/7", "0/7", "Correct", "Incorrect", etc.
        grade_patterns = [
            r'(Correct|Incorrect|Partial credit).*?(\d+/\d+)',
            r'(\d+/\d+)',
            r'(full credit|partial credit|no credit)',
            r'(pass|fail)',
            r'(grade|score)\s*[:=]?\s*(\d+[/\s]*\d*)',
        ]
        
        for pattern in grade_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        # Look for any text that looks like a grade or evaluation
        # Common patterns: "7 points", "3 out of 7", "zero", etc.
        evaluation_patterns = [
            r'(\d+)\s*(?:points?|pts?)',
            r'(\d+)\s+out\s+of\s+(\d+)',
            r'(zero|one|two|three|four|five|six|seven)\s*(?:points?)?',
        ]
        
        for pattern in evaluation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        # If no grade pattern found, return first sentence or first 200 chars
        sentences = text.split('.')
        if sentences:
            first = sentences[0].strip()
            if len(first) > 10:  # Ensure it's meaningful
                return first[:200]
        
        # Last resort: return first 200 chars of text
        cleaned = text.strip()
        if len(cleaned) > 10:
            return cleaned[:200]
        
        return "Unable to extract valid response"
