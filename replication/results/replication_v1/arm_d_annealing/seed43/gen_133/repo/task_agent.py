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


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
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
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to clean common JSON issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues."""
    cleaned = text.strip()
    
    # Remove markdown code block markers
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes to double quotes for keys and string values
    cleaned = re.sub(r"'([^']*?)'(?=\s*:)", r'"\1"', cleaned)
    cleaned = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', cleaned)
    
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    # Fix escaped newlines
    cleaned = cleaned.replace('\\n', '\n')
    cleaned = cleaned.replace('\\t', '\t')
    
    # Fix double-escaped quotes
    cleaned = cleaned.replace('\\"', '"')
    
    # Remove BOM if present
    cleaned = cleaned.lstrip('\ufeff')
    
    return cleaned


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    """
    results = []
    
    def extract_balanced_json(content: str) -> list[dict]:
        """Extract all balanced JSON objects from content using stack-based parsing."""
        objects = []
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_obj = json.loads(content[start_idx:i+1])
                        if isinstance(json_obj, dict):
                            objects.append(json_obj)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing
                        try:
                            cleaned = _clean_json_string(content[start_idx:i+1])
                            json_obj = json.loads(cleaned)
                            if isinstance(json_obj, dict):
                                objects.append(json_obj)
                        except json.JSONDecodeError:
                            pass
                    start_idx = -1
        return objects
    
    # Try to find JSON objects in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        objects = extract_balanced_json(content)
        results.extend(objects)
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        results = extract_balanced_json(text)
    
    # Final fallback: try to find key-value patterns for response, reasoning, and confidence
    if not results:
        result = {}
        # Look for response pattern with more flexible matching
        response_patterns = [
            r'["\']response["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']response["\']\s*:\s*(\d+)',
            r'response\s*:\s*([\w\s-]+)(?:\n|$)',
            r'grade["\']?\s*[=:]\s*["\']?([\w\s-]+)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip()
                break
        
        # Look for reasoning pattern
        reasoning_patterns = [
            r'["\']reasoning["\']\s*:\s*"([^"]*)"',
            r'["\']reasoning["\']\s*:\s*\'([^\']*)\'',
            r'reasoning["\']?\s*[=:]\s*["\']?([^"\']{10,})',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result["reasoning"] = match.group(1).strip()
                break
        
        # Look for confidence pattern
        confidence_patterns = [
            r'["\']confidence["\']\s*:\s*(0?\.\d+|1\.0|1|0)',
            r'confidence\s*:\s*(0?\.\d+|1\.0|1|0)',
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    result["confidence"] = float(match.group(1))
                except (ValueError, TypeError):
                    pass
                break
        
        if result:
            results.append(result)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _normalize_grade(self, grade: str) -> str:
        """Normalize grade to standard format."""
        grade_lower = str(grade).lower().strip()
        
        # Map various grade formats to standard ones
        if any(x in grade_lower for x in ["correct", "right", "true", "yes", "1", "full", "accurate", "valid"]):
            return "Correct"
        elif any(x in grade_lower for x in ["partial", "half", "some", "0.5", "incomplete", "partly", "mostly"]):
            return "Partial"
        elif any(x in grade_lower for x in ["incorrect", "wrong", "false", "no", "0", "none", "error", "invalid", "fail"]):
            return "Incorrect"
        else:
            # Try to extract numeric score
            try:
                num = float(grade)
                if num >= 0.8:
                    return "Correct"
                elif num >= 0.4:
                    return "Partial"
                else:
                    return "Incorrect"
            except (ValueError, TypeError):
                # Check if grade contains any of the keywords as substrings
                if "correct" in grade_lower and "incorrect" not in grade_lower and "partial" not in grade_lower:
                    return "Correct"
                elif "partial" in grade_lower:
                    return "Partial"
                elif "incorrect" in grade_lower or "wrong" in grade_lower:
                    return "Incorrect"
                return "Incorrect"  # Default fallback

    def _build_grading_prompt(self, inputs: dict, is_retry: bool = False, previous_error: str = "") -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        base_prompt = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)",
    "confidence": 0.95
}}
</json>

IMPORTANT: 
- Ensure your JSON is valid and properly formatted. 
- The 'response' field should contain only the grade, not the reasoning.
- The 'confidence' field should be a number between 0 and 1 indicating your confidence in the grade.
- Use EXACTLY one of these values for 'response': Correct, Partial, or Incorrect"""

        if is_retry and previous_error:
            return f"""Your previous response had an error: {previous_error}

Please correct this and provide a valid response following the exact format below.

{base_prompt}"""
        
        return base_prompt

    def _extract_prediction(self, text: str) -> tuple[str, str, float]:
        """Extract prediction, reasoning, and confidence from response text.
        
        Returns:
            (prediction, reasoning, confidence) tuple
        """
        prediction = "None"
        reasoning = ""
        confidence = 0.5
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            # Find the best JSON object (prefer one with all required fields)
            best_json = None
            for json_obj in extracted:
                if "response" in json_obj:
                    best_json = json_obj
                    break
            
            # If no JSON with response found, use the last one
            if best_json is None:
                best_json = extracted[-1]
            
            if "response" in best_json:
                raw_prediction = str(best_json["response"])
                prediction = self._normalize_grade(raw_prediction)
            if "reasoning" in best_json:
                reasoning = str(best_json["reasoning"])
            if "confidence" in best_json:
                try:
                    confidence = float(best_json["confidence"])
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except (ValueError, TypeError):
                    confidence = 0.5
        
        return prediction, reasoning, confidence

    def _calculate_confidence(self, prediction: str, reasoning: str, original_confidence: float) -> float:
        """Calculate a more reliable confidence score based on multiple factors."""
        # Base confidence from LLM
        confidence = original_confidence
        
        # Adjust based on reasoning quality
        if reasoning:
            reasoning_length = len(reasoning.strip())
            # Longer reasoning (within reason) suggests more thorough analysis
            if reasoning_length > 100:
                confidence = min(1.0, confidence + 0.05)
            if reasoning_length > 200:
                confidence = min(1.0, confidence + 0.05)
            
            # Check for uncertainty indicators in reasoning
            uncertainty_words = ["maybe", "perhaps", "unclear", "ambiguous", "difficult to tell", 
                               "not sure", "uncertain", "possibly", "might be"]
            reasoning_lower = reasoning.lower()
            uncertainty_count = sum(1 for word in uncertainty_words if word in reasoning_lower)
            if uncertainty_count > 0:
                confidence = max(0.0, confidence - 0.1 * uncertainty_count)
        else:
            # No reasoning provided, reduce confidence
            confidence = max(0.0, confidence - 0.2)
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        confidence = 0.5
        best_prediction = "None"
        best_confidence = 0.0
        best_reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, confidence = self._extract_prediction(last_text)
                
                # Calculate improved confidence
                confidence = self._calculate_confidence(prediction, reasoning, confidence)
                
                # Track best prediction by confidence
                if prediction != "None" and confidence > best_confidence:
                    best_prediction = prediction
                    best_confidence = confidence
                    best_reasoning = reasoning
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction} (confidence: {confidence:.2f})")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    # If high confidence, accept immediately
                    if confidence >= 0.85:
                        break
                    # Otherwise continue to get more samples
                    if attempt < self.max_retries - 1:
                        self.log_fn(f"Confidence {confidence:.2f} < 0.85, retrying for better confidence...")
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = self._build_grading_prompt(
                            inputs, 
                            is_retry=True, 
                            previous_error="Response did not contain valid JSON with a 'response' field"
                        )
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    instruction = self._build_grading_prompt(
                        inputs, 
                        is_retry=True, 
                        previous_error=f"Exception occurred: {str(e)}"
                    )
                else:
                    prediction = f"Error: {e}"
        
        # Use best prediction if we got any valid ones
        if best_prediction != "None":
            prediction = best_prediction
            confidence = best_confidence
            reasoning = best_reasoning
            self.log_fn(f"Final prediction: {prediction} (best confidence: {best_confidence:.2f})")
        else:
            # If all attempts failed, default to Incorrect
            prediction = "Incorrect"
            self.log_fn(f"All attempts failed, defaulting to 'Incorrect'")
        
        return str(prediction), msg_history
