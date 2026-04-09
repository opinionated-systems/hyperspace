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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM error)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction for when standard extraction fails.
    
    Tries to find JSON objects even without proper <json> tags.
    """
    results = []
    # Try to find JSON-like structures with curly braces
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = text[start_idx:i+1]
                    results.append(json.loads(json_str))
                except json.JSONDecodeError:
                    # Try common fixes
                    try:
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        fixed = re.sub(r"(?<!\\)'", '"', fixed)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        pass
                start_idx = None
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt with chain-of-thought instructions."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert mathematical grader for {domain} problems.

Your task is to evaluate a student's answer to a mathematical problem with precision and accuracy.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

INSTRUCTIONS:
1. Carefully read the problem and understand what is being asked.
2. Study the official solution to understand the correct approach and final answer.
3. Analyze the student's answer step by step:
   - Does the student understand the problem?
   - Is the approach mathematically sound?
   - Are the calculations correct?
   - Does the student arrive at the correct final answer?
4. Check for partial credit:
   - Correct approach but calculation error
   - Correct final answer but incomplete reasoning
   - Partial progress toward solution
5. Be strict but fair in your evaluation.
6. Provide your final grade in the exact JSON format below.

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": "Your final grade/assessment here"
}}
</json>

The "response" field should contain only the final grade (e.g., "7", "5", "0", "Correct", "Incorrect", "Partial", etc.).
The "reasoning" field should contain your detailed analysis of why you gave that grade."""

    def _validate_prediction(self, prediction: str, inputs: dict) -> tuple[str, str]:
        """Validate and normalize the prediction against grading guidelines.
        
        Args:
            prediction: The raw prediction string
            inputs: The task inputs containing grading guidelines
            
        Returns:
            (validated_prediction, validation_note) tuple
        """
        if not prediction or prediction == "None":
            return "None", ""
        
        prediction = prediction.strip()
        grading_guidelines = inputs.get("grading_guidelines", "")
        
        # Check if this is a numeric grade (0-7 scale common in IMO)
        try:
            numeric_grade = float(prediction)
            # Ensure it's within reasonable bounds
            if 0 <= numeric_grade <= 7:
                return str(int(numeric_grade)), ""
            else:
                return str(int(numeric_grade)), f"Note: Grade {numeric_grade} outside typical 0-7 range"
        except ValueError:
            pass
        
        # Check for common non-numeric grades
        valid_non_numeric = ["correct", "incorrect", "partial", "incomplete", 
                            "full credit", "no credit", "pass", "fail"]
        pred_lower = prediction.lower()
        
        for valid in valid_non_numeric:
            if valid in pred_lower:
                return prediction, ""
        
        # If prediction doesn't match known patterns, flag it
        return prediction, f"Note: Unusual grade format '{prediction}'"

    def _extract_prediction(self, msg_history: list[dict], inputs: dict | None = None) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Args:
            msg_history: List of message dictionaries
            inputs: Optional task inputs for validation
            
        Returns:
            (prediction, reasoning) tuple
        """
        if not msg_history:
            return "None", ""
        
        # Search through all messages in reverse order to find the most recent valid JSON
        for msg in reversed(msg_history):
            text = msg.get("text", "")
            if not text:
                continue
            
            # Try standard extraction first
            extracted = _extract_jsons(text)
            if not extracted:
                # Try fuzzy extraction as fallback
                extracted = _extract_json_fuzzy(text)
            
            if extracted:
                last_json = extracted[-1]
                
                # Try multiple possible keys for the response
                prediction = None
                for key in ["response", "grade", "answer", "result", "prediction", "score"]:
                    if key in last_json:
                        prediction = last_json[key]
                        break
                
                if prediction is None:
                    prediction = "None"
                
                # Get reasoning
                reasoning = last_json.get("reasoning", "")
                if not reasoning:
                    # Try alternative keys for reasoning
                    for key in ["analysis", "explanation", "thought", "thoughts", "evaluation"]:
                        if key in last_json:
                            reasoning = last_json[key]
                            break
                
                # Clean up prediction
                if isinstance(prediction, (int, float)):
                    prediction = str(prediction)
                elif isinstance(prediction, str):
                    prediction = prediction.strip()
                else:
                    prediction = str(prediction)
                
                # Validate prediction is not empty
                if prediction and prediction != "None":
                    # Apply validation if inputs provided
                    if inputs:
                        prediction, validation_note = self._validate_prediction(prediction, inputs)
                        if validation_note:
                            reasoning = f"{reasoning}\n\n[Validation: {validation_note}]".strip()
                    return prediction, str(reasoning)
        
        return "None", ""

    def _calculate_confidence(self, prediction: str, reasoning: str, msg_history: list[dict]) -> float:
        """Calculate a confidence score for the prediction.
        
        Returns a score between 0.0 and 1.0 based on:
        - Presence of detailed reasoning
        - Prediction format validity
        - Consistency across multiple extractions (if available)
        
        Args:
            prediction: The extracted prediction
            reasoning: The extracted reasoning
            msg_history: Full message history
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Start with neutral confidence
        
        # Factor 1: Reasoning quality (0.0 - 0.3)
        if reasoning:
            reasoning_len = len(reasoning.strip())
            if reasoning_len > 200:
                confidence += 0.3  # Detailed reasoning
            elif reasoning_len > 100:
                confidence += 0.2  # Moderate reasoning
            elif reasoning_len > 50:
                confidence += 0.1  # Brief reasoning
        
        # Factor 2: Prediction format validity (0.0 - 0.3)
        try:
            # Numeric predictions are more reliable
            float(prediction)
            confidence += 0.3
        except ValueError:
            # Check for known valid non-numeric formats
            valid_patterns = ["correct", "incorrect", "partial", "full", "pass", "fail"]
            pred_lower = prediction.lower()
            if any(pattern in pred_lower for pattern in valid_patterns):
                confidence += 0.25
            else:
                confidence += 0.1  # Unknown format, lower confidence
        
        # Factor 3: JSON extraction quality (0.0 - 0.2)
        if msg_history:
            last_msg = msg_history[-1].get("text", "")
            if "<json>" in last_msg and "</json>" in last_msg:
                confidence += 0.2  # Properly formatted JSON
            elif "<json>" in last_msg or "</json>" in last_msg:
                confidence += 0.1  # Partially formatted
        
        # Factor 4: Prediction not None (0.0 - 0.2)
        if prediction and prediction != "None":
            confidence += 0.2
        
        return min(1.0, max(0.0, confidence))

    def forward(self, inputs: dict) -> tuple[str, list[dict], dict]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history, metadata) where metadata includes confidence score
        """
        base_instruction = self._build_prompt(inputs)
        instruction = base_instruction
        
        prediction = "None"
        reasoning = ""
        all_msg_history = []
        confidence = 0.0
        
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                all_msg_history.extend(msg_history)
                
                prediction, reasoning = self._extract_prediction(msg_history, inputs)
                
                # Calculate confidence for this prediction
                confidence = self._calculate_confidence(prediction, reasoning, msg_history)
                
                # Validate that we got a meaningful prediction
                if prediction != "None" and prediction.strip():
                    self.log_fn(f"Successfully extracted prediction: {prediction} (confidence: {confidence:.2f})")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Build a more specific hint based on what went wrong
                    if not msg_history or not msg_history[-1].get("text", "").strip():
                        instruction = base_instruction + "\n\nIMPORTANT: Please provide a complete response with your analysis and grade."
                    else:
                        last_text = msg_history[-1].get("text", "")
                        if "<json>" not in last_text:
                            instruction = base_instruction + "\n\nIMPORTANT: Your response MUST be wrapped in <json>...</json> tags."
                        elif "</json>" not in last_text:
                            instruction = base_instruction + "\n\nIMPORTANT: You started with <json> but forgot to close with </json>."
                        else:
                            instruction = base_instruction + "\n\nIMPORTANT: Make sure your JSON is valid and includes both 'reasoning' and 'response' fields."
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1}: Error during LLM call: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return what we have
                    break
        
        # Log final result
        if prediction == "None" or not prediction.strip():
            self.log_fn("Warning: Could not extract valid prediction after all retries")
        
        metadata = {
            "confidence": confidence,
            "reasoning": reasoning,
            "attempts": attempt + 1,
        }
        
        return str(prediction), all_msg_history, metadata
