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
            continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Look for standalone 0 or 1 in the text
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON with "reasoning" key (full schema)
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(full_json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for any JSON object at the end of text (last resort)
    # This handles cases where the model outputs JSON without any markers
    last_brace_idx = text.rfind('}')
    if last_brace_idx != -1:
        # Find the matching opening brace
        brace_count = 0
        for i in range(last_brace_idx, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        candidate = text[i:last_brace_idx + 1]
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "response" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue
                    break
    
    # Strategy 6: Look for standalone 0 or 1 as last resort
    # This handles cases where the model just outputs the number
    text_stripped = text.strip()
    if text_stripped == "0" or text_stripped == "1":
        return {"response": int(text_stripped), "reasoning": "Direct numeric response"}
    
    # Look for 0 or 1 at the very end of the text
    last_char = text_stripped[-1:] if text_stripped else ""
    if last_char in ["0", "1"]:
        # Check if it's a standalone digit (not part of a larger number)
        if len(text_stripped) == 1 or not text_stripped[-2].isdigit():
            return {"response": int(last_char), "reasoning": "Extracted from end of response"}
    
    return None


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that all required fields are present and non-empty.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
    
    for field in required_fields:
        if field not in inputs:
            return False, f"Missing required field: {field}"
        if not inputs[field] or not str(inputs[field]).strip():
            return False, f"Empty required field: {field}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 5  # Increased retries for better accuracy
        self.confidence_threshold = 0.75  # Slightly higher threshold for auto-acceptance

    def _calculate_confidence(self, extracted: dict | None, response_text: str) -> float:
        """Calculate confidence score for the prediction.
        
        Returns a score between 0 and 1 based on:
        - Presence of reasoning
        - Response format quality
        - Length and coherence of reasoning
        - Mathematical analysis depth
        """
        if extracted is None:
            return 0.0
        
        confidence = 0.4  # Base confidence (slightly lower to encourage thorough analysis)
        
        # Bonus for having reasoning
        if "reasoning" in extracted and extracted["reasoning"]:
            reasoning = extracted["reasoning"]
            confidence += 0.25
            
            # Bonus for detailed reasoning (at least 100 chars for thorough analysis)
            if len(reasoning) >= 100:
                confidence += 0.15
            elif len(reasoning) >= 50:
                confidence += 0.05
            
            # Bonus for structured reasoning (contains step numbers or bullet points)
            if any(marker in reasoning.lower() for marker in ["1.", "2.", "3.", "step", "first", "second", "third"]):
                confidence += 0.1
            
            # Bonus for mathematical analysis keywords
            math_keywords = ["correct", "incorrect", "solution", "answer", "compare", "match", 
                           "equivalent", "different", "error", "mistake", "valid", "invalid"]
            keyword_count = sum(1 for kw in math_keywords if kw in reasoning.lower())
            confidence += min(keyword_count * 0.02, 0.1)  # Max 0.1 bonus for keywords
        
        # Bonus for proper JSON format with <json> tags
        if "<json>" in response_text and "</json>" in response_text:
            confidence += 0.1
        
        # Penalty for very short responses (likely incomplete analysis)
        if len(response_text) < 50:
            confidence -= 0.2
        
        return max(0.0, min(confidence, 1.0))

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return "0", []
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions for mathematical olympiad problems.

Your task is to grade a student's answer by carefully comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. Analyze what the problem is asking for - identify key requirements and expected outcomes
2. Review the correct solution approach - understand the method and final answer
3. Compare the student's answer to the correct solution - check both the method and final result
4. Check if the student followed the grading guidelines - look for specific criteria mentioned
5. Identify any errors or misconceptions in the student's work
6. Determine if the student's answer demonstrates sufficient understanding (1) or is incorrect (0)

Important grading considerations:
- The student's answer should be mathematically equivalent to the correct solution
- Partial credit is NOT awarded - the answer must be fully correct to receive 1
- Check for common errors: calculation mistakes, wrong formulas, missing steps, incorrect units
- If the student answer is blank or completely unrelated, mark as 0

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis explaining your grading decision. Be specific about what the student did right or wrong.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Be strict in your grading - only award 1 if the answer is fully correct."""

        best_prediction = None
        best_confidence = 0.0
        best_history = []
        predictions_count = {"0": 0, "1": 0}  # Track vote counts
        
        # Try with retries for robustness, using increasing temperature for diversity
        for attempt in range(self.max_retries):
            try:
                # Increase temperature slightly with each retry for more diverse responses
                temperature = min(0.3, attempt * 0.05)
                
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                    temperature=temperature,
                )
                
                response_text = msg_history[-1]["text"]
                
                # Extract prediction from JSON using flexible extraction
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    confidence = self._calculate_confidence(extracted, response_text)
                    
                    # Validate prediction is 0 or 1
                    if prediction in [0, 1, "0", "1"]:
                        pred_str = str(prediction)
                        predictions_count[pred_str] += 1
                        
                        # Track best prediction by confidence
                        if confidence > best_confidence:
                            best_prediction = pred_str
                            best_confidence = confidence
                            best_history = msg_history
                            
                        # If we have high confidence, return immediately
                        if confidence >= self.confidence_threshold:
                            self.log_fn(f"High confidence prediction: {pred_str} (confidence: {confidence:.2f}, attempt {attempt + 1})")
                            return pred_str, msg_history
                        else:
                            self.log_fn(f"Low confidence prediction: {pred_str} (confidence: {confidence:.2f}, attempt {attempt + 1}), retrying...")
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Return best prediction if we have one with sufficient confidence
        if best_prediction is not None and best_confidence >= 0.5:
            self.log_fn(f"Returning best prediction: {best_prediction} (confidence: {best_confidence:.2f})")
            return best_prediction, best_history
        
        # If no high-confidence prediction, use majority voting
        if predictions_count["0"] > predictions_count["1"]:
            self.log_fn(f"Using majority vote: 0 (votes: {predictions_count['0']} vs {predictions_count['1']})")
            return "0", best_history if best_history else []
        elif predictions_count["1"] > predictions_count["0"]:
            self.log_fn(f"Using majority vote: 1 (votes: {predictions_count['1']} vs {predictions_count['0']})")
            return "1", best_history if best_history else []
        
        # Tie or no predictions: default to "0" (conservative grading)
        self.log_fn(f"Tie or no clear prediction, defaulting to 0 (votes: {predictions_count})")
        return "0", best_history if best_history else []
