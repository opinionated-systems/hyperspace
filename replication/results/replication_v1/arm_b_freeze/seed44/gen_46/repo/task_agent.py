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
    7. Look for "correct" or "incorrect" keywords
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
    
    # Strategy 7: Look for explicit correctness keywords in the text
    # This helps when the model describes the answer in natural language
    text_lower = text.lower()
    
    # Check for explicit correctness statements
    correct_patterns = [
        r'the\s+student[\'"\s]+s?\s+answer\s+is\s+correct',
        r'the\s+answer\s+is\s+correct',
        r'student\s+is\s+correct',
        r'answer\s+is\s+right',
        r'correct\s+answer',
        r'should\s+be\s+marked\s+as\s+correct',
        r'grade.*as\s+correct',
        r'response.*is\s+1',
        r'\bis\s+correct\b',
    ]
    
    incorrect_patterns = [
        r'the\s+student[\'"\s]+s?\s+answer\s+is\s+incorrect',
        r'the\s+answer\s+is\s+incorrect',
        r'student\s+is\s+incorrect',
        r'answer\s+is\s+wrong',
        r'incorrect\s+answer',
        r'should\s+be\s+marked\s+as\s+incorrect',
        r'grade.*as\s+incorrect',
        r'response.*is\s+0',
        r'\bis\s+incorrect\b',
        r'\bis\s+wrong\b',
    ]
    
    # Count matches for each category
    correct_count = sum(1 for pattern in correct_patterns if re.search(pattern, text_lower))
    incorrect_count = sum(1 for pattern in incorrect_patterns if re.search(pattern, text_lower))
    
    # Only use keyword detection if we have a clear signal
    if correct_count > incorrect_count and correct_count > 0:
        return {"response": 1, "reasoning": f"Detected correctness keywords ({correct_count} matches)"}
    elif incorrect_count > correct_count and incorrect_count > 0:
        return {"response": 0, "reasoning": f"Detected incorrectness keywords ({incorrect_count} matches)"}
    
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
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Features:
    - Multi-strategy JSON extraction for robust parsing
    - Adaptive confidence scoring with multiple quality signals
    - Grading statistics tracking for performance monitoring
    - Retry logic with confidence-based early termination
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.confidence_threshold = 0.7  # Minimum confidence for auto-acceptance
        
        # Statistics tracking for performance monitoring
        self.stats = {
            "total_graded": 0,
            "high_confidence": 0,
            "low_confidence_retries": 0,
            "fallback_to_default": 0,
            "json_extraction_failures": 0,
            "reasoning_quality_scores": [],
        }

    def _calculate_confidence(self, extracted: dict | None, response_text: str) -> float:
        """Calculate confidence score for the prediction using multiple quality signals.
        
        Returns a score between 0 and 1 based on:
        - Presence and quality of reasoning
        - Response format quality
        - Semantic coherence indicators
        - Mathematical content indicators
        - Extraction method reliability
        """
        if extracted is None:
            self.stats["json_extraction_failures"] += 1
            return 0.0
        
        confidence = 0.35  # Base confidence (slightly lower to allow more granularity)
        
        # Signal 1: Reasoning presence and quality (up to 0.35)
        if "reasoning" in extracted and extracted["reasoning"]:
            reasoning = extracted["reasoning"]
            confidence += 0.15
            
            # Bonus for detailed reasoning (at least 100 chars for thorough analysis)
            reasoning_len = len(reasoning)
            if reasoning_len >= 150:
                confidence += 0.1
            elif reasoning_len >= 100:
                confidence += 0.075
            elif reasoning_len >= 50:
                confidence += 0.05
            
            # Bonus for structured reasoning with clear steps
            structure_markers = ["1.", "2.", "3.", "step", "first", "second", "third", 
                               "therefore", "because", "since", "thus", "hence", 
                               "conclusion", "analysis", "compare", "evaluate"]
            structure_count = sum(1 for marker in structure_markers if marker in reasoning.lower())
            if structure_count >= 3:
                confidence += 0.1
            elif structure_count >= 2:
                confidence += 0.075
            elif structure_count >= 1:
                confidence += 0.03
            
            # Track reasoning quality for statistics
            self.stats["reasoning_quality_scores"].append(min(reasoning_len / 200, 1.0))
        
        # Signal 2: Format quality (up to 0.15)
        if "<json>" in response_text and "</json>" in response_text:
            confidence += 0.1
            # Bonus for properly formatted JSON with newlines
            if '"reasoning":' in response_text and '"response":' in response_text:
                confidence += 0.05
        
        # Signal 3: Mathematical content indicators (up to 0.1)
        math_indicators = ["=", "+", "-", "*", "/", "^", "√", "∑", "∫", "lim", "sin", "cos", "tan",
                          "π", "∞", "∈", "⊂", "∪", "∩", "∀", "∃", "⇒", "⇔"]
        math_count = sum(1 for indicator in math_indicators if indicator in response_text)
        if math_count >= 3:
            confidence += 0.1
        elif math_count >= 1:
            confidence += 0.05
        
        # Signal 4: Response value validation (up to 0.05)
        if "response" in extracted:
            response_val = extracted["response"]
            if response_val in [0, 1, "0", "1"]:
                confidence += 0.05
        
        # Signal 5: Extraction method reliability (up to 0.1)
        # Higher confidence for structured JSON vs keyword fallback
        if extracted.get("reasoning", "").startswith("Detected"):
            # Keyword-based extraction - lower confidence
            confidence += 0.02
        elif extracted.get("reasoning", "").startswith("Direct numeric"):
            # Direct numeric extraction - medium confidence
            confidence += 0.05
        elif extracted.get("reasoning", "").startswith("Extracted from end"):
            # End-of-text extraction - medium-low confidence
            confidence += 0.03
        else:
            # Full JSON extraction - higher confidence
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_statistics(self) -> dict:
        """Return grading statistics for performance monitoring."""
        stats = self.stats.copy()
        if stats["reasoning_quality_scores"]:
            stats["avg_reasoning_quality"] = sum(stats["reasoning_quality_scores"]) / len(stats["reasoning_quality_scores"])
        else:
            stats["avg_reasoning_quality"] = 0.0
        del stats["reasoning_quality_scores"]  # Remove raw scores from output
        return stats
    
    def reset_statistics(self) -> None:
        """Reset grading statistics."""
        self.stats = {
            "total_graded": 0,
            "high_confidence": 0,
            "low_confidence_retries": 0,
            "fallback_to_default": 0,
            "json_extraction_failures": 0,
            "reasoning_quality_scores": [],
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with statistics tracking.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_graded"] += 1
        
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            self.stats["fallback_to_default"] += 1
            return "0", []
        
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions for mathematical olympiad problems.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide detailed analysis:
1. Analyze what the problem is asking for and identify key requirements
2. Review the correct solution approach and identify critical steps
3. Compare the student's answer to the correct solution:
   - Check if the final answer matches
   - Verify the reasoning and methodology
   - Look for any errors or omissions
4. Apply the grading guidelines strictly:
   - Award points only for correct work
   - Deduct for errors, even if the final answer is correct
   - Consider partial credit where applicable
5. Make a final determination:
   - Use response=1 if the answer is fully correct or meets passing criteria
   - Use response=0 if the answer is incorrect, incomplete, or fails criteria

IMPORTANT: Be strict in your evaluation. A response of 1 means the student has demonstrated full understanding. A response of 0 means there are significant errors or the answer does not meet requirements.

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Include specific observations about what the student did right or wrong.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Do not include any other text outside the JSON block."""

        best_prediction = None
        best_confidence = 0.0
        best_history = []
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
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
                        
                        # Track best prediction by confidence
                        if confidence > best_confidence:
                            best_prediction = pred_str
                            best_confidence = confidence
                            best_history = msg_history
                            
                        # If we have high confidence, return immediately
                        if confidence >= self.confidence_threshold:
                            self.stats["high_confidence"] += 1
                            self.log_fn(f"High confidence prediction: {pred_str} (confidence: {confidence:.2f})")
                            return pred_str, msg_history
                        else:
                            self.stats["low_confidence_retries"] += 1
                            self.log_fn(f"Low confidence prediction: {pred_str} (confidence: {confidence:.2f}), retrying...")
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Return best prediction if we have one, otherwise default to "0"
        if best_prediction is not None:
            self.log_fn(f"Returning best prediction: {best_prediction} (confidence: {best_confidence:.2f})")
            return best_prediction, best_history
        
        # Fallback: return "0" if all retries failed
        self.stats["fallback_to_default"] += 1
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", best_history if best_history else []
