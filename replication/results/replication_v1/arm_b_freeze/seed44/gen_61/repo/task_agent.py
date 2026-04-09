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


def _extract_json_flexible(text: str) -> tuple[dict | None, str]:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple approaches in order:
    1. Standard <json>...</json> blocks
    2. Markdown code blocks with json
    3. Raw JSON objects with "response" key
    4. JSON objects with "reasoning" key
    5. Any valid JSON object at the end of the text
    6. Look for standalone 0 or 1 in the text
    7. Look for "correct" or "incorrect" keywords
    
    Returns:
        Tuple of (extracted_dict or None, extraction_method_name)
    """
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        return results[-1], "json_tags"
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip()), "markdown"
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0)), "json_pattern"
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON with "reasoning" key (full schema)
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|\d+)\s*\}'
    for match in re.finditer(full_json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0)), "full_json_pattern"
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
                            return parsed, "last_brace"
                    except json.JSONDecodeError:
                        continue
                    break
    
    # Strategy 6: Look for standalone 0 or 1 as last resort
    # This handles cases where the model just outputs the number
    text_stripped = text.strip()
    if text_stripped == "0" or text_stripped == "1":
        return {"response": int(text_stripped), "reasoning": "Direct numeric response"}, "standalone_digit"
    
    # Look for 0 or 1 at the very end of the text
    last_char = text_stripped[-1:] if text_stripped else ""
    if last_char in ["0", "1"]:
        # Check if it's a standalone digit (not part of a larger number)
        if len(text_stripped) == 1 or not text_stripped[-2].isdigit():
            return {"response": int(last_char), "reasoning": "Extracted from end of response"}, "end_digit"
    
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
        r'\bmark\s+as\s+correct\b',
        r'\baward\s+full\s+points\b',
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
        r'\bmark\s+as\s+incorrect\b',
        r'\bno\s+points\b',
        r'\bzero\s+points\b',
    ]
    
    # Count matches for each category
    correct_count = sum(1 for pattern in correct_patterns if re.search(pattern, text_lower))
    incorrect_count = sum(1 for pattern in incorrect_patterns if re.search(pattern, text_lower))
    
    # Only use keyword detection if we have a clear signal
    if correct_count > incorrect_count and correct_count > 0:
        return {"response": 1, "reasoning": f"Detected correctness keywords ({correct_count} matches)"}, "keywords"
    elif incorrect_count > correct_count and incorrect_count > 0:
        return {"response": 0, "reasoning": f"Detected incorrectness keywords ({incorrect_count} matches)"}, "keywords"
    
    return None, "none"


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
        self.max_retries = 5  # Increased for better robustness on difficult cases
        self.confidence_threshold = 0.65  # Slightly lowered to reduce unnecessary retries while maintaining quality
        
        # Statistics tracking for performance monitoring
        self.stats = {
            "total_graded": 0,
            "high_confidence": 0,
            "low_confidence_retries": 0,
            "fallback_to_default": 0,
            "json_extraction_failures": 0,
            "reasoning_quality_scores": [],
            "extraction_methods_used": [],  # Track which extraction methods succeed
        }
        
        # Store last reasoning for debugging
        self._last_reasoning = ""

    def _calculate_confidence(self, extracted: dict | None, response_text: str, extraction_method: str = "unknown") -> float:
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
        
        # Track extraction method used
        if extraction_method not in self.stats["extraction_methods_used"]:
            self.stats["extraction_methods_used"].append(extraction_method)
        
        confidence = 0.30  # Base confidence
        
        # Signal 1: Reasoning presence and quality (up to 0.40)
        if "reasoning" in extracted and extracted["reasoning"]:
            reasoning = extracted["reasoning"]
            self._last_reasoning = reasoning  # Store for debugging
            confidence += 0.15
            
            # Bonus for detailed reasoning (at least 100 chars for thorough analysis)
            reasoning_len = len(reasoning)
            if reasoning_len >= 200:
                confidence += 0.15
            elif reasoning_len >= 150:
                confidence += 0.12
            elif reasoning_len >= 100:
                confidence += 0.08
            elif reasoning_len >= 50:
                confidence += 0.04
            
            # Bonus for structured reasoning with clear steps
            structure_markers = ["1.", "2.", "3.", "step", "first", "second", "third", "fourth", "fifth",
                               "therefore", "because", "since", "thus", "hence", "so", "consequently",
                               "conclusion", "analysis", "compare", "evaluate", "assess", "determine",
                               "check", "verify", "examine", "review", "note", "observe", "see"]
            structure_count = sum(1 for marker in structure_markers if marker in reasoning.lower())
            if structure_count >= 5:
                confidence += 0.1
            elif structure_count >= 3:
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
        
        # Signal 3: Mathematical content indicators (up to 0.15)
        math_indicators = ["=", "+", "-", "*", "/", "^", "√", "∑", "∫", "lim", "sin", "cos", "tan",
                          "π", "∞", "∈", "⊂", "∪", "∩", "∀", "∃", "⇒", "⇔", "≤", "≥", "≠", "≡",
                          "mod", "gcd", "lcm", "prime", "factor", "divisible", "congruent"]
        math_count = sum(1 for indicator in math_indicators if indicator in response_text)
        if math_count >= 5:
            confidence += 0.15
        elif math_count >= 3:
            confidence += 0.1
        elif math_count >= 1:
            confidence += 0.05
        
        # Signal 4: Response value validation (up to 0.05)
        if "response" in extracted:
            response_val = extracted["response"]
            if response_val in [0, 1, "0", "1"]:
                confidence += 0.05
        
        # Signal 5: Extraction method reliability (up to 0.15)
        # Higher confidence for structured JSON vs keyword fallback
        if extraction_method == "json_tags":
            confidence += 0.15  # Best method
        elif extraction_method == "markdown":
            confidence += 0.12
        elif extraction_method == "json_pattern":
            confidence += 0.10
        elif extraction_method == "full_json_pattern":
            confidence += 0.10
        elif extraction_method == "last_brace":
            confidence += 0.08
        elif extraction_method == "standalone_digit":
            confidence += 0.05
        elif extraction_method == "end_digit":
            confidence += 0.04
        elif extraction_method == "keywords":
            confidence += 0.03  # Lowest - keyword-based extraction
        else:
            confidence += 0.05
        
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
    
    def get_last_reasoning(self) -> str:
        """Return the reasoning from the last graded problem for debugging."""
        return self._last_reasoning
    
    def reset_statistics(self) -> None:
        """Reset grading statistics."""
        self.stats = {
            "total_graded": 0,
            "high_confidence": 0,
            "low_confidence_retries": 0,
            "fallback_to_default": 0,
            "json_extraction_failures": 0,
            "reasoning_quality_scores": [],
            "extraction_methods_used": [],
        }
        self._last_reasoning = ""

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
   - Check if the final answer matches numerically or symbolically
   - Verify the reasoning and methodology step by step
   - Look for any errors, omissions, or logical gaps
   - Check if the student used valid mathematical techniques
4. Apply the grading guidelines strictly:
   - Award points only for correct work
   - Deduct for errors, even if the final answer is correct
   - Consider partial credit only if explicitly allowed by guidelines
5. Make a final determination:
   - Use response=1 ONLY if the answer is fully correct with valid reasoning
   - Use response=0 if there are ANY significant errors, missing steps, or invalid reasoning

IMPORTANT: Be strict and conservative in your evaluation. 
- A response of 1 requires the student to have demonstrated complete and correct understanding.
- A response of 0 should be used if there is ANY doubt about correctness.
- When in doubt, mark as 0 (incorrect).

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Include specific observations about what the student did right or wrong. Be thorough and mention specific mathematical details.",
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
                extracted, extraction_method = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    confidence = self._calculate_confidence(extracted, response_text, extraction_method)
                    
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
                            self.log_fn(f"High confidence prediction: {pred_str} (confidence: {confidence:.2f}, method: {extraction_method})")
                            return pred_str, msg_history
                        else:
                            self.stats["low_confidence_retries"] += 1
                            self.log_fn(f"Low confidence prediction: {pred_str} (confidence: {confidence:.2f}, method: {extraction_method}), retrying...")
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
