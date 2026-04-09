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
    Includes detailed logging for debugging extraction failures.
    """
    results = []
    search_from = 0
    extraction_attempts = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Found opening <json> at position {start} but no closing </json>")
            break
        
        inner = text[start + 6:end].strip()
        search_from = end + 7
        extraction_attempts += 1
        
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully extracted JSON block #{len(results)}")
            else:
                logger.warning(f"Extracted JSON is not a dict, got {type(parsed).__name__}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in block #{extraction_attempts}: {e}")
            # Try to extract partial JSON if possible
            try:
                # Look for nested JSON objects
                brace_start = inner.find('{')
                brace_end = inner.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    partial = inner[brace_start:brace_end + 1]
                    parsed = json.loads(partial)
                    if isinstance(parsed, dict):
                        results.append(parsed)
                        logger.info(f"Recovered partial JSON from block #{extraction_attempts}")
            except Exception:
                pass  # Partial extraction failed, continue
            continue
    
    if extraction_attempts > 0 and not results:
        logger.warning(f"Attempted to extract {extraction_attempts} JSON blocks but all failed")
    
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
        self.max_retries = 5  # Increased retries for better coverage
        self.confidence_threshold = 0.65  # Slightly lowered for better efficiency
        self.min_acceptable_confidence = 0.45  # Minimum to accept after retries
        
        # Statistics tracking for performance monitoring
        self.stats = {
            "total_graded": 0,
            "high_confidence": 0,
            "low_confidence_retries": 0,
            "fallback_to_default": 0,
            "json_extraction_failures": 0,
            "reasoning_quality_scores": [],
            "early_termination": 0,  # Track when we accept lower confidence
        }

    def _calculate_confidence(self, extracted: dict | None, response_text: str) -> float:
        """Calculate confidence score for the prediction using multiple quality signals.
        
        Returns a score between 0 and 1 based on:
        - Presence and quality of reasoning
        - Response format quality
        - Semantic coherence indicators
        - Mathematical content indicators
        - Extraction method reliability
        - Consistency between reasoning and response
        """
        if extracted is None:
            self.stats["json_extraction_failures"] += 1
            return 0.0
        
        confidence = 0.30  # Base confidence
        
        # Signal 1: Reasoning presence and quality (up to 0.40)
        if "reasoning" in extracted and extracted["reasoning"]:
            reasoning = extracted["reasoning"]
            confidence += 0.15
            
            # Bonus for detailed reasoning
            reasoning_len = len(reasoning)
            if reasoning_len >= 200:
                confidence += 0.12
            elif reasoning_len >= 150:
                confidence += 0.10
            elif reasoning_len >= 100:
                confidence += 0.075
            elif reasoning_len >= 50:
                confidence += 0.05
            
            # Bonus for structured reasoning with clear steps
            structure_markers = ["1.", "2.", "3.", "4.", "5.", "step", "first", "second", "third", 
                               "therefore", "because", "since", "thus", "hence", 
                               "conclusion", "analysis", "compare", "evaluate", "verify"]
            structure_count = sum(1 for marker in structure_markers if marker in reasoning.lower())
            if structure_count >= 4:
                confidence += 0.13
            elif structure_count >= 3:
                confidence += 0.10
            elif structure_count >= 2:
                confidence += 0.07
            elif structure_count >= 1:
                confidence += 0.03
            
            # Track reasoning quality for statistics
            self.stats["reasoning_quality_scores"].append(min(reasoning_len / 200, 1.0))
        
        # Signal 2: Format quality (up to 0.15)
        if "<json>" in response_text and "</json>" in response_text:
            confidence += 0.10
            # Bonus for properly formatted JSON with newlines
            if '"reasoning":' in response_text and '"response":' in response_text:
                confidence += 0.05
        elif "```json" in response_text:
            confidence += 0.08
        
        # Signal 3: Mathematical content indicators (up to 0.10)
        math_indicators = ["=", "+", "-", "*", "/", "^", "√", "∑", "∫", "lim", "sin", "cos", "tan",
                          "π", "∞", "∈", "⊂", "∪", "∩", "∀", "∃", "⇒", "⇔", "≤", "≥", "≠"]
        math_count = sum(1 for indicator in math_indicators if indicator in response_text)
        if math_count >= 5:
            confidence += 0.10
        elif math_count >= 3:
            confidence += 0.08
        elif math_count >= 1:
            confidence += 0.04
        
        # Signal 4: Response value validation (up to 0.05)
        if "response" in extracted:
            response_val = extracted["response"]
            if response_val in [0, 1, "0", "1"]:
                confidence += 0.05
        
        # Signal 5: Consistency check between reasoning and response (up to 0.10)
        reasoning_text = extracted.get("reasoning", "").lower()
        response_val = extracted.get("response")
        if response_val is not None:
            # Check if reasoning mentions correctness/incorrectness that matches response
            correct_keywords = ["correct", "right", "valid", "properly", "accurate", "matches"]
            incorrect_keywords = ["incorrect", "wrong", "invalid", "error", "mistake", "does not match"]
            
            has_correct = any(kw in reasoning_text for kw in correct_keywords)
            has_incorrect = any(kw in reasoning_text for kw in incorrect_keywords)
            
            if response_val == 1 and has_correct and not has_incorrect:
                confidence += 0.10  # Strong consistency
            elif response_val == 0 and has_incorrect and not has_correct:
                confidence += 0.10  # Strong consistency
            elif (response_val == 1 and has_correct) or (response_val == 0 and has_incorrect):
                confidence += 0.05  # Moderate consistency
            elif has_correct and has_incorrect:
                confidence -= 0.05  # Contradictory reasoning
        
        # Signal 6: Extraction method reliability (up to 0.10)
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
            confidence += 0.10
        
        # Signal 7: Mathematical proof depth indicators (up to 0.08)
        # IMO problems require rigorous proof-based reasoning
        proof_depth_keywords = [
            "proof", "theorem", "lemma", "corollary", "proposition",
            "by contradiction", "induction", "base case", "inductive step",
            "without loss of generality", "w.l.o.g.", "wlog",
            "sufficient to show", "necessary condition", "sufficient condition",
            "if and only if", "iff", "equivalent to",
            "assume", "suppose", "let", "define", "denote",
            "observe that", "note that", "recall", "remember",
            "we claim", "we show", "we prove", "it follows",
            "substituting", "rearranging", "simplifying", "expanding",
            "factor", "factorize", "expand", "collect",
            "boundary", "extreme", "maximum", "minimum", "optimal",
            "invariant", "monotonic", "convex", "concave",
            "divisible", "prime", "composite", "gcd", "lcm",
            "modulo", "congruence", "residue", "remainder",
            "bijection", "injection", "surjection", "permutation",
            "combination", "binomial", "multinomial", "catalan"
        ]
        proof_depth_count = sum(1 for kw in proof_depth_keywords if kw in reasoning_text)
        if proof_depth_count >= 8:
            confidence += 0.08  # Strong proof-based reasoning
        elif proof_depth_count >= 5:
            confidence += 0.06  # Good proof depth
        elif proof_depth_count >= 3:
            confidence += 0.04  # Moderate proof depth
        elif proof_depth_count >= 1:
            confidence += 0.02  # Some proof elements
        
        return max(0.0, min(confidence, 1.0))
    
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
            "early_termination": 0,
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

Think step by step and provide detailed analysis following this structure:

1. PROBLEM ANALYSIS: Identify what the problem is asking and key requirements
2. CORRECT SOLUTION REVIEW: Identify the critical steps and expected approach
3. STUDENT ANSWER EVALUATION:
   - Compare the student's final answer to the correct answer
   - Verify the student's reasoning and methodology
   - Identify any errors, omissions, or misconceptions
4. GRADING DECISION:
   - Apply the grading guidelines strictly
   - Award credit only for correct work
   - Consider partial credit if guidelines allow
5. FINAL VERDICT:
   - State clearly whether the answer is CORRECT or INCORRECT
   - Provide the response value (1 for correct, 0 for incorrect)

IMPORTANT GRADING PRINCIPLES:
- Be thorough but fair in your evaluation
- A response of 1 means the student demonstrated full understanding with no significant errors
- A response of 0 means there are significant errors, missing steps, or the answer fails to meet requirements
- Even if the final answer is correct, deduct if the reasoning is flawed
- Even if the final answer is wrong, credit correct partial work if guidelines allow

Respond ONLY in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis following the structure above. Be specific about what the student did right or wrong.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect). Do not include any other text outside the JSON block."""

        best_prediction = None
        best_confidence = 0.0
        best_history = []
        best_extracted = None
        
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
                            best_extracted = extracted
                            
                        # If we have high confidence, return immediately
                        if confidence >= self.confidence_threshold:
                            self.stats["high_confidence"] += 1
                            self.log_fn(f"High confidence prediction: {pred_str} (confidence: {confidence:.2f})")
                            return pred_str, msg_history
                        
                        # Early termination: if we've tried enough and have acceptable confidence
                        if attempt >= 2 and confidence >= self.min_acceptable_confidence:
                            self.stats["early_termination"] += 1
                            self.log_fn(f"Early termination with acceptable confidence: {pred_str} (confidence: {confidence:.2f})")
                            return pred_str, msg_history
                        
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
