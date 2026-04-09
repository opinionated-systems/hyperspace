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
import time
from typing import Literal

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
    
    Tries multiple approaches in order of reliability:
    1. Standard <json>...</json> blocks (most reliable)
    2. Markdown code blocks with json
    3. Bracket-balanced JSON extraction with field scoring
    4. Relaxed pattern matching for common LLM output formats
    """
    # Strategy 1: Standard <json> tags (most reliable)
    results = _extract_jsons(text)
    if results:
        # Return the one with the most complete schema
        best = max(results, key=lambda r: ("reasoning" in r) + ("response" in r))
        if "response" in best:
            return best
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Find all potential JSON objects with proper brace balancing
    # This is the most robust approach for nested structures
    potential_jsons = []
    for match in re.finditer(r'\{', text):
        start = match.start()
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "response" in parsed:
                            # Score based on having both required fields
                            score = ("reasoning" in parsed) * 2 + ("response" in parsed)
                            potential_jsons.append((score, parsed))
                    except json.JSONDecodeError:
                        pass
                    break
    
    # Return the highest-scoring valid JSON (prefer ones with both fields)
    if potential_jsons:
        potential_jsons.sort(reverse=True, key=lambda x: x[0])
        return potential_jsons[0][1]
    
    # Strategy 4: Relaxed pattern matching for common LLM output formats
    # Handle cases where JSON might be malformed but extractable
    relaxed_patterns = [
        # Pattern: "response": 0.5 or "response": 1
        r'"response"\s*:\s*([0-9.]+)',
        # Pattern: response: 0.5 (without quotes)
        r'response\s*:\s*([0-9.]+)',
    ]
    
    for pattern in relaxed_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = match.group(1)
                # Try to extract reasoning if available
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL)
                reasoning = reasoning_match.group(1) if reasoning_match else "Extracted from relaxed pattern"
                return {"reasoning": reasoning, "response": float(value) if '.' in value else int(value)}
            except (ValueError, IndexError):
                continue
    
    return None


def _categorize_error(error: Exception, extracted: dict | None) -> tuple[Literal["llm_error", "json_error", "validation_error", "unknown"], str]:
    """Categorize the type of error for better retry decisions.
    
    Returns:
        Tuple of (error_category, error_message)
    """
    error_msg = str(error).lower()
    error_type = type(error).__name__
    
    # LLM API errors - these benefit from retry with backoff
    llm_error_indicators = [
        "rate limit", "timeout", "connection", "server error", 
        "503", "502", "429", "500", "503", "504",
        "api error", "service unavailable", "too many requests",
        "context length exceeded", "max tokens", "token limit"
    ]
    if any(x in error_msg for x in llm_error_indicators):
        return "llm_error", f"LLM API error ({error_type}): {error}"
    
    # JSON extraction errors - may benefit from retry with different prompt
    if extracted is None:
        return "json_error", f"JSON extraction failed ({error_type}): {error}"
    
    # Validation errors - the response format is wrong, retry may help
    if isinstance(error, (ValueError, TypeError)):
        return "validation_error", f"Validation error ({error_type}): {error}"
    
    # JSON decode errors
    if "json" in error_msg or "decode" in error_msg:
        return "json_error", f"JSON parsing error ({error_type}): {error}"
    
    return "unknown", f"Unexpected error ({error_type}): {error}"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Supports both binary (0/1) and partial credit (0.0-1.0) scoring modes.
    Implements progressive retry with enhanced prompts for better robustness.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", partial_credit: bool = False) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.partial_credit = partial_credit
        # Exponential backoff: base delay in seconds
        self._base_delay = 1.0
        self._max_delay = 30.0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Check if problem supports partial credit
        supports_partial = inputs.get("supports_partial_credit", self.partial_credit)

        # Build base prompt
        if supports_partial:
            base_prompt = self._build_partial_credit_prompt(
                domain, problem, solution, grading_guidelines, student_answer
            )
        else:
            base_prompt = self._build_binary_prompt(
                domain, problem, solution, grading_guidelines, student_answer
            )

        # Try with retries for robustness, using exponential backoff
        last_error = None
        last_extracted = None
        all_msg_histories = []
        
        for attempt in range(self.max_retries):
            # Progressive prompt enhancement on retries
            instruction = self._enhance_prompt_for_retry(base_prompt, attempt, last_error)
            
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                all_msg_histories.extend(msg_history)
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                last_extracted = extracted
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Validate prediction based on scoring mode
                    if supports_partial:
                        # Partial credit: 0.0 to 1.0
                        try:
                            pred_val = float(prediction)
                            if 0.0 <= pred_val <= 1.0:
                                return str(pred_val), all_msg_histories
                        except (ValueError, TypeError) as e:
                            last_error = e
                            self.log_fn(f"Invalid partial credit value: {prediction}, retrying...")
                    else:
                        # Binary: 0 or 1
                        if prediction in [0, 1, "0", "1"]:
                            return str(prediction), all_msg_histories
                        last_error = ValueError(f"Invalid binary prediction: {prediction}")
                        self.log_fn(f"Invalid binary prediction value: {prediction}, retrying...")
                else:
                    last_error = ValueError("No valid JSON with 'response' field found")
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    
            except Exception as e:
                last_error = e
                error_category, error_msg = _categorize_error(e, last_extracted)
                self.log_fn(f"Error on attempt {attempt + 1}/{self.max_retries} ({error_category}): {error_msg}")
                
                # Exponential backoff with jitter for LLM errors
                if error_category == "llm_error" and attempt < self.max_retries - 1:
                    delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                    self.log_fn(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return default prediction if all retries failed
        default_pred = "0.0" if supports_partial else "0"
        if last_error:
            error_category, _ = _categorize_error(last_error, last_extracted)
            self.log_fn(f"All retries failed ({error_category}), returning default prediction {default_pred}")
        else:
            self.log_fn(f"All retries failed, returning default prediction {default_pred}")
        return default_pred, all_msg_histories
    
    def _enhance_prompt_for_retry(self, base_prompt: str, attempt: int, last_error: Exception | None) -> str:
        """Enhance the prompt for retry attempts with additional guidance."""
        if attempt == 0:
            return base_prompt
        
        # Add retry-specific guidance
        enhancements = [
            "\n\nIMPORTANT: Your previous response could not be parsed. Please ensure you respond ONLY with valid JSON in the exact format specified above.",
            "\n\nCRITICAL: This is a retry attempt. You MUST output valid JSON with 'reasoning' and 'response' fields. No other text before or after the JSON.",
            "\n\nFINAL ATTEMPT: Output ONLY the JSON object. Do not include any explanations, markdown formatting, or other text. Just the raw JSON.",
        ]
        
        enhancement = enhancements[min(attempt - 1, len(enhancements) - 1)]
        
        # Add error-specific guidance
        if last_error:
            error_str = str(last_error).lower()
            if "json" in error_str or "extract" in error_str:
                enhancement += "\nMake sure to wrap your JSON in <json>...</json> tags."
            elif "valid" in error_str or "range" in error_str:
                enhancement += "\nDouble-check that your response value is within the valid range."
        
        return base_prompt + enhancement

    def _build_binary_prompt(self, domain: str, problem: str, solution: str, 
                            grading_guidelines: str, student_answer: str) -> str:
        """Build prompt for binary (0/1) scoring mode."""
        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

You MUST respond with ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "response": 1 or 0
}}
</json>

Requirements:
- The "response" field must be either 1 (correct) or 0 (incorrect)
- The "reasoning" field must contain your analysis
- Output ONLY the JSON wrapped in <json> tags, nothing else"""

    def _build_partial_credit_prompt(self, domain: str, problem: str, solution: str,
                                   grading_guidelines: str, student_answer: str) -> str:
        """Build prompt for partial credit (0.0-1.0) scoring mode."""
        return f"""You are an expert {domain} grader evaluating student solutions with partial credit.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Identify which parts the student got correct vs incorrect
5. Assess the student's reasoning quality and approach
6. Assign a partial credit score from 0.0 (completely wrong) to 1.0 (completely correct)

Scoring guidelines:
- 1.0: Fully correct solution with proper reasoning
- 0.7-0.9: Mostly correct with minor errors or gaps
- 0.4-0.6: Partially correct, significant progress but major gaps
- 0.1-0.3: Minimal correct elements, mostly wrong approach
- 0.0: Completely wrong or no valid attempt

You MUST respond with ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here, including what was correct and what was wrong",
    "response": 0.0 to 1.0
}}
</json>

Requirements:
- The "response" field must be a decimal number between 0.0 and 1.0 (inclusive)
- The "reasoning" field must contain your detailed analysis
- Output ONLY the JSON wrapped in <json> tags, nothing else"""
