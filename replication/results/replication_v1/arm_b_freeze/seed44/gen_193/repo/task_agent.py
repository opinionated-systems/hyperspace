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
    3. Raw JSON objects in text (with nested brace handling)
    4. Look for JSON with "reasoning" and "response" keys
    5. Direct text analysis for yes/no/correct/incorrect patterns
    6. Look for standalone 0 or 1 in the response
    """
    if not text or not text.strip():
        logger.debug("Empty text provided to _extract_json_flexible")
        return None
    
    text_stripped = text.strip()
    
    # Strategy 1: Standard <json> tags
    results = _extract_jsons(text)
    if results:
        logger.debug(f"Extracted JSON from <json> tags: {results[-1]}")
        return results[-1]
    
    # Strategy 2: Markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            logger.debug(f"Extracted JSON from markdown block: {parsed}")
            return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Look for JSON-like structures with balanced braces
    # This handles nested JSON objects properly
    def find_json_objects(s: str) -> list[str]:
        """Find all JSON-like objects with balanced braces."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(s) and brace_count > 0:
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    for obj_str in find_json_objects(text):
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and "response" in parsed:
                logger.debug(f"Extracted JSON from balanced braces: {parsed}")
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON with "response" key (simple pattern)
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            logger.debug(f"Extracted JSON from simple pattern: {parsed}")
            return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Direct text analysis for grading decisions
    # This handles cases where the model doesn't output proper JSON
    text_lower = text.lower()
    
    # Look for explicit grading keywords
    correct_indicators = [
        "the student's answer is correct",
        "the answer is correct",
        "this is correct",
        "mark as correct",
        "should be marked correct",
        "grade: correct",
        "correct (1)",
        "correct: 1",
        "result: 1",
        "score: 1",
        "is correct",
        "correct.",
        "correctly",
    ]
    incorrect_indicators = [
        "the student's answer is incorrect",
        "the answer is incorrect",
        "this is incorrect",
        "mark as incorrect",
        "should be marked incorrect",
        "grade: incorrect",
        "incorrect (0)",
        "incorrect: 0",
        "result: 0",
        "score: 0",
        "is incorrect",
        "incorrect.",
        "wrong",
        "not correct",
    ]
    
    # Check for indicators
    has_correct = any(ind in text_lower for ind in correct_indicators)
    has_incorrect = any(ind in text_lower for ind in incorrect_indicators)
    
    # Also check for standalone numbers at the end of the response
    final_number_pattern = r'(?:final\s+(?:answer|grade|score)|conclusion|therefore)[\s:]*([01])'
    final_match = re.search(final_number_pattern, text_lower)
    
    if final_match:
        result = {"response": int(final_match.group(1)), "reasoning": "Extracted from text analysis"}
        logger.debug(f"Extracted from final number pattern: {result}")
        return result
    
    # If we have clear indicators, use them
    if has_correct and not has_incorrect:
        result = {"response": 1, "reasoning": "Text analysis indicates correct answer"}
        logger.debug(f"Extracted from correct indicators: {result}")
        return result
    elif has_incorrect and not has_correct:
        result = {"response": 0, "reasoning": "Text analysis indicates incorrect answer"}
        logger.debug(f"Extracted from incorrect indicators: {result}")
        return result
    
    # Strategy 6: Look for standalone 0 or 1 at the end of the text
    # This catches cases where the model just outputs the number
    standalone_pattern = r'(?:^|\s)([01])(?:\s*$|\s*\.)'
    standalone_match = re.search(standalone_pattern, text_stripped)
    if standalone_match:
        result = {"response": int(standalone_match.group(1)), "reasoning": "Extracted from standalone number"}
        logger.debug(f"Extracted from standalone number: {result}")
        return result
    
    logger.debug(f"No JSON or pattern found in text: {text[:200]}...")
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Build a structured grading prompt with clear instructions."""
        # Check for empty/invalid student answers early
        student_clean = str(student_answer).strip().lower() if student_answer else ""
        is_empty = not student_clean or student_clean in (
            "i don't know", "i dont know", "idk", "n/a", "none", "null", 
            "empty", "blank", "no answer", "not sure", "unsure"
        )
        
        empty_note = ""
        if is_empty:
            empty_note = """
⚠️ IMPORTANT: The student's answer appears to be empty or invalid. 
According to the grading guidelines, empty/invalid answers should be marked as INCORRECT (0).
"""

        return f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer by comparing it to the correct solution and following the grading guidelines.

=== PROBLEM ===
{problem}

=== CORRECT SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}
{empty_note}

=== GRADING INSTRUCTIONS ===
Think step by step:
1. Analyze what the problem is asking for
2. Review the correct solution approach
3. Compare the student's answer to the correct solution
4. Check if the student followed the grading guidelines
5. Determine if the student's answer is correct (1) or incorrect (0)

Important grading criteria:
- The answer must be mathematically correct
- The reasoning must be sound and complete
- Partial credit is NOT given - the answer is either fully correct (1) or incorrect (0)
- If the student answer is empty, blank, or says "I don't know", mark as incorrect (0)
- Be strict: only mark as correct (1) if the answer is fully correct

=== RESPONSE FORMAT ===
You MUST respond with a valid JSON object wrapped in <json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your thought process clearly.",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain why the answer is wrong.",
    "response": 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect). No other values are accepted.
Remember: Only output the JSON block, nothing else."""

    def _validate_prediction(self, prediction: any) -> str | None:
        """Validate and normalize prediction value.
        
        Returns normalized prediction string or None if invalid.
        """
        # Handle various formats
        if isinstance(prediction, bool):
            return "1" if prediction else "0"
        if isinstance(prediction, (int, float)):
            if prediction == 1 or prediction == 1.0:
                return "1"
            elif prediction == 0 or prediction == 0.0:
                return "0"
        if isinstance(prediction, str):
            pred_clean = prediction.strip().lower()
            if pred_clean in ("1", "true", "correct", "yes"):
                return "1"
            elif pred_clean in ("0", "false", "incorrect", "no"):
                return "0"
        return None

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

        # Early detection of empty/invalid answers
        student_clean = str(student_answer).strip().lower() if student_answer else ""
        is_empty = not student_clean or student_clean in (
            "i don't know", "i dont know", "idk", "n/a", "none", "null",
            "empty", "blank", "no answer", "not sure", "unsure", ""
        )
        
        # Build structured prompt
        instruction = self._build_grading_prompt(
            domain=domain,
            problem=problem,
            solution=solution,
            grading_guidelines=grading_guidelines,
            student_answer=student_answer,
        )

        msg_history = []
        
        # Try with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if msg_history else [],
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    validated = self._validate_prediction(prediction)
                    
                    if validated is not None:
                        self.log_fn(f"Valid prediction: {validated} (raw: {prediction})")
                        return validated, msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: for empty/invalid answers, return 0 immediately
        if is_empty:
            self.log_fn("Empty or invalid student answer detected, returning 0")
            return "0", msg_history
        
        # Try one more time with a simplified prompt if all retries failed
        try:
            simplified_prompt = f"""Grade this student answer. Respond ONLY with a JSON object.

Problem: {problem[:500]}

Correct Solution: {solution[:500]}

Student Answer: {student_answer[:500]}

Is the student's answer correct? Output ONLY:
<json>{{"response": 1}}</json> if correct, or <json>{{"response": 0}}</json> if incorrect."""
            
            response, msg_history, info = get_response_from_llm(
                msg=simplified_prompt,
                model=self.model,
                msg_history=[],
            )
            
            response_text = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_json_flexible(response_text)
            
            if extracted and "response" in extracted:
                prediction = extracted["response"]
                validated = self._validate_prediction(prediction)
                if validated is not None:
                    self.log_fn(f"Simplified prompt valid prediction: {validated}")
                    return validated, msg_history
        except Exception as e:
            self.log_fn(f"Simplified prompt also failed: {e}")
        
        # Final fallback: return "0" if all retries failed
        self.log_fn("All retries failed, returning default prediction 0")
        return "0", msg_history
