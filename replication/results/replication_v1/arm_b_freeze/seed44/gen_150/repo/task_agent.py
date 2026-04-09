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
    3. Raw JSON objects in text (with balanced braces)
    4. Direct text pattern matching for yes/no/correct/incorrect
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
    
    # Strategy 3: Look for JSON-like structures with balanced braces
    # Find all potential JSON objects by tracking brace balance
    def find_json_objects(s: str) -> list[str]:
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
    
    for json_str in find_json_objects(text):
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for explicit verdict patterns in text
    text_lower = text.lower()
    # Check for explicit verdict statements
    if re.search(r'\bverdict\s*[:=]\s*(1|correct|true|yes)\b', text_lower):
        return {"response": 1}
    if re.search(r'\bverdict\s*[:=]\s*(0|incorrect|false|no)\b', text_lower):
        return {"response": 0}
    
    # Strategy 5: Look for standalone "response": 1 or "response": 0
    if re.search(r'"response"\s*:\s*1\b', text):
        return {"response": 1}
    if re.search(r'"response"\s*:\s*0\b', text):
        return {"response": 0}
    if re.search(r"'response'\s*:\s*1\b", text):
        return {"response": 1}
    if re.search(r"'response'\s*:\s*0\b", text):
        return {"response": 0}
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

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

        instruction = f"""You are an expert {domain} grader evaluating student solutions with high precision and consistency.

Your task is to grade a student's answer by carefully comparing it to the correct solution and strictly following the grading guidelines.

PROBLEM:
{problem}

CORRECT SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Follow this systematic evaluation process:
1. **Problem Analysis**: Identify the key concepts, required steps, and expected answer format
2. **Solution Review**: Understand the correct approach and final answer
3. **Student Work Evaluation**: 
   - Check if the student demonstrated understanding of core concepts
   - Verify if the final answer matches the correct solution (accounting for equivalent forms)
   - Identify any conceptual errors or missing steps
4. **Guideline Compliance**: Apply the grading guidelines strictly
5. **Final Verdict**: Determine if the answer is CORRECT (1) or INCORRECT (0)

IMPORTANT GRADING PRINCIPLES:
- The answer is CORRECT (1) if it matches the solution or is mathematically equivalent
- The answer is INCORRECT (0) if it contains errors, is incomplete, or doesn't match
- Be precise: partial credit is not awarded unless explicitly allowed by guidelines
- Consider equivalent forms: 1/2 = 0.5 = 2/4, x² = x*x, etc.
- Check for correct reasoning even if the final format differs slightly

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your evaluation clearly.",
    "response": 1 or 0
}}
</json>

The "response" field MUST be either 1 (correct) or 0 (incorrect). No other values are accepted."""

        # Try with retries for robustness
        msg_history = []
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Extract prediction from JSON using flexible extraction
                response_text = msg_history[-1]["text"]
                extracted = _extract_json_flexible(response_text)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    # Validate prediction is 0 or 1
                    if prediction in [0, 1, "0", "1", True, False]:
                        # Normalize to string "0" or "1"
                        if prediction in [1, "1", True]:
                            return "1", msg_history
                        else:
                            return "0", msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                        last_error = f"Invalid prediction: {prediction}"
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    last_error = "No valid JSON extracted"
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: use heuristics if all retries failed
        self.log_fn(f"All retries failed ({last_error}), using heuristic fallback")
        return self._heuristic_fallback(inputs, msg_history)
    
    def _heuristic_fallback(self, inputs: dict, msg_history: list[dict]) -> tuple[str, list[dict]]:
        """Fallback grading using simple heuristics when LLM fails."""
        student_answer = inputs.get("student_answer", "").strip().lower()
        solution = inputs.get("solution", "").strip().lower()
        
        # Extract final answers (last line or after "answer:")
        student_final = self._extract_final_answer(student_answer)
        solution_final = self._extract_final_answer(solution)
        
        # Simple string comparison
        if student_final and solution_final:
            if student_final == solution_final:
                return "1", msg_history
            # Check for numeric equivalence
            try:
                if float(student_final) == float(solution_final):
                    return "1", msg_history
            except ValueError:
                pass
            # Check for fraction equivalence (e.g., "1/2" vs "0.5")
            try:
                if self._compare_fractions(student_final, solution_final):
                    return "1", msg_history
            except Exception:
                pass
        
        # Check if solution is contained in student answer
        if solution_final and solution_final in student_answer:
            return "1", msg_history
        
        # Check for key terms/numbers in solution appearing in student answer
        solution_numbers = self._extract_numbers(solution)
        student_numbers = self._extract_numbers(student_answer)
        if solution_numbers and student_numbers:
            # If all key numbers from solution appear in student answer
            if all(num in student_numbers for num in solution_numbers):
                return "1", msg_history
            
        return "0", msg_history
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from text."""
        # Look for "answer:" pattern
        match = re.search(r'answer[:=]\s*([^\n]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Return last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            return lines[-1]
        return text
    
    def _compare_fractions(self, val1: str, val2: str) -> bool:
        """Compare two values that might be fractions or decimals."""
        def parse_value(v: str) -> float:
            v = v.strip()
            # Handle fractions like "1/2"
            if '/' in v and ' ' not in v:
                parts = v.split('/')
                if len(parts) == 2:
                    try:
                        return float(parts[0]) / float(parts[1])
                    except (ValueError, ZeroDivisionError):
                        pass
            # Handle mixed numbers like "1 1/2"
            if ' ' in v and '/' in v:
                parts = v.split()
                if len(parts) == 2:
                    try:
                        whole = float(parts[0])
                        frac_parts = parts[1].split('/')
                        if len(frac_parts) == 2:
                            frac = float(frac_parts[0]) / float(frac_parts[1])
                            return whole + frac if whole >= 0 else whole - frac
                    except (ValueError, ZeroDivisionError):
                        pass
            return float(v)
        
        try:
            return abs(parse_value(val1) - parse_value(val2)) < 1e-9
        except (ValueError, TypeError):
            return False
    
    def _extract_numbers(self, text: str) -> list[str]:
        """Extract all numbers from text."""
        # Match integers, decimals, and fractions
        pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
        return re.findall(pattern, text)
