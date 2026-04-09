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
    3. Raw JSON objects in text
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
    
    # Strategy 3: Look for JSON-like structures with "response" key
    json_pattern = r'\{\s*"response"\s*:[^\}]+\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for explicit verdict patterns in text
    text_lower = text.lower()
    # Check for explicit verdict statements
    if re.search(r'\bverdict\s*[:=]\s*(1|correct|true|yes)\b', text_lower):
        return {"response": 1}
    if re.search(r'\bverdict\s*[:=]\s*(0|incorrect|false|no)\b', text_lower):
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
        """Fallback grading using enhanced heuristics when LLM fails.
        
        Implements multiple equivalence checking strategies:
        1. Exact string match
        2. Numeric equivalence (integers, floats, fractions)
        3. Mathematical expression normalization
        4. LaTeX expression comparison
        5. Substring containment for key results
        """
        import re
        import math
        
        student_answer = inputs.get("student_answer", "").strip()
        solution = inputs.get("solution", "").strip()
        
        # Extract final answers
        student_final = self._extract_final_answer(student_answer.lower())
        solution_final = self._extract_final_answer(solution.lower())
        
        if not student_final or not solution_final:
            return "0", msg_history
        
        # Strategy 1: Exact match (case-insensitive)
        if student_final == solution_final:
            return "1", msg_history
        
        # Strategy 2: Numeric equivalence
        try:
            # Try direct float comparison
            if abs(float(student_final) - float(solution_final)) < 1e-9:
                return "1", msg_history
        except ValueError:
            pass
        
        # Strategy 3: Fraction equivalence (e.g., "1/2" vs "0.5")
        try:
            student_frac = self._parse_fraction(student_final)
            solution_frac = self._parse_fraction(solution_final)
            if student_frac is not None and solution_frac is not None:
                if abs(student_frac - solution_frac) < 1e-9:
                    return "1", msg_history
        except Exception:
            pass
        
        # Strategy 4: Normalize and compare mathematical expressions
        student_norm = self._normalize_math_expr(student_final)
        solution_norm = self._normalize_math_expr(solution_final)
        
        if student_norm == solution_norm:
            return "1", msg_history
        
        # Strategy 5: Check if normalized solution is in normalized student answer
        if solution_norm and solution_norm in student_norm:
            return "1", msg_history
        
        # Strategy 6: Check for key mathematical patterns
        if self._check_math_equivalence(student_final, solution_final):
            return "1", msg_history
            
        return "0", msg_history
    
    def _parse_fraction(self, text: str) -> float | None:
        """Parse a fraction like '1/2' or '3/4' into a float."""
        text = text.strip()
        # Match patterns like "1/2", "-3/4", "10/5"
        match = re.match(r'^-?\d+\/\d+$', text)
        if match:
            try:
                num, denom = text.split('/')
                return float(num) / float(denom)
            except (ValueError, ZeroDivisionError):
                return None
        return None
    
    def _normalize_math_expr(self, text: str) -> str:
        """Normalize a mathematical expression for comparison.
        
        Removes common formatting differences:
        - Whitespace normalization
        - LaTeX formatting removal
        - Common symbol standardization
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove LaTeX formatting
        text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', text)  # \text{}, \mathbf{}, etc.
        text = re.sub(r'[{}]', '', text)  # Remove braces
        text = text.replace('\\', '')  # Remove backslashes
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Standardize common mathematical notations
        replacements = [
            (r'\^\s*2', '²'),  # x^2 -> x²
            (r'\^\s*3', '³'),  # x^3 -> x³
            (r'\*\*\s*2', '²'),  # x**2 -> x²
            (r'\*\*\s*3', '³'),  # x**3 -> x³
            (r'\*', '·'),  # Standardize multiplication
            (r'\\cdot', '·'),
            (r'\\times', '·'),
            (r'\\div', '/'),
            (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)'),  # \frac{a}{b} -> (a)/(b)
            (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),  # \sqrt{x} -> sqrt(x)
            (r'\\pi', 'π'),
            (r'\\infty', '∞'),
            (r'\\in', '∈'),
            (r'\\notin', '∉'),
            (r'\\subset', '⊂'),
            (r'\\subseteq', '⊆'),
            (r'\\cup', '∪'),
            (r'\\cap', '∩'),
            (r'\\emptyset', '∅'),
            (r'\\varnothing', '∅'),
            (r'\\mathbb\{r\}', 'ℝ'),  # \mathbb{R} -> ℝ
            (r'\\mathbb\{n\}', 'ℕ'),  # \mathbb{N} -> ℕ
            (r'\\mathbb\{z\}', 'ℤ'),  # \mathbb{Z} -> ℤ
            (r'\\mathbb\{q\}', 'ℚ'),  # \mathbb{Q} -> ℚ
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        # Remove common wrapper phrases
        wrappers = [
            r'^the answer is\s*[:=]?\s*',
            r'^answer[:=]?\s*',
            r'^final answer[:=]?\s*',
            r'^therefore,?\s*',
            r'^thus,?\s*',
            r'^so,?\s*',
            r'^hence,?\s*',
            r'^we have\s*',
            r'\.$',  # Trailing period
        ]
        for wrapper in wrappers:
            text = re.sub(wrapper, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _check_math_equivalence(self, student: str, solution: str) -> bool:
        """Check for mathematical equivalence using various methods."""
        # Remove all whitespace for comparison
        student_nows = ''.join(student.split())
        solution_nows = ''.join(solution.split())
        
        if student_nows == solution_nows:
            return True
        
        # Check for common equivalent forms
        # e.g., "x = 5" vs "5" vs "x=5"
        student_clean = re.sub(r'^[a-z]\s*=\s*', '', student_nows)
        solution_clean = re.sub(r'^[a-z]\s*=\s*', '', solution_nows)
        
        if student_clean == solution_clean:
            return True
        
        # Check for ordered pair equivalence (x, y) vs x, y
        student_coords = re.findall(r'-?\d+\.?\d*', student_nows)
        solution_coords = re.findall(r'-?\d+\.?\d*', solution_nows)
        
        if student_coords and solution_coords:
            try:
                if len(student_coords) == len(solution_coords):
                    matches = all(
                        abs(float(s) - float(t)) < 1e-9 
                        for s, t in zip(student_coords, solution_coords)
                    )
                    if matches:
                        return True
            except ValueError:
                pass
        
        # Check for set equivalence {a, b} vs {b, a}
        student_set = re.search(r'\{([^}]+)\}', student_nows)
        solution_set = re.search(r'\{([^}]+)\}', solution_nows)
        
        if student_set and solution_set:
            student_elems = set(e.strip() for e in student_set.group(1).split(','))
            solution_elems = set(e.strip() for e in solution_set.group(1).split(','))
            if student_elems == solution_elems:
                return True
        
        return False
    
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
