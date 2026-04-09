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
        1. Exact string match (case-insensitive, whitespace-normalized)
        2. Numeric equivalence (handles integers, floats, scientific notation)
        3. Fraction equivalence (converts fractions to decimals)
        4. Mathematical expression normalization
        5. Substring containment (for complex answers)
        """
        student_answer = inputs.get("student_answer", "").strip()
        solution = inputs.get("solution", "").strip()
        
        # Extract final answers using multiple strategies
        student_final = self._extract_final_answer(student_answer)
        solution_final = self._extract_final_answer(solution)
        
        if not student_final or not solution_final:
            return "0", msg_history
        
        # Normalize answers for comparison
        student_norm = self._normalize_answer(student_final)
        solution_norm = self._normalize_answer(solution_final)
        
        # Strategy 1: Exact normalized match
        if student_norm == solution_norm:
            return "1", msg_history
        
        # Strategy 2: Numeric equivalence
        if self._check_numeric_equivalence(student_norm, solution_norm):
            return "1", msg_history
        
        # Strategy 3: Fraction equivalence
        if self._check_fraction_equivalence(student_norm, solution_norm):
            return "1", msg_history
        
        # Strategy 4: Mathematical expression equivalence
        if self._check_expression_equivalence(student_norm, solution_norm):
            return "1", msg_history
        
        # Strategy 5: Containment check (for complex/multi-part answers)
        if solution_norm in student_norm or student_norm in solution_norm:
            # Additional check: ensure they're not just single characters
            if len(solution_norm) > 2 and len(student_norm) > 2:
                return "1", msg_history
        
        return "0", msg_history
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from text using multiple patterns."""
        text_lower = text.lower()
        
        # Pattern 1: Look for "answer:" or "final answer:" with various formats
        answer_patterns = [
            r'(?:final\s+)?answer\s*[:=]\s*([\s\S]+?)(?:\n\n|\Z|$)',
            r'(?:final\s+)?answer\s+is\s*[:=]?\s*([\s\S]+?)(?:\n\n|\Z|$)',
            r'\\boxed\{([^}]+)\}',  # LaTeX boxed answer
            r'\*\*answer\*\*\s*[:=]?\s*([\s\S]+?)(?:\n\n|\Z|$)',  # Markdown bold
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer (take first line if multiple)
                answer = answer.split('\n')[0].strip()
                if answer:
                    return answer
        
        # Pattern 2: Look for common conclusion markers
        conclusion_markers = ['therefore', 'thus', 'hence', 'so', 'we have']
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in conclusion_markers):
                # Return this line or the next one if this is just a marker
                if len(line) < 20 and i + 1 < len(lines):
                    return lines[i + 1]
                return line
        
        # Pattern 3: Return last non-empty line (often contains the answer)
        if lines:
            # Skip common trailing phrases
            for line in reversed(lines):
                if not any(phrase in line.lower() for phrase in ['thank', 'hope', 'please', 'best', 'regards']):
                    return line
            return lines[-1]
        
        return text.strip()
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove common wrappers and decorations
        answer = answer.strip()
        
        # Remove LaTeX formatting
        answer = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', answer)  # \text{}, \mathbf{}, etc.
        answer = re.sub(r'\$+', '', answer)  # Remove $ delimiters
        
        # Remove units and explanatory text in parentheses
        answer = re.sub(r'\s*\([^)]*\)\s*', ' ', answer)
        
        # Normalize whitespace
        answer = ' '.join(answer.split())
        
        # Normalize common mathematical notations
        answer = answer.replace('×', '*').replace('·', '*').replace('÷', '/')
        answer = answer.replace('−', '-').replace('—', '-')
        
        return answer.strip().lower()
    
    def _check_numeric_equivalence(self, a: str, b: str) -> bool:
        """Check if two strings represent the same numeric value."""
        # Try direct float comparison
        try:
            # Handle scientific notation and various formats
            val_a = float(a.replace(',', ''))
            val_b = float(b.replace(',', ''))
            # Use tolerance for floating point comparison
            return abs(val_a - val_b) < 1e-9
        except (ValueError, OverflowError):
            pass
        
        # Try integer comparison
        try:
            return int(a.replace(',', '')) == int(b.replace(',', ''))
        except (ValueError, OverflowError):
            pass
        
        return False
    
    def _check_fraction_equivalence(self, a: str, b: str) -> bool:
        """Check if two strings represent equivalent fractions."""
        # Pattern: number/number
        frac_pattern = r'^(-?\d+)\s*/\s*(\d+)$'
        
        match_a = re.match(frac_pattern, a)
        match_b = re.match(frac_pattern, b)
        
        if match_a and match_b:
            num_a, den_a = int(match_a.group(1)), int(match_a.group(2))
            num_b, den_b = int(match_b.group(1)), int(match_b.group(2))
            # Check cross-multiplication for equivalence
            return num_a * den_b == num_b * den_a
        
        # Check if one is a fraction and the other is decimal
        if match_a:
            num_a, den_a = int(match_a.group(1)), int(match_a.group(2))
            if den_a != 0:
                frac_val = num_a / den_a
                try:
                    return abs(frac_val - float(b.replace(',', ''))) < 1e-9
                except ValueError:
                    pass
        
        if match_b:
            num_b, den_b = int(match_b.group(1)), int(match_b.group(2))
            if den_b != 0:
                frac_val = num_b / den_b
                try:
                    return abs(frac_val - float(a.replace(',', ''))) < 1e-9
                except ValueError:
                    pass
        
        return False
    
    def _check_expression_equivalence(self, a: str, b: str) -> bool:
        """Check if two mathematical expressions are equivalent."""
        # Normalize common expressions
        def normalize_expr(expr: str) -> str:
            # Remove spaces around operators
            expr = re.sub(r'\s*([+\-*/=])\s*', r'\1', expr)
            # Normalize power notation
            expr = expr.replace('^', '**')
            # Handle square roots
            expr = re.sub(r'sqrt\(([^)]+)\)', r'(\1)**0.5', expr)
            return expr
        
        norm_a = normalize_expr(a)
        norm_b = normalize_expr(b)
        
        # Check if they're identical after normalization
        if norm_a == norm_b:
            return True
        
        # Check for simple algebraic equivalence (a+b vs b+a)
        # Split by operators and compare sets of terms
        def get_terms(expr: str) -> frozenset:
            # Simple term extraction for sums
            terms = re.split(r'[+\-]', expr)
            return frozenset(t.strip() for t in terms if t.strip())
        
        try:
            terms_a = get_terms(norm_a)
            terms_b = get_terms(norm_b)
            if terms_a == terms_b and len(terms_a) > 1:
                return True
        except Exception:
            pass
        
        return False
