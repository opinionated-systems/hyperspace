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
    # More permissive pattern to handle nested braces
    json_pattern = r'\{[^{}]*"response"\s*:\s*(?:0|1|"0"|"1"|true|false)[^{}]*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for JSON with reasoning field
    full_json_pattern = r'\{\s*"reasoning"\s*:\s*"[^"]*"\s*,\s*"response"\s*:\s*(?:0|1|"0"|"1"|true|false)\s*\}'
    for match in re.finditer(full_json_pattern, text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Look for explicit verdict patterns in text
    text_lower = text.lower()
    # Check for explicit verdict statements
    if re.search(r'\bverdict\s*[:=]\s*(1|correct|true|yes)\b', text_lower):
        return {"response": 1}
    if re.search(r'\bverdict\s*[:=]\s*(0|incorrect|false|no)\b', text_lower):
        return {"response": 0}
    
    # Strategy 6: Look for standalone response values
    if re.search(r'"response"\s*:\s*(1|true)\b', text_lower):
        return {"response": 1}
    if re.search(r'"response"\s*:\s*(0|false)\b', text_lower):
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
- Consider equivalent forms: fractions vs decimals, different variable names, equivalent expressions
- Check for common student mistakes: sign errors, arithmetic errors, missing steps

Respond ONLY in the following JSON format. Do not include any text outside the JSON tags:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your evaluation clearly, including what the student did right or wrong.",
    "response": 1
}}
</json>

OR

<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Explain your evaluation clearly, including what the student did right or wrong.",
    "response": 0
}}
</json>

CRITICAL: The "response" field MUST be exactly 1 (correct) or 0 (incorrect). Use integer values, not strings."""

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
                    # Validate prediction is 0 or 1 (handle various types)
                    if prediction in [0, 1, "0", "1", True, False, "true", "false"]:
                        # Normalize to string "0" or "1"
                        if prediction in [1, "1", True, "true"]:
                            return "1", msg_history
                        else:
                            return "0", msg_history
                    else:
                        self.log_fn(f"Invalid prediction value: {prediction}, retrying...")
                        last_error = f"Invalid prediction: {prediction}"
                        # Add feedback to history for retry
                        msg_history.append({
                            "role": "user",
                            "text": f"Error: The response value must be exactly 1 or 0 (integer), not '{prediction}'. Please provide your evaluation again with the correct JSON format."
                        })
                else:
                    self.log_fn(f"No valid JSON found in response, retrying...")
                    last_error = "No valid JSON extracted"
                    # Add feedback to history for retry
                    msg_history.append({
                        "role": "user",
                        "text": "Error: Could not extract valid JSON from your response. Please respond ONLY with the JSON format specified, using <json>...</json> tags."
                    })
                    
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
        
        Uses multiple strategies: exact match, numeric equivalence,
        symbolic equivalence, and containment checks.
        """
        student_answer = inputs.get("student_answer", "").strip().lower()
        solution = inputs.get("solution", "").strip().lower()
        
        # Extract final answers using multiple strategies
        student_final = self._extract_final_answer(student_answer)
        solution_final = self._extract_final_answer(solution)
        
        # Strategy 1: Exact string match (case-insensitive, whitespace-normalized)
        if student_final and solution_final:
            student_norm = self._normalize_answer(student_final)
            solution_norm = self._normalize_answer(solution_final)
            
            if student_norm == solution_norm:
                return "1", msg_history
            
            # Strategy 2: Numeric equivalence with tolerance
            numeric_match = self._check_numeric_equivalence(student_norm, solution_norm)
            if numeric_match:
                return "1", msg_history
            
            # Strategy 3: Symbolic/mathematical equivalence
            symbolic_match = self._check_symbolic_equivalence(student_norm, solution_norm)
            if symbolic_match:
                return "1", msg_history
        
        # Strategy 4: Check if solution is contained in student answer
        if solution_final and solution_final in student_answer:
            return "1", msg_history
        
        # Strategy 5: Check for key mathematical expressions
        key_match = self._check_key_expressions(solution, student_answer)
        if key_match:
            return "1", msg_history
            
        return "0", msg_history
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison."""
        # Remove common wrappers
        text = text.strip()
        text = re.sub(r'^\$+|\$+$', '', text)  # Remove LaTeX delimiters
        text = re.sub(r'^\\?\(|\\?\)$', '', text)  # Remove parentheses
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.lower().strip()
    
    def _check_numeric_equivalence(self, a: str, b: str) -> bool:
        """Check if two strings represent the same numeric value."""
        # Try direct float comparison
        try:
            val_a = float(a.replace(',', ''))
            val_b = float(b.replace(',', ''))
            # Use relative tolerance for floating point comparison
            if val_a == 0 and val_b == 0:
                return True
            if val_a == 0 or val_b == 0:
                return abs(val_a - val_b) < 1e-10
            rel_diff = abs(val_a - val_b) / max(abs(val_a), abs(val_b))
            return rel_diff < 1e-9
        except (ValueError, OverflowError):
            pass
        
        # Try fraction comparison (e.g., "1/2" vs "0.5")
        frac_match_a = re.match(r'^(-?\d+)\s*/\s*(\d+)$', a)
        frac_match_b = re.match(r'^(-?\d+)\s*/\s*(\d+)$', b)
        
        if frac_match_a and frac_match_b:
            try:
                num_a, den_a = int(frac_match_a.group(1)), int(frac_match_a.group(2))
                num_b, den_b = int(frac_match_b.group(1)), int(frac_match_b.group(2))
                # Cross multiply to compare fractions
                return num_a * den_b == num_b * den_a
            except (ValueError, ZeroDivisionError):
                pass
        
        # Check if one is a fraction and the other is decimal
        if frac_match_a:
            try:
                frac_val = int(frac_match_a.group(1)) / int(frac_match_a.group(2))
                dec_val = float(b.replace(',', ''))
                return abs(frac_val - dec_val) < 1e-9
            except (ValueError, ZeroDivisionError):
                pass
        
        if frac_match_b:
            try:
                frac_val = int(frac_match_b.group(1)) / int(frac_match_b.group(2))
                dec_val = float(a.replace(',', ''))
                return abs(frac_val - dec_val) < 1e-9
            except (ValueError, ZeroDivisionError):
                pass
        
        return False
    
    def _check_symbolic_equivalence(self, a: str, b: str) -> bool:
        """Check for symbolic/mathematical equivalence."""
        # Remove common mathematical notation differences
        def simplify(expr: str) -> str:
            expr = expr.replace('**', '^')
            expr = expr.replace('×', '*')
            expr = expr.replace('⋅', '*')
            expr = expr.replace(' ', '')
            expr = re.sub(r'\^(\d+)', r'^\1', expr)
            return expr
        
        return simplify(a) == simplify(b)
    
    def _check_key_expressions(self, solution: str, student_answer: str) -> bool:
        """Check if key mathematical expressions from solution appear in student answer."""
        # Extract mathematical expressions (numbers, variables, simple formulas)
        patterns = [
            r'\b\d+\s*[=\-]\s*\d+\b',  # Simple equations like "x = 5"
            r'\b[a-z]\s*=\s*[-+]?\d+\.?\d*\b',  # Variable assignments
            r'\b\d+\.?\d*\s*[/\*\^]\s*\d+\.?\d*\b',  # Simple operations
        ]
        
        key_exprs = []
        for pattern in patterns:
            key_exprs.extend(re.findall(pattern, solution.lower()))
        
        # If we found key expressions, check if they appear in student answer
        if key_exprs:
            matches = sum(1 for expr in key_exprs if expr in student_answer)
            # Require at least 50% of key expressions to match
            return matches >= len(key_exprs) * 0.5
        
        return False
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from text using multiple strategies."""
        text = text.strip()
        
        # Strategy 1: Look for "answer:" or "final answer:" pattern
        patterns = [
            r'(?:final\s+)?answer[:=]\s*([^\n]+)',
            r'(?:the\s+)?answer\s+(?:is|equals?)\s*[:=]?\s*([^\n.]+)',
            r'\\boxed\{([^}]+)\}',  # LaTeX boxed answer
            r'\*\*answer:\*\*\s*([^\n]+)',  # Markdown bold
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Strategy 2: Look for common conclusion markers
        conclusion_markers = ['therefore', 'thus', 'hence', 'so', 'conclusion:', 'in conclusion']
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for marker in conclusion_markers:
                if marker in line_lower:
                    # Return the rest of this line or the next line
                    parts = line_lower.split(marker, 1)
                    if len(parts) > 1 and parts[1].strip():
                        return line[len(parts[0]) + len(marker):].strip()
                    if i + 1 < len(lines):
                        return lines[i + 1]
        
        # Strategy 3: Return last non-empty line (but filter out common endings)
        if lines:
            for line in reversed(lines):
                line_lower = line.lower()
                # Skip common non-answer endings
                if not any(skip in line_lower for skip in ['thank', 'hope', 'please', 'let me know', 'best regards']):
                    return line
        
        return text
