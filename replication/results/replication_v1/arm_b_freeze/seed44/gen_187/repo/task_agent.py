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
    6. Bracket-balanced JSON extraction with nested structure support
    7. LLM-based extraction as final fallback
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
        # Find the matching opening brace using bracket counting
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(last_brace_idx, -1, -1):
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
            if char == '}':
                brace_count += 1
            elif char == '{':
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
    
    # Strategy 6: Find all potential JSON objects with proper brace balancing
    # This handles nested JSON structures more robustly
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
                            score = 0
                            if "reasoning" in parsed:
                                score += 2
                            if "response" in parsed:
                                score += 1
                            potential_jsons.append((score, parsed))
                    except json.JSONDecodeError:
                        pass
                    break
    
    # Return the highest-scoring valid JSON (prefer ones with both fields)
    if potential_jsons:
        potential_jsons.sort(reverse=True, key=lambda x: x[0])
        return potential_jsons[0][1]
    
    return None


def _extract_with_llm(text: str, model: str = EVAL_MODEL) -> dict | None:
    """Use LLM to extract structured JSON from unstructured text.
    
    This is a final fallback when regex-based extraction fails.
    """
    try:
        extraction_prompt = f"""Extract the grading result from the following text.
The text should contain a JSON object with "reasoning" and "response" fields.

Text to extract from:
{text}

Respond ONLY with a valid JSON object in this exact format:
<json>
{{
    "reasoning": "the reasoning text found in the input",
    "response": the numeric value (0, 1, or decimal between 0.0-1.0)
}}
</json>"""
        
        response, _, _ = get_response_from_llm(
            msg=extraction_prompt,
            model=model,
            msg_history=[],
        )
        # _extract_jsons returns a list, get the last element like other strategies
        results = _extract_jsons(response)
        return results[-1] if results else None
    except Exception:
        return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Supports both binary (0/1) and partial credit (0.0-1.0) scoring modes.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "", partial_credit: bool = False) -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self.partial_credit = partial_credit

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

        if supports_partial:
            instruction = self._build_partial_credit_prompt(
                domain, problem, solution, grading_guidelines, student_answer
            )
        else:
            instruction = self._build_binary_prompt(
                domain, problem, solution, grading_guidelines, student_answer
            )

        # Try with retries for robustness
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract prediction from JSON using flexible extraction
                raw_response_text = msg_history[-1]["text"] if msg_history else ""
                extracted = _extract_json_flexible(raw_response_text)
                
                # If regex extraction fails, try LLM-based extraction
                if extracted is None:
                    extracted = _extract_with_llm(raw_response_text, self.model)
                
                if extracted and "response" in extracted:
                    prediction = extracted["response"]
                    reasoning = extracted.get("reasoning", "No reasoning provided")
                    
                    # Validate prediction based on scoring mode
                    if supports_partial:
                        # Partial credit: 0.0 to 1.0
                        try:
                            pred_val = float(prediction)
                            if 0.0 <= pred_val <= 1.0:
                                self.log_fn(f"Graded with score {pred_val}: {reasoning[:100]}...")
                                return str(pred_val), msg_history
                        except (ValueError, TypeError):
                            pass
                        self.log_fn(f"Invalid partial credit value: {prediction}, retrying...")
                        last_error = f"Invalid partial credit value: {prediction}"
                    else:
                        # Binary: 0 or 1
                        if prediction in [0, 1, "0", "1"]:
                            self.log_fn(f"Graded as {'correct' if str(prediction) in ['1', 1] else 'incorrect'}: {reasoning[:100]}...")
                            return str(prediction), msg_history
                        self.log_fn(f"Invalid binary prediction value: {prediction}, retrying...")
                        last_error = f"Invalid binary prediction value: {prediction}"
                else:
                    self.log_fn(f"No valid JSON found in response (attempt {attempt + 1}), retrying...")
                    last_error = "No valid JSON found in response"
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    break
        
        # Fallback: return default prediction if all retries failed
        default_pred = "0.0" if supports_partial else "0"
        self.log_fn(f"All retries failed (last error: {last_error}), returning default prediction {default_pred}")
        return default_pred, msg_history if 'msg_history' in locals() else []

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
1. Analyze what the problem is asking for - identify the key concepts and required steps
2. Review the correct solution approach - understand the logic and methodology
3. Compare the student's answer to the correct solution - check for:
   - Correct final answer (numerical or symbolic)
   - Valid reasoning and logical steps
   - Proper application of mathematical concepts
   - Correct use of formulas and theorems
4. Check if the student followed the grading guidelines - look for specific requirements
5. Determine if the student's answer is correct (1) or incorrect (0)

IMPORTANT GRADING CRITERIA:
- Award 1 (correct) if:
  * The final answer matches the correct solution exactly, OR
  * The answer is mathematically equivalent to the correct solution, OR
  * The reasoning is sound and leads to the correct conclusion, even if formatted differently
- Award 0 (incorrect) if:
  * The final answer is wrong or missing
  * The reasoning contains critical logical errors
  * The student used incorrect formulas or methods
  * The answer is incomplete or doesn't address the problem

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be specific about what the student did correctly or incorrectly.",
    "response": 1 or 0
}}
</json>

The "response" field must be either 1 (correct) or 0 (incorrect)."""

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
1. Analyze what the problem is asking for - identify key concepts, required steps, and scoring rubric elements
2. Review the correct solution approach - understand the logic, methodology, and expected answer format
3. Compare the student's answer to the correct solution - evaluate:
   - Final answer correctness (numerical or symbolic)
   - Reasoning quality and logical flow
   - Proper application of mathematical concepts
   - Correct use of formulas and theorems
   - Completeness of the solution
4. Identify which parts the student got correct vs incorrect
5. Assess the student's reasoning quality and approach
6. Assign a partial credit score from 0.0 (completely wrong) to 1.0 (completely correct)

DETAILED SCORING RUBRIC:
- 1.0: Fully correct solution with proper reasoning, complete and accurate
- 0.8-0.9: Mostly correct with minor errors (e.g., arithmetic mistake, notation issue) or small gaps
- 0.6-0.7: Partially correct with significant progress but some major gaps or errors
- 0.4-0.5: Some correct elements but major conceptual errors or incomplete solution
- 0.2-0.3: Minimal correct elements, mostly wrong approach but some valid steps
- 0.0-0.1: Completely wrong, no valid attempt, or entirely missing the point

GRADING PRINCIPLES:
- Award credit for correct reasoning even if the final answer has minor errors
- Consider mathematical equivalence (different forms of the same answer)
- Penalize major conceptual errors more heavily than calculation mistakes
- Ensure the score reflects the proportion of the problem the student solved correctly

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be specific about what was correct, what was wrong, and how you determined the score.",
    "response": 0.0 to 1.0
}}
</json>

The "response" field must be a decimal number between 0.0 and 1.0 (inclusive)."""
