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
    Includes enhanced error recovery for malformed JSON.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON issues
        fixed = inner
        
        # Fix trailing commas before closing braces/brackets
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Fix single quotes to double quotes (but not within string values)
        # Use a safer approach: replace only unquoted single quotes
        fixed = re.sub(r"(?<!\\)'", '"', fixed)
        
        # Fix common escape sequence issues
        fixed = fixed.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        
        try:
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try extracting just the JSON object using brace counting
        try:
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                fixed = inner[json_start:json_end+1]
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                fixed = re.sub(r"(?<!\\)'", '"', fixed)
                results.append(json.loads(fixed))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: try to find and parse individual JSON objects
        try:
            # Look for JSON-like structures with balanced braces
            brace_count = 0
            obj_start = -1
            for i, char in enumerate(inner):
                if char == '{':
                    if brace_count == 0:
                        obj_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and obj_start != -1:
                        obj_str = inner[obj_start:i+1]
                        try:
                            obj_str = re.sub(r',(\s*[}\]])', r'\1', obj_str)
                            obj_str = re.sub(r"(?<!\\)'", '"', obj_str)
                            results.append(json.loads(obj_str))
                        except json.JSONDecodeError:
                            pass
                        obj_start = -1
        except Exception:
            pass
            
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses a robust brace-counting approach to find complete JSON objects,
    handling nested braces correctly. Also attempts to fix common JSON
    formatting issues before parsing.
    """
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                try:
                    obj = json.loads(json_str)
                    results.append(obj)
                except json.JSONDecodeError:
                    # Try to fix common issues and retry
                    try:
                        # Fix trailing commas before closing braces
                        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        # Fix single quotes to double quotes
                        fixed = fixed.replace("'", '"')
                        obj = json.loads(fixed)
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                start_idx = -1
    return results or None


def _extract_from_markdown_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Handles cases where the model outputs JSON in markdown format.
    """
    results = []
    # Find markdown code blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        match = match.strip()
        if not match.startswith('{'):
            continue
        try:
            obj = json.loads(match)
            results.append(obj)
        except json.JSONDecodeError:
            # Try the fallback extraction on the content
            nested = _extract_any_json(match)
            if nested:
                results.extend(nested)
    
    return results or None


def _normalize_grade(prediction: str) -> str:
    """Normalize various grade formats to a standard set.
    
    Handles common variations like 'Correct'/'correct'/'CORRECT',
    numeric scores, and partial credit indicators.
    Returns one of: "Correct", "Incorrect", "Partial", or "None".
    """
    import re
    
    if not prediction or not isinstance(prediction, str):
        return "None"
    
    pred = prediction.strip()
    pred_lower = pred.lower()
    
    # First check for exact matches (case-insensitive)
    exact_correct = ['correct', 'right', 'true', 'yes', 'pass', 'full credit', 'full', '100%']
    exact_incorrect = ['incorrect', 'wrong', 'false', 'fail', 'no', 'none', 'zero', '0%', '0']
    exact_partial = ['partial', 'partial credit', 'half', 'incomplete', 'partially correct', '50%']
    
    if pred_lower in exact_correct:
        return "Correct"
    if pred_lower in exact_incorrect:
        return "Incorrect"
    if pred_lower in exact_partial:
        return "Partial"
    
    # Check for numeric grades
    # Try to extract the first number from the prediction
    numbers = re.findall(r'-?\d+\.?\d*', pred)
    if numbers:
        try:
            num = float(numbers[0])
            # Handle common numeric scales
            if num >= 0.9 or num == 1.0:
                return "Correct"
            elif num >= 0.4 and num < 0.9:
                return "Partial"
            elif num < 0.4 and num >= 0:
                return "Incorrect"
        except ValueError:
            pass
    
    # Check for substring matches (more lenient)
    # Check for incorrect first (before correct, since "incorrect" contains "correct")
    incorrect_patterns = [
        'incorrect', 'wrong', 'false', 'fail', 'error', 'invalid',
        r'\bno\b',  # word boundary for "no"
        r'\b0\b',  # standalone 0
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, pred_lower):
            return "Incorrect"
    
    # Check for correct patterns
    correct_patterns = [
        'correct', 'right', 'true', 'yes', 'valid', 'accurate',
        r'\b1\b',  # standalone 1
        r'\b1\.0\b',
    ]
    for pattern in correct_patterns:
        if re.search(pattern, pred_lower):
            return "Correct"
    
    # Check for partial patterns
    partial_patterns = [
        'partial', 'half', 'incomplete', 'partially', 'some credit',
        'minor error', 'small mistake', 'mostly correct',
        r'\b0\.5\b',
    ]
    for pattern in partial_patterns:
        if re.search(pattern, pred_lower):
            return "Partial"
    
    # If we can't normalize, return "None" to indicate uncertainty
    return "None"


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade with detailed reasoning.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step and provide a thorough analysis:
1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required
2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format
3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps or valid insights
   - Identify errors, gaps, or misconceptions
4. GRADING CRITERIA CHECK:
   - Verify if the student met each criterion in the grading guidelines
   - Note partial credit for incomplete but valid reasoning
5. FINAL DETERMINATION: Assign grade based on completeness, correctness, and adherence to guidelines

GRADE DEFINITIONS (use EXACTLY one of these labels):
- "Correct": The student's answer is fully correct, complete, and matches the official solution. All required steps are present and valid.
- "Incorrect": The student's answer is wrong, incomplete in a critical way, or contains fundamental errors that invalidate the solution.
- "Partial": The student shows valid reasoning and made progress toward the solution, but the answer is incomplete or contains non-critical errors. Award this when the student demonstrates understanding of key concepts but hasn't fully solved the problem.

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade - MUST be exactly one of: Correct, Incorrect, or Partial"
}}
</json>

IMPORTANT FORMATTING RULES:
- Use double quotes for all JSON strings, never single quotes
- Do not include trailing commas
- Ensure the JSON is valid and parseable
- The response field MUST contain exactly one of: "Correct", "Incorrect", or "Partial" (case-sensitive)

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect. When in doubt between "Partial" and "Incorrect", choose "Partial" if the student demonstrated meaningful understanding of the problem.

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines
- Your JSON output follows the exact format specified above
- Your response field contains exactly one of the three allowed values: Correct, Incorrect, or Partial"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        extraction_method = "none"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (from <json> tags)
            extracted = _extract_jsons(last_message)
            if extracted:
                extraction_method = "json_tags"
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
                if extracted:
                    extraction_method = "any_json"
            
            # Fallback 2: markdown code blocks
            if extracted is None:
                extracted = _extract_from_markdown_code_blocks(last_message)
                if extracted:
                    extraction_method = "markdown"
            
            if extracted:
                # Prefer response field, but accept other common field names
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                elif "grade" in last_json:
                    prediction = last_json["grade"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                elif "score" in last_json:
                    prediction = last_json["score"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and len(value) < 100:
                            prediction = value
                            break
                
                self.log_fn(f"Extracted prediction via {extraction_method}: {prediction}")
                
                # Normalize the grade to standard format
                normalized = _normalize_grade(prediction)
                if normalized != prediction:
                    self.log_fn(f"Normalized grade: {prediction} -> {normalized}")
                    prediction = normalized
                
                # Validate that we have a valid grade
                valid_grades = ["Correct", "Incorrect", "Partial"]
                if prediction not in valid_grades and prediction != "None":
                    # Try to extract from reasoning field if available
                    if "reasoning" in last_json:
                        reasoning = last_json["reasoning"]
                        if isinstance(reasoning, str):
                            # Try to find grade mentions in reasoning
                            reasoning_lower = reasoning.lower()
                            if "grade" in reasoning_lower or "determination" in reasoning_lower:
                                # Look for explicit grade statements
                                for grade in valid_grades:
                                    if grade.lower() in reasoning_lower:
                                        prediction = grade
                                        self.log_fn(f"Extracted grade from reasoning: {prediction}")
                                        break
                    
                    # If still not valid, default to None for manual review
                    if prediction not in valid_grades:
                        self.log_fn(f"Warning: Could not normalize grade '{prediction}', defaulting to None")
                        prediction = "None"
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
