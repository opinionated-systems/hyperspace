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
    Includes multiple fallback strategies for robust extraction.
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
        
        # Strategy 1: Fix trailing commas
        try:
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Fix single quotes to double quotes
        try:
            fixed = inner.replace("'", '"')
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Combine both fixes
        try:
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
            fixed = fixed.replace("'", '"')
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Extract just the JSON object between first { and last }
        try:
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                fixed = inner[json_start:json_end+1]
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                fixed = fixed.replace("'", '"')
                results.append(json.loads(fixed))
                continue
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Handle escaped characters and newlines in strings
        try:
            # Replace literal newlines within strings with escaped newlines
            fixed = inner.replace('\n', '\\n').replace('\t', '\\t')
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
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
                
                # Try multiple parsing strategies
                for fix_strategy in [
                    lambda s: s,  # No fix
                    lambda s: re.sub(r',(\s*[}\]])', r'\1', s),  # Fix trailing commas
                    lambda s: s.replace("'", '"'),  # Fix single quotes
                    lambda s: re.sub(r',(\s*[}\]])', r'\1', s).replace("'", '"'),  # Both fixes
                    lambda s: s.replace('\n', '\\n').replace('\t', '\\t'),  # Fix newlines
                ]:
                    try:
                        fixed = fix_strategy(json_str)
                        obj = json.loads(fixed)
                        results.append(obj)
                        break
                    except json.JSONDecodeError:
                        continue
                
                start_idx = -1
    return results or None


def _extract_from_markdown_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Handles cases where the model outputs JSON in markdown format.
    Uses multiple parsing strategies for robust extraction.
    """
    results = []
    # Find markdown code blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        match = match.strip()
        if not match.startswith('{'):
            continue
        
        # Try multiple parsing strategies
        parsed = False
        for fix_strategy in [
            lambda s: s,  # No fix
            lambda s: re.sub(r',(\s*[}\]])', r'\1', s),  # Fix trailing commas
            lambda s: s.replace("'", '"'),  # Fix single quotes
            lambda s: re.sub(r',(\s*[}\]])', r'\1', s).replace("'", '"'),  # Both fixes
            lambda s: s.replace('\n', '\\n').replace('\t', '\\t'),  # Fix newlines
        ]:
            try:
                fixed = fix_strategy(match)
                obj = json.loads(fixed)
                results.append(obj)
                parsed = True
                break
            except json.JSONDecodeError:
                continue
        
        # If direct parsing failed, try nested extraction
        if not parsed:
            nested = _extract_any_json(match)
            if nested:
                results.extend(nested)
    
    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract a grade from raw text when JSON parsing fails.
    
    Looks for grade-related keywords in the text as a last resort.
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Look for explicit grade statements
    grade_patterns = [
        (r'\bgrade\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 1),
        (r'\bthe\s+(correct|partial|incorrect)\s+grade\b', 1),
        (r'\b(final\s+)?grade\s+(is|should\s+be)\s+["\']?(correct|partial|incorrect)["\']?', 3),
        (r'\b(response|prediction)\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?', 2),
        (r'\bassign\s+["\']?(correct|partial|incorrect)["\']?\b', 1),
        (r'\b(correct|partial|incorrect)\s+(?:grade|score|evaluation)\b', 1),
    ]
    
    for pattern, group_idx in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(group_idx).lower()
            if grade in ('correct', 'partial', 'incorrect'):
                return grade.capitalize()
    
    # Count occurrences of each grade term
    correct_count = len(re.findall(r'\bcorrect\b', text_lower))
    partial_count = len(re.findall(r'\bpartial\b', text_lower))
    incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
    
    # If one appears significantly more than others, use it
    total = correct_count + partial_count + incorrect_count
    if total > 0:
        if correct_count > partial_count and correct_count > incorrect_count:
            return "Correct"
        elif partial_count > correct_count and partial_count > incorrect_count:
            return "Partial"
        elif incorrect_count > correct_count and incorrect_count > partial_count:
            return "Incorrect"
    
    return None


def _normalize_grade(prediction: str) -> str:
    """Normalize various grade formats to a standard set.
    
    Handles common variations like 'Correct'/'correct'/'CORRECT',
    numeric scores, and partial credit indicators.
    """
    if not prediction or not isinstance(prediction, str):
        return "None"
    
    pred_lower = prediction.lower().strip()
    
    # Check for incorrect first (before correct, since "incorrect" contains "correct")
    # Use word boundaries to avoid matching "no" inside "unknown"
    incorrect_patterns = [
        'incorrect', 'wrong', 'false', 'fail',
        r'\bno\b',  # word boundary for "no"
        r'^0$', r'^0\.0$',  # exact 0 matches only (not part of decimals)
        'zero',
    ]
    for pattern in incorrect_patterns:
        if re.search(pattern, pred_lower):
            return "Incorrect"
    
    # Check for 'none' as a special case
    if pred_lower == 'none':
        return "Incorrect"
    
    # Map common variations to standard grades
    correct_variants = ['correct', 'right', 'true', 'yes', 'full credit', '100%', 'pass']
    for variant in correct_variants:
        if variant in pred_lower:
            return "Correct"
    
    # Check for exact matches of 1
    if pred_lower == '1' or pred_lower == '1.0':
        return "Correct"
    
    partial_variants = ['partial', 'partial credit', 'half', 'incomplete', 'partially correct']
    for variant in partial_variants:
        if variant in pred_lower:
            return "Partial"
    
    # Check for exact match of 0.5
    if pred_lower == '0.5':
        return "Partial"
    
    # Check for fraction 1/2
    if '1/2' in pred_lower or 'half' in pred_lower:
        return "Partial"
    
    # If it contains a number, try to interpret it
    numbers = re.findall(r'\d+\.?\d*', prediction)
    if numbers:
        try:
            num = float(numbers[0])
            # Handle 0-1 scale
            if 0 <= num <= 1:
                if num >= 0.8:
                    return "Correct"
                elif num >= 0.3:
                    return "Partial"
                else:
                    return "Incorrect"
            # Handle 0-100 scale (percentage)
            elif 0 <= num <= 100:
                if num >= 80:
                    return "Correct"
                elif num >= 30:
                    return "Partial"
                else:
                    return "Incorrect"
            # Handle 0-10 scale
            elif 0 <= num <= 10:
                if num >= 8:
                    return "Correct"
                elif num >= 3:
                    return "Partial"
                else:
                    return "Incorrect"
        except ValueError:
            pass
    
    # Check for letter grades
    if re.match(r'^[abc][+-]?$', pred_lower):
        if pred_lower.startswith('a'):
            return "Correct"
        elif pred_lower.startswith('b'):
            return "Partial"
        else:
            return "Incorrect"
    
    # Return original if no normalization applied
    return prediction


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

GRADE DEFINITIONS:
- "Correct": The answer is fully correct, complete, and follows all requirements. Minor notation issues that don't affect mathematical correctness are acceptable.
- "Partial": The answer shows significant valid reasoning but is incomplete, has minor errors, or misses some requirements. Award this when the student demonstrates understanding of the core concepts but falls short of a complete solution.
- "Incorrect": The answer is fundamentally wrong, shows no valid reasoning, or completely misses the problem requirements.

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade - must be exactly one of: 'Correct', 'Partial', or 'Incorrect'"
}}
</json>

IMPORTANT FORMATTING RULES:
- Use double quotes for all JSON strings, never single quotes
- Do not include trailing commas
- Ensure the JSON is valid and parseable
- The response field MUST contain exactly one of: 'Correct', 'Partial', or 'Incorrect'

Important: Be objective and consistent. Award partial credit generously when the student shows valid reasoning even if the final answer is incorrect. When in doubt between two grades, prefer the more lenient grade that acknowledges the student's effort and understanding.

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines
- Your JSON output follows the exact format specified above
- Your response field contains exactly one of the three allowed values"""

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
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            
            # Last resort: try to extract grade from raw text
            try:
                last_message = msg_history[-1]["text"]
                text_grade = _extract_grade_from_text(last_message)
                if text_grade:
                    self.log_fn(f"Extracted grade from text analysis: {text_grade}")
                    prediction = text_grade
            except Exception as text_e:
                self.log_fn(f"Text extraction also failed: {text_e}")

        return str(prediction), msg_history
