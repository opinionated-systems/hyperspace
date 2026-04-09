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
            # Try to fix common JSON issues before giving up
            try:
                # Fix trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (carefully)
                fixed = fixed.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try one more aggressive fix: extract just the JSON object
                try:
                    # Find the first { and last }
                    json_start = inner.find('{')
                    json_end = inner.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        fixed = inner[json_start:json_end+1]
                        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                        fixed = fixed.replace("'", '"')
                        results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
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
    
    # If it contains a number, try to interpret it
    numbers = re.findall(r'\d+\.?\d*', prediction)
    if numbers:
        try:
            num = float(numbers[0])
            if num >= 0.8:
                return "Correct"
            elif num >= 0.4:
                return "Partial"
            else:
                return "Incorrect"
        except ValueError:
            pass
    
    # Return original if no normalization applied
    return prediction


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.stats["total_calls"] += 1
        
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Validate inputs
        if not student_answer or not student_answer.strip():
            self.log_fn("Warning: Empty student answer received")
            return "Incorrect", []

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

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "response": "The final grade/prediction (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score)",
    "confidence": "High|Medium|Low - your confidence in this grade"
}}
</json>

IMPORTANT FORMATTING RULES:
- Use double quotes for all JSON strings, never single quotes
- Do not include trailing commas
- Ensure the JSON is valid and parseable
- The response field must contain a concise grade label
- The confidence field helps track grading reliability

Important: Be objective and consistent. Award partial credit when the student shows valid reasoning even if the final answer is incorrect.

SELF-CORRECTION CHECK: Before finalizing your grade, verify that:
- You haven't missed any subtle errors in the student's work
- You haven't overlooked partial credit opportunities
- Your grade aligns with the official solution's requirements
- Your reasoning is consistent with the grading guidelines
- Your JSON output follows the exact format specified above"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            self.stats["failed_extractions"] += 1
            return "None", []

        # Extract prediction from JSON with fallback mechanisms
        prediction = "None"
        confidence = "Unknown"
        extraction_method = "none"
        
        try:
            if not msg_history or len(msg_history) < 2:
                self.log_fn("Warning: Empty message history from LLM")
                self.stats["failed_extractions"] += 1
                return "None", msg_history
                
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method (from <json> tags)
            extracted = _extract_jsons(last_message)
            if extracted:
                extraction_method = "json_tags"
                self.stats["successful_extractions"] += 1
            
            # Fallback 1: generic JSON extraction from braces
            if extracted is None:
                extracted = _extract_any_json(last_message)
                if extracted:
                    extraction_method = "any_json"
                    self.stats["fallback_extractions"] += 1
            
            # Fallback 2: markdown code blocks
            if extracted is None:
                extracted = _extract_from_markdown_code_blocks(last_message)
                if extracted:
                    extraction_method = "markdown"
                    self.stats["fallback_extractions"] += 1
            
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
                
                # Extract confidence if available
                if "confidence" in last_json:
                    confidence = last_json["confidence"]
                
                self.log_fn(f"Extracted prediction via {extraction_method}: {prediction} (confidence: {confidence})")
                
                # Normalize the grade to standard format
                normalized = _normalize_grade(prediction)
                if normalized != prediction:
                    self.log_fn(f"Normalized grade: {prediction} -> {normalized}")
                    prediction = normalized
            else:
                self.stats["failed_extractions"] += 1
                self.log_fn(f"Failed to extract JSON from response. Raw response preview: {last_message[:200]}...")
                    
        except Exception as e:
            self.stats["failed_extractions"] += 1
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
    
    def get_stats(self) -> dict:
        """Return extraction statistics for monitoring."""
        total = self.stats["total_calls"]
        if total > 0:
            success_rate = (self.stats["successful_extractions"] / total) * 100
            fallback_rate = (self.stats["fallback_extractions"] / total) * 100
            fail_rate = (self.stats["failed_extractions"] / total) * 100
        else:
            success_rate = fallback_rate = fail_rate = 0.0
        
        return {
            **self.stats,
            "success_rate_pct": round(success_rate, 2),
            "fallback_rate_pct": round(fallback_rate, 2),
            "fail_rate_pct": round(fail_rate, 2),
        }
