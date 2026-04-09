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
    Includes multiple fallback strategies for malformed JSON.
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
        
        # Try parsing directly first
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
        
        # Strategy 5: Try to find and fix unescaped newlines in strings
        try:
            # Replace newlines within the JSON with escaped newlines
            fixed = re.sub(r'(?<=")\n(?=")', '\\n', inner)
            fixed = re.sub(r'(?<=")\n(?=\w)', '\\n', fixed)
            results.append(json.loads(fixed))
            continue
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Extract just the response field if present
        try:
            response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
            if response_match:
                response_val = response_match.group(1)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', inner, re.DOTALL)
                reasoning_val = reasoning_match.group(1) if reasoning_match else "Extracted from partial JSON"
                results.append({"reasoning": reasoning_val, "response": response_val})
                continue
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
                
                # Try multiple parsing strategies
                for attempt in range(4):
                    try:
                        if attempt == 0:
                            # Direct parse
                            obj = json.loads(json_str)
                        elif attempt == 1:
                            # Fix trailing commas
                            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            obj = json.loads(fixed)
                        elif attempt == 2:
                            # Fix single quotes
                            fixed = json_str.replace("'", '"')
                            obj = json.loads(fixed)
                        else:
                            # Combined fixes
                            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            fixed = fixed.replace("'", '"')
                            obj = json.loads(fixed)
                        
                        # Check if it has expected fields for our use case
                        if isinstance(obj, dict):
                            results.append(obj)
                        break
                    except json.JSONDecodeError:
                        if attempt == 3:
                            # Final attempt: try to extract just key fields
                            try:
                                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', json_str)
                                if response_match:
                                    response_val = response_match.group(1)
                                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', json_str, re.DOTALL)
                                    reasoning_val = reasoning_match.group(1) if reasoning_match else "Extracted from partial JSON"
                                    results.append({"reasoning": reasoning_val, "response": response_val})
                            except Exception:
                                pass
                start_idx = -1
    return results or None


def _extract_from_markdown_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Handles cases where the model outputs JSON in markdown format.
    Also handles nested code blocks and multiple JSON objects.
    """
    results = []
    # Find markdown code blocks - be more flexible with the pattern
    # Handle both ```json and ``` (no language specifier)
    pattern = r'```(?:json|JSON)?\s*\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        match = match.strip()
        if not match:
            continue
            
        # Try to parse the content directly
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                results.append(obj)
            elif isinstance(obj, list):
                results.extend([item for item in obj if isinstance(item, dict)])
            continue
        except json.JSONDecodeError:
            pass
        
        # If direct parse fails, try to extract JSON objects from within
        if match.startswith('{'):
            try:
                # Try with common fixes
                fixed = re.sub(r',(\s*[}\]])', r'\1', match)
                fixed = fixed.replace("'", '"')
                obj = json.loads(fixed)
                if isinstance(obj, dict):
                    results.append(obj)
                continue
            except json.JSONDecodeError:
                pass
        
        # Try the fallback extraction on the content
        nested = _extract_any_json(match)
        if nested:
            results.extend(nested)
    
    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Last resort: extract grade directly from text using pattern matching.
    
    This is used when all JSON extraction methods fail.
    Looks for explicit grade statements in the text.
    """
    text_lower = text.lower()
    
    # Look for explicit grade statements
    patterns = [
        # Direct grade assignments
        r'grade\s*[=:]\s*"?(correct|incorrect|partial)"?',
        r'response\s*[=:]\s*"?(correct|incorrect|partial)"?',
        r'final\s*(?:grade|determination|answer)\s*[=:]\s*"?(correct|incorrect|partial)"?',
        # Conclusion statements
        r'(?:therefore|thus|hence|conclusion|final).*?(correct|incorrect|partial)',
        r'(?:the\s+student\s+(?:should\s+)?(?:receive|get|be\s+graded)).*?"?(correct|incorrect|partial)"?',
        # Standalone grades at end of text
        r'(?:^|\n)\s*"?(correct|incorrect|partial)"?\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            grade = match.group(1).capitalize()
            if grade in ["Correct", "Incorrect", "Partial"]:
                return grade
    
    # Check for keywords in the last 200 characters (often where conclusion is)
    conclusion = text_lower[-200:] if len(text_lower) > 200 else text_lower
    if re.search(r'\b(correct|right|true|pass)\b', conclusion) and not re.search(r'\b(incorrect|wrong|false|fail)\b', conclusion):
        return "Correct"
    if re.search(r'\b(incorrect|wrong|false|fail)\b', conclusion) and not re.search(r'\b(correct|right|true|pass)\b', conclusion):
        return "Incorrect"
    if re.search(r'\b(partial|partially|some|incomplete)\b', conclusion):
        return "Partial"
    
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
    # Use word boundaries to avoid partial matches
    if re.search(r'\b(incorrect|wrong|error|false|fail|no|0\s*\/\s*\d+|0\s*points?)\b', pred_lower):
        return "Incorrect"
    
    # Check for correct
    if re.search(r'\b(correct|right|true|yes|pass|1\s*\/\s*1|full\s*(credit|score|points?)|perfect|excellent)\b', pred_lower):
        return "Correct"
    
    # Check for partial credit - look for fraction patterns first
    fraction_match = re.search(r'(\d+)\s*\/\s*(\d+)', pred_lower)
    if fraction_match:
        num, den = int(fraction_match.group(1)), int(fraction_match.group(2))
        if den > 0:  # Avoid division by zero
            ratio = num / den
            if num == 0 or ratio < 0.3:
                return "Incorrect"
            elif ratio >= 0.9 or num >= den:
                return "Correct"
            else:
                return f"Partial ({num}/{den})"
    
    # Check for partial credit keywords
    if re.search(r'\b(partial|partially|some|incomplete|partial\s*(credit|score)|half|minor|partially\s*correct)\b', pred_lower):
        return "Partial"
    
    # Check for numeric scores (0-10 scale or 0-100 scale)
    match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:\/\s*\d+|out\s+of\s*\d+|points?|%|percent)?\b', pred_lower)
    if match:
        score = float(match.group(1))
        # Detect scale based on value
        if score <= 10 and '100' not in pred_lower:
            # Likely 0-10 scale
            if score == 0:
                return "Incorrect"
            elif score >= 8:  # 8-10 is correct
                return "Correct"
            elif score >= 4:  # 4-7 is partial
                return "Partial"
            else:
                return "Incorrect"
        else:
            # Likely 0-100 scale or percentage
            if score == 0:
                return "Incorrect"
            elif score >= 80:  # 80%+ is correct
                return "Correct"
            elif score >= 40:  # 40-79% is partial
                return "Partial"
            else:
                return "Incorrect"
    
    # Check for letter grades
    if re.search(r'\b(a\+|a|excellent|outstanding)\b', pred_lower):
        return "Correct"
    if re.search(r'\b(b|c|satisfactory|good)\b', pred_lower):
        return "Partial"
    if re.search(r'\b(d|f|unsatisfactory|poor|failing)\b', pred_lower):
        return "Incorrect"
    
    # Default: return original if no pattern matched
    return prediction.strip()


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

Think step by step and provide a thorough analysis following this structure:

STEP 1 - PROBLEM ANALYSIS:
Identify the key mathematical concepts, theorems, and techniques required to solve this problem. What makes this problem challenging?

STEP 2 - SOLUTION REVIEW:
Analyze the official solution's approach, key steps, and expected answer format. What are the critical elements that must be present in a correct solution?

STEP 3 - STUDENT WORK ANALYSIS:
- What approach did the student take?
- What correct steps or valid insights did they demonstrate?
- What errors, gaps, or misconceptions are present?
- Did they reach the correct final answer?

STEP 4 - GRADING CRITERIA CHECK:
- Which criteria from the grading guidelines did the student meet?
- Which criteria were not met?
- What partial credit should be awarded for incomplete but valid reasoning?

STEP 5 - FINAL DETERMINATION:
Based on your analysis, assign one of these grades:
- "Correct" - Complete and correct solution with proper reasoning
- "Incorrect" - Wrong answer with flawed or missing reasoning
- "Partial" - Partial credit for valid reasoning even if final answer is wrong

Respond ONLY in JSON format wrapped in <json> tags with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be thorough and specific.",
    "response": "The final grade - must be exactly one of: Correct, Incorrect, or Partial"
}}
</json>

CRITICAL FORMATTING RULES:
- Use double quotes for all JSON strings, never single quotes
- Do not include trailing commas
- Ensure the JSON is valid and parseable
- The response field MUST contain ONLY: "Correct", "Incorrect", or "Partial" (case-sensitive)
- Do not add extra text after the JSON block

GRADING PRINCIPLES:
- Be objective, consistent, and fair
- Award partial credit when the student shows valid mathematical reasoning even if the final answer is incorrect
- A correct final answer with no reasoning shown may warrant partial credit depending on the problem
- An incorrect final answer with good reasoning may warrant partial credit
- Follow the grading guidelines strictly for full credit determination

FINAL VERIFICATION: Before outputting your JSON, verify:
1. Your reasoning addresses all 5 steps above
2. Your grade is one of: Correct, Incorrect, or Partial
3. Your JSON uses double quotes only
4. There are no trailing commas
5. The JSON is properly closed with }}</json>"""

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
            else:
                # Fallback 3: extract grade directly from text
                text_grade = _extract_grade_from_text(last_message)
                if text_grade:
                    prediction = text_grade
                    extraction_method = "text_pattern"
                    self.log_fn(f"Extracted prediction via {extraction_method}: {prediction}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
