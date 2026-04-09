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
    Also handles markdown code blocks with json tag.
    Includes advanced cleanup for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First try to find explicit <json> tags
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
            # Try to clean up common issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            end = text.find("```", start + 7)
            if end == -1:
                break
            inner = text[start + 7:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                try:
                    cleaned = _clean_json_string(inner)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    continue
    
    # Try to find JSON objects in plain text as last resort
    if not results:
        results = _extract_any_json(text)
    
    return results or None


def _clean_json_string(text: str) -> str:
    """Clean up common JSON formatting issues from LLM outputs.
    
    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Comments (// and /* */)
    """
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', text)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Remove single-line comments
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    
    # Remove multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Replace single quotes with double quotes (carefully)
    # This is a simplified approach - handles common cases
    cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
    
    # Normalize whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses brace counting to find balanced JSON objects, with additional
    validation and cleanup attempts for malformed JSON.
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
                candidate = text[start_idx:i+1]
                try:
                    obj = json.loads(candidate)
                    results.append(obj)
                except json.JSONDecodeError:
                    # Try with cleanup
                    try:
                        cleaned = _clean_json_string(candidate)
                        obj = json.loads(cleaned)
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                start_idx = -1
    return results or None


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

        # Check for empty student answer early
        if not student_answer or not student_answer.strip():
            return "Incorrect", [{"role": "system", "text": "Empty student answer detected, returning Incorrect"}]

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

1. PROBLEM ANALYSIS: Identify the key mathematical concepts, theorems, and techniques required to solve this problem.

2. SOLUTION REVIEW: Analyze the official solution's approach, key steps, and expected answer format. Note what constitutes a complete and correct solution.

3. STUDENT WORK ANALYSIS: 
   - Identify what approach the student took
   - Note any correct steps, valid insights, or correct intermediate results
   - Identify errors, gaps, misconceptions, or missing steps
   - Check if the final answer matches the expected form

4. GRADING CRITERIA CHECK:
   - Systematically verify if the student met each criterion in the grading guidelines
   - Award partial credit for incomplete but valid reasoning
   - Note any specific point deductions for errors or omissions

5. FINAL DETERMINATION: Assign a clear grade based on:
   - Completeness of the solution
   - Mathematical correctness
   - Adherence to grading guidelines
   - Quality of reasoning shown

Respond ONLY in JSON format with the following exact schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above. Be specific about what the student did right and wrong.",
    "response": "The final grade/prediction. Use exactly one of: 'Correct', 'Incorrect', 'Partial', or a numeric score if specified in guidelines."
}}
</json>

Important guidelines:
- Be objective, consistent, and thorough in your analysis
- Award partial credit when the student shows valid reasoning even if the final answer is incorrect
- If the student answer is empty or completely irrelevant, grade as 'Incorrect'
- If the student shows significant correct work but has minor errors, consider 'Partial'
- Only output the JSON block, no additional text before or after
- Ensure your JSON is valid with no trailing commas"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies.
        
        Args:
            msg_history: List of message dicts with 'role' and 'text' keys
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            return "None"
        
        try:
            last_message = msg_history[-1].get("text", "")
            if not last_message:
                return "None"
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            if not extracted:
                # Try to extract grade from plain text as last resort
                return self._extract_from_text(last_message)
            
            # Prefer response field, but accept other common field names
            last_json = extracted[-1]
            
            # Priority order for grade fields
            grade_fields = ["response", "grade", "answer", "result", "evaluation", 
                           "prediction", "score", "verdict", "assessment"]
            
            for field in grade_fields:
                if field in last_json:
                    field_value = last_json[field]
                    # Handle both string and numeric values
                    if isinstance(field_value, (str, int, float, bool)):
                        return str(field_value)
            
            # If no known field, use the first string or numeric value found
            for key, value in last_json.items():
                if isinstance(value, (str, int, float)) and key != "reasoning":
                    return str(value)
            
            # Last resort: check if there's a value that looks like a grade
            for key, value in last_json.items():
                if isinstance(value, str) and value.lower() in ["correct", "incorrect", "partial", "true", "false"]:
                    return value
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None"
    
    def _extract_from_text(self, text: str) -> str:
        """Extract grade from plain text when JSON parsing fails.
        
        Args:
            text: Raw text to search for grade indicators
            
        Returns:
            Extracted grade or "None"
        """
        text_lower = text.lower()
        
        # Look for explicit grade statements with more comprehensive patterns
        grade_patterns = [
            (r'grade\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 1),
            (r'final\s*(?:grade|determination|verdict)\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 1),
            (r'response\s*[:=]\s*["\']?(correct|incorrect|partial)["\']?', 1),
            (r'(?:the\s+)?(?:answer|grade)\s+is\s+["\']?(correct|incorrect|partial)["\']?', 1),
            (r'["\']?(correct|incorrect|partial)["\']?\s*(?:grade|score|verdict)', 1),
            (r'(?:i\s+)?(?:conclude|determine|assess|judge)\s+(?:that\s+)?(?:it\s+is\s+)?["\']?(correct|incorrect|partial)["\']?', 1),
            (r'(?:this\s+is\s+)?["\']?(correct|incorrect|partial)["\']?(?:\s+answer)?', 1),
        ]
        
        for pattern, group in grade_patterns:
            match = re.search(pattern, text_lower)
            if match:
                grade = match.group(group).lower()
                # Normalize grade
                if grade in ["correct", "true", "right", "valid"]:
                    return "Correct"
                elif grade in ["incorrect", "false", "wrong", "invalid", "error"]:
                    return "Incorrect"
                elif grade == "partial":
                    return "Partial"
        
        # Count mentions as fallback, with context awareness
        # Weight phrases that appear near keywords like "student", "answer", "solution"
        context_keywords = ["student", "answer", "solution", "response", "submission"]
        has_context = any(kw in text_lower for kw in context_keywords)
        
        correct_count = len(re.findall(r'\bcorrect\b', text_lower))
        incorrect_count = len(re.findall(r'\bincorrect\b', text_lower))
        partial_count = len(re.findall(r'\bpartial\b', text_lower))
        
        # Boost counts if they appear in context
        if has_context:
            for kw in context_keywords:
                # Look for patterns like "student answer is correct"
                correct_count += len(re.findall(rf'{kw}[^.]*correct', text_lower))
                incorrect_count += len(re.findall(rf'{kw}[^.]*incorrect', text_lower))
        
        if incorrect_count > correct_count and incorrect_count > partial_count:
            return "Incorrect"
        elif partial_count > correct_count and partial_count > incorrect_count:
            return "Partial"
        elif correct_count > 0:
            return "Correct"
        
        return "None"
