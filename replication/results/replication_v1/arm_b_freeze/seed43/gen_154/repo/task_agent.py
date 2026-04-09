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
    Also handles markdown code blocks with json tag and inline JSON.
    Includes multiple fallback strategies for robust extraction.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
            # Try to fix common JSON issues
            fixed = _fix_json(inner)
            if fixed:
                results.append(fixed)
    
    # Also try markdown code blocks ```json ... ```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                break
            start = start + 7  # Skip past ```json
            end = text.find("```", start)
            if end == -1:
                break
            inner = text[start:end].strip()
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                fixed = _fix_json(inner)
                if fixed:
                    results.append(fixed)
    
    return results or None


def _fix_json(text: str) -> dict | None:
    """Attempt to fix common JSON formatting issues.
    
    Applies multiple repair strategies in order of aggressiveness.
    Returns the parsed dict if successful, None otherwise.
    """
    fixes = [
        # Strategy 1: Remove trailing commas before closing braces/brackets
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        # Strategy 2: Fix unescaped newlines in strings
        lambda t: re.sub(r'(?<!\\)\n', '\\n', t),
        # Strategy 3: Fix unescaped tabs in strings
        lambda t: re.sub(r'(?<!\\)\t', '\\t', t),
        # Strategy 4: Fix unescaped carriage returns
        lambda t: re.sub(r'(?<!\\)\r', '\\r', t),
        # Strategy 5: Remove control characters
        lambda t: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', t),
        # Strategy 6: Fix single quotes used as JSON delimiters
        lambda t: t.replace("'", '"'),
        # Strategy 7: Remove BOM if present
        lambda t: t.lstrip('\ufeff'),
    ]
    
    current = text
    for fix in fixes:
        try:
            current = fix(current)
            return json.loads(current)
        except json.JSONDecodeError:
            continue
    
    return None


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text.
    
    Uses proper brace counting with string awareness to handle nested objects
    and braces inside strings correctly.
    """
    results = []
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        obj = json.loads(text[start_idx:i+1])
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

        # Analyze grading guidelines to determine expected format
        expected_format = self._analyze_grading_format(grading_guidelines)

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step:
1. Analyze what the problem is asking - identify key concepts and required steps
2. Review the official solution approach - understand the expected reasoning
3. Compare the student's answer to the official solution - check for correctness and completeness
4. Check if the student followed the grading guidelines - look for partial credit criteria
5. Determine the appropriate grade - be precise and consistent with guidelines

IMPORTANT: Your response MUST be valid JSON wrapped in <json> tags. Do not include any text outside the JSON tags.

Respond in this exact format:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis here. Be thorough and specific about what the student did right or wrong.",
    "response": "{expected_format}"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with fallback mechanisms
        prediction = self._extract_prediction(msg_history)

        return str(prediction), msg_history

    def _analyze_grading_format(self, grading_guidelines: str) -> str:
        """Analyze grading guidelines to determine expected response format."""
        guidelines_lower = grading_guidelines.lower()
        
        # Check for numeric scoring
        if any(x in guidelines_lower for x in ["score", "points", "mark", "out of", "/", "0-", "1-", "2-", "3-", "4-", "5-", "6-", "7-", "8-", "9-"]):
            return "The final numeric score (e.g., '5', '3/7', '2 points')"
        
        # Check for binary grading
        if any(x in guidelines_lower for x in ["correct", "incorrect", "right", "wrong", "true", "false", "pass", "fail"]):
            return "Correct or Incorrect"
        
        # Check for letter grades
        if any(x in guidelines_lower for x in ["grade a", "grade b", "grade c", "grade d", "grade f"]):
            return "The letter grade (A, B, C, D, or F)"
        
        # Default to flexible format
        return "The final grade/prediction. Use the exact format specified in the grading guidelines"

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history with multiple fallback strategies."""
        if not msg_history:
            return "None"
        
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if not extracted:
                self.log_fn("No JSON found in response, attempting text extraction")
                return self._extract_from_text(last_message)
            
            # Get the last JSON object (most likely to be the final answer)
            last_json = extracted[-1]
            
            # Priority order for field names
            priority_fields = [
                "response", "grade", "answer", "result", 
                "evaluation", "score", "verdict", "prediction",
                "output", "decision", "assessment", "conclusion"
            ]
            
            # Try priority fields first
            for field in priority_fields:
                if field in last_json:
                    value = last_json[field]
                    if isinstance(value, str):
                        return value.strip()
                    elif isinstance(value, (int, float)):
                        return str(value)
                    elif isinstance(value, bool):
                        return "Correct" if value else "Incorrect"
            
            # If no priority field found, look for any string or numeric value
            for key, value in last_json.items():
                if isinstance(value, str) and key not in ["reasoning", "explanation", "analysis", "thought"]:
                    return value.strip()
                elif isinstance(value, (int, float)):
                    return str(value)
                elif isinstance(value, bool) and key not in ["reasoning", "explanation", "analysis", "thought"]:
                    return "Correct" if value else "Incorrect"
            
            # Last resort: check if reasoning contains a clear verdict
            if "reasoning" in last_json and isinstance(last_json["reasoning"], str):
                return self._extract_from_text(last_json["reasoning"])
            
            self.log_fn(f"Could not extract prediction from JSON: {last_json}")
            return "None"
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            return "None"
    
    def _extract_from_text(self, text: str) -> str:
        """Extract a grade/prediction from plain text when JSON parsing fails."""
        text_lower = text.lower()
        
        # Look for explicit grade statements
        patterns = [
            # Binary grades
            (r'(?:grade|score|verdict|assessment|evaluation)[:\s]+(correct|incorrect|right|wrong|true|false|pass|fail)', 1),
            (r'(?:is|would be|should be|final)[:\s]+(correct|incorrect|right|wrong|true|false|pass|fail)', 1),
            (r'\b(correct|incorrect)\b', 0),
            # Numeric grades
            (r'(?:grade|score|points?)[:\s]+(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)', 1),
            (r'(?:score|grade) of (\d+(?:\.\d+)?)', 1),
            (r'(?:awarded?|given?|assigned?)[:\s]+(\d+(?:\.\d+)?(?:\s*points?)?)', 1),
            # Letter grades
            (r'(?:grade|score)[:\s]+([abcdf][+-]?)', 1),
            (r'\b([abcdf][+-]?)\s*(?:grade|score)\b', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = match.group(group)
                # Normalize binary results
                if result in ["right", "true", "pass"]:
                    return "Correct"
                elif result in ["wrong", "false", "fail"]:
                    return "Incorrect"
                return result.capitalize() if result.islower() else result
        
        # Check for partial credit indicators
        if any(word in text_lower for word in ["partial", "partially", "some credit", "incomplete"]):
            return "Partial"
        
        return "None"
