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
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes multiple cleanup strategies for common JSON formatting issues.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_json(match.strip())
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like object in the text
    if not results:
        # Look for patterns like {"key": "value"} with nested support
        # Use a more sophisticated approach: find balanced braces
        json_objects = _find_balanced_json_objects(text)
        for obj_text in json_objects:
            parsed = _try_parse_json(obj_text)
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple cleanup strategies."""
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes (common LLM mistake)
    try:
        # Replace single quotes around keys and string values
        # This is a simplified approach - be careful with nested quotes
        cleaned = re.sub(r"(?<!\\)'", '"', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Remove comments (// and /* */)
    try:
        # Remove single-line comments
        cleaned = re.sub(r'//.*?\n', '\n', text)
        # Remove multi-line comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Fix unescaped newlines in strings
    try:
        # Replace newlines within string values with escaped newlines
        cleaned = re.sub(r'(?<=")\n(?=")', '\\n', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def _find_balanced_json_objects(text: str) -> list[str]:
    """Find JSON objects by tracking balanced braces."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Found potential start of JSON object
            start = i
            brace_count = 1
            i += 1
            in_string = False
            escape_next = False
            
            while i < len(text) and brace_count > 0:
                char = text[i]
                
                if escape_next:
                    escape_next = False
                elif char == '\\' and in_string:
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                i += 1
            
            if brace_count == 0:
                # Found a balanced object
                obj_text = text[start:i]
                # Check if it looks like a JSON object (has key-value pairs)
                if '"' in obj_text or "'" in obj_text or ':' in obj_text:
                    results.append(obj_text)
        else:
            i += 1
    
    return results


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

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
        # Extract fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem and assign an appropriate score based on the official solution and grading guidelines.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Your Task

Follow this structured evaluation process carefully:

### Step 1: Problem Understanding
- Identify the key mathematical concepts and techniques required
- Note the critical steps in the official solution
- Understand the scoring rubric from the grading guidelines
- Determine the maximum possible score for this problem
- Identify what constitutes a complete, correct solution

### Step 2: Student Answer Analysis
- Check if the student stated the final answer correctly
- Identify which solution steps the student completed
- Note any missing or incorrect steps
- Evaluate the logical flow and mathematical rigor
- Look for alternative valid approaches that differ from the official solution
- Check for calculation errors, logical gaps, or unstated assumptions

### Step 3: Partial Credit Assessment
- Award points for each correct step completed
- Deduct points for logical gaps or errors
- Consider alternative valid approaches
- Be generous with partial credit when reasoning is sound
- Note: Even incomplete solutions may earn significant partial credit
- For each deduction, explicitly state what was missing or incorrect

### Step 4: Final Score Determination
- Sum the points earned across all steps
- Verify against the grading guidelines
- Ensure consistency with the official scoring rubric
- Double-check that your score reflects the student's actual work
- Consider: Would another expert grader assign the same score?

## Critical Grading Principles
1. **Correct final answer** with valid reasoning → Full marks
2. **Correct approach** with minor errors → Deduct 1-2 points
3. **Partial progress** → Award proportional points
4. **Alternative valid solutions** → Award full credit if mathematically sound
5. **No valid work shown** → Score of 0
6. **Significant progress** → Award at least half the points even if final answer is wrong
7. **Correct ideas but poor execution** → Award partial credit based on merit

## Common Mistakes to Avoid
- Do not penalize for notation differences if the mathematics is correct
- Do not require the student to use the exact same method as the official solution
- Do not award full marks if there are logical gaps, even with correct final answer
- Do not give zero if the student made meaningful progress

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis following the steps above. Include: (1) key concepts identified, (2) steps completed by student with specific point values, (3) errors or gaps found with point deductions, (4) alternative approaches considered, (5) justification for final score",
    "response": "The final score (e.g., '0', '1', '2', '7', etc.)"
}}
</json>

Be thorough in your reasoning, generous with partial credit for correct reasoning, and precise in your final scoring. Always verify your score against the grading guidelines before finalizing."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        extraction_method = "none"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                # Try to get response field first, fallback to other common fields
                last_extract = extracted[-1]
                
                # Priority order for score fields
                score_fields = ["response", "score", "answer", "grade", "mark", "points", "result"]
                for field in score_fields:
                    if field in last_extract:
                        prediction = last_extract[field]
                        extraction_method = f"json_field:{field}"
                        break
                else:
                    # If no recognized field, look for numeric values
                    for key, value in last_extract.items():
                        if isinstance(value, (str, int, float)):
                            str_val = str(value).strip()
                            # Check if it looks like a score (numeric, possibly with decimal or negative)
                            if _is_valid_score(str_val):
                                prediction = str_val
                                extraction_method = f"json_numeric:{key}"
                                break
                    else:
                        # Last resort: use first string value
                        for key, value in last_extract.items():
                            if isinstance(value, str):
                                prediction = value
                                extraction_method = f"json_first_string:{key}"
                                break
                        else:
                            prediction = str(last_extract) if last_extract else "None"
                            extraction_method = "json_str_repr"
            
            # Fallback: try to extract a score directly from the text using patterns
            if prediction == "None" or prediction is None:
                text = msg_history[-1]["text"]
                prediction, extraction_method = _extract_score_from_text(text)
                if prediction != "None":
                    self.log_fn(f"Extracted score via pattern '{extraction_method}': {prediction}")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try emergency text extraction
            try:
                text = msg_history[-1].get("text", "")
                prediction, extraction_method = _extract_score_from_text(text)
            except:
                pass

        # Clean up the prediction
        prediction = _clean_prediction(prediction)
        
        if extraction_method != "none":
            self.log_fn(f"Final prediction ({extraction_method}): {prediction}")

        return prediction, msg_history


def _is_valid_score(value: str) -> bool:
    """Check if a string value looks like a valid score."""
    if not value:
        return False
    # Remove common prefixes/suffixes
    cleaned = value.strip().lower()
    cleaned = re.sub(r'^(score|grade|mark|points|value)[:\s=]+', '', cleaned)
    cleaned = re.sub(r'\s*(points?|marks?|score)$', '', cleaned)
    cleaned = cleaned.strip()
    
    # Check if it's a valid number (integer or decimal, possibly negative)
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _extract_score_from_text(text: str) -> tuple[str, str]:
    """Extract a score from text using various patterns.
    
    Returns (score, method) tuple.
    """
    # Pattern priority order - more specific patterns first
    score_patterns = [
        # Explicit final score mentions
        (r'[Ff]inal\s+[Ss]core[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "final_score"),
        (r'[Ff]inal\s+[Gg]rade[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "final_grade"),
        (r'[Ff]inal\s+[Mm]ark[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "final_mark"),
        
        # Response/answer field with number
        (r'[Rr]esponse[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "response_field"),
        (r'[Aa]nswer[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "answer_field"),
        
        # Score/grade/mark with number
        (r'[Ss]core[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "score_field"),
        (r'[Gg]rade[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "grade_field"),
        (r'[Mm]ark[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "mark_field"),
        (r'[Pp]oints[:\s=]+["\']?(-?\d+(?:\.\d+)?)["\']?', "points_field"),
        
        # Number followed by points/marks/score
        (r'["\']?(-?\d+(?:\.\d+)?)["\']?\s*(?:points?|marks?|score)', "number_then_unit"),
        
        # Standalone numbers at end of sentences (often scores)
        (r'(?:score|grade|mark|point)s?\s+(?:is|of|equals?)\s+["\']?(-?\d+(?:\.\d+)?)["\']?', "score_is_number"),
        
        # Numbers in parentheses after score mentions
        (r'[Ss]core.*?\((-?\d+(?:\.\d+)?)\)', "score_in_parens"),
        
        # Simple standalone numbers (last resort - be careful)
        (r'(?:^|\n)\s*(-?\d+(?:\.\d+)?)\s*(?:\n|$)', "standalone_number"),
    ]
    
    for pattern, method in score_patterns:
        match = re.search(pattern, text)
        if match:
            score = match.group(1).strip()
            if _is_valid_score(score):
                return score, method
    
    return "None", "none"


def _clean_prediction(prediction) -> str:
    """Clean and validate the prediction value."""
    if prediction is None:
        return "None"
    
    prediction = str(prediction).strip()
    
    # Handle common null-like values
    if prediction.lower() in ["none", "null", "undefined", "", "nan", "inf"]:
        return "None"
    
    # Try to extract just the numeric part if there's extra text
    numeric_match = re.match(r'^(-?\d+(?:\.\d+)?)', prediction)
    if numeric_match:
        return numeric_match.group(1)
    
    return prediction
