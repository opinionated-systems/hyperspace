"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Enhanced with structured reasoning and few-shot examples for better performance.

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
    Also handles markdown code blocks and plain JSON.
    Enhanced with nested brace counting for more robust extraction.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
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
            # Try to extract JSON from within the content using brace counting
            json_str = _extract_json_with_brace_counting(inner)
            if json_str:
                try:
                    results.append(json.loads(json_str))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        search_from = 0
        while True:
            # Try ```json blocks
            start = text.find("```json", search_from)
            if start == -1:
                # Try plain ``` blocks
                start = text.find("```", search_from)
                if start == -1:
                    break
                end = text.find("```", start + 3)
                if end == -1:
                    break
                inner = text[start + 3:end].strip()
            else:
                end = text.find("```", start + 7)
                if end == -1:
                    break
                inner = text[start + 7:end].strip()
            
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to extract JSON object within the content using brace counting
                json_str = _extract_json_with_brace_counting(inner)
                if json_str:
                    try:
                        results.append(json.loads(json_str))
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    # Last resort: try to find any JSON object in the text using brace counting
    if not results:
        json_str = _extract_json_with_brace_counting(text)
        if json_str:
            try:
                results.append(json.loads(json_str))
            except (json.JSONDecodeError, ValueError):
                pass
    
    return results or None


def _extract_json_with_brace_counting(text: str) -> str | None:
    """Extract a JSON object from text using brace counting.
    
    This is more robust than simple first/last brace matching as it
    properly handles nested structures.
    """
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        return None
    
    # Use brace counting to find the matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\" and in_string:
            escape_next = True
            continue
        
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i + 1]
    
    return None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1 - Correct Answer:
Problem: Find the sum of 2 + 3.
Solution: The sum is 5.
Grading Guidelines: Award 1 point for correct answer.
Student Answer: 5
Analysis: The student's answer matches the correct solution exactly. The answer is complete and correct.
Score: 1/1

Example 2 - Partial Credit:
Problem: Solve x^2 = 4.
Solution: x = 2 or x = -2.
Grading Guidelines: Award 1 point for each correct solution (max 2 points).
Student Answer: x = 2
Analysis: The student found one correct solution (x = 2) but missed the negative solution (x = -2). According to the guidelines, 1 point is awarded for each correct solution.
Score: 1/2

Example 3 - Insufficient Work:
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof with parallel line construction]
Grading Guidelines: Award 1 point for stating the theorem, 2 points for correct proof steps.
Student Answer: The angles add up to 180 because it's a triangle.
Analysis: The student stated the correct result (1 point) but provided no valid proof or reasoning. The answer lacks the required geometric construction and logical steps needed for full credit.
Score: 1/3

Example 4 - Complex Partial Credit:
Problem: Find all integer solutions to x^2 + y^2 = 25.
Solution: The integer solutions are (0,±5), (±5,0), (±3,±4), (±4,±3) - total 12 solutions.
Grading Guidelines: Award 1 point for finding solutions with 0, 1 point for solutions with 5, 2 points for solutions with 3 and 4.
Student Answer: (3,4) and (4,3)
Analysis: The student found 2 valid solutions involving 3 and 4, earning 1 point out of 2 for that category. However, they missed all solutions with 0 and 5, losing 2 more points. The answer shows some understanding but is incomplete.
Score: 1/4

Example 5 - Wrong Format:
Problem: Compute the definite integral of x^2 from 0 to 3.
Solution: ∫[0 to 3] x^2 dx = [x^3/3][0 to 3] = 9
Grading Guidelines: Award 1 point for correct setup, 1 point for correct antiderivative, 1 point for correct evaluation.
Student Answer: 27/3
Analysis: The student's answer simplifies to 9, which is correct, but shows only the final evaluation without the setup or antiderivative steps. The answer is mathematically correct but doesn't demonstrate the required work for full credit.
Score: 1/3

Example 6 - Complete Proof:
Problem: Prove that for any positive integer n, n^3 - n is divisible by 6.
Solution: Factor as n(n-1)(n+1). Among three consecutive integers, one is divisible by 2 and one by 3, so product is divisible by 6.
Grading Guidelines: Award 2 points for correct factorization, 2 points for recognizing consecutive integers property, 2 points for divisibility argument.
Student Answer: n^3 - n = n(n^2 - 1) = n(n-1)(n+1). These are three consecutive integers. One must be even (divisible by 2) and one must be divisible by 3. Therefore the product is divisible by 6.
Analysis: The student provided a complete and correct proof. They correctly factorized (2 points), identified the consecutive integers property (2 points), and gave a valid divisibility argument (2 points).
Score: 6/6

Example 7 - Zero Credit:
Problem: Find the area of a circle with radius 5.
Solution: Area = πr^2 = 25π
Grading Guidelines: Award 1 point for correct formula, 1 point for correct substitution, 1 point for correct answer.
Student Answer: 10π
Analysis: The student used an incorrect formula (circumference instead of area). No points awarded as the fundamental approach was wrong.
Score: 0/3
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with structured reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution using the provided grading guidelines.

{FEW_SHOT_EXAMPLES}

Now evaluate the following:

Problem:
```
{problem}
```

Correct Solution:
```
{solution}
```

Grading Guidelines:
```
{grading_guidelines}
```

Student's Answer:
```
{student_answer}
```

Think step by step and provide a thorough evaluation:
1. Analyze what the problem is asking and identify key concepts
2. Review the correct solution approach and identify all required components
3. Compare the student's answer to the correct solution element by element
4. Check against each criterion in the grading guidelines carefully
5. Consider partial credit - did the student show any correct reasoning even if the final answer is wrong?
6. Look for common errors: calculation mistakes, missing cases, incomplete proofs, wrong format
7. Determine the score based strictly on the grading guidelines provided

Important grading principles:
- Award partial credit when the student demonstrates correct understanding of some concepts
- Deduct points for missing required components mentioned in the guidelines
- Consider the effort and reasoning shown, not just the final answer
- If the answer is in a different but equivalent form, it may still be correct
- Check if the student answered a simpler version of the problem correctly
- Be precise with point allocation - follow the guidelines exactly
- If the guidelines specify "X points for Y", award exactly X points when Y is present
- If the guidelines specify "deduct X points for Y", deduct exactly X points when Y is present

CRITICAL: You must respond ONLY in valid JSON format with the following schema. Do not include any text before or after the JSON:

<json>
{{
    "analysis": "Your detailed analysis of the student's answer, including what they did right and what they missed",
    "score_breakdown": "Specific explanation of how points were awarded/deducted based on each guideline criterion",
    "response": "The final score in the exact format specified in the guidelines (e.g., '3/5' or '2 points')"
}}
</json>

The "response" field must contain ONLY the score value, such as "3/5" or "2 points". Do not add extra text in this field."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1] if msg_history else {}
            last_text = last_msg.get("text", "") if isinstance(last_msg, dict) else str(last_msg)
            
            extracted = _extract_jsons(last_text)
            if extracted:
                last_json = extracted[-1]
                
                # Try multiple possible field names for the score
                score_fields = ["response", "score", "grade", "points", "result", "evaluation"]
                for field in score_fields:
                    if field in last_json and last_json[field]:
                        prediction = str(last_json[field]).strip()
                        break
                
                # Log analysis if available
                if "analysis" in last_json:
                    self.log_fn(f"Analysis: {last_json['analysis'][:200]}...")
                if "score_breakdown" in last_json:
                    self.log_fn(f"Score breakdown: {last_json['score_breakdown'][:200]}...")
            else:
                # Fallback: try to extract score pattern from raw text
                prediction = _extract_score_from_text(last_text)
                if prediction != "None":
                    self.log_fn(f"Extracted score via pattern: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history


def _extract_score_from_text(text: str) -> str:
    """Extract score from raw text using multiple patterns.
    
    Returns the score string or "None" if no score found.
    """
    score_patterns = [
        # Match "Score: X/Y" or "Score: X points"
        (r'Score:\s*(\d+)\s*[/\\]\s*(\d+)', lambda m: f"{m.group(1)}/{m.group(2)}"),
        (r'Score:\s*(\d+)\s*points?', lambda m: m.group(1)),
        # Match "X/Y" format
        (r'(?:^|\s)(\d+)\s*[/\\]\s*(\d+)(?:\s|$|[^\d])', lambda m: f"{m.group(1)}/{m.group(2)}"),
        # Match "X out of Y" format
        (r'(\d+)\s+out\s+of\s+(\d+)', lambda m: f"{m.group(1)}/{m.group(2)}"),
        # Match "X points" or "X point"
        (r'(\d+)\s*points?', lambda m: m.group(1)),
        # Match grade/score with colon or equals
        (r'(?:score|grade|points?)\s*[:=]\s*(\d+)', lambda m: m.group(1)),
        (r'(?:score|grade|points?)\s*[:=]\s*(\d+)\s*[/\\]\s*(\d+)', lambda m: f"{m.group(1)}/{m.group(2)}"),
    ]
    
    for pattern, extractor in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return extractor(match)
    
    return "None"
