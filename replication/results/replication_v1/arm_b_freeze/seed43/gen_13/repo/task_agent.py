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
    Includes robust error recovery for malformed JSON.
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
            # Try to extract JSON from within the content
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
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
                # Try to find JSON object within the content
                try:
                    json_start = inner.find("{")
                    json_end = inner.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(inner[json_start:json_end + 1]))
                except (json.JSONDecodeError, ValueError):
                    continue
    
    # Try to find JSON objects by looking for balanced braces
    if not results:
        # Find all potential JSON objects by tracking brace balance
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Found a potential start
                start = i
                brace_count = 1
                i += 1
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    # Found a balanced JSON object
                    try:
                        json_str = text[start:i]
                        results.append(json.loads(json_str))
                    except (json.JSONDecodeError, ValueError):
                        pass
            else:
                i += 1
    
    return results or None


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

Example 6 - Equivalent Answer:
Problem: Simplify the fraction 4/8.
Solution: 1/2 or 0.5
Grading Guidelines: Award 1 point for correct simplified form.
Student Answer: 2/4
Analysis: The student's answer 2/4 is equivalent to 1/2 but is not in fully simplified form. The student demonstrated understanding of equivalent fractions but didn't complete the simplification. Partial credit is not awarded as the guidelines require the simplified form.
Score: 0/1

Example 7 - Complete Solution with All Cases:
Problem: Find all real solutions to |x| = 5.
Solution: x = 5 or x = -5
Grading Guidelines: Award 1 point for each correct solution (max 2 points).
Student Answer: x = ±5
Analysis: The student's answer x = ±5 correctly represents both solutions (x = 5 and x = -5) in compact notation. This is mathematically equivalent to listing both solutions separately.
Score: 2/2
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
- If the answer is in a different but equivalent form, it may still be correct (e.g., x = ±5 is equivalent to x = 5 or x = -5)
- Check if the student answered a simpler version of the problem correctly
- Be precise with mathematical notation: ± notation, set notation, interval notation are all valid if mathematically equivalent
- For "find all" problems, check if the student found ALL required solutions, not just some
- Award full points only when the answer completely satisfies the grading guidelines

CRITICAL: You must respond ONLY in valid JSON format with the following schema. Do not include any text before or after the JSON:
<json>
{{
    "analysis": "Your detailed analysis of the student's answer, including what they did right and what they missed",
    "score_breakdown": "Specific explanation of how points were awarded/deducted based on each guideline criterion",
    "response": "The final score in the exact format specified in the guidelines (e.g., '3/5' or '2 points')"
}}
</json>"""

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
                        prediction = last_json[field]
                        break
                
                # Log analysis if available
                if "analysis" in last_json:
                    self.log_fn(f"Analysis: {last_json['analysis'][:200]}...")
                if "score_breakdown" in last_json:
                    self.log_fn(f"Score breakdown: {last_json['score_breakdown'][:200]}...")
            else:
                # Fallback: try to extract score pattern from raw text
                import re
                score_patterns = [
                    # Standard "Score: X/Y" format
                    r'Score:\s*(\d+[/\\]\d+)',
                    r'Score:\s*(\d+)\s*points?',
                    r'Score:\s*(\d+)\s*/\s*(\d+)',
                    # "X/Y points" format
                    r'(\d+)[/\\](\d+)\s*points?',
                    # "X out of Y" format
                    r'(\d+)\s+out\s+of\s+(\d+)',
                    # Generic score/grade/points patterns
                    r'(?:score|grade|points?)\s*[:=]\s*(\d+[/\\]\d+)',
                    r'(?:score|grade|points?)\s*[:=]\s*(\d+)',
                    r'(?:score|grade|points?)\s*[:=]\s*(\d+)\s*/\s*(\d+)',
                    # Fraction patterns at end of lines
                    r'(?:^|\n)\s*(\d+)[/\\](\d+)\s*(?:$|\n)',
                    # "X points" format
                    r'(\d+)\s*points?',
                    # Decimal scores
                    r'Score:\s*(\d+\.\d+)',
                    r'(?:score|grade|points?)\s*[:=]\s*(\d+\.\d+)',
                    # Additional patterns for edge cases
                    r'(?:^|\n)\s*(\d+)\s*/\s*(\d+)\s*(?:$|\n)',  # Standalone X / Y
                    r'(?:total|final|awarded)?\s*(?:score|points)?\s*[:=]?\s*(\d+)[/\\](\d+)',  # Flexible format
                    r'(?:^|\n)\s*Score\s+is\s+[:=]?\s*(\d+)[/\\]?(\d*)',  # "Score is X" or "Score is X/Y"
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, last_text, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) > 1 and groups[1] and groups[1].strip():
                            prediction = f"{groups[0]}/{groups[1]}"
                        else:
                            prediction = groups[0]
                        self.log_fn(f"Extracted score via pattern: {prediction}")
                        break
                
                # If still no match, try to find any fraction-like pattern
                if prediction == "None":
                    # Look for patterns like "X/Y" near keywords
                    context_patterns = [
                        r'(?:score|grade|points|result|evaluation|mark).*?(\d+)\s*[/\\]\s*(\d+)',
                        r'(\d+)\s*[/\\]\s*(\d+).*?(?:score|grade|points|result|evaluation|mark)',
                    ]
                    for pattern in context_patterns:
                        match = re.search(pattern, last_text, re.IGNORECASE | re.DOTALL)
                        if match:
                            prediction = f"{match.group(1)}/{match.group(2)}"
                            self.log_fn(f"Extracted score via context pattern: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
