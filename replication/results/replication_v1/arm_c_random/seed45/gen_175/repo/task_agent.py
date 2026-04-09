"""
Task agent: solves a given task with chain-of-thought reasoning and self-reflection.

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
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery for malformed JSON with enhanced validation.
    """
    if not text or not isinstance(text, str):
        return None
        
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
            # Try to extract JSON from within the text if it's wrapped in other content
            try:
                # Look for JSON object pattern with balanced braces
                json_start = inner.find("{")
                if json_start != -1:
                    # Find matching closing brace by counting
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(inner[json_start:], start=json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i
                                break
                    if json_end != -1:
                        results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Match ```json ... ``` or just ``` ... ``` blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try to find JSON object within the match
                try:
                    json_start = match.find("{")
                    json_end = match.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        results.append(json.loads(match[json_start:json_end + 1].strip()))
                except json.JSONDecodeError:
                    continue
    
    # Last resort: try to find any JSON object in the text
    if not results:
        # Look for JSON-like patterns with improved matching
        # This pattern handles nested braces better
        json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                # Only include if it has expected keys
                if any(key in parsed for key in ["score", "max_score", "thinking", "rationale", "response", "final_response", "revised_score", "revised_max_score"]):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    # Validate extracted results have required fields for grading
    validated_results = []
    for result in results:
        if isinstance(result, dict):
            # Ensure score and max_score are present and numeric if possible
            if "score" in result:
                try:
                    result["score"] = float(result["score"])
                except (ValueError, TypeError):
                    pass
            if "max_score" in result:
                try:
                    result["max_score"] = float(result["max_score"])
                except (ValueError, TypeError):
                    pass
            # Also validate revised_score fields
            if "revised_score" in result:
                try:
                    result["revised_score"] = float(result["revised_score"])
                except (ValueError, TypeError):
                    pass
            if "revised_max_score" in result:
                try:
                    result["revised_max_score"] = float(result["revised_max_score"])
                except (ValueError, TypeError):
                    pass
            validated_results.append(result)
    
    return validated_results if validated_results else None


# Few-shot examples for grading
FEW_SHOT_EXAMPLES = """
Here are some examples of proper grading:

Example 1:
Problem: Find the sum of 2+2.
Official Solution: The sum is 4.
Student Answer: 4
Grade: <json>{"thinking": "The student correctly answered 4, which matches the official solution.", "score": 1, "max_score": 1, "rationale": "Correct answer", "response": "1/1 - Correct"}</json>

Example 2:
Problem: Solve x^2 = 4.
Official Solution: x = 2 or x = -2.
Student Answer: x = 2
Grade: <json>{"thinking": "The student found one correct solution but missed x = -2.", "score": 1, "max_score": 2, "rationale": "Partial credit for finding one solution", "response": "1/2 - Partial credit"}</json>
"""


class TaskAgent:
    """Task agent that grades mathematical problems with reasoning and reflection."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract the score/max_score prediction from message history."""
        try:
            if not msg_history:
                return "None"
            
            # Get the last assistant message
            last_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant":
                    last_msg = msg
                    break
            
            if not last_msg:
                return "None"
            
            text = last_msg.get("text", "")
            if not text:
                return "None"
            
            # Try to extract JSON from the response
            jsons = _extract_jsons(text)
            if not jsons:
                return "None"
            
            # Use the last JSON object (most recent)
            data = jsons[-1]
            
            # Check for revised_score first (from reflection), then score
            score = data.get("revised_score", data.get("score"))
            max_score = data.get("revised_max_score", data.get("max_score"))
            
            if score is None or max_score is None:
                return "None"
            
            # Validate and convert to numeric
            try:
                score_val = float(score) if not isinstance(score, (int, float)) else score
                max_val = float(max_score) if not isinstance(max_score, (int, float)) else max_score
                
                # Sanity check: score should not exceed max_score by too much
                if score_val > max_val * 1.5:
                    self.log_fn(f"Warning: Score {score_val} seems unusually high compared to max {max_val}")
                
                return f"{int(score_val)}/{int(max_val)}"
            except (ValueError, TypeError):
                return f"{score}/{max_score}"
        
        except Exception as e:
            self.log_fn(f"Error in _extract_prediction: {e}")
            return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with reasoning and reflection.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Step 1: Initial grading with chain-of-thought
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to grade a student's answer to a mathematical problem.

{FEW_SHOT_EXAMPLES}

Now grade the following problem:

Domain: {inputs.get('domain', 'Mathematics')}

Problem:
{inputs.get('problem', '')}

Official Solution:
{inputs.get('solution', '')}

Grading Guidelines:
{inputs.get('grading_guidelines', '')}

Student Answer:
{inputs.get('student_answer', '')}

Think step by step:
1. Analyze what the student did correctly according to the official solution
2. Identify any errors, gaps, or missing steps
3. Compare against the grading guidelines
4. Determine the score and provide detailed rationale

Respond in JSON format with the following schema:
<json>
{{
    "thinking": "Your detailed step-by-step analysis here",
    "score": <numerical_score_as_number>,
    "max_score": <maximum_possible_score_as_number>,
    "rationale": "Detailed explanation of why this score was awarded",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>

IMPORTANT: score and max_score must be numeric values (integers or floats), NOT strings."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced error handling
        prediction = self._extract_prediction(msg_history)
        
        # Log extraction issues for debugging
        if prediction == "None":
            self.log_fn(f"Warning: Could not extract valid prediction from response. Raw text: {msg_history[-1]['text'][:500]}...")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above carefully. Perform a critical self-check:

1. ACCURACY: Did you award points the student didn't actually earn? Check each claim against the official solution.
2. ERRORS: Did you miss any mathematical errors, logical gaps, or incorrect statements in the student's work?
3. CONSISTENCY: Is your score strictly consistent with the grading guidelines? Are you being too lenient or too harsh?
4. COMPLETENESS: Did the student address all parts of the problem? Are there missing cases or incomplete reasoning?
5. ALTERNATIVE APPROACHES: If the student used a different valid method than the official solution, did you fairly credit it?

IMPORTANT: Your revised_score MUST be a numeric value (integer or float), not a string.
Your revised_max_score MUST also be a numeric value.

Be honest and critical. If you find any issues, provide a corrected grade with clear justification.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review addressing each point above",
    "revised_score": <numeric_score>,
    "revised_max_score": <numeric_max_score>,
    "final_response": "<score>/<max_score> - <brief summary of final assessment>"
}}
</json>"""
            
            reflection_response, msg_history, _ = get_response_from_llm(
                msg=reflection_msg,
                model=self.model,
                msg_history=msg_history,
            )
            
            # Extract revised prediction using the helper method
            revised_prediction = self._extract_prediction(msg_history)
            if revised_prediction != "None":
                prediction = revised_prediction
                self.log_fn(f"Reflection updated prediction to: {prediction}")
            else:
                self.log_fn("Reflection did not produce a valid revised prediction, keeping original")

        return str(prediction), msg_history
