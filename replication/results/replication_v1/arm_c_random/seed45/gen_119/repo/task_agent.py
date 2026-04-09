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
    Also handles markdown code blocks as a fallback.
    Includes robust error recovery for malformed JSON with multiple fix strategies.
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
        
        # Try parsing with multiple recovery strategies
        parsed = _try_parse_json_with_fixes(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Fallback: try markdown code blocks if no <json> blocks found
    if not results:
        # Look for ```json ... ``` blocks (with or without trailing newline)
        markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            parsed = _try_parse_json_with_fixes(match.group(1).strip())
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like object in the text
    if not results:
        # Look for patterns like {"key": value} or {"key": "value"}
        # Improved pattern to handle nested braces more robustly
        json_pattern = r'\{(?:[^{}]|\{[^{}]*\})*"[^"]+"\s*:\s*(?:[^}]|\{[^{}]*\})*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _try_parse_json_with_fixes(text: str) -> dict | None:
    """Try to parse JSON with multiple fix strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes (carefully)
    try:
        # Only replace single quotes that are likely JSON string delimiters
        # (not apostrophes in words like "don't")
        fixed = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Combined fixes
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        fixed = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Handle unquoted keys
    try:
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 6: Handle escaped quotes and newlines in strings
    try:
        fixed = text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 7: Extract first complete JSON object by brace counting
    try:
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
                    return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    
    logger.debug(f"All JSON parse strategies failed for: {text[:100]}...")
    return None


# Few-shot examples for IMO grading
FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find all positive integers n such that n^2 + 3n + 2 is divisible by 4.
Solution: n^2 + 3n + 2 = (n+1)(n+2). For divisibility by 4, either n+1 or n+2 must be even, and one of them must be divisible by 4. This happens when n ≡ 0 or 3 (mod 4).
Grading Guidelines: Award 1 point for factoring, 1 point for analyzing cases, 1 point for correct answer.
Student Answer: "I factored it as (n+1)(n+2). Since these are consecutive integers, one is even. For divisibility by 4, we need one factor divisible by 4. This happens when n=3,7,11,... or n=0,4,8,... So n ≡ 0 or 3 (mod 4)."
Grade: {"score": 3, "max_score": 3, "rationale": "Complete solution with correct factoring, case analysis, and answer."}

Example 2:
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the odd numbers be 2k+1 and 2m+1. Their sum is 2k+1+2m+1 = 2(k+m+1), which is even.
Grading Guidelines: Award 1 point for setting up odd number representation, 1 point for algebraic manipulation, 1 point for conclusion.
Student Answer: "Odd numbers end in 1,3,5,7,9. Adding two odd numbers gives an even number. For example, 3+5=8 which is even."
Grade: {"score": 1, "max_score": 3, "rationale": "Student only provided examples without general proof. Missing algebraic representation and general reasoning."}
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _extract_prediction(self, msg_history: list[dict], stage: str = "initial") -> str:
        """Extract prediction from message history with robust fallback mechanisms.
        
        Args:
            msg_history: The message history from LLM calls
            stage: Description of the extraction stage for logging
            
        Returns:
            Extracted prediction string or "None" if extraction fails
        """
        if not msg_history:
            self.log_fn(f"Warning: Empty message history in {stage} stage")
            return "None"
            
        try:
            last_msg = msg_history[-1].get("text", "")
            if not last_msg:
                self.log_fn(f"Warning: Empty last message in {stage} stage")
                return "None"
                
            extracted = _extract_jsons(last_msg)
            if extracted:
                result = extracted[-1]
                self.log_fn(f"[{stage}] Extracted JSON result: {result}")
                
                # Try response field first
                if "response" in result and result["response"]:
                    pred = str(result["response"])
                    self.log_fn(f"[{stage}] Using 'response' field: {pred}")
                    return pred
                    
                # Try final_response field (for reflection stage)
                if "final_response" in result and result["final_response"]:
                    pred = str(result["final_response"])
                    self.log_fn(f"[{stage}] Using 'final_response' field: {pred}")
                    return pred
                    
                # Try score/max_score fields
                if "score" in result and "max_score" in result:
                    try:
                        score = float(result["score"])
                        max_score = float(result["max_score"])
                        pred = f"{int(score)}/{int(max_score)}"
                        self.log_fn(f"[{stage}] Using score/max_score fields: {pred}")
                        return pred
                    except (ValueError, TypeError) as ve:
                        self.log_fn(f"[{stage}] Invalid numeric score: {ve}")
                        
                # Try revised_score/revised_max_score fields (for reflection stage)
                if "revised_score" in result and "revised_max_score" in result:
                    try:
                        score = float(result["revised_score"])
                        max_score = float(result["revised_max_score"])
                        pred = f"{int(score)}/{int(max_score)}"
                        self.log_fn(f"[{stage}] Using revised_score/revised_max_score fields: {pred}")
                        return pred
                    except (ValueError, TypeError) as ve:
                        self.log_fn(f"[{stage}] Invalid numeric revised_score: {ve}")
                        
                self.log_fn(f"[{stage}] Warning: JSON missing expected fields. Keys: {list(result.keys())}")
            else:
                self.log_fn(f"[{stage}] Warning: No JSON blocks found in response")
                
            # Fallback: Try to extract any numeric score pattern
            score_match = re.search(r'(\d+)\s*/\s*(\d+)', last_msg)
            if score_match:
                pred = f"{score_match.group(1)}/{score_match.group(2)}"
                self.log_fn(f"[{stage}] Fallback extraction: {pred}")
                return pred
                
        except Exception as e:
            self.log_fn(f"[{stage}] Error extracting prediction: {e}")
            import traceback
            self.log_fn(f"[{stage}] Traceback: {traceback.format_exc()}")
            
        return "None"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with reasoning and reflection.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        required_fields = ['problem', 'solution', 'grading_guidelines', 'student_answer']
        missing_fields = [f for f in required_fields if f not in inputs or not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Warning: Missing required fields: {missing_fields}")

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
    "score": <numerical score>,
    "max_score": <maximum possible score>,
    "rationale": "Detailed explanation of why this score was awarded",
    "response": "<score>/<max_score> - <brief summary>"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error in initial LLM call: {e}")
            return "None", []

        # Extract prediction from JSON with detailed logging
        prediction = self._extract_prediction(msg_history, stage="initial")

        # Step 2: Self-reflection to verify the grade
        if prediction != "None" and len(msg_history) >= 2:
            reflection_msg = f"""Review your grading above. Check for:
1. Did you award points the student didn't earn?
2. Did you miss any errors in the student's work?
3. Is your score consistent with the grading guidelines?
4. Would another grader agree with your assessment?

If you need to revise your grade, provide the corrected JSON. If your grade is correct, confirm it by returning the same score.

Respond in JSON format:
<json>
{{
    "reflection": "Your detailed self-review here - analyze each point above",
    "revised_score": <numerical score>,
    "revised_max_score": <maximum possible score>,
    "final_response": "<score>/<max_score> - <brief summary of decision>"
}}
</json>

Important: revised_score must be a number, not a string."""
            
            try:
                reflection_response, msg_history, _ = get_response_from_llm(
                    msg=reflection_msg,
                    model=self.model,
                    msg_history=msg_history,
                )
                
                # Extract revised prediction
                revised_prediction = self._extract_prediction(msg_history, stage="reflection")
                if revised_prediction != "None":
                    prediction = revised_prediction
            except Exception as e:
                self.log_fn(f"Error in reflection LLM call: {e}")
                # Keep original prediction on error

        return str(prediction), msg_history
