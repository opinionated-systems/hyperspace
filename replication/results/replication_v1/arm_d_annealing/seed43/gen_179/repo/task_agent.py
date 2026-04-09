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
            continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    """
    results = []
    
    # Try to find JSON objects in code blocks with proper brace balancing
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # Try to find balanced JSON objects within the code block
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_obj = json.loads(content[start_idx:i+1])
                        if isinstance(json_obj, dict):
                            results.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
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
                    try:
                        json_obj = json.loads(text[start_idx:i+1])
                        if isinstance(json_obj, dict):
                            results.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
    
    # Final fallback: try to find any response-like pattern
    if not results:
        # Look for "response": "value" pattern with flexible whitespace
        response_match = re.search(r'["\']response["\']\s*:\s*["\']([^"\']+)["\']', text)
        if response_match:
            results.append({"response": response_match.group(1)})
        # Look for reasoning pattern too
        reasoning_match = re.search(r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']', text, re.DOTALL)
        if reasoning_match:
            if results:
                results[0]["reasoning"] = reasoning_match.group(1)
            else:
                results.append({"reasoning": reasoning_match.group(1)})
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: 
- Ensure your JSON is valid and properly formatted with double quotes around keys and string values.
- The 'response' field should contain only the grade/assessment, not the reasoning.
- Do not include markdown formatting (like ```json) inside the <json> tags."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses a multi-layer extraction strategy:
        1. Primary: Extract JSON from <json>...</json> tags
        2. Fallback: Extract JSON from code blocks or raw text with brace balancing
        3. Final fallback: Extract key-value patterns directly from text
        
        Returns:
            (prediction, reasoning) tuple. prediction="None" if extraction fails.
        """
        prediction = "None"
        reasoning = ""
        
        # Layer 1: Try primary extraction method (exact <json> tags)
        extracted = _extract_jsons(text)
        
        # Layer 2: Fallback to regex extraction for malformed responses
        if extracted is None:
            extracted = _extract_json_with_regex(text)
        
        # Layer 3: Extract from the last valid JSON object found
        if extracted:
            last_json = extracted[-1]
            
            # Extract response field (handle various types)
            if "response" in last_json:
                response_val = last_json["response"]
                # Handle different response types gracefully
                if isinstance(response_val, (str, int, float, bool)):
                    prediction = str(response_val).strip()
                elif isinstance(response_val, list) and response_val:
                    prediction = str(response_val[0]).strip()
                elif response_val is None:
                    prediction = "None"
                else:
                    prediction = str(response_val).strip()
            
            # Extract reasoning field (handle various types)
            if "reasoning" in last_json:
                reasoning_val = last_json["reasoning"]
                if isinstance(reasoning_val, str):
                    reasoning = reasoning_val.strip()
                elif isinstance(reasoning_val, (int, float, bool)):
                    reasoning = str(reasoning_val)
                elif isinstance(reasoning_val, list):
                    reasoning = " ".join(str(r) for r in reasoning_val)
                else:
                    reasoning = str(reasoning_val)
        
        # Validate prediction isn't empty or whitespace-only
        if not prediction or prediction.strip() == "":
            prediction = "None"
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    # Store reasoning in msg_history for potential downstream use
                    msg_history.append({"role": "system", "text": f"[Extraction] Reasoning: {reasoning[:500]}"})
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

Common mistakes to avoid:
1. Do NOT use markdown code blocks (```json) inside <json> tags
2. Do NOT include any text before <json> or after </json>
3. Ensure all string values use double quotes, not single quotes
4. The JSON must be a single object with "reasoning" and "response" keys

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
