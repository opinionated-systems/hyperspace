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
    Also handles nested JSON objects and common formatting issues.
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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try common fixes: remove trailing commas, fix quotes
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting just the outermost JSON object
                try:
                    brace_start = inner.find('{')
                    brace_end = inner.rfind('}')
                    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                        results.append(json.loads(inner[brace_start:brace_end+1]))
                except json.JSONDecodeError:
                    continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    
    # Try to find JSON objects in code blocks
    json_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_pattern:
        try:
            results.append(json.loads(json_pattern.group(1)))
        except json.JSONDecodeError:
            # Try with trailing comma fix
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', json_pattern.group(1))
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        results.append({"response": response_pattern.group(1)})
    
    # Try to find any JSON-like structure with single quotes
    if not results:
        single_quote_pattern = re.search(r"'response'\s*:\s*'([^']*)'", text)
        if single_quote_pattern:
            results.append({"response": single_quote_pattern.group(1)})
    
    # Last resort: look for grade keywords in the text
    if not results:
        text_lower = text.lower()
        if any(word in text_lower for word in ['correct', 'right', 'accurate', 'valid']):
            # Check if it's clearly marked as partial
            if any(word in text_lower for word in ['partial', 'partially', 'incomplete']):
                results.append({"response": "Partial"})
            else:
                results.append({"response": "Correct"})
        elif any(word in text_lower for word in ['incorrect', 'wrong', 'error', 'invalid']):
            results.append({"response": "Incorrect"})
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3
        self._call_count = 0

    def _validate_inputs(self, inputs: dict) -> tuple[bool, str]:
        """Validate that all required fields are present in inputs.
        
        Returns:
            (is_valid, error_message) tuple
        """
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing = [f for f in required_fields if f not in inputs or not inputs[f]]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
        return True, ""

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
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result. All steps are logically sound and the final answer is correct.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results. The student demonstrates understanding of key concepts but makes computational or logical mistakes.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results. The student fails to demonstrate understanding of the core concepts.

## Response Format (REQUIRED)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

IMPORTANT: 
- Ensure your JSON is valid and properly formatted with no trailing commas.
- The 'response' field should contain only the grade, not the reasoning.
- Be objective and consistent in your grading based on the provided guidelines.
- If the student uses an alternative valid approach that differs from the provided solution but arrives at the correct answer with sound reasoning, mark it as Correct."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            # Use the last valid JSON object
            last_json = extracted[-1]
            
            # Extract response/grade
            if "response" in last_json:
                prediction = str(last_json["response"]).strip()
            elif "grade" in last_json:
                prediction = str(last_json["grade"]).strip()
            elif "result" in last_json:
                prediction = str(last_json["result"]).strip()
            elif "answer" in last_json:
                prediction = str(last_json["answer"]).strip()
            
            # Extract reasoning
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"]).strip()
            elif "analysis" in last_json:
                reasoning = str(last_json["analysis"]).strip()
            elif "explanation" in last_json:
                reasoning = str(last_json["explanation"]).strip()
        
        # Normalize common grade variations
        if prediction != "None":
            pred_lower = prediction.lower()
            if pred_lower in ["correct", "right", "true", "valid", "accurate", "yes"]:
                prediction = "Correct"
            elif pred_lower in ["incorrect", "wrong", "false", "invalid", "inaccurate", "no", "error"]:
                prediction = "Incorrect"
            elif pred_lower in ["partial", "partially correct", "partial credit", "incomplete"]:
                prediction = "Partial"
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Validate inputs first
        is_valid, error_msg = self._validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
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
                    self.log_fn(f"[Call {self._call_count}] Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"[Call {self._call_count}] Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"[Call {self._call_count}] Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Common mistakes to avoid:
- Do not include trailing commas (e.g., {{"response": "Correct",}} is invalid)
- Do not include markdown formatting inside the JSON tags
- Ensure all quotes are straight double quotes (")
- The response field must be a string in quotes

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"[Call {self._call_count}] Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
