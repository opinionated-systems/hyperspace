"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Includes robust error recovery for malformed JSON with multiple fix strategies.
    """
    results = []
    search_from = 0
    
    def _try_fix_json(json_str: str) -> dict | None:
        """Try multiple strategies to fix and parse malformed JSON."""
        # Strategy 1: Direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from within the content (find first { and last })
        json_start = json_str.find("{")
        json_end = json_str.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                return json.loads(json_str[json_start:json_end + 1])
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Remove trailing commas before closing braces/brackets
        try:
            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Fix single quotes to double quotes
        try:
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Fix unescaped newlines in strings
        try:
            fixed = re.sub(r'(?<!\\)\n', '\\n', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Extract only the outermost JSON object
        if json_start != -1 and json_end != -1:
            try:
                outermost = json_str[json_start:json_end + 1]
                # Apply all fixes to the outermost object
                fixed = re.sub(r',(\s*[}\]])', r'\1', outermost)
                fixed = fixed.replace("'", '"')
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        
        return None
    
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
        
        parsed = _try_fix_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            parsed = _try_fix_json(block.strip())
            if parsed:
                results.append(parsed)
        
        # Try bare ``` ... ``` blocks that might contain JSON
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') or block.startswith('['):
                    parsed = _try_fix_json(block)
                    if parsed:
                        results.append(parsed)
        
        # Try bare JSON objects as fallback
        if not results:
            # Find JSON-like structures with nested support
            # Use a more robust approach: find all { and } positions
            brace_positions = []
            for i, char in enumerate(text):
                if char == '{':
                    brace_positions.append((i, 'open'))
                elif char == '}':
                    brace_positions.append((i, 'close'))
            
            # Try to find valid JSON by matching braces (longest first)
            if brace_positions:
                # Sort by length descending to try largest objects first
                candidates = []
                stack = []
                for pos, kind in brace_positions:
                    if kind == 'open':
                        stack.append(pos)
                    elif kind == 'close' and stack:
                        start_pos = stack.pop()
                        candidates.append((start_pos, pos))
                
                # Sort by length descending
                candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
                
                for start_pos, end_pos in candidates:
                    candidate = text[start_pos:end_pos+1]
                    parsed = _try_fix_json(candidate)
                    if parsed and isinstance(parsed, dict):
                        results.append(parsed)
                        break  # Only take the first (largest) valid JSON
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased retries for better reliability
        self.system_prompt = """You are an expert grader evaluating student solutions to academic problems.

Your responsibilities:
1. Carefully analyze the student's answer against the official solution
2. Follow the grading guidelines precisely
3. Provide detailed reasoning for your assessment
4. Give a clear, definitive grade

Always respond in the exact JSON format requested."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert {domain} grader evaluating student solutions.

Your task is to grade a student's answer to a problem by comparing it against the official solution and following the grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Instructions:
1. First, analyze the student's answer step by step. Identify what they did correctly and incorrectly.
2. Compare their approach to the official solution.
3. Consider the grading guidelines carefully.
4. Provide your reasoning before giving the final grade.
5. Respond in JSON format with the following schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Think carefully and provide a fair assessment based on the official solution and grading guidelines."""

        # Try with retries for better reliability
        prediction = "None"
        reasoning = ""
        msg_history = []
        
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
                system_msg=self.system_prompt if attempt == 0 else None,
            )

            # Extract prediction from JSON
            prediction, reasoning = self._extract_prediction(msg_history)
            
            if prediction != "None":
                break
            
            if attempt < self.max_retries:
                self.log_fn(f"Retry {attempt + 1}: No valid JSON found, retrying with stronger prompt...")
                # Add a reminder to the instruction for the retry
                instruction += f"\n\nIMPORTANT (Attempt {attempt + 2}): You MUST respond with valid JSON in the <json>...</json> format shown above. Do not include any text outside the JSON tags."

        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")

        return str(prediction), msg_history

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Try the last assistant message first
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            # If not found, try searching all messages from most recent to oldest
            if not extracted:
                for msg in reversed(msg_history):
                    text = msg.get("text", "")
                    if text:
                        extracted = _extract_jsons(text)
                        if extracted:
                            break
            
            if extracted:
                # Use the last extracted JSON (most likely to be the final answer)
                result = extracted[-1]
                
                # Try multiple possible keys for the response/grade
                response_keys = ["response", "grade", "answer", "result", "assessment", 
                                "evaluation", "score", "verdict", "conclusion", "decision"]
                for key in response_keys:
                    if key in result:
                        value = result[key]
                        # Handle different value types
                        if isinstance(value, (str, int, float, bool)):
                            prediction = str(value)
                        elif isinstance(value, list) and len(value) > 0:
                            prediction = str(value[0])
                        break
                
                # Extract reasoning if available
                reasoning_keys = ["reasoning", "analysis", "thought", "explanation", 
                                 "rationale", "thinking", "justification", "notes"]
                for key in reasoning_keys:
                    if key in result:
                        value = result[key]
                        if isinstance(value, str):
                            reasoning = value
                        elif isinstance(value, list):
                            reasoning = " ".join(str(v) for v in value)
                        break
                
                # If no explicit reasoning key, try to construct from other fields
                if not reasoning and isinstance(result, dict):
                    other_fields = {k: v for k, v in result.items() 
                                   if k not in response_keys and k not in reasoning_keys}
                    if other_fields:
                        reasoning_parts = [f"{k}: {v}" for k, v in other_fields.items() 
                                          if isinstance(v, (str, int, float, bool))]
                        if reasoning_parts:
                            reasoning = "; ".join(reasoning_parts)
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning
