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
    Includes robust error recovery for malformed JSON.
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
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    # Remove trailing commas before closing braces/brackets
                    fixed = re.sub(r',(\s*[}\]])', r'\1', inner[json_start:json_end + 1] if json_start != -1 and json_end != -1 else inner)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try fixing common issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', block.strip())
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    continue
        
        # Try bare ``` ... ``` blocks that might contain JSON
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') or block.startswith('['):
                    try:
                        results.append(json.loads(block))
                    except json.JSONDecodeError:
                        try:
                            fixed = re.sub(r',(\s*[}\]])', r'\1', block)
                            results.append(json.loads(fixed))
                        except json.JSONDecodeError:
                            continue
        
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
            
            # Try to find valid JSON by matching braces
            if brace_positions:
                stack = []
                for pos, kind in brace_positions:
                    if kind == 'open':
                        stack.append(pos)
                    elif kind == 'close' and stack:
                        start_pos = stack.pop()
                        candidate = text[start_pos:pos+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            # Try fixing trailing commas
                            try:
                                fixed = re.sub(r',(\s*[}\]])', r'\1', candidate)
                                results.append(json.loads(fixed))
                            except json.JSONDecodeError:
                                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 2  # Number of retries for failed JSON extraction

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
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )

            # Extract prediction from JSON
            prediction, reasoning = self._extract_prediction(msg_history)
            
            if prediction != "None":
                break
            
            if attempt < self.max_retries:
                self.log_fn(f"Retry {attempt + 1}: No valid JSON found, retrying with stronger prompt...")
                # Add a reminder to the instruction for the retry
                instruction += "\n\nIMPORTANT: You MUST respond with valid JSON in the <json>...</json> format shown above."

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
            # Try the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_jsons(last_msg)
            
            # If not found, try searching all messages
            if not extracted:
                for msg in reversed(msg_history):
                    text = msg.get("text", "")
                    extracted = _extract_jsons(text)
                    if extracted:
                        break
            
            if extracted:
                result = extracted[-1]
                # Try multiple possible keys for the response
                for key in ["response", "grade", "answer", "result", "assessment", "evaluation", "score", "verdict"]:
                    if key in result:
                        prediction = result[key]
                        break
                
                # Extract reasoning if available
                for key in ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]:
                    if key in result:
                        reasoning = result[key]
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning
