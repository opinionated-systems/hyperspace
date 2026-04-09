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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks and bare JSON objects.
    Includes enhanced error recovery for malformed JSON.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from within the content
        try:
            # Find first { and last }
            json_start = inner.find("{")
            json_end = inner.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end + 1]))
                continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON errors and re-parse
        try:
            fixed = _fix_common_json_errors(inner)
            if fixed != inner:
                results.append(json.loads(fixed))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try fixing common errors
                try:
                    fixed = _fix_common_json_errors(block.strip())
                    results.append(json.loads(fixed))
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Try bare ``` ... ``` blocks that might contain JSON
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{'):
                    try:
                        results.append(json.loads(block))
                    except json.JSONDecodeError:
                        try:
                            fixed = _fix_common_json_errors(block)
                            results.append(json.loads(fixed))
                        except (json.JSONDecodeError, ValueError):
                            continue
        
        # Try bare JSON objects as fallback - improved pattern for nested braces
        if not results:
            # Find JSON-like structures with balanced braces
            potential_jsons = re.findall(r'\{[\s\S]*?"[^"]+"[\s\S]*?\}', text)
            for pj in potential_jsons:
                try:
                    # Validate braces are balanced
                    if pj.count('{') == pj.count('}'):
                        results.append(json.loads(pj))
                except json.JSONDecodeError:
                    try:
                        fixed = _fix_common_json_errors(pj)
                        if fixed != pj:
                            results.append(json.loads(fixed))
                    except (json.JSONDecodeError, ValueError):
                        continue
    
    return results or None


def _fix_common_json_errors(text: str) -> str:
    """Fix common JSON formatting errors that LLMs make.
    
    Args:
        text: Potentially malformed JSON string
        
    Returns:
        Fixed JSON string
    """
    text = text.strip()
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix single quotes to double quotes (but be careful with apostrophes in text)
    # Only replace single quotes that appear to be used as JSON delimiters
    text = re.sub(r"(?<=[\{\[,\s])'([^']*?)'(?=\s*[\}\],:])", r'"\1"', text)
    
    # Fix unquoted keys (common LLM error: {key: "value"} -> {"key": "value"})
    text = re.sub(r'([\{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Fix escaped newlines that might be double-escaped
    text = text.replace('\\n', '\n').replace('\\t', '\t')
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    return text


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate that required inputs are present and non-empty.
    
    Args:
        inputs: Dictionary containing problem inputs
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["problem", "solution", "grading_guidelines", "student_answer"]
    
    for field in required_fields:
        if field not in inputs:
            return False, f"Missing required field: {field}"
        value = inputs.get(field, "")
        if not value or not str(value).strip():
            return False, f"Empty required field: {field}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Validate inputs first
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            self.log_fn(f"Input validation failed: {error_msg}")
            return f"Error: {error_msg}", []
        
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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        extraction_errors = []
        
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
                for key in ["response", "grade", "answer", "result", "assessment", "evaluation"]:
                    if key in result:
                        prediction = result[key]
                        break
                else:
                    # No recognized key found - log available keys
                    available_keys = list(result.keys())
                    extraction_errors.append(f"No recognized response key found. Available keys: {available_keys}")
                
                # Extract reasoning if available
                for key in ["reasoning", "analysis", "thought", "explanation", "rationale"]:
                    if key in result:
                        reasoning = result[key]
                        break
                
                # Log reasoning if available
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                extraction_errors.append("No JSON found in LLM response")
                # Try to extract any text that might be the answer
                if msg_history:
                    last_text = msg_history[-1].get("text", "")
                    # Look for common patterns like "Grade: X" or "Answer: X"
                    grade_patterns = [
                        r'[Gg]rade[:\s]+([\w\s-]+)',
                        r'[Aa]nswer[:\s]+([\w\s-]+)',
                        r'[Rr]esult[:\s]+([\w\s-]+)',
                        r'[Ss]core[:\s]+([\w\s-]+)',
                    ]
                    for pattern in grade_patterns:
                        match = re.search(pattern, last_text)
                        if match:
                            prediction = match.group(1).strip()
                            self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                            break
        except Exception as e:
            extraction_errors.append(f"Error extracting prediction: {e}")
            self.log_fn(f"Error extracting prediction: {e}")
        
        # Log extraction errors if any
        if extraction_errors and prediction == "None":
            self.log_fn(f"Extraction issues: {'; '.join(extraction_errors)}")

        return str(prediction), msg_history
