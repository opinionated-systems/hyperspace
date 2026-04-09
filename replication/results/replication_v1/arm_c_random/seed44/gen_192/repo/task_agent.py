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
    Also handles markdown code blocks with json tag.
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
            continue
    
    # If no results, try markdown code blocks ```json ... ```
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Also try without 'json' specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = "```"
            else:
                end_marker = "```"
            
            # Find the closing ```
            end = text.find(end_marker, start + len("```json") if "json" in text[start:start+7] else start + 3)
            if end == -1:
                break
            
            # Extract content between markers
            if "json" in text[start:start+7]:
                inner = text[start + 7:end].strip()
            else:
                inner = text[start + 3:end].strip()
            
            search_from = end + 3
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                continue
    
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, missing quotes,
    and unescaped special characters.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Retry with progressively more aggressive fixes
    fixes = [
        # Level 1: Basic fixes
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),  # Remove trailing commas
        # Level 2: Quote fixes
        lambda t: re.sub(r"(?<!\\)'", '"', t),  # Fix single quotes
        # Level 3: Remove comments and normalize whitespace
        lambda t: re.sub(r'//.*?\n', '\n', re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL)),
    ]
    
    for attempt in range(min(max_retries, len(fixes))):
        try:
            fixed_text = text
            
            # Apply fixes up to current level
            for i in range(attempt + 1):
                fixed_text = fixes[i](fixed_text)
            
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix attempt {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix attempt {attempt + 1} failed: {e}")
            continue
    
    # Final attempt: Try to extract any JSON-like structure
    try:
        # Look for content between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            # Apply all fixes
            for fix in fixes:
                potential_json = fix(potential_json)
            parsed = json.loads(potential_json)
            logger.debug("JSON extraction succeeded with brute force parsing")
            return [parsed]
    except Exception as e:
        logger.debug(f"Brute force parsing failed: {e}")
    
    # Log the problematic text for debugging (truncated)
    preview = text[:500] + "..." if len(text) > 500 else text
    logger.warning(f"Failed to extract JSON from text: {preview}")
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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
        # Validate required input fields
        required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
        missing_fields = [f for f in required_fields if f not in inputs]
        if missing_fields:
            logger.warning(f"Missing input fields: {missing_fields}")
        
        # Extract fields for better prompt construction
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert grading agent specializing in {domain}. Your task is to evaluate a student's answer to a problem with careful reasoning.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task:
1. Analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your evaluation with clear reasoning

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your step-by-step analysis and comparison",
    "evaluation": "Your final evaluation/grade based on the guidelines",
    "response": "Your complete evaluation result (this will be the final output)"
}}
</json>

Important: 
- Ensure your response is valid JSON with double quotes around keys and string values
- The "response" field should contain your complete evaluation
- Be thorough in your reasoning before providing the final evaluation
- Do not use markdown formatting inside the JSON values
- Escape any double quotes within string values with backslash
- The JSON must be parseable by a standard JSON parser"""

        self.log_fn(f"Processing task with model: {self.model}")
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            
            # Log token usage if available
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                          f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                          f"Total: {usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", []

        # Extract prediction from JSON with retry mechanism
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                # Search through message history from the end to find the first valid JSON
                for msg in reversed(msg_history):
                    text_content = msg.get("text", "")
                    if not text_content:
                        continue
                    
                    extracted = _extract_json_with_retry(text_content)
                    if extracted:
                        last_json = extracted[-1]
                        if "response" in last_json:
                            prediction = last_json["response"]
                            self.log_fn(f"Successfully extracted prediction: {str(prediction)[:100]}...")
                            break
                        elif "evaluation" in last_json:
                            # Fallback to evaluation field if response not present
                            prediction = last_json["evaluation"]
                            self.log_fn(f"Using 'evaluation' field: {str(prediction)[:100]}...")
                            break
                        elif "reasoning" in last_json:
                            # If only reasoning is present, use the whole JSON as string
                            prediction = json.dumps(last_json)
                            self.log_fn(f"Using full JSON (reasoning only): {prediction[:100]}...")
                            break
                        else:
                            logger.warning(f"JSON missing expected keys. Keys found: {list(last_json.keys())}")
                            prediction = str(last_json)
                            break
                else:
                    # No valid JSON found in any message
                    logger.warning("No valid JSON found in response")
                    # Fallback: return raw text from last message if no JSON found
                    last_text = msg_history[-1].get("text", "")
                    prediction = last_text[:1000] if last_text else "None"
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Full traceback:")

        return str(prediction), msg_history
