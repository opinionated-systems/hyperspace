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

# Compile regex patterns once for efficiency
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)

# Additional patterns for robust extraction
_REASONING_PATTERN = re.compile(r'(?:reasoning|analysis|explanation|thought)[\s:]+([^\n]+)', re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r'(?:confidence|certainty|sure)[\s:]+(\d+)', re.IGNORECASE)

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes enhanced error recovery for malformed JSON.
    """
    results = []
    search_from = 0
    parse_errors = []
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            continue
            
        # Try to parse the JSON
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
        else:
            parse_errors.append(f"Block at {start}: failed to parse")
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            parsed = _try_parse_json(block.strip())
            if parsed is not None:
                results.append(parsed)
    
    # Log parsing errors for debugging if no results found
    if not results and parse_errors:
        logger.debug(f"JSON parsing errors: {parse_errors}")
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try multiple strategies to parse JSON from text.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from within the text if it's wrapped in other content
    try:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return json.loads(text[brace_start:brace_end + 1])
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Clean up common JSON-breaking patterns
    try:
        cleaned = _clean_json_text(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def _clean_json_text(text: str) -> str:
    """Clean up common JSON-breaking patterns."""
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', text)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    # Remove single-line comments
    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
    # Remove multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    # Fix common quote issues
    cleaned = re.sub(r'"""', '"', cleaned)
    cleaned = re.sub(r"''", "'", cleaned)
    return cleaned


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    # Class-level cache for similar problems
    _cache: dict[str, tuple[str, list[dict]]] = {}
    _cache_hits = 0
    _cache_misses = 0

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._max_retries = 3

    def _generate_cache_key(self, inputs: dict) -> str:
        """Generate a cache key from inputs."""
        # Use problem + student_answer as cache key
        key_parts = [
            inputs.get("problem", "")[:100],
            inputs.get("student_answer", "")[:100]
        ]
        return hash(tuple(key_parts))

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Check cache first
        cache_key = self._generate_cache_key(inputs)
        if cache_key in self._cache:
            TaskAgent._cache_hits += 1
            self.log_fn(f"Cache hit! (hits: {TaskAgent._cache_hits}, misses: {TaskAgent._cache_misses})")
            return self._cache[cache_key]
        
        TaskAgent._cache_misses += 1

        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Build the instruction with improved prompting
        instruction = self._build_instruction(
            domain, problem, solution, grading_guidelines, student_answer
        )

        # Try multiple times with different strategies if needed
        for attempt in range(self._max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )

                # Extract prediction from JSON
                prediction, reasoning, confidence = self._extract_prediction(msg_history)
                
                # If we got a valid prediction, cache and return
                if prediction != "None" and prediction:
                    result = (str(prediction), msg_history)
                    self._cache[cache_key] = result
                    return result
                
                # If no valid prediction and we have retries left, try again
                if attempt < self._max_retries - 1:
                    self.log_fn(f"No valid prediction on attempt {attempt + 1}, retrying...")
                    # Add a hint to the instruction for the retry
                    instruction += "\n\nIMPORTANT: Your previous response did not contain valid JSON. Please ensure your response is valid JSON wrapped in <json> tags."
                    
            except Exception as e:
                self.log_fn(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self._max_retries - 1:
                    # Last attempt failed, return error
                    return f"Error: {e}", []

        # All retries exhausted, return best effort
        prediction, reasoning, confidence = self._extract_prediction(msg_history)
        result = (str(prediction) if prediction else "None", msg_history)
        self._cache[cache_key] = result
        return result

    def _build_instruction(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str
    ) -> str:
        """Build the grading instruction prompt."""
        return f"""You are an expert {domain} grader evaluating student solutions.

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
5. Respond ONLY in JSON format wrapped in <json>...</json> tags with the following exact schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)",
    "confidence": "high|medium|low"
}}
</json>

## Example Response:
<json>
{{
    "reasoning": "The student correctly identified the key theorem but made an error in the algebraic manipulation at step 3. The final answer is incorrect due to this calculation error.",
    "response": "Partially Correct",
    "confidence": "high"
}}
</json>

Think carefully and provide a fair assessment based on the official solution and grading guidelines. Your response MUST be valid JSON inside <json> tags."""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str, str]:
        """Extract prediction, reasoning, and confidence from message history.
        
        Returns:
            (prediction, reasoning, confidence)
        """
        prediction = "None"
        reasoning = ""
        confidence = "unknown"
        
        if not msg_history:
            return prediction, reasoning, confidence
        
        try:
            response_text = msg_history[-1].get("text", "")
            extracted = _extract_jsons(response_text)
            
            if extracted:
                last_json = extracted[-1]
                
                # Extract prediction - try multiple possible keys
                prediction_keys = ["response", "grade", "answer", "result", 
                                  "assessment", "evaluation", "score", "verdict"]
                for key in prediction_keys:
                    if key in last_json:
                        prediction = str(last_json[key])
                        break
                
                # Extract reasoning - try multiple possible keys
                reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking"]
                for key in reasoning_keys:
                    if key in last_json:
                        reasoning = str(last_json[key])
                        self.log_fn(f"{key.capitalize()}: {reasoning[:200]}...")
                        break
                
                # Extract confidence
                if "confidence" in last_json:
                    confidence = str(last_json["confidence"])
                    self.log_fn(f"Confidence: {confidence}")
            else:
                # Fallback: try to extract any meaningful text from the response
                grade_match = _GRADE_PATTERN.search(response_text)
                if grade_match:
                    prediction = grade_match.group(1).strip()
                    self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                else:
                    # Last resort: use first 100 chars of response
                    prediction = response_text[:100].strip()
                    self.log_fn(f"Using raw response (no JSON found): {prediction[:50]}...")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning, confidence

    @classmethod
    def get_cache_stats(cls) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": cls._cache_hits,
            "misses": cls._cache_misses,
            "size": len(cls._cache)
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cache."""
        cls._cache.clear()
        cls._cache_hits = 0
        cls._cache_misses = 0
