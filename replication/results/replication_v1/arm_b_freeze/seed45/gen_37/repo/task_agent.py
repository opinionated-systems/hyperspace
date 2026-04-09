"""
Task agent: solves a given task with chain-of-thought reasoning.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Version: 1.1.0 - Enhanced JSON parsing with text normalization
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

# Version constant
__version__ = "1.1.0"

# Compile regex patterns once for efficiency
_GRADE_PATTERN = re.compile(r'(?:grade|score|result|answer)[\s:]+([^\n]+)', re.IGNORECASE)
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to extract JSON from markdown code blocks as fallback.
    Includes enhanced error recovery for malformed JSON with multiple fallback strategies.
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
            
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
        else:
            parse_errors.append(f"Block at {start}: failed all parsing strategies")
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            parsed = _try_parse_json(block.strip())
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        # Look for content between outermost braces
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            parsed = _try_parse_json(text[brace_start:brace_end + 1])
            if parsed is not None:
                results.append(parsed)
    
    # Log parsing errors for debugging if no results found
    if not results and parse_errors:
        logger.debug(f"JSON parsing errors: {parse_errors}")
    
    return results or None


def _normalize_json_text(text: str) -> str:
    """Normalize text for JSON parsing by handling common LLM output issues.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Normalized text with common issues fixed
    """
    # Remove null bytes and control characters
    text = text.replace('\x00', '').replace('\x01', '').replace('\x02', '')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove BOM if present
    if text.startswith('\ufeff'):
        text = text[1:]
    
    return text


def _try_parse_json(text: str) -> dict | None:
    """Try multiple strategies to parse JSON text.
    
    Returns the parsed dict if successful, None otherwise.
    """
    # Normalize text first
    text = _normalize_json_text(text)
    
    # Strategy 1: Direct parsing
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
        cleaned = text.replace(",\n}", "\n}").replace(",\n]", "\n]")
        # Remove single-line comments
        cleaned = re.sub(r'//.*?\n', '\n', cleaned)
        # Remove multi-line comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Handle common LLM output issues (trailing commas, unquoted keys)
    try:
        cleaned = text
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        # Fix unquoted keys (simple heuristic)
        cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._log_file = log_file
    
    def reset(self) -> None:
        """Reset the agent state for a fresh evaluation run."""
        self._call_count = 0
        
    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "call_count": self._call_count,
            "model": self.model,
        }

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self._call_count += 1
        
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Validate required fields
        if not problem or not solution:
            self.log_fn(f"Call {self._call_count}: Missing required fields (problem or solution)")
            return "Error: Missing required fields", []

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

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Call {self._call_count}: LLM call failed: {e}")
            return f"Error: LLM call failed - {e}", []

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        confidence = "unknown"
        extraction_method = "unknown"
        
        try:
            if not msg_history or len(msg_history) < 2:
                self.log_fn(f"Call {self._call_count}: Empty message history")
                return "Error: Empty message history", msg_history
                
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                extraction_method = "json"
                
                # Try multiple possible keys for the response
                response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score", "verdict"]
                for key in response_keys:
                    if key in last_json:
                        prediction = last_json[key]
                        break
                
                # Log reasoning if available
                reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "rationale"]
                for key in reasoning_keys:
                    if key in last_json:
                        reasoning = last_json[key]
                        self.log_fn(f"Call {self._call_count}: {key.capitalize()}: {reasoning[:200]}...")
                        break
                
                # Extract confidence if available
                if "confidence" in last_json:
                    confidence = str(last_json["confidence"])
                    self.log_fn(f"Call {self._call_count}: Confidence: {confidence}")
            else:
                # Fallback: try to extract any meaningful text from the response
                extraction_method = "fallback"
                response_text = msg_history[-1]["text"]
                # Look for common patterns like "Grade: X" or "Answer: X"
                grade_match = _GRADE_PATTERN.search(response_text)
                if grade_match:
                    prediction = grade_match.group(1).strip()
                    self.log_fn(f"Call {self._call_count}: Extracted grade via pattern matching: {prediction}")
                else:
                    # Last resort: use first 100 chars of response
                    prediction = response_text[:100].strip()
                    self.log_fn(f"Call {self._call_count}: Using raw response (no JSON found): {prediction[:50]}...")
        except Exception as e:
            self.log_fn(f"Call {self._call_count}: Error extracting prediction: {e}")
            extraction_method = "error"

        # Log summary
        self.log_fn(f"Call {self._call_count}: Prediction='{prediction}', Method={extraction_method}")
        
        return str(prediction), msg_history
