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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also attempts to parse raw JSON objects if no <json> tags are found.
    Includes additional heuristics for malformed JSON.
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (carefully)
                fixed = re.sub(r"(?<!\\)'([^']*?)'(?<!\\)", r'"\1"', fixed)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Fallback: try to find JSON objects directly if no <json> tags
    if not results:
        try:
            # Look for JSON-like structures with braces
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                potential_json = text[brace_start:brace_end + 1]
                try:
                    results.append(json.loads(potential_json))
                except json.JSONDecodeError:
                    # Try fixing common issues
                    fixed = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                    fixed = re.sub(r"(?<!\\)'([^']*?)'(?<!\\)", r'"\1"', fixed)
                    results.append(json.loads(fixed))
        except (json.JSONDecodeError, ValueError):
            pass
    
    return results or None


def _format_inputs(inputs: dict) -> str:
    """Format task inputs into a structured prompt.
    
    Provides better structure and context for the LLM with clear section headers.
    """
    parts = []
    # Order matters for grading - present in logical sequence
    key_order = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    
    for key in key_order:
        if key in inputs:
            value = inputs[key]
            # Format the key name nicely
            display_key = key.replace("_", " ").title()
            separator = "=" * 60
            parts.append(f"{separator}\n{display_key}\n{separator}\n{value}\n")
    
    # Add any remaining keys not in the standard order
    for key, value in inputs.items():
        if key not in key_order:
            display_key = key.replace("_", " ").title()
            separator = "=" * 60
            parts.append(f"{separator}\n{display_key}\n{separator}\n{value}\n")
    
    return "\n".join(parts)


def _validate_grading_response(response: dict) -> tuple[bool, str]:
    """Validate that the grading response has the required structure.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    if "response" not in response:
        return False, "Missing 'response' key in JSON"
    
    response_value = response["response"]
    if not isinstance(response_value, (str, int, float, bool)):
        return False, f"'response' value has unsupported type: {type(response_value)}"
    
    # Additional validation: check for empty responses
    if isinstance(response_value, str) and not response_value.strip():
        return False, "'response' value is empty string"
    
    return True, ""


def _score_response_quality(response_text: str) -> dict:
    """Score the quality of a grading response.
    
    Returns a dictionary with quality metrics:
    - length_score: based on appropriate length (not too short, not too long)
    - structure_score: based on presence of structured elements
    - detail_score: based on specific grading indicators
    
    This helps identify high-quality responses for potential caching or
    use as examples for few-shot prompting.
    """
    scores = {
        "length_score": 0.0,
        "structure_score": 0.0,
        "detail_score": 0.0,
        "overall_score": 0.0,
    }
    
    # Length scoring: ideal range is 200-800 characters
    length = len(response_text)
    if 200 <= length <= 800:
        scores["length_score"] = 1.0
    elif 100 <= length < 200 or 800 < length <= 1200:
        scores["length_score"] = 0.7
    elif 50 <= length < 100 or 1200 < length <= 2000:
        scores["length_score"] = 0.4
    else:
        scores["length_score"] = 0.2
    
    # Structure scoring: check for grading-specific structure
    structure_indicators = [
        r"\b(correct|incorrect|partial|error|mistake)\b",
        r"\b(score|grade|point|mark)\b",
        r"\b(step|process|reasoning|logic)\b",
        r"\b(answer|solution|result)\b",
        r"\n\s*[-•*]\s+",  # Bullet points
        r"\n\s*\d+\.\s+",  # Numbered lists
    ]
    
    structure_matches = sum(1 for pattern in structure_indicators 
                           if re.search(pattern, response_text, re.IGNORECASE))
    scores["structure_score"] = min(1.0, structure_matches / 4)
    
    # Detail scoring: check for specific grading elements
    detail_indicators = [
        r"\b(explanation|because|since|therefore|thus)\b",
        r"\b(compare|contrast|versus|against)\b",
        r"\b(identify|found|observed|noticed)\b",
        r"\b(suggest|recommend|improve|fix)\b",
        r"\d+\s*/\s*\d+",  # Score fractions like "3/5"
        r"\b\d+%\b",  # Percentages
    ]
    
    detail_matches = sum(1 for pattern in detail_indicators 
                        if re.search(pattern, response_text, re.IGNORECASE))
    scores["detail_score"] = min(1.0, detail_matches / 3)
    
    # Overall score: weighted average
    scores["overall_score"] = (
        scores["length_score"] * 0.3 +
        scores["structure_score"] * 0.4 +
        scores["detail_score"] * 0.3
    )
    
    return scores


def _extract_response_heuristic(text: str) -> str | None:
    """Extract a response using heuristics when JSON parsing fails.
    
    Looks for patterns like "Answer: X", "Response: X", "The answer is X", etc.
    """
    # Common patterns for answer extraction
    patterns = [
        r'(?:answer|response|result|evaluation)[:\s]+(.+?)(?:\n|$)',
        r'(?:the answer is|the response is|the result is)[:\s]+(.+?)(?:\n|$)',
        r'(?:conclusion|verdict)[:\s]+(.+?)(?:\n|$)',
        r'["\']([^"\']+)["\']\s*(?:is the answer|is correct|is incorrect)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with improved error handling."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.heuristic_extraction_count = 0
        self.response_quality_scores: list[float] = []  # Track quality of successful responses

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        self.log_fn(f"TaskAgent call #{self.call_count} starting")
        
        # Format inputs more clearly
        formatted_inputs = _format_inputs(inputs)
        
        instruction = f"""You are an expert grading agent specializing in mathematical problem evaluation. Your task is to evaluate student answers based on the provided problem, solution, and grading guidelines.

{formatted_inputs}

Your evaluation must be provided in the following JSON format:
<json>
{{
    "response": "Your detailed evaluation result here"
}}
</json>

Important Instructions:
1. First, carefully read and understand the problem statement
2. Study the provided solution to understand the correct approach and expected answer
3. Review the grading guidelines to understand the criteria for evaluation
4. Compare the student's answer against the solution:
   - Check if the final answer matches
   - Evaluate the reasoning process if visible
   - Identify any errors or misconceptions
   - Note any partial credit considerations
5. Provide a clear, detailed evaluation in the "response" field
6. Ensure your JSON is valid - use double quotes, no trailing commas

Think step by step before providing your final evaluation."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            self.log_fn(f"LLM call completed, response length: {len(response)}")
        except Exception as e:
            self.log_fn(f"Error in LLM call: {e}")
            self.error_count += 1
            return "Error: LLM call failed", []

        # Extract prediction from JSON with better error handling
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                # Try JSON extraction first
                extracted = _extract_jsons(text_content)
                if extracted:
                    last_extracted = extracted[-1]
                    is_valid, error_msg = _validate_grading_response(last_extracted)
                    if is_valid and isinstance(last_extracted, dict) and "response" in last_extracted:
                        prediction = last_extracted["response"]
                        extraction_method = "json"
                        self.success_count += 1
                        self.log_fn(f"Successfully extracted prediction via JSON: {str(prediction)[:100]}")
                    else:
                        self.log_fn(f"Invalid grading response from JSON: {error_msg}")
                        # Try heuristic extraction as fallback
                        heuristic = _extract_response_heuristic(text_content)
                        if heuristic:
                            prediction = heuristic
                            extraction_method = "heuristic"
                            self.heuristic_extraction_count += 1
                            self.success_count += 1
                            self.log_fn(f"Extracted prediction via heuristic: {str(prediction)[:100]}")
                        else:
                            self.error_count += 1
                            self.log_fn("No valid response found via JSON or heuristics")
                else:
                    # Try heuristic extraction when no JSON found
                    heuristic = _extract_response_heuristic(text_content)
                    if heuristic:
                        prediction = heuristic
                        extraction_method = "heuristic"
                        self.heuristic_extraction_count += 1
                        self.success_count += 1
                        self.log_fn(f"Extracted prediction via heuristic (no JSON): {str(prediction)[:100]}")
                    else:
                        self.error_count += 1
                        self.log_fn("No JSON found in response and heuristic extraction failed")
            else:
                self.error_count += 1
                self.log_fn("Empty message history")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        # Track response quality for successful extractions
        if extraction_method in ("json", "heuristic") and prediction != "None":
            quality_scores = _score_response_quality(str(prediction))
            self.response_quality_scores.append(quality_scores["overall_score"])
            self.log_fn(f"Response quality score: {quality_scores['overall_score']:.2f} "
                       f"(length: {quality_scores['length_score']:.2f}, "
                       f"structure: {quality_scores['structure_score']:.2f}, "
                       f"detail: {quality_scores['detail_score']:.2f})")

        self.log_fn(f"TaskAgent stats: {self.success_count} successes, {self.error_count} errors, {self.heuristic_extraction_count} heuristic extractions out of {self.call_count} calls (method: {extraction_method})")
        return str(prediction), msg_history

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        stats = {
            "total_calls": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "heuristic_extraction_count": self.heuristic_extraction_count,
            "success_rate": self.success_count / max(1, self.call_count),
        }
        
        # Add quality metrics if available
        if self.response_quality_scores:
            stats["avg_quality_score"] = sum(self.response_quality_scores) / len(self.response_quality_scores)
            stats["high_quality_responses"] = sum(1 for s in self.response_quality_scores if s >= 0.7)
            stats["quality_score_distribution"] = {
                "excellent (>=0.8)": sum(1 for s in self.response_quality_scores if s >= 0.8),
                "good (0.6-0.8)": sum(1 for s in self.response_quality_scores if 0.6 <= s < 0.8),
                "fair (0.4-0.6)": sum(1 for s in self.response_quality_scores if 0.4 <= s < 0.6),
                "poor (<0.4)": sum(1 for s in self.response_quality_scores if s < 0.4),
            }
        
        return stats
