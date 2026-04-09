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
    if not text or not isinstance(text, str):
        logger.debug("JSON extraction: empty or invalid input text")
        return None
        
    results = []
    search_from = 0
    parse_errors = []
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"JSON extraction: found <json> at {start} but no closing </json>")
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            logger.debug(f"JSON extraction: empty block at position {start}")
            continue
            
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
            logger.debug(f"JSON extraction: successfully parsed block at position {start}")
        else:
            parse_errors.append(f"Block at {start}: failed all parsing strategies")
            logger.debug(f"JSON extraction: failed to parse block at position {start}")
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        logger.debug(f"JSON extraction: trying markdown fallback, found {len(json_blocks)} code blocks")
        for i, block in enumerate(json_blocks):
            parsed = _try_parse_json(block.strip())
            if parsed is not None:
                results.append(parsed)
                logger.debug(f"JSON extraction: successfully parsed markdown block {i}")
    
    # Last resort: try to find any JSON-like structure in the text
    if not results:
        # Look for content between outermost braces
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            logger.debug(f"JSON extraction: trying brace extraction from {brace_start} to {brace_end}")
            parsed = _try_parse_json(text[brace_start:brace_end + 1])
            if parsed is not None:
                results.append(parsed)
                logger.debug("JSON extraction: successfully parsed brace content")
    
    # Log parsing errors for debugging if no results found
    if not results and parse_errors:
        logger.debug(f"JSON parsing errors: {parse_errors}")
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try multiple strategies to parse JSON text.
    
    Returns the parsed dict if successful, None otherwise.
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    if not text:
        return None
    
    # Strategy 1: Direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # If it's a list, return the last dict element if any
        if isinstance(result, list) and result:
            for item in reversed(result):
                if isinstance(item, dict):
                    return item
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from within the text if it's wrapped in other content
    try:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            result = json.loads(text[brace_start:brace_end + 1])
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Clean up common JSON-breaking patterns
    try:
        cleaned = text.replace(",\n}", "\n}").replace(",\n]", "\n]")
        # Remove single-line comments
        cleaned = re.sub(r'//.*?\n', '\n', cleaned)
        # Remove multi-line comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
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
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Handle nested JSON strings that might be double-encoded
    try:
        # Sometimes LLMs return JSON as a string inside a JSON response
        if text.startswith('"') and text.endswith('"'):
            # Try to unescape and parse
            unescaped = json.loads(text)
            if isinstance(unescaped, str):
                return _try_parse_json(unescaped)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning.
    
    Features:
    - Retry mechanism with temperature variation for robustness
    - Self-consistency: multiple grading calls with majority voting for uncertain cases
    - Enhanced JSON extraction with multiple fallback strategies
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._call_count = 0
        self._max_retries = 3
        self._consistency_calls = 3  # Number of calls for self-consistency

    def _normalize_grade(self, grade: str) -> str:
        """Normalize a grade string for comparison."""
        grade = grade.lower().strip()
        # Map common variations to canonical forms
        if any(x in grade for x in ["correct", "right", "true", "yes"]):
            if "partial" in grade or "partially" in grade:
                return "partially_correct"
            return "correct"
        elif any(x in grade for x in ["incorrect", "wrong", "false", "no", "error"]):
            return "incorrect"
        elif any(x in grade for x in ["partial", "partially", "half", "some"]):
            return "partially_correct"
        return grade

    def _aggregate_predictions(self, predictions: list[str], confidences: list[str]) -> tuple[str, str]:
        """Aggregate multiple predictions using weighted majority voting.
        
        Returns:
            (aggregated_prediction, aggregation_method)
        """
        if not predictions:
            return "Error: No predictions to aggregate", "error"
        
        if len(predictions) == 1:
            return predictions[0], "single"
        
        # Weight by confidence
        weights = []
        for conf in confidences:
            conf_lower = conf.lower()
            if "high" in conf_lower:
                weights.append(1.0)
            elif "medium" in conf_lower:
                weights.append(0.6)
            elif "low" in conf_lower:
                weights.append(0.3)
            else:
                weights.append(0.5)
        
        # Count weighted votes for each normalized grade
        from collections import defaultdict
        votes = defaultdict(float)
        for pred, weight in zip(predictions, weights):
            normalized = self._normalize_grade(pred)
            votes[normalized] += weight
        
        # Find the winner
        if votes:
            winner = max(votes.items(), key=lambda x: x[1])
            total_votes = sum(votes.values())
            confidence_ratio = winner[1] / total_votes if total_votes > 0 else 0
            
            # Map back to a representative original prediction
            for pred in predictions:
                if self._normalize_grade(pred) == winner[0]:
                    return pred, f"consensus_{confidence_ratio:.2f}"
        
        # Fallback: return first prediction
        return predictions[0], "fallback_first"

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

        # Self-consistency: make multiple calls and aggregate results
        all_predictions = []
        all_confidences = []
        all_msg_histories = []
        
        for consistency_attempt in range(self._consistency_calls):
            # Retry loop with temperature variation for robustness
            msg_history = []
            for retry_attempt in range(self._max_retries):
                try:
                    # Vary temperature: low for first attempt, higher for retries and consistency calls
                    temperature = 0.0 if (retry_attempt == 0 and consistency_attempt == 0) else 0.2 + (consistency_attempt * 0.15) + (retry_attempt * 0.1)
                    response, msg_history, info = get_response_from_llm(
                        msg=instruction,
                        model=self.model,
                        msg_history=[],
                    )
                    break
                except Exception as e:
                    self.log_fn(f"Call {self._call_count}.{consistency_attempt}: LLM call failed (retry {retry_attempt + 1}/{self._max_retries}): {e}")
                    if retry_attempt == self._max_retries - 1:
                        if consistency_attempt == 0:
                            return f"Error: LLM call failed after {self._max_retries} attempts - {e}", msg_history
                        else:
                            # Skip this consistency attempt but continue with others
                            continue
                    # Continue to next retry
            else:
                # All retries failed for this consistency attempt
                if consistency_attempt == 0:
                    return f"Error: LLM call failed after {self._max_retries} attempts", msg_history
                continue
            
            all_msg_histories.append(msg_history)
            
            # Extract prediction from JSON
            prediction = "None"
            reasoning = ""
            confidence = "unknown"
            extraction_method = "unknown"
            
            try:
                if not msg_history or len(msg_history) < 2:
                    self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Empty message history")
                    prediction = "Error: Empty message history"
                    confidence = "low"
                else:
                    # Get the last assistant response
                    last_message = msg_history[-1]
                    if not isinstance(last_message, dict) or "text" not in last_message:
                        self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Invalid message format in history")
                        prediction = "Error: Invalid message format"
                        confidence = "low"
                    else:
                        response_text = last_message["text"]
                        if not response_text or not response_text.strip():
                            self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Empty response text")
                            prediction = "Error: Empty LLM response"
                            confidence = "low"
                        else:
                            extracted = _extract_jsons(response_text)
                            if extracted:
                                last_json = extracted[-1]
                                extraction_method = "json"
                                
                                # Validate that extracted JSON is a dictionary
                                if not isinstance(last_json, dict):
                                    self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Extracted JSON is not a dictionary")
                                    extraction_method = "fallback"
                                    prediction = str(last_json)[:100]
                                    confidence = "low"
                                else:
                                    # Try multiple possible keys for the response
                                    response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score", "verdict"]
                                    for key in response_keys:
                                        if key in last_json and last_json[key] is not None:
                                            prediction = str(last_json[key]).strip()
                                            break
                                    
                                    # Log reasoning if available
                                    reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "rationale"]
                                    for key in reasoning_keys:
                                        if key in last_json and last_json[key] is not None:
                                            reasoning = str(last_json[key])
                                            self.log_fn(f"Call {self._call_count}.{consistency_attempt}: {key.capitalize()}: {reasoning[:200]}...")
                                    
                                    # Extract confidence if available
                                    if "confidence" in last_json and last_json["confidence"] is not None:
                                        confidence = str(last_json["confidence"])
                                        self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Confidence: {confidence}")
                            else:
                                # Fallback: try to extract any meaningful text from the response
                                extraction_method = "fallback"
                                # Look for common patterns like "Grade: X" or "Answer: X"
                                grade_match = _GRADE_PATTERN.search(response_text)
                                if grade_match:
                                    prediction = grade_match.group(1).strip()
                                    self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Extracted grade via pattern matching: {prediction}")
                                else:
                                    # Last resort: use first 100 chars of response
                                    prediction = response_text[:100].strip()
                                    self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Using raw response (no JSON found): {prediction[:50]}...")
                                confidence = "low"
            except Exception as e:
                self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Error extracting prediction: {e}")
                extraction_method = "error"
                prediction = "Error: Extraction failed"
                confidence = "low"
            
            all_predictions.append(prediction)
            all_confidences.append(confidence)
            self.log_fn(f"Call {self._call_count}.{consistency_attempt}: Prediction='{prediction}', Method={extraction_method}, Confidence={confidence}")
        
        # Aggregate predictions using self-consistency
        if len(all_predictions) == 1:
            final_prediction = all_predictions[0]
            aggregation_method = "single"
        else:
            final_prediction, aggregation_method = self._aggregate_predictions(all_predictions, all_confidences)
        
        # Log summary
        self.log_fn(f"Call {self._call_count}: Final Prediction='{final_prediction}', Method={aggregation_method}, Votes={len(all_predictions)}")
        if len(all_predictions) > 1:
            self.log_fn(f"Call {self._call_count}: All predictions: {all_predictions}")
        
        # Return the first message history (or the one with highest confidence if we tracked that)
        return str(final_prediction), all_msg_histories[0] if all_msg_histories else []
