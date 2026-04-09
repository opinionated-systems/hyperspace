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
            
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError as e:
            parse_errors.append(f"Block at {start}: {str(e)[:50]}")
            # Try to extract JSON from within the text if it's wrapped in other content
            try:
                # Look for JSON-like content with braces
                brace_start = inner.find("{")
                brace_end = inner.rfind("}")
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    results.append(json.loads(inner[brace_start:brace_end + 1]))
            except json.JSONDecodeError:
                # Try one more fallback: look for JSON with trailing commas or comments
                try:
                    # Remove common JSON-breaking patterns
                    cleaned = inner.replace(",\n}", "\n}").replace(",\n]", "\n]")
                    # Remove single-line comments
                    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    # Final fallback: try to fix common LLM JSON errors
                    try:
                        # Fix unquoted keys (simple heuristic)
                        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
                        # Fix single quotes to double quotes
                        fixed = fixed.replace("'", '"')
                        # Fix trailing commas before closing braces/brackets
                        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        continue
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        json_blocks = _JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                # Try the same cleanup for markdown blocks
                try:
                    cleaned = block.strip().replace(",\n}", "\n}").replace(",\n]", "\n]")
                    cleaned = re.sub(r'//.*?\n', '\n', cleaned)
                    results.append(json.loads(cleaned))
                except json.JSONDecodeError:
                    # Final fallback for markdown blocks too
                    try:
                        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
                        fixed = fixed.replace("'", '"')
                        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        continue
    
    # Log parsing errors for debugging if no results found
    if not results and parse_errors:
        logger.debug(f"JSON parsing errors: {parse_errors}")
    
    return results or None


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
5. Respond ONLY in JSON format wrapped in <json>...</json> tags with the following exact schema:

<json>
{{
    "reasoning": "Your detailed analysis and reasoning about the student's answer...",
    "response": "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

## Example Response:
<json>
{{
    "reasoning": "The student correctly identified the key theorem but made an error in the algebraic manipulation at step 3. The final answer is incorrect due to this calculation error.",
    "response": "Partially Correct"
}}
</json>

Think carefully and provide a fair assessment based on the official solution and grading guidelines. Your response MUST be valid JSON inside <json> tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = "None"
        reasoning = ""
        confidence = "unknown"
        try:
            # Get the last assistant message
            last_msg = None
            for msg in reversed(msg_history):
                if msg.get("role") == "assistant" or msg.get("role") is None:
                    last_msg = msg
                    break
            
            if last_msg is None:
                raise ValueError("No assistant message found in history")
            
            response_text = last_msg.get("text", "") or last_msg.get("content", "")
            
            extracted = _extract_jsons(response_text)
            if extracted:
                last_json = extracted[-1]
                # Try multiple possible keys for the response (prioritized by likelihood)
                response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score", "verdict", "output", "prediction"]
                for key in response_keys:
                    if key in last_json and last_json[key] is not None:
                        prediction = str(last_json[key]).strip()
                        if prediction and prediction.lower() not in ["none", "null", "", "n/a"]:
                            break
                
                # Log reasoning if available (check multiple possible keys)
                reasoning_keys = ["reasoning", "analysis", "explanation", "thought", "thinking", "rationale", "justification", "evaluation"]
                for key in reasoning_keys:
                    if key in last_json and last_json[key] is not None:
                        reasoning = str(last_json[key])
                        if reasoning.strip():
                            self.log_fn(f"{key.capitalize()}: {reasoning[:200]}...")
                            break
                
                # Extract confidence if available (handle both string and numeric)
                if "confidence" in last_json and last_json["confidence"] is not None:
                    confidence = str(last_json["confidence"])
                    self.log_fn(f"Confidence: {confidence}")
                elif "certainty" in last_json and last_json["certainty"] is not None:
                    confidence = str(last_json["certainty"])
                    self.log_fn(f"Certainty: {confidence}")
            else:
                # Fallback: try to extract any meaningful text from the response
                # Look for common patterns like "Grade: X" or "Answer: X"
                grade_match = _GRADE_PATTERN.search(response_text)
                if grade_match:
                    prediction = grade_match.group(1).strip()
                    self.log_fn(f"Extracted grade via pattern matching: {prediction}")
                else:
                    # Try to find standalone grades in the text
                    grade_keywords = ["correct", "incorrect", "partially correct", "partial", "wrong", "right", "full credit", "no credit"]
                    text_lower = response_text.lower()
                    for keyword in grade_keywords:
                        if keyword in text_lower:
                            # Extract surrounding context
                            idx = text_lower.find(keyword)
                            start = max(0, idx - 20)
                            end = min(len(response_text), idx + len(keyword) + 20)
                            prediction = response_text[start:end].strip()
                            self.log_fn(f"Extracted grade via keyword '{keyword}': {prediction}")
                            break
                    else:
                        # Last resort: use first 100 chars of response
                        prediction = response_text[:100].strip()
                        self.log_fn(f"Using raw response (no JSON found): {prediction[:50]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Ensure we still return something usable even on error
            try:
                if msg_history:
                    last_msg = msg_history[-1]
                    response_text = last_msg.get("text", "") or last_msg.get("content", "")
                    if response_text:
                        prediction = response_text[:100].strip()
            except Exception:
                prediction = "Error: Extraction failed"

        return str(prediction), msg_history
