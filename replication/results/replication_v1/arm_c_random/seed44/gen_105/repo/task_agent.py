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
            continue
    return results or None


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    First tries standard extraction, then attempts to fix common
    JSON formatting issues like trailing commas, missing quotes,
    unescaped special characters, and nested structure issues.
    
    Enhanced with additional heuristics for LLM-generated JSON.
    """
    # First attempt: standard extraction
    result = _extract_jsons(text)
    if result:
        return result
    
    # Preprocessing: Clean up common LLM JSON issues
    cleaned_text = text
    
    # Remove markdown code blocks if present
    cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
    
    # Retry with progressively more aggressive fixes
    fixes = [
        # Level 1: Basic fixes (trailing commas, basic quote issues)
        lambda t: re.sub(r',(\s*[}\]])', r'\1', t),
        
        # Level 2: Handle single quotes and common escape issues
        lambda t: re.sub(r"(?<!\\)'", '"', t),
        
        # Level 3: Remove comments and fix more edge cases
        lambda t: re.sub(r'//.*?\n', '\n', re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL)),
        
        # Level 4: Fix unescaped quotes in string values
        lambda t: re.sub(r'(?<=")([^"]*?)"([^"]*?)(?=")', lambda m: m.group(0).replace('"', '\\"') if m.group(0).count('"') > 0 else m.group(0), t),
    ]
    
    for attempt in range(min(max_retries, len(fixes))):
        try:
            fixed_text = cleaned_text
            
            # Apply fixes cumulatively
            for i in range(attempt + 1):
                fixed_text = fixes[i](fixed_text)
            
            # Additional fix: handle unescaped newlines and tabs in strings
            def escape_special_chars(match):
                content = match.group(1)
                # Escape newlines and tabs that aren't already escaped
                content = content.replace('\n', '\\n').replace('\t', '\\t')
                # Handle actual newlines and tabs
                content = content.replace('\n', '\\n').replace('\t', '\\t')
                return '"' + content + '"'
            
            # Find string content and escape special chars
            fixed_text = re.sub(r'"([^"]*(?:\n[^"]*)*)"', escape_special_chars, fixed_text)
            
            result = _extract_jsons(fixed_text)
            if result:
                logger.debug(f"JSON extraction succeeded after fix level {attempt + 1}")
                return result
        except Exception as e:
            logger.debug(f"Fix level {attempt + 1} failed: {e}")
            continue
    
    # Final attempts: Try multiple regex patterns for JSON extraction
    json_patterns = [
        # Pattern 1: Standard JSON object
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        # Pattern 2: JSON with nested objects (one level deeper)
        r'\{(?:[^{}]|\{[^{}]*\})*\}',
        # Pattern 3: JSON array
        r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]',
    ]
    
    for pattern in json_patterns:
        try:
            matches = re.findall(pattern, text, re.DOTALL)
            for potential_json in matches:
                try:
                    parsed = json.loads(potential_json)
                    logger.debug(f"JSON extraction succeeded using pattern: {pattern[:30]}...")
                    return [parsed] if not isinstance(parsed, list) else parsed
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue
    
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
- Be thorough in your reasoning before providing the final evaluation"""

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
        extraction_metadata = {
            "success": False,
            "method": None,
            "keys_found": [],
            "error": None,
        }
        
        try:
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1]
                text_content = last_message.get("text", "")
                
                # Validate text content exists
                if not text_content or not text_content.strip():
                    logger.warning("Empty text content in last message")
                    extraction_metadata["error"] = "Empty text content"
                else:
                    extracted = _extract_json_with_retry(text_content)
                    if extracted:
                        extraction_metadata["success"] = True
                        extraction_metadata["method"] = "json_extraction"
                        last_json = extracted[-1]
                        extraction_metadata["keys_found"] = list(last_json.keys())
                        
                        # Priority order for extracting the prediction
                        if "response" in last_json:
                            prediction = last_json["response"]
                            self.log_fn(f"Successfully extracted prediction from 'response' field: {str(prediction)[:100]}...")
                        elif "evaluation" in last_json:
                            prediction = last_json["evaluation"]
                            self.log_fn(f"Using 'evaluation' field: {str(prediction)[:100]}...")
                        elif "answer" in last_json:
                            prediction = last_json["answer"]
                            self.log_fn(f"Using 'answer' field: {str(prediction)[:100]}...")
                        elif "result" in last_json:
                            prediction = last_json["result"]
                            self.log_fn(f"Using 'result' field: {str(prediction)[:100]}...")
                        else:
                            # No recognized key found, use the entire JSON as string
                            logger.warning(f"JSON missing expected keys. Keys found: {list(last_json.keys())}")
                            prediction = json.dumps(last_json)
                            extraction_metadata["error"] = f"No recognized prediction key. Available: {list(last_json.keys())}"
                    else:
                        logger.warning("No valid JSON found in response, attempting text extraction")
                        # Fallback: Try to extract meaningful content from raw text
                        # Look for common patterns like "Answer: ..." or "Result: ..."
                        text_patterns = [
                            r'(?:Answer|Result|Evaluation|Response)[:\s]+(.+?)(?:\n|$)',
                            r'(?:The answer is|Therefore)[:\s]+(.+?)(?:\n|$)',
                        ]
                        for pattern in text_patterns:
                            match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
                            if match:
                                prediction = match.group(1).strip()
                                extraction_metadata["success"] = True
                                extraction_metadata["method"] = "text_pattern"
                                self.log_fn(f"Extracted prediction using text pattern: {str(prediction)[:100]}...")
                                break
                        else:
                            # No pattern matched, use truncated raw text
                            prediction = text_content[:1000] if text_content else "None"
                            extraction_metadata["error"] = "No JSON or text pattern found"
            else:
                logger.warning("Empty message history from LLM")
                extraction_metadata["error"] = "Empty message history"
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Full traceback:")
            extraction_metadata["error"] = str(e)

        # Log extraction metadata for debugging
        logger.debug(f"Extraction metadata: {extraction_metadata}")

        return str(prediction), msg_history
