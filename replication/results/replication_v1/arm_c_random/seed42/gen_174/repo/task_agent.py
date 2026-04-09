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
    
    Enhanced with detailed logging and multiple parsing strategies.
    """
    results = []
    search_from = 0
    json_blocks_found = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.warning(f"Unclosed <json> tag found at position {start}")
            break
        
        inner = text[start + 6:end].strip()
        json_blocks_found += 1
        search_from = end + 7
        
        # Try primary parsing
        try:
            parsed = json.loads(inner)
            results.append(parsed)
            logger.debug(f"Successfully parsed JSON block #{json_blocks_found}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error at position {start}: {e}")
            
            # Attempt recovery: try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                # Fix single quotes to double quotes (common LLM mistake)
                fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', fixed)
                fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
                parsed = json.loads(fixed)
                results.append(parsed)
                logger.info(f"Recovered JSON block #{json_blocks_found} with auto-fixes")
            except json.JSONDecodeError:
                logger.debug(f"Failed to recover JSON block #{json_blocks_found}")
                continue
    
    if json_blocks_found > 0 and not results:
        logger.warning(f"Found {json_blocks_found} JSON blocks but none were valid")
    
    return results or None


def _extract_jsons_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for edge cases.
    
    Attempts to find JSON objects even without proper <json> tags.
    """
    results = []
    # Try to find JSON objects between curly braces
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems.
    
    This agent uses an LLM to evaluate mathematical problem solutions
    and returns structured feedback in JSON format.
    
    Attributes:
        model: The LLM model identifier to use for inference.
        log_fn: Logging function for agent activity (defaults to logger.info).
        max_retries: Maximum number of retry attempts for failed LLM calls.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: The LLM model identifier to use. Defaults to EVAL_MODEL.
            log_file: Optional path to a log file (currently unused, for interface compatibility).
        """
        self.model: str = model
        self.log_fn = logger.info
        self.max_retries: int = 2

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not isinstance(inputs, dict):
            raise ValueError(f"Expected dict for inputs, got {type(inputs).__name__}")
        
        logger.info(f"TaskAgent.forward called with inputs keys: {list(inputs.keys())}")
        
        instruction = f"""You are an expert grading agent for mathematical problems.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>

Important: Ensure your response is valid JSON and properly formatted within the <json> tags."""

        # Call LLM with retry logic
        retries = 0
        last_error = None
        msg_history = []
        
        while retries <= self.max_retries:
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
                logger.info(f"LLM call successful on attempt {retries + 1}")
                break
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"LLM call failed (attempt {retries}/{self.max_retries + 1}): {e}")
                if retries > self.max_retries:
                    logger.error(f"All LLM call attempts failed: {last_error}")
                    return f"Error: LLM call failed after {self.max_retries + 1} attempts", []
        
        # Extract prediction from JSON with fallback
        prediction = "None"
        extraction_method = "none"
        
        try:
            if msg_history:
                text = msg_history[-1].get("text", "")
                text_preview = text[:200] + "..." if len(text) > 200 else text
                logger.debug(f"Raw LLM response preview: {text_preview}")
                
                # Primary extraction
                extracted = _extract_jsons(text)
                extraction_method = "primary"
                
                # Try fallback if primary extraction fails
                if not extracted:
                    extracted = _extract_jsons_fallback(text)
                    if extracted:
                        extraction_method = "fallback"
                        self.log_fn("Used fallback JSON extraction")
                
                if extracted:
                    logger.info(f"Successfully extracted {len(extracted)} JSON object(s) using {extraction_method} method")
                    
                    if "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        pred_preview = prediction[:100] + "..." if len(str(prediction)) > 100 else prediction
                        self.log_fn(f"Successfully extracted prediction: {pred_preview}")
                    else:
                        available_keys = list(extracted[-1].keys())
                        self.log_fn(f"No 'response' field found. Available keys: {available_keys}")
                        # Try to use the first available value as prediction
                        if available_keys:
                            prediction = str(extracted[-1][available_keys[0]])
                            self.log_fn(f"Using alternative field '{available_keys[0]}' as prediction")
                else:
                    self.log_fn("No valid JSON found in response")
                    # Last resort: use raw text if no JSON found
                    if text.strip():
                        prediction = text.strip()[:500]  # Limit length
                        self.log_fn("Using raw text as prediction (limited to 500 chars)")
            else:
                self.log_fn("Empty message history received")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Full traceback for prediction extraction error:")

        return str(prediction), msg_history
