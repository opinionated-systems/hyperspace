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
    Also handles nested JSON objects within the tags.
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
        
        # Try to parse the inner content as JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to extract JSON objects using brace matching for nested structures
            try:
                # Find the outermost JSON object by matching braces
                brace_count = 0
                json_start = -1
                for i, char in enumerate(inner):
                    if char == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start != -1:
                            json_str = inner[json_start:i+1]
                            try:
                                results.append(json.loads(json_str))
                                break
                            except json.JSONDecodeError:
                                continue
            except Exception:
                continue
    return results or None


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for unwrapped JSON objects.

    This handles cases where the model outputs valid JSON without <json> tags.
    Searches for JSON objects with a "response" key using multiple strategies.
    """
    results = []
    
    # Strategy 1: Pattern to match JSON objects (handles nested braces)
    pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match.group())
            if "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    # Strategy 2: If regex fails, try to find any JSON-like structure
    if not results:
        try:
            # Try to parse the entire text as JSON
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to extract JSON using brace matching for complex nested structures
    if not results:
        try:
            brace_count = 0
            json_start = -1
            for i, char in enumerate(text):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_str = text[json_start:i+1]
                        try:
                            obj = json.loads(json_str)
                            if isinstance(obj, dict) and "response" in obj:
                                results.append(obj)
                                break  # Take the first valid one
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction.
    
    This agent processes grading tasks by calling an LLM and extracting
    structured JSON responses. It includes fallback mechanisms for robust
    extraction even when the model output format varies.
    
    Attributes:
        model: The LLM model identifier to use for task solving.
        log_fn: Logging function for agent operations.
    """

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an agent.

Task input:
```
{inputs}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": ...
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "failed"
        try:
            extracted = _extract_jsons(msg_history[-1]["text"])
            if extracted and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                extraction_method = "primary"
                self._extraction_stats["primary"] += 1
            else:
                # Try fallback extraction
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                else:
                    self._extraction_stats["failed"] += 1
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(msg_history[-1]["text"])
                if extracted and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self._extraction_stats["fallback"] += 1
                else:
                    self._extraction_stats["failed"] += 1
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")
                self._extraction_stats["failed"] += 1

        self.log_fn(f"Extraction method used: {extraction_method}")
        self.log_fn(f"Extraction stats: {self._extraction_stats}")
        
        # Validate the prediction before returning
        validated_prediction = self._validate_prediction(prediction, inputs)
        return validated_prediction, msg_history

    def get_extraction_stats(self) -> dict:
        """Return the current extraction statistics.
        
        Returns:
            Dictionary with counts of primary, fallback, and failed extractions.
        """
        return dict(self._extraction_stats)

    def reset_extraction_stats(self) -> None:
        """Reset extraction statistics to zero."""
        self._extraction_stats = {"primary": 0, "fallback": 0, "failed": 0}

    def _validate_prediction(self, prediction: str, inputs: dict) -> str:
        """Validate and normalize the prediction output.
        
        Args:
            prediction: The raw prediction string from LLM
            inputs: The original task inputs for context
            
        Returns:
            Validated and normalized prediction string
        """
        if prediction is None or prediction == "None":
            return "Error: No prediction generated"
        
        # Strip whitespace and normalize
        prediction = str(prediction).strip()
        
        # Check for empty predictions
        if not prediction:
            return "Error: Empty prediction"
        
        # Log prediction length for monitoring
        pred_len = len(prediction)
        if pred_len > 1000:
            self.log_fn(f"Warning: Very long prediction ({pred_len} chars)")
        
        return prediction

    def process_batch(self, batch_inputs: list[dict]) -> list[tuple[str, list[dict]]]:
        """Process multiple tasks in batch with shared context.
        
        This method enables efficient batch processing by reusing
        the same agent configuration across multiple inputs.
        
        Args:
            batch_inputs: List of input dictionaries, each containing
                         domain, problem, solution, grading_guidelines, student_answer
                         
        Returns:
            List of (prediction, msg_history) tuples, one per input
        """
        results = []
        self.log_fn(f"Starting batch processing of {len(batch_inputs)} tasks")
        
        for idx, inputs in enumerate(batch_inputs):
            self.log_fn(f"Processing batch item {idx + 1}/{len(batch_inputs)}")
            try:
                prediction, msg_history = self.forward(inputs)
                results.append((prediction, msg_history))
            except Exception as e:
                self.log_fn(f"Error processing batch item {idx + 1}: {e}")
                results.append((f"Error: {e}", []))
        
        self.log_fn(f"Completed batch processing. Success: {len([r for r in results if not str(r[0]).startswith('Error')])}/{len(batch_inputs)}")
        return results
