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
from collections import Counter
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


class TaskAgentError(Exception):
    """Custom exception for TaskAgent errors."""
    pass


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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue
    return results or None


def _validate_inputs(inputs: dict) -> None:
    """Validate that required fields are present in inputs.
    
    Args:
        inputs: Dictionary containing task inputs
        
    Raises:
        TaskAgentError: If required fields are missing
    """
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [field for field in required_fields if field not in inputs]
    if missing:
        raise TaskAgentError(f"Missing required fields: {missing}")


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
            
        Raises:
            TaskAgentError: If inputs are invalid or LLM call fails
        """
        # Validate inputs
        try:
            _validate_inputs(inputs)
        except TaskAgentError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        self.call_count += 1
        logger.info(f"TaskAgent call #{self.call_count} with model={self.model}")

        instruction = f"""You are an expert grading agent for mathematical olympiad problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise TaskAgentError(f"Failed to get response from LLM: {e}") from e

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                extracted = _extract_jsons(msg_history[-1]["text"])
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    logger.info(f"Successfully extracted prediction (length={len(str(prediction))})")
                else:
                    logger.warning("No valid JSON response found in LLM output")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            logger.error(f"Error extracting prediction: {e}")

        return str(prediction), msg_history

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "model": self.model,
            "call_count": self.call_count,
        }

    def evaluate_with_confidence(self, inputs: dict) -> tuple[str, float, list[dict]]:
        """Run evaluation with confidence scoring.
        
        This enhanced method performs multiple evaluations and computes
        a confidence score based on consistency across runs.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            
        Returns:
            (prediction, confidence_score, msg_history)
            confidence_score is between 0.0 and 1.0
        """
        # Validate inputs
        try:
            _validate_inputs(inputs)
        except TaskAgentError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        self.call_count += 1
        logger.info(f"TaskAgent evaluate_with_confidence call #{self.call_count}")

        # Run multiple evaluations for confidence scoring
        num_runs = 3
        predictions = []
        all_histories = []
        
        instruction = f"""You are an expert grading agent for mathematical olympiad problems.

Your task is to evaluate a student's answer based on the provided problem, solution, and grading guidelines.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here",
    "score": "Numeric score if applicable",
    "reasoning": "Brief reasoning for the grade"
}}
</json>"""

        for run in range(num_runs):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                    temperature=0.2,  # Slight variation for confidence measurement
                )
                
                # Extract prediction from JSON
                prediction = "None"
                if msg_history and len(msg_history) > 0:
                    extracted = _extract_jsons(msg_history[-1]["text"])
                    if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                
                predictions.append(prediction)
                all_histories.append(msg_history)
                logger.info(f"Run {run + 1}/{num_runs} completed")
                
            except Exception as e:
                logger.error(f"Run {run + 1} failed: {e}")
                predictions.append("ERROR")

        # Compute confidence based on consistency
        valid_predictions = [p for p in predictions if p != "ERROR" and p != "None"]
        
        if len(valid_predictions) == 0:
            confidence = 0.0
            final_prediction = "None"
        elif len(valid_predictions) == 1:
            confidence = 0.5
            final_prediction = valid_predictions[0]
        else:
            # Simple confidence: ratio of identical predictions
            pred_counts = Counter(valid_predictions)
            most_common = pred_counts.most_common(1)[0]
            confidence = most_common[1] / len(valid_predictions)
            final_prediction = most_common[0]
            
            # Add confidence indicator to prediction
            confidence_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
            final_prediction = f"[CONFIDENCE: {confidence_label} ({confidence:.2f})] {final_prediction}"

        logger.info(f"Final confidence score: {confidence:.2f}")
        
        return final_prediction, confidence, all_histories[0] if all_histories else []
