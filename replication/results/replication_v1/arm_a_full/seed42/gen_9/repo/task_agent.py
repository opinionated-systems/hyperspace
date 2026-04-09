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
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from a task agent run."""
    prediction: str = "None"
    msg_history: list[dict] = field(default_factory=list)
    success: bool = False
    duration: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "prediction": self.prediction,
            "success": self.success,
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata,
        }


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


def _validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate task inputs.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, "Inputs must be a dictionary"
    
    required_fields = ["domain", "problem", "solution", "grading_guidelines", "student_answer"]
    missing = [f for f in required_fields if f not in inputs]
    
    if missing:
        return False, f"Missing required fields: {missing}"
    
    return True, ""


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(
        self, 
        model: str = EVAL_MODEL, 
        log_file: str = "",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.log_fn = logger.info
        self.temperature = temperature

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        result = self.forward_with_result(inputs)
        return result.prediction, result.msg_history

    def forward_with_result(self, inputs: dict) -> TaskResult:
        """Run the task agent with full result information.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            
        Returns:
            TaskResult with prediction and metadata
        """
        start_time = time.time()
        
        # Validate inputs
        is_valid, error_msg = _validate_inputs(inputs)
        if not is_valid:
            return TaskResult(
                prediction="None",
                success=False,
                duration=time.time() - start_time,
                error=error_msg,
            )
        
        instruction = f"""You are an expert grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's answer to a mathematical problem.

Task input:
```
{json.dumps(inputs, indent=2)}
```

Carefully analyze the student's answer against the provided solution and grading guidelines.

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
                temperature=self.temperature,
                msg_history=[],
            )

            # Extract prediction from JSON
            prediction = "None"
            extracted = None
            try:
                if msg_history and len(msg_history) > 0:
                    extracted = _extract_jsons(msg_history[-1].get("text", ""))
                    if extracted and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
            except Exception as e:
                self.log_fn(f"Error extracting prediction: {e}")

            duration = time.time() - start_time
            
            metadata = {
                "usage": info.get("usage", {}),
                "thinking": info.get("thinking", ""),
                "extracted_json_count": len(extracted) if extracted else 0,
            }
            
            return TaskResult(
                prediction=str(prediction),
                msg_history=msg_history,
                success=True,
                duration=duration,
                metadata=metadata,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task agent failed: {e}")
            
            return TaskResult(
                prediction="None",
                success=False,
                duration=duration,
                error=str(e),
            )

    def batch_forward(self, inputs_list: list[dict]) -> list[TaskResult]:
        """Process multiple tasks in batch.
        
        Args:
            inputs_list: List of input dictionaries
            
        Returns:
            List of TaskResult objects
        """
        results = []
        for i, inputs in enumerate(inputs_list):
            self.log_fn(f"Processing task {i+1}/{len(inputs_list)}")
            result = self.forward_with_result(inputs)
            results.append(result)
        return results
