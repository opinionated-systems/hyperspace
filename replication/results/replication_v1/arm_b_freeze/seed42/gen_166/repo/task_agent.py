"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.

Enhanced with better error handling, response validation, and logging.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

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


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects from text using multiple strategies.
    
    Tries <json> tags first, then looks for JSON objects in code blocks,
    then tries to find JSON objects directly in the text.
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            inner = match.group(1).strip()
            results.append(json.loads(inner))
        except (json.JSONDecodeError, IndexError):
            continue
    
    # Try to find JSON objects directly
    if not results:
        # Look for patterns like {"key": value}
        json_pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
        for match in re.finditer(json_pattern, text):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    
    return results or None


@dataclass
class TaskResult:
    """Result of a task execution."""
    prediction: str
    message_history: list[dict]
    duration_seconds: float
    tokens_used: int
    success: bool
    error: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error": self.error,
        }


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._execution_history: list[TaskResult] = []

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        start_time = time.time()
        
        self.log_fn(f"Starting task execution with model: {self.model}")
        self.log_fn(f"Input keys: {list(inputs.keys())}")
        
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
            
            duration = time.time() - start_time
            tokens_used = info.get("usage", {}).get("total_tokens", 0)
            
            self.log_fn(f"LLM call completed in {duration:.2f}s, {tokens_used} tokens used")

            # Extract prediction from JSON
            prediction = "None"
            extraction_success = False
            
            try:
                if msg_history:
                    last_msg = msg_history[-1]
                    text = last_msg.get("text", "")
                    extracted = _extract_any_json(text)
                    if extracted and "response" in extracted[-1]:
                        prediction = extracted[-1]["response"]
                        extraction_success = True
                        self.log_fn(f"Successfully extracted prediction: {prediction}")
                    else:
                        self.log_fn(f"No valid JSON found in response")
            except Exception as e:
                self.log_fn(f"Error extracting prediction: {e}")
            
            result = TaskResult(
                prediction=str(prediction),
                message_history=msg_history,
                duration_seconds=duration,
                tokens_used=tokens_used,
                success=extraction_success,
                error="" if extraction_success else "Failed to extract prediction",
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_fn(f"Task execution failed: {e}")
            
            result = TaskResult(
                prediction="None",
                message_history=[],
                duration_seconds=duration,
                tokens_used=0,
                success=False,
                error=str(e),
            )
        
        finally:
            self._execution_history.append(result)
            self.log_fn(f"Task result: {result.to_dict()}")

        return result.prediction, result.message_history
    
    def get_execution_history(self) -> list[TaskResult]:
        """Get history of all task executions."""
        return list(self._execution_history)
    
    def get_average_duration(self) -> float:
        """Get average execution duration."""
        if not self._execution_history:
            return 0.0
        return sum(r.duration_seconds for r in self._execution_history) / len(self._execution_history)
    
    def get_success_rate(self) -> float:
        """Get success rate of task executions."""
        if not self._execution_history:
            return 0.0
        successful = sum(1 for r in self._execution_history if r.success)
        return successful / len(self._execution_history)
