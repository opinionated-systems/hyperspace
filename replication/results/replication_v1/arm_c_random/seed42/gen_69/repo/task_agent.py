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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionMetrics:
    """Tracks execution metrics for a single task."""
    start_time: float = field(default_factory=time.time)
    llm_latency_ms: float = 0.0
    extraction_time_ms: float = 0.0
    extraction_attempts: int = 0
    extraction_method: str = ""
    cache_hit: bool = False
    token_usage: dict = field(default_factory=dict)
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of execution metrics."""
        total_time = time.time() - self.start_time
        return {
            "total_time_ms": round(total_time * 1000, 2),
            "llm_latency_ms": round(self.llm_latency_ms, 2),
            "extraction_time_ms": round(self.extraction_time_ms, 2),
            "extraction_attempts": self.extraction_attempts,
            "extraction_method": self.extraction_method,
            "cache_hit": self.cache_hit,
            "token_usage": self.token_usage,
        }


def _extract_jsons(text: str) -> tuple[list[dict] | None, str]:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    
    Also attempts fallback extraction from markdown code blocks and 
    raw JSON objects if <json> tags are not found.
    
    Returns:
        Tuple of (extracted JSON objects or None, method name used)
    """
    results = []
    search_from = 0
    
    # Primary: Extract from <json>...</json> blocks
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
            logger.debug(f"Failed to parse JSON from <json> block: {e}")
            continue
    
    if results:
        return results, "json_tags"
    
    # Fallback 1: Extract from markdown code blocks ```json ... ```
    markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    if results:
        return results, "markdown_code_block"
    
    # Fallback 2: Extract raw JSON objects from text
    # Match JSON objects: starts with { and ends with }
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(json_pattern, text):
        try:
            parsed = json.loads(match.group(0))
            results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    if results:
        return results, "raw_json"
    
    return None, "none"


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self._execution_history: list[dict] = []

    def forward(self, inputs: dict, track_metrics: bool = True) -> tuple[str, list[dict], dict | None]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            track_metrics: whether to track detailed execution metrics

        Returns:
            (prediction, msg_history, metrics_summary or None)
        """
        self.call_count += 1
        metrics = TaskExecutionMetrics() if track_metrics else None
        
        self.log_fn(f"[TaskAgent #{self.call_count}] Processing new task")
        
        # Build structured instruction with clearer formatting
        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer.

Task Input:
{json.dumps(inputs, indent=2, default=str)}

Instructions:
1. Carefully analyze the problem, solution, grading guidelines, and student answer
2. Provide your evaluation in the exact JSON format below
3. Your response must be wrapped in <json>...</json> tags

Response Format:
<json>
{{
    "response": "Your detailed evaluation here"
}}
</json>"""

        try:
            llm_start = time.time()
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            llm_latency = (time.time() - llm_start) * 1000  # Convert to ms
            
            if metrics:
                metrics.llm_latency_ms = llm_latency
                metrics.cache_hit = info.get("cached", False)
                metrics.token_usage = info.get("usage", {})
            
            self.log_fn(f"[TaskAgent #{self.call_count}] LLM call successful, latency: {llm_latency:.2f}ms, usage: {info.get('usage', {})}")
        except Exception as e:
            self.log_fn(f"[TaskAgent #{self.call_count}] LLM call failed: {e}")
            if metrics:
                metrics.extraction_method = "llm_error"
                self._execution_history.append({
                    "call": self.call_count,
                    "metrics": metrics.get_summary(),
                })
            return "Error: LLM call failed", [{"role": "error", "text": str(e)}], metrics.get_summary() if metrics else None

        # Extract prediction from JSON with enhanced error handling
        prediction = "None"
        extraction_errors = []
        
        try:
            extraction_start = time.time()
            
            if not msg_history:
                extraction_errors.append("Empty message history")
            else:
                last_message = msg_history[-1].get("text", "")
                if not last_message:
                    extraction_errors.append("Last message has no text content")
                else:
                    extracted, method = _extract_jsons(last_message)
                    if metrics:
                        metrics.extraction_method = method
                        metrics.extraction_attempts = 1
                    
                    if not extracted:
                        extraction_errors.append("No valid JSON found in response")
                    elif "response" not in extracted[-1]:
                        extraction_errors.append(f"JSON missing 'response' key. Keys found: {list(extracted[-1].keys())}")
                    else:
                        prediction = extracted[-1]["response"]
                        self.log_fn(f"[TaskAgent #{self.call_count}] Successfully extracted prediction using {method}")
            
            if metrics:
                metrics.extraction_time_ms = (time.time() - extraction_start) * 1000
                
        except Exception as e:
            extraction_errors.append(f"Exception during extraction: {e}")
            self.log_fn(f"[TaskAgent #{self.call_count}] Error extracting prediction: {e}")
        
        if extraction_errors:
            self.log_fn(f"[TaskAgent #{self.call_count}] Extraction issues: {'; '.join(extraction_errors)}")

        # Store metrics in history
        metrics_summary = None
        if metrics:
            metrics_summary = metrics.get_summary()
            self._execution_history.append({
                "call": self.call_count,
                "metrics": metrics_summary,
            })
            # Attach metrics to the last message for external access
            if msg_history:
                msg_history[-1]["_execution_metrics"] = metrics_summary

        return str(prediction), msg_history, metrics_summary
    
    def get_execution_history(self) -> list[dict]:
        """Get the execution history with metrics for all tasks."""
        return self._execution_history
    
    def get_average_latency(self) -> float:
        """Get the average LLM latency across all executions."""
        if not self._execution_history:
            return 0.0
        latencies = [
            entry["metrics"]["llm_latency_ms"]
            for entry in self._execution_history
            if "metrics" in entry and "llm_latency_ms" in entry["metrics"]
        ]
        return sum(latencies) / len(latencies) if latencies else 0.0
    
    def get_cache_hit_rate(self) -> float:
        """Get the cache hit rate across all executions."""
        if not self._execution_history:
            return 0.0
        cache_hits = sum(
            1 for entry in self._execution_history
            if entry.get("metrics", {}).get("cache_hit", False)
        )
        return cache_hits / len(self._execution_history)
