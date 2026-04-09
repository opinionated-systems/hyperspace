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


def _extract_json_fuzzy(text: str) -> list[dict] | None:
    """Fuzzy JSON extraction as fallback for malformed responses.
    
    Attempts to find JSON-like structures even without proper tags.
    """
    results = []
    # Try to find JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON objects with curly braces
    if not results:
        # Look for outermost JSON objects
        brace_count = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        results.append(json.loads(text[start_idx:i+1]))
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Build a structured prompt with chain-of-thought reasoning
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematical grader for the International Mathematical Olympiad (IMO).

Your task is to evaluate a student's solution to a mathematical problem. Follow these steps:

1. **Understand the Problem**: Read the problem statement carefully and identify what is being asked.

2. **Review the Official Solution**: Study the provided official solution to understand the expected approach and key insights.

3. **Analyze the Student's Answer**: Examine the student's solution step by step.

4. **Apply Grading Guidelines**: Use the provided grading guidelines to assess the student's work.

5. **Provide Your Evaluation**: Give a clear assessment of the student's solution.

---

**Domain**: {domain}

**Problem**:
```
{problem}
```

**Official Solution**:
```
{solution}
```

**Grading Guidelines**:
```
{grading_guidelines}
```

**Student's Answer**:
```
{student_answer}
```

---

Think through this problem step by step. First, provide your reasoning about the student's solution, then give your final evaluation.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's solution...",
    "response": "Your final evaluation/grade for the student's answer"
}}
</json>

The "response" field should contain your final assessment (e.g., a score, grade, or evaluation summary)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        reasoning = ""
        
        try:
            response_text = msg_history[-1]["text"]
            
            # First try standard extraction
            extracted = _extract_jsons(response_text)
            
            # If that fails, try fuzzy extraction
            if not extracted:
                extracted = _extract_json_fuzzy(response_text)
            
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = last_json["response"]
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # If only reasoning is present, use it as the response
                if prediction == "None" and "reasoning" in last_json:
                    prediction = last_json["reasoning"]
            
            # Log the reasoning for debugging
            if reasoning:
                self.log_fn(f"Reasoning: {reasoning[:200]}...")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to use the raw response
            try:
                raw_text = msg_history[-1]["text"]
                # Remove JSON tags and use the content
                cleaned = re.sub(r'<json>|</json>', '', raw_text).strip()
                if len(cleaned) < 1000:  # Only use if reasonably short
                    prediction = cleaned
            except Exception:
                pass

        return str(prediction), msg_history
