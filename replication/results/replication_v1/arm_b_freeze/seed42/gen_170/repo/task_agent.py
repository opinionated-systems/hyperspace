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
    Also handles markdown code blocks as a fallback.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            logger.debug(f"Failed to parse JSON: {e}, content: {inner[:100]}...")
            continue
    
    # Fallback: try to find JSON in markdown code blocks
    if not results:
        import re
        # Match ```json ... ``` or ``` ... ``` blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

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
        # Format the task input nicely
        task_input = json.dumps(inputs, indent=2, default=str)
        
        # Log the task input for debugging
        logger.info(f"Processing task with inputs: domain={inputs.get('domain', 'unknown')}")

        instruction = f"""You are an expert grading agent. Your task is to evaluate a student's answer to a problem.

Task input:
```json
{task_input}
```

Instructions:
1. Carefully read the problem, solution, grading guidelines, and student answer
2. Evaluate the student's answer based on the grading guidelines
3. Provide your evaluation in the JSON format below

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your detailed evaluation and grade here"
}}
</json>

Important: Ensure your response is valid JSON inside the <json> tags."""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            logger.info(f"LLM call successful, usage: {info.get('usage', {})}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Extract prediction from JSON
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                last_msg = msg_history[-1]
                text_content = last_msg.get("text", "")
                
                # Log the raw response for debugging
                logger.debug(f"Raw response length: {len(text_content)}")
                
                extracted = _extract_jsons(text_content)
                if extracted and len(extracted) > 0:
                    last_extracted = extracted[-1]
                    if "response" in last_extracted:
                        prediction = last_extracted["response"]
                        logger.info(f"Successfully extracted prediction, length: {len(str(prediction))}")
                    else:
                        logger.warning(f"No 'response' key in extracted JSON: {last_extracted.keys()}")
                        # Use the first available key as fallback
                        if last_extracted:
                            first_key = list(last_extracted.keys())[0]
                            prediction = last_extracted[first_key]
                            logger.info(f"Using fallback key '{first_key}' for prediction")
                else:
                    logger.warning("No JSON extracted from response")
                    # Fallback: use raw response if no JSON found
                    prediction = text_content[:500]  # Truncate for safety
            else:
                logger.warning("Empty message history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            logger.exception("Prediction extraction failed")

        return str(prediction), msg_history
