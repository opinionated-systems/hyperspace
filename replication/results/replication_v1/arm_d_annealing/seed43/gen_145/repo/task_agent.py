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
    Includes detailed logging for debugging extraction failures.
    """
    results = []
    search_from = 0
    blocks_found = 0
    
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            logger.debug(f"Unclosed <json> tag found at position {start}")
            break
        
        blocks_found += 1
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            logger.debug(f"Empty JSON block #{blocks_found} found, skipping")
            continue
            
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON block #{blocks_found}")
            else:
                logger.debug(f"JSON block #{blocks_found} is not a dict, skipping")
        except json.JSONDecodeError as e:
            # Log the error with context for debugging
            preview = inner[:100].replace('\n', ' ')
            logger.debug(f"JSON decode error in block #{blocks_found}: {e}. Content preview: {preview}...")
            continue
    
    if blocks_found > 0 and not results:
        logger.warning(f"Found {blocks_found} JSON blocks but none parsed successfully")
    
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses.
    
    Handles nested braces, code blocks, and various formatting edge cases.
    Improved with better brace balancing and more robust pattern matching.
    """
    results = []
    
    def extract_balanced_json(content: str) -> list[dict]:
        """Extract all balanced JSON objects from content using stack-based parsing."""
        objects = []
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
                continue
            if char == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_obj = json.loads(content[start_idx:i+1])
                        if isinstance(json_obj, dict):
                            objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        return objects
    
    # Try to find JSON objects in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        objects = extract_balanced_json(content)
        results.extend(objects)
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        results = extract_balanced_json(text)
    
    # Final fallback: try to find key-value patterns for response and reasoning
    if not results:
        result = {}
        # Look for response pattern with more flexible matching - prioritize exact grade words
        response_patterns = [
            r'["\']response["\']\s*:\s*["\'](Correct|Partial|Incorrect)["\']',
            r'["\']response["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']response["\']\s*:\s*(\d+)',
            r'response\s*:\s*([\w\s-]+)(?:\n|$)',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip()
                break
        
        # Look for reasoning pattern
        reasoning_patterns = [
            r'["\']reasoning["\']\s*:\s*"((?:[^"\\]|\\.)*)"',
            r'["\']reasoning["\']\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            r'reasoning\s*:\s*(.+?)(?:\n\s*["\']|$)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result["reasoning"] = match.group(1).strip()
                break
        
        if result:
            results.append(result)
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

## Grading Rubric (STRICT)
You must assign EXACTLY one of these three grades:

- **Correct**: Use ONLY when:
  * The final answer matches the solution exactly, OR
  * The answer uses an equivalent valid approach with correct reasoning AND arrives at the correct final result
  * The reasoning is sound and complete
  * Example: "Correct"

- **Partial**: Use ONLY when:
  * The answer shows some correct reasoning or approach but has minor errors
  * The answer is incomplete but partially correct
  * The final result is wrong but the approach has merit
  * Example: "Partial"

- **Incorrect**: Use ONLY when:
  * The answer contains fundamental conceptual errors
  * The approach is completely wrong or irrelevant
  * The final result is completely wrong with no valid reasoning
  * Example: "Incorrect"

## Response Format (REQUIRED - STRICT)
You MUST respond with EXACTLY this format - valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Correct"
}}
</json>

The "response" field MUST be exactly one of: "Correct", "Partial", or "Incorrect" (case-sensitive, no quotes around the value in the field).

IMPORTANT: 
- Ensure your JSON is valid and properly formatted
- The 'response' field should contain ONLY the grade word, nothing else
- Do not add extra text before or after the JSON block
- Escape any quotes within your reasoning with backslash
"""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                raw_prediction = str(last_json["response"]).strip()
                # Normalize prediction to one of the three valid grades
                prediction_lower = raw_prediction.lower()
                if prediction_lower in ["correct", "right", "true", "yes", "1", "full"]:
                    prediction = "Correct"
                elif prediction_lower in ["partial", "partially", "half", "0.5", "some"]:
                    prediction = "Partial"
                elif prediction_lower in ["incorrect", "wrong", "false", "no", "0", "none", "error"]:
                    prediction = "Incorrect"
                else:
                    # Keep original if it doesn't match known patterns
                    prediction = raw_prediction
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning = self._extract_prediction(last_text)
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract valid prediction, retrying...")
                    # Add feedback for retry with clearer instructions
                    if attempt < self.max_retries - 1:
                        base_prompt = self._build_grading_prompt(inputs)
                        instruction = (
                            "ERROR: Your previous response did not contain a valid grade in the 'response' field.\n\n"
                            'The \'response\' field MUST contain exactly one of these three values (case-sensitive):\n'
                            '- "Correct" - for fully correct answers\n'
                            '- "Partial" - for partially correct answers  \n'
                            '- "Incorrect" - for wrong answers\n\n'
                            "You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. No text before or after.\n\n"
                            "Correct format example:\n"
                            "<json>\n"
                            '{\n    "reasoning": "The student correctly applied the quadratic formula but made an arithmetic error in the final step...",\n    "response": "Partial"\n}\n'
                            "</json>\n\n"
                            "Now try again with the original task:\n\n" + base_prompt
                        )
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        # Final fallback: if we still don't have a valid prediction, try one more extraction
        # from the last response with more lenient parsing
        if prediction in ["None", ""] and msg_history:
            last_text = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else str(msg_history[-1])
            # Look for any of the three grade words in the text
            text_lower = last_text.lower()
            if "correct" in text_lower and "incorrect" not in text_lower:
                prediction = "Correct"
                self.log_fn("Final fallback: extracted 'Correct' from text")
            elif "partial" in text_lower:
                prediction = "Partial"
                self.log_fn("Final fallback: extracted 'Partial' from text")
            elif "incorrect" in text_lower or "wrong" in text_lower:
                prediction = "Incorrect"
                self.log_fn("Final fallback: extracted 'Incorrect' from text")
        
        return str(prediction), msg_history
