"""
Task agent: solves a given task with chain-of-thought reasoning.

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
    Also handles markdown code blocks and bare JSON objects.
    Includes robust error recovery for malformed JSON with multiple fallback strategies.
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
        
        # Try multiple parsing strategies
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            parsed = _try_parse_json(block.strip())
            if parsed:
                results.append(parsed)
        
        # Try bare ``` ... ``` blocks that might contain JSON
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') or block.startswith('['):
                    parsed = _try_parse_json(block)
                    if parsed:
                        results.append(parsed)
        
        # Try bare JSON objects as fallback
        if not results:
            results.extend(_extract_bare_jsons(text))
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies.
    
    Returns the parsed dict if successful, None otherwise.
    """
    if not text or not text.strip():
        return None
        
    text = text.strip()
    
    # Strategy 1: Direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from within the content
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            inner_json = text[json_start:json_end + 1]
            result = json.loads(inner_json)
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix trailing commas before closing braces/brackets
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1:
            inner_json = text[json_start:json_end + 1]
            # Remove trailing commas before } or ]
            fixed = re.sub(r',(\s*[}\]])', r'\1', inner_json)
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Fix single quotes to double quotes
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1:
            inner_json = text[json_start:json_end + 1]
            # Replace single quotes with double quotes (carefully)
            fixed = re.sub(r"(?<!\\)'", '"', inner_json)
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Fix unescaped newlines in strings
    try:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1:
            inner_json = text[json_start:json_end + 1]
            # Escape unescaped newlines within strings
            fixed = re.sub(r'(?<!\\)\n', r'\\n', inner_json)
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass
    
    return None


def _extract_bare_jsons(text: str) -> list[dict]:
    """Extract bare JSON objects from text using brace matching.
    
    Returns a list of successfully parsed dicts.
    """
    results = []
    
    # Find all { and } positions
    brace_positions = []
    for i, char in enumerate(text):
        if char == '{':
            brace_positions.append((i, 'open'))
        elif char == '}':
            brace_positions.append((i, 'close'))
    
    # Try to find valid JSON by matching braces
    if brace_positions:
        stack = []
        for pos, kind in brace_positions:
            if kind == 'open':
                stack.append(pos)
            elif kind == 'close' and stack:
                start_pos = stack.pop()
                candidate = text[start_pos:pos+1]
                
                # Try to parse this candidate
                parsed = _try_parse_json(candidate)
                if parsed:
                    results.append(parsed)
    
    return results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3  # Increased retries for better reliability

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with chain-of-thought reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields for better prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Build a comprehensive grading instruction
        instruction = self._build_grading_instruction(
            domain, problem, solution, grading_guidelines, student_answer
        )

        msg_history = []
        prediction = "None"
        reasoning = ""

        # Try with retries for better reliability
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON
            prediction, reasoning = self._extract_prediction(msg_history)
            
            if prediction != "None":
                break
            
            if attempt < self.max_retries:
                self.log_fn(f"Retry {attempt + 1}: No valid JSON found, retrying with stronger prompt...")
                # Build a stronger reminder for the retry
                instruction = self._build_retry_instruction(
                    domain, problem, solution, grading_guidelines, student_answer, attempt + 1
                )

        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")

        return str(prediction), msg_history

    def _build_grading_instruction(
        self, domain: str, problem: str, solution: str, 
        grading_guidelines: str, student_answer: str
    ) -> str:
        """Build a comprehensive grading instruction with clear structure."""
        return f"""You are an expert {domain} grader evaluating student solutions with precision and fairness.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Your Task
Evaluate the student's answer by following this structured approach:

### Step 1: Understanding Check
- Identify the key concepts and methods required by the problem
- Note the expected approach from the official solution

### Step 2: Student's Approach Analysis
- What approach did the student take?
- Did they identify the correct method?
- Where did their reasoning diverge from the official solution (if at all)?

### Step 3: Correctness Assessment
- Identify all correct steps in the student's work
- Identify any errors, omissions, or misconceptions
- Check if partial credit should be awarded based on the grading guidelines

### Step 4: Final Grade Determination
- Synthesize your analysis into a clear grade
- The grade should align with the grading guidelines

## Response Format (REQUIRED)
You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering the points above...",
    "response": "The final grade (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"
}}
</json>

Important:
- The JSON must be valid and properly formatted
- Use double quotes for all strings
- The "response" field must contain only the final grade/assessment
- Be objective and fair in your evaluation"""

    def _build_retry_instruction(
        self, domain: str, problem: str, solution: str,
        grading_guidelines: str, student_answer: str, attempt: int
    ) -> str:
        """Build a stronger instruction for retry attempts."""
        base = self._build_grading_instruction(
            domain, problem, solution, grading_guidelines, student_answer
        )
        
        emphasis = [
            "\n\nCRITICAL: Your previous response did not contain valid JSON. You MUST use the exact format shown above with <json>...</json> tags.",
            "\n\nVERY IMPORTANT: I could not parse your previous response. Please ensure your entire response is valid JSON inside <json>...</json> tags. No text outside the tags.",
            "\n\nFINAL ATTEMPT: You MUST output ONLY the JSON object wrapped in <json>...</json> tags. Nothing else. The JSON must be syntactically valid.",
        ]
        
        return base + emphasis[min(attempt - 1, len(emphasis) - 1)]

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced robustness.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Collect all text from messages for comprehensive search
            all_texts = []
            for msg in reversed(msg_history):
                text = msg.get("text", "")
                if text:
                    all_texts.append(text)
            
            # Try to extract JSON from all messages
            extracted = None
            for text in all_texts:
                extracted = _extract_jsons(text)
                if extracted:
                    break
            
            if extracted and len(extracted) > 0:
                # Use the last valid JSON object found
                result = extracted[-1]
                
                # Try multiple possible keys for the response (ordered by likelihood)
                response_keys = [
                    "response", "grade", "answer", "result", "assessment", 
                    "evaluation", "score", "verdict", "final_grade", 
                    "grading_result", "output", "conclusion"
                ]
                for key in response_keys:
                    if key in result:
                        value = result[key]
                        # Ensure we get a string representation
                        if isinstance(value, (str, int, float, bool)):
                            prediction = str(value)
                        elif isinstance(value, list) and len(value) > 0:
                            prediction = str(value[0])
                        elif isinstance(value, dict):
                            prediction = str(value)
                        break
                
                # Extract reasoning if available
                reasoning_keys = [
                    "reasoning", "analysis", "thought", "explanation", 
                    "rationale", "thinking", "evaluation", "assessment",
                    "reason", "justification", "notes"
                ]
                for key in reasoning_keys:
                    if key in result:
                        reasoning_value = result[key]
                        if isinstance(reasoning_value, str):
                            reasoning = reasoning_value
                        else:
                            reasoning = str(reasoning_value)
                        break
                
                # If no reasoning found but we have a prediction, try to extract from other fields
                if not reasoning and prediction != "None":
                    # Combine all other string fields as reasoning
                    other_fields = []
                    for key, value in result.items():
                        if key not in response_keys and isinstance(value, str):
                            other_fields.append(f"{key}: {value}")
                    if other_fields:
                        reasoning = " | ".join(other_fields)
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning
