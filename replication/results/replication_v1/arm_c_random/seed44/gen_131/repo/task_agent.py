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
    Includes robust JSON repair for common LLM formatting errors.
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
        
        # Try parsing the JSON directly first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix common JSON issues with comprehensive repair
        try:
            fixed = inner
            
            # Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            
            # Fix single quotes to double quotes for keys and string values
            # Handle nested quotes carefully
            fixed = re.sub(r"(?<=[{\s,])'([^']*?)'(?=\s*:)", r'"\1"', fixed)
            fixed = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', fixed)
            
            # Fix unquoted keys (common in LLM outputs)
            fixed = re.sub(r'(?<=[{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', fixed)
            
            # Fix Python-style True/False/None to JSON true/false/null
            fixed = re.sub(r'(?<=[\s\[:,])True(?=[\s\],}])', 'true', fixed)
            fixed = re.sub(r'(?<=[\s\[:,])False(?=[\s\],}])', 'false', fixed)
            fixed = re.sub(r'(?<=[\s\[:,])None(?=[\s\],}])', 'null', fixed)
            
            # Remove comments (// style and /* */ style)
            fixed = re.sub(r'//.*?\n', '\n', fixed)
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
            
            results.append(json.loads(fixed))
        except json.JSONDecodeError:
            # If all repairs fail, try to extract just the fields we need
            try:
                # Extract reasoning field
                reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"response"|\s*})', fixed, re.DOTALL)
                response_match = re.search(r'"response"\s*:\s*"([^"]*)"', fixed)
                
                if reasoning_match and response_match:
                    results.append({
                        "reasoning": reasoning_match.group(1).replace('\\n', '\n'),
                        "response": response_match.group(1)
                    })
            except Exception:
                continue
    
    return results or None


def _find_json_objects(s: str) -> list[str]:
    """Find potential JSON objects by tracking brace balance."""
    objects = []
    start = -1
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(s[start:i+1])
                    start = -1
    return objects


def _repair_json(text: str) -> str:
    """Apply common JSON repairs."""
    fixed = text
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    # Fix single quotes to double quotes for keys and string values
    fixed = re.sub(r"(?<=[{\s,])'([^']*?)'(?=\s*:)", r'"\1"', fixed)
    fixed = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', fixed)
    # Fix unquoted keys
    fixed = re.sub(r'(?<=[{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', fixed)
    # Fix Python-style True/False/None
    fixed = re.sub(r'(?<=[\s\[:,])True(?=[\s\],}])', 'true', fixed)
    fixed = re.sub(r'(?<=[\s\[:,])False(?=[\s\],}])', 'false', fixed)
    fixed = re.sub(r'(?<=[\s\[:,])None(?=[\s\],}])', 'null', fixed)
    return fixed


def _extract_json_with_retry(text: str, max_retries: int = 3) -> list[dict] | None:
    """Extract JSON with multiple fallback strategies.
    
    Tries multiple extraction methods:
    1. Standard <json>...</json> blocks
    2. JSON code blocks ```json...```
    3. Raw JSON objects in text
    4. Pattern-based field extraction as last resort
    """
    results = []
    
    # Try standard extraction first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try JSON code blocks
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try repair on code block content
            try:
                fixed = _repair_json(match.strip())
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    if results:
        return results
    
    # Try to find raw JSON objects with balanced braces
    raw_objects = _find_json_objects(text)
    for obj in raw_objects:
        try:
            results.append(json.loads(obj))
        except json.JSONDecodeError:
            # Try repair
            try:
                fixed = _repair_json(obj)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    if results:
        return results
    
    # Last resort: try to extract fields directly from text
    # Look for reasoning and response patterns
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"response"|\s*})',
        r'[Rr]easoning[:\s]+(.*?)(?=[Rr]esponse|$)',
        r'[Aa]nalysis[:\s]+(.*?)(?=[Gg]rade|[Ss]core|[Rr]esponse|$)',
    ]
    response_patterns = [
        r'"response"\s*:\s*"([^"]*)"',
        r'"response"\s*:\s*(\d+)',
        r'[Rr]esponse[:\s]+"?([^"\n]+)"?',
        r'[Gg]rade[:\s]+"?([^"\n]+)"?',
        r'[Ss]core[:\s]+"?([^"\n]+)"?',
    ]
    
    reasoning = None
    response = None
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            break
    
    for pattern in response_patterns:
        match = re.search(pattern, text)
        if match:
            response = match.group(1).strip()
            break
    
    if reasoning or response:
        return [{"reasoning": reasoning or "", "response": response or "None"}]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a comprehensive prompt for the grading task."""
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions
1. Carefully analyze the student's answer step by step
2. Compare it against the correct solution
3. Apply the grading guidelines strictly
4. Provide your reasoning before giving the final grade

## Response Format (MANDATORY)
You MUST respond using EXACTLY this format with <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning. Explain what the student did correctly and incorrectly, and how you applied the grading guidelines.",
    "response": "The final grade/score (a number or specific grade value like '7' or '0')"
}}
</json>

IMPORTANT RULES:
- Start your response with <json> and end with </json>
- Use double quotes (") for all strings, never single quotes (')
- The "response" field must contain ONLY the final grade/score
- Do not include any text before <json> or after </json>
- Ensure valid JSON syntax"""

    def _try_extract_prediction(self, text: str) -> tuple[str, str | None]:
        """Try to extract prediction from response text.
        
        Returns:
            (prediction, reasoning)
        """
        try:
            extracted = _extract_json_with_retry(text)
            if extracted:
                last = extracted[-1]
                prediction = last.get("response", "None")
                reasoning = last.get("reasoning")
                return str(prediction), reasoning
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return "None", None

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)
        msg_history = []
        prediction = "None"
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                # Extract prediction
                text = msg_history[-1]["text"] if msg_history else ""
                pred, reasoning = self._try_extract_prediction(text)
                
                if pred != "None":
                    prediction = pred
                    if reasoning:
                        self.log_fn(f"Grading reasoning: {reasoning[:200]}...")
                    break
                
                # If extraction failed, provide more specific guidance
                if attempt < self.max_retries - 1:
                    has_json_tags = "<json>" in text and "</json>" in text
                    has_reasoning = '"reasoning"' in text
                    has_response = '"response"' in text
                    
                    if not has_json_tags:
                        instruction = "ERROR: Your response must start with <json> and end with </json>. Example:\n<json>\n{\"reasoning\": \"Your analysis here\", \"response\": \"grade here\"}\n</json>"
                    elif not has_reasoning:
                        instruction = "ERROR: Your JSON is missing the 'reasoning' field. Include: \"reasoning\": \"your analysis\""
                    elif not has_response:
                        instruction = "ERROR: Your JSON is missing the 'response' field. Include: \"response\": \"grade\""
                    else:
                        instruction = "ERROR: Your JSON could not be parsed. Check for: trailing commas, single quotes instead of double quotes, or unescaped quotes in strings. Use this exact format:\n<json>\n{\"reasoning\": \"your analysis\", \"response\": \"grade\"}\n</json>"
                    
            except Exception as e:
                self.log_fn(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    break
        
        return str(prediction), msg_history
