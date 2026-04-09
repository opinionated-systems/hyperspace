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
import hashlib
from functools import lru_cache

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM responses to avoid redundant API calls
# This improves performance and reduces API costs during repeated evaluations
_RESPONSE_CACHE: dict[str, tuple[str, list[dict], dict]] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _get_cache_key(msg: str, model: str, temperature: float) -> str:
    """Generate a cache key from the message and parameters."""
    key_data = f"{model}:{temperature}:{msg}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _get_cached_response(msg: str, model: str, temperature: float) -> tuple[str, list[dict], dict] | None:
    """Get cached response if available."""
    global _CACHE_HITS
    key = _get_cache_key(msg, model, temperature)
    if key in _RESPONSE_CACHE:
        _CACHE_HITS += 1
        logger.debug(f"Cache hit! Total hits: {_CACHE_HITS}")
        return _RESPONSE_CACHE[key]
    return None


def _cache_response(msg: str, model: str, temperature: float, response: tuple[str, list[dict], dict]) -> None:
    """Cache an LLM response."""
    global _CACHE_MISSES
    key = _get_cache_key(msg, model, temperature)
    _RESPONSE_CACHE[key] = response
    _CACHE_MISSES += 1
    # Limit cache size to prevent memory issues
    if len(_RESPONSE_CACHE) > 1000:
        # Remove oldest entries (first 100)
        keys_to_remove = list(_RESPONSE_CACHE.keys())[:100]
        for k in keys_to_remove:
            del _RESPONSE_CACHE[k]


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring performance."""
    global _CACHE_HITS, _CACHE_MISSES
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = _CACHE_HITS / total if total > 0 else 0
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "total": total,
        "hit_rate": hit_rate,
        "size": len(_RESPONSE_CACHE),
    }


def clear_cache() -> None:
    """Clear the response cache."""
    global _CACHE_HITS, _CACHE_MISSES
    _RESPONSE_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Includes detailed logging for debugging extraction failures.
    Also attempts to repair common JSON formatting issues.
    """
    results = []
    search_from = 0
    blocks_found = 0
    
    def repair_json(content: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        # Fix single quotes to double quotes (simple cases)
        content = re.sub(r"'([^']*?)':", r'"\1":', content)
        content = re.sub(r":\s*'([^']*?)'", r': "\1"', content)
        # Fix unquoted keys (but be careful not to break valid JSON)
        content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', content)
        return content
    
    # Handle case-insensitive tag matching
    text_lower = text.lower()
    
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            logger.debug(f"Unclosed <json> tag found at position {start}")
            break
        
        blocks_found += 1
        # Use original text for content extraction (preserve case)
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Skip empty blocks
        if not inner:
            logger.debug(f"Empty JSON block #{blocks_found} found, skipping")
            continue
            
        # Try parsing original content first
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                results.append(parsed)
                logger.debug(f"Successfully parsed JSON block #{blocks_found}")
                continue
            else:
                logger.debug(f"JSON block #{blocks_found} is not a dict, skipping")
                continue
        except json.JSONDecodeError as e:
            # Try repairing the JSON
            repaired = repair_json(inner)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    results.append(parsed)
                    logger.debug(f"Successfully parsed repaired JSON block #{blocks_found}")
                    continue
            except json.JSONDecodeError:
                pass
            
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
    Improved with better brace balancing, multi-level fallbacks, and LLM-based repair.
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
    
    def try_repair_json(content: str) -> dict | None:
        """Attempt to repair common JSON formatting issues."""
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        # Fix single quotes to double quotes (simple cases)
        content = re.sub(r"'([^']*?)':", r'"\1":', content)
        content = re.sub(r":\s*'([^']*?)'", r': "\1"', content)
        # Fix unquoted keys (but be careful not to break valid JSON)
        content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', content)
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return None
    
    def extract_from_angle_brackets(content: str) -> list[dict]:
        """Extract JSON from <json>...</json> tags."""
        objects = []
        pattern = r'<json>(.*?)</json>'
        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    objects.append(parsed)
            except json.JSONDecodeError:
                # Try repair
                repaired = try_repair_json(json_str)
                if repaired:
                    objects.append(repaired)
        return objects
    
    # Try to find JSON objects in <json> tags first (most reliable)
    angle_bracket_objects = extract_from_angle_brackets(text)
    results.extend(angle_bracket_objects)
    
    # Try to find JSON objects in code blocks
    if not results:
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            content = match.group(1).strip()
            objects = extract_balanced_json(content)
            results.extend(objects)
            # Try repair on failed extractions
            if not objects:
                repaired = try_repair_json(content)
                if repaired:
                    results.append(repaired)
    
    # If no results from code blocks, try to find any balanced JSON object in text
    if not results:
        results = extract_balanced_json(text)
    
    # Try repair on the full text if still no results
    if not results:
        repaired = try_repair_json(text)
        if repaired:
            results.append(repaired)
    
    # Final fallback: try to find key-value patterns for response and reasoning
    if not results:
        result = {}
        # Look for response pattern with more flexible matching
        response_patterns = [
            r'["\']response["\']\s*:\s*["\']([^"\']+)["\']',
            r'["\']response["\']\s*:\s*(\d+)',
            r'response\s*:\s*([\w\s-]+)(?:\n|$)',
            r'(?:^|\n)\s*response\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            r'(?:grade|assessment|score|result)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in response_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["response"] = match.group(1).strip()
                break
        
        # Look for reasoning pattern
        reasoning_patterns = [
            r'["\']reasoning["\']\s*:\s*["\']([^"\']*(?:\.[^"\']*)*)["\']',
            r'["\']reasoning["\']\s*:\s*"([^"]*)"',
            r'reasoning\s*:\s*(.+?)(?:\n\s*["\']|$)',
            r'(?:^|\n)\s*reasoning\s*[:=]\s*["\']?(.+?)(?:\n\s*(?:response|grade)|$)',
            r'(?:analysis|explanation|evaluation)\s*[:=]\s*["\']?(.+?)(?:\n\s*(?:response|grade|score)|$)',
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

## Grading Rubric
When assigning grades, consider:
- **Correct**: The answer matches the solution or uses an equivalent valid approach with correct reasoning and final result.
- **Partial**: The answer shows some correct reasoning but has minor errors, incomplete steps, or partially correct results.
- **Incorrect**: The answer contains fundamental errors, wrong approach, or completely wrong results.

## Response Format (REQUIRED - READ CAREFULLY)
You MUST respond with valid JSON wrapped in <json>...</json> tags. Do not include any text outside the JSON tags.

<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)"
}}
</json>

## CRITICAL FORMATTING RULES:
1. Use <json> and </json> tags (NOT markdown code blocks like ```json)
2. Use straight double quotes (") for all strings (NOT single quotes)
3. Do NOT include trailing commas in JSON objects
4. The 'response' field must contain ONLY the grade (e.g., "Correct", "Partial", "Incorrect")
5. The 'reasoning' field must contain your detailed analysis
6. Do NOT include any text before <json> or after </json>
7. Ensure your JSON is valid - you can test it at jsonlint.com

Your entire response must be ONLY the JSON block above, nothing else."""

    def _extract_prediction(self, text: str) -> tuple[str, str]:
        """Extract prediction and reasoning from response text.
        
        Uses a multi-tier extraction strategy:
        1. Primary: Extract from <json> tags
        2. Secondary: Regex-based extraction for malformed JSON
        3. Tertiary: Direct text pattern matching
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        
        # Tier 1: Try primary extraction method from <json> tags
        extracted = _extract_jsons(text)
        
        # Tier 2: Fallback to regex extraction if primary fails
        if extracted is None:
            extracted = _extract_json_with_regex(text)
        
        # Tier 3: Direct pattern extraction as last resort
        if not extracted:
            # Try to find any grade/assessment in the text
            grade_patterns = [
                r'(?:grade|assessment|score|result)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
                r'(?:the answer is|final grade|assessment)\s*[:=]?\s*["\']?([^"\'\n]+)["\']?',
            ]
            for pattern in grade_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    prediction = match.group(1).strip()
                    break
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
        
        # Validate prediction - if it's empty or just whitespace, mark as None
        if prediction and prediction.strip():
            prediction = prediction.strip()
        else:
            prediction = "None"
        
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
        
        # Check cache first for identical prompts (only on first attempt)
        cached = _get_cached_response(instruction, self.model, 0.0)
        if cached is not None:
            response_text, cached_history, info = cached
            last_text = cached_history[-1]["text"] if cached_history else ""
            prediction, reasoning = self._extract_prediction(last_text)
            if prediction != "None":
                self.log_fn(f"Cache hit! Using cached prediction: {prediction}")
                return str(prediction), cached_history
        
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
                    # Cache successful responses
                    if attempt == 0:
                        _cache_response(instruction, self.model, 0.0, (response, msg_history, info))
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry with more specific guidance
                    if attempt < self.max_retries - 1:
                        instruction = f"""ERROR: Your previous response did not contain valid JSON with a 'response' field.

You MUST respond with ONLY valid JSON wrapped in <json>...</json> tags. Do not include any text before or after the JSON tags.

Common mistakes to avoid:
- Do not use markdown code blocks (```json) - use <json> tags instead
- Ensure all quotes are straight double quotes (")
- Do not include trailing commas
- The 'response' field must contain the grade (e.g., "Correct", "Partial", "Incorrect")
- The 'reasoning' field must contain your analysis

Correct format:
<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "Correct"
}}
</json>

Now try again with the original task:

{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        return str(prediction), msg_history
