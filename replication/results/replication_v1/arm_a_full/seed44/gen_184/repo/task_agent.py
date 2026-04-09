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
from typing import Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


class JSONExtractor:
    """Robust JSON extraction from LLM responses with multiple fallback strategies."""
    
    @staticmethod
    def extract_from_tags(text: str) -> list[dict]:
        """Extract JSON objects from <json>...</json> blocks."""
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
                # Try with brace matching for nested structures
                json_obj = JSONExtractor._extract_with_brace_matching(inner)
                if json_obj:
                    results.append(json_obj)
        return results
    
    @staticmethod
    def _extract_with_brace_matching(text: str) -> dict | None:
        """Extract JSON using brace counting with string handling."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 1
        i = start_idx + 1
        in_string = False
        escape_next = False
        string_delim = None
        
        while i < len(text) and brace_count > 0:
            char = text[i]
            
            if escape_next:
                escape_next = False
            elif char == '\\' and in_string:
                escape_next = True
            elif char == '"' and not in_string:
                in_string = True
                string_delim = '"'
            elif char == '"' and in_string and string_delim == '"':
                in_string = False
                string_delim = None
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            i += 1
        
        if brace_count == 0:
            candidate = text[start_idx:i]
            for parser in [json.loads, JSONExtractor._parse_with_sanitization]:
                try:
                    return parser(candidate)
                except json.JSONDecodeError:
                    continue
        return None
    
    @staticmethod
    def _parse_with_sanitization(text: str) -> dict:
        """Parse JSON after applying sanitization fixes."""
        sanitized = JSONExtractor._sanitize_json(text)
        return json.loads(sanitized)
    
    @staticmethod
    def _sanitize_json(text: str) -> str:
        """Fix common JSON formatting issues from LLM outputs."""
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix single quotes for keys
        text = re.sub(r"(?<=[{,\s])'([^']+)'(?=\s*:)", r'"\1"', text)
        # Remove comments
        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text.strip()
    
    @staticmethod
    def extract_from_code_blocks(text: str) -> list[dict]:
        """Extract JSON from markdown code blocks."""
        results = []
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                results.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                # Try with sanitization
                try:
                    results.append(JSONExtractor._parse_with_sanitization(match))
                except json.JSONDecodeError:
                    continue
        return results
    
    @staticmethod
    def extract_response_value(text: str) -> str | None:
        """Direct extraction of response value using regex patterns."""
        patterns = [
            (r'"response"\s*:\s*(-?\d+(?:\.\d+)?)', lambda m: str(int(float(m.group(1))))),
            (r'"response"\s*:\s*"([^"]*)"', lambda m: m.group(1)),
            (r'"response"\s*:\s*(true|false|null)', lambda m: m.group(1)),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return converter(match)
                except (ValueError, TypeError):
                    return match.group(1)
        return None
    
    @staticmethod
    def extract_all(text: str) -> tuple[list[dict], str | None]:
        """Run all extraction strategies and return best results."""
        # Try tag extraction first
        results = JSONExtractor.extract_from_tags(text)
        if results:
            return results, None
        
        # Try code blocks
        results = JSONExtractor.extract_from_code_blocks(text)
        if results:
            return results, None
        
        # Try direct value extraction
        direct_value = JSONExtractor.extract_response_value(text)
        if direct_value is not None:
            return [], direct_value
        
        return [], None


# Backwards compatibility - module-level functions
def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
    results = JSONExtractor.extract_from_tags(text)
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract JSON using brace matching."""
    return JSONExtractor._extract_with_brace_matching(text)


def _sanitize_json_string(text: str) -> str:
    """Sanitize JSON string."""
    return JSONExtractor._sanitize_json(text)


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Extract JSON using regex patterns."""
    results = JSONExtractor.extract_from_code_blocks(text)
    return results or None


def _extract_response_value(text: str) -> str | None:
    """Extract response value directly."""
    return JSONExtractor.extract_response_value(text)


def _extract_response_from_code_blocks(text: str) -> str | None:
    """Extract response from code blocks."""
    # Look for code blocks with json tag
    json_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    
    for match in matches:
        value = JSONExtractor.extract_response_value(match)
        if value is not None:
            return value
        
        # Try to find just a number
        number_match = re.search(r'^\s*(-?\d+(?:\.\d+)?)\s*$', match.strip())
        if number_match:
            return number_match.group(1)
    
    return None


def _aggressive_json_clean(text: str) -> str:
    """Aggressively clean JSON string."""
    # Remove control characters
    cleaned = ''.join(char for char in text if char in '\t\n\r' or ord(char) >= 32)
    # Remove trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Escape special characters in strings
    result = []
    in_string = False
    string_delim = None
    escape_next = False
    
    for char in cleaned:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
            
        if not in_string:
            if char in '"\'':
                in_string = True
                string_delim = char
            result.append(char)
        else:
            if char == string_delim:
                in_string = False
                string_delim = None
                result.append(char)
            elif char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(char)
    
    return ''.join(result)


class TaskAgent:
    """Task agent that solves IMO grading problems with improved extraction and reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self._extraction_stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "fallback_extractions": 0,
            "failed_extractions": 0,
        }

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build an optimized grading prompt with clear structure."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert mathematical grader for IMO-level competition problems.

Your task is to evaluate a student's solution and assign an appropriate score based on the official grading guidelines.

## INPUTS

**Problem Domain:** {domain}

**Problem Statement:**
```
{problem}
```

**Official Solution:**
```
{solution}
```

**Grading Guidelines (Rubric):**
```
{grading_guidelines}
```

**Student's Answer:**
```
{student_answer}
```

## EVALUATION PROCESS

Follow this systematic approach:

1. **Problem Understanding**: Verify the student correctly interpreted the problem requirements.

2. **Approach Analysis**: 
   - Did they use the official solution method or a valid alternative?
   - Is their overall strategy sound?

3. **Step-by-Step Verification**:
   - Check each claim and calculation
   - Identify any logical gaps or errors
   - Note correct intermediate results

4. **Partial Credit Assessment**:
   - IMO problems award partial points for progress
   - Award credit for correct lemmas, even if the full solution isn't reached
   - Consider partial proofs of key claims

5. **Final Score Determination**:
   - Map the student's work to the rubric point-by-point
   - Sum the points earned
   - Verify the total matches the guidelines

## GRADING PRINCIPLES

- **Precision**: Deduct only for actual mathematical errors
- **Generosity**: Award partial credit generously for correct reasoning
- **Flexibility**: Accept valid alternative approaches
- **Completeness**: Check if claims are proved or merely stated

## OUTPUT FORMAT

Respond ONLY with a valid JSON object wrapped in <json> tags:

<json>
{{
    "response": <numerical_score>
}}
</json>

The response must be a single number matching the rubric's scoring system (typically 0-7 for IMO problems).

Examples:
- Full marks: <json>{{"response": 7}}</json>
- Partial credit: <json>{{"response": 3}}</json>
- No credit: <json>{{"response": 0}}</json>"""

    def _extract_prediction(self, raw_text: str) -> tuple[str, str]:
        """Extract prediction using the unified JSONExtractor.
        
        Returns:
            Tuple of (prediction, method_used)
        """
        self._extraction_stats["total_calls"] += 1
        
        # Use the unified extraction method
        json_objects, direct_value = JSONExtractor.extract_all(raw_text)
        
        # Try to get response from JSON objects
        if json_objects:
            for obj in reversed(json_objects):  # Check from last to first
                if isinstance(obj, dict) and "response" in obj:
                    self._extraction_stats["successful_extractions"] += 1
                    return str(obj["response"]), "json_extractor"
        
        # Use direct value if available
        if direct_value is not None:
            self._extraction_stats["successful_extractions"] += 1
            return str(direct_value), "direct_value"
        
        # Fallback: try legacy methods
        self._extraction_stats["fallback_extractions"] += 1
        
        # Try sanitization
        try:
            sanitized = JSONExtractor._sanitize_json(raw_text)
            sanitized_objects, _ = JSONExtractor.extract_all(sanitized)
            if sanitized_objects:
                for obj in reversed(sanitized_objects):
                    if isinstance(obj, dict) and "response" in obj:
                        return str(obj["response"]), "sanitized"
        except Exception:
            pass
        
        # Try code blocks specifically
        code_value = _extract_response_from_code_blocks(raw_text)
        if code_value is not None:
            return str(code_value), "code_block"
        
        self._extraction_stats["failed_extractions"] += 1
        return "None", "failed"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_grading_prompt(inputs)

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using improved extraction
        raw_text = msg_history[-1]["text"]
        
        # Log the raw response for debugging (truncated)
        preview = raw_text[:500].replace('\n', ' ')
        self.log_fn(f"Raw LLM response preview: {preview}...")
        
        prediction, method = self._extract_prediction(raw_text)
        self.log_fn(f"Extracted prediction: {prediction} (method: {method})")

        return str(prediction), msg_history
    
    def get_extraction_stats(self) -> dict:
        """Return extraction statistics for monitoring."""
        stats = self._extraction_stats.copy()
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_calls"]
        return stats
