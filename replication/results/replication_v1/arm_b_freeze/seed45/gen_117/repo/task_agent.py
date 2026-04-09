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
    Includes robust error recovery for malformed JSON with multiple fix strategies.
    Enhanced with better handling for nested structures and edge cases.
    """
    results = []
    search_from = 0
    
    def _try_parse_json(json_str: str) -> dict | None:
        """Try to parse JSON with multiple recovery strategies."""
        json_str = json_str.strip()
        if not json_str:
            return None
            
        # Strategy 1: Direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from first { to last }
        try:
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(json_str[start:end + 1])
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix trailing commas before } or ]
        try:
            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Fix single quotes to double quotes (but preserve escaped quotes)
        try:
            # Replace single quotes with double quotes, but be careful with apostrophes in text
            fixed = re.sub(r"(?<!\\)'", '"', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Fix unquoted keys (simple cases)
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Remove comments
        try:
            fixed = re.sub(r'//.*?\n', '\n', json_str)
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 7: Fix common escape sequence issues
        try:
            # Fix double-escaped newlines and other sequences
            fixed = json_str.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            # But preserve valid escaped quotes
            fixed = fixed.replace('\\"', '\"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 8: Extract only the first valid JSON object for malformed responses
        try:
            # Find the first complete JSON object
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        candidate = json_str[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass
            
        return None
    
    def _find_json_blocks(text: str, start_tag: str, end_tag: str) -> list[str]:
        """Find all blocks between start_tag and end_tag."""
        blocks = []
        search_pos = 0
        while True:
            start = text.find(start_tag, search_pos)
            if start == -1:
                break
            end = text.find(end_tag, start)
            if end == -1:
                break
            inner = text[start + len(start_tag):end].strip()
            if inner:
                blocks.append(inner)
            search_pos = end + len(end_tag)
        return blocks
    
    # First try to find <json>...</json> blocks
    json_blocks = _find_json_blocks(text, "<json>", "</json>")
    for block in json_blocks:
        parsed = _try_parse_json(block)
        if parsed:
            results.append(parsed)
    
    # If no <json> blocks found or none parsed successfully, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_code_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_code_blocks:
            parsed = _try_parse_json(block)
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
            # Find JSON-like structures with nested support using improved algorithm
            brace_positions = []
            for i, char in enumerate(text):
                if char == '{':
                    brace_positions.append((i, 'open'))
                elif char == '}':
                    brace_positions.append((i, 'close'))
            
            # Try to find valid JSON by matching braces
            if brace_positions:
                stack = []
                candidates = []
                for pos, kind in brace_positions:
                    if kind == 'open':
                        stack.append(pos)
                    elif kind == 'close' and stack:
                        start_pos = stack.pop()
                        candidates.append((start_pos, pos + 1))
                
                # Sort by length (longest first) to prioritize complete objects
                candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
                
                for start_pos, end_pos in candidates:
                    candidate = text[start_pos:end_pos]
                    parsed = _try_parse_json(candidate)
                    if parsed and isinstance(parsed, dict):
                        results.append(parsed)
                        break  # Take the first valid complete object
    
    return results or None


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

        # Build a structured, detailed prompt for better grading accuracy
        instruction = self._build_grading_prompt(
            domain, problem, solution, grading_guidelines, student_answer
        )

        # Try with retries for better reliability
        msg_history = []
        for attempt in range(self.max_retries + 1):
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=msg_history if attempt > 0 else [],
            )

            # Extract prediction from JSON
            prediction, reasoning = self._extract_prediction(msg_history)
            
            if prediction != "None":
                # Validate the prediction format
                if self._validate_prediction(prediction, grading_guidelines):
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Invalid prediction format, retrying...")
            
            if attempt < self.max_retries:
                self.log_fn(f"Retry {attempt + 1}: No valid JSON found, retrying with stronger prompt...")
                # Build a more explicit retry instruction
                instruction = self._build_retry_prompt(
                    domain, problem, solution, grading_guidelines, student_answer,
                    attempt + 1
                )

        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")

        return str(prediction), msg_history

    def _build_grading_prompt(
        self, domain: str, problem: str, solution: str, 
        grading_guidelines: str, student_answer: str
    ) -> str:
        """Build a comprehensive grading prompt with clear structure."""
        return f"""You are an expert {domain} grader evaluating student solutions with precision and consistency.

Your task is to analyze a student's answer and provide a fair, accurate grade based on the official solution and grading guidelines.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Instructions:

### Step 1: Analysis
Carefully analyze the student's answer:
- What mathematical concepts or techniques did they use?
- Did they follow a valid approach to solve the problem?
- What steps did they complete correctly?
- What errors or omissions did they make?
- How does their approach compare to the official solution?

### Step 2: Scoring
Based on the grading guidelines, determine:
- Which parts of the solution did they complete correctly?
- Are there partial credits to be awarded?
- What is the appropriate final grade?

### Step 3: Response Format
You MUST respond with a valid JSON object in the following format:

<json>
{{
    "reasoning": "Provide a detailed analysis of the student's work. Explain what they did correctly, what errors they made, and how you arrived at your grading decision. Be specific about which parts of the solution match or deviate from the official solution.",
    "response": "The final grade as specified in the grading guidelines (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score like '7/7', '3/7', etc.)"
}}
</json>

IMPORTANT:
- Your response field must match the expected answer format from the grading guidelines
- Be objective and consistent with the official solution
- Award partial credit where appropriate according to the guidelines"""

    def _build_retry_prompt(
        self, domain: str, problem: str, solution: str,
        grading_guidelines: str, student_answer: str,
        retry_num: int
    ) -> str:
        """Build a more explicit retry prompt when previous attempts failed."""
        return f"""You are an expert {domain} grader. This is retry attempt {retry_num}.

The previous response did not contain valid JSON. You MUST respond ONLY with a valid JSON object.

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## CRITICAL INSTRUCTIONS:
1. Analyze the student's answer carefully
2. Compare with the official solution
3. Apply the grading guidelines
4. Respond EXACTLY in this JSON format (no other text before or after):

<json>
{{
    "reasoning": "Your detailed analysis here...",
    "response": "The final grade here..."
}}
</json>

The JSON must be valid and properly formatted with double quotes around keys and string values."""

    def _validate_prediction(self, prediction: str, grading_guidelines: str) -> bool:
        """Validate that the prediction format matches expected grading format.
        
        Enhanced to handle more grading formats and edge cases.
        """
        if not prediction or prediction == "None":
            return False
        
        prediction_str = str(prediction).strip()
        
        # Check for common valid formats (case-insensitive)
        valid_patterns = [
            r'^Correct$', r'^Incorrect$', r'^Partially\s+Correct$',
            r'^True$', r'^False$',
            r'^Yes$', r'^No$',
            r'^Pass$', r'^Fail$',
            r'^\d+/\d+$',  # e.g., 7/7, 3/7
            r'^\d+$',  # e.g., 0, 1, 7
            r'^\d+\.\d+$',  # e.g., 3.5, 2.0
            r'^\d+\.\d+/\d+$',  # e.g., 3.5/7
            r'^\d+/\d+\.\d+$',  # e.g., 3/7.0
            r'^\d+\.\d+/\d+\.\d+$',  # e.g., 3.5/7.0
            r'^[A-F][+-]?$',  # Letter grades: A, B+, C-, etc.
            r'^\d+%$',  # Percentage: 85%
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, prediction_str, re.IGNORECASE):
                return True
        
        # If grading guidelines mention specific formats, check those
        guidelines_lower = grading_guidelines.lower()
        if "points" in guidelines_lower or "/" in guidelines_lower:
            # Likely expects a score format
            return bool(re.match(r'^[\d/\.\s]+$', prediction_str))
        
        if "percent" in guidelines_lower or "%" in guidelines_lower:
            # Likely expects a percentage
            return bool(re.match(r'^[\d\.%]+$', prediction_str))
        
        # Check for letter grade mentions in guidelines
        if any(grade in guidelines_lower for grade in ['grade', 'a', 'b', 'c', 'd', 'f']):
            return bool(re.match(r'^[A-F][+-]?$', prediction_str, re.IGNORECASE))
        
        # Accept any non-empty, non-whitespace string as potentially valid
        # (at least 1 character that's not whitespace)
        return len(prediction_str) > 0 and not prediction_str.isspace()

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced error handling.
        
        Returns:
            (prediction, reasoning) tuple
        """
        prediction = "None"
        reasoning = ""
        confidence = 0.0
        
        try:
            # Collect all text from messages to search for JSON
            all_texts = []
            
            # Try the last assistant message first (most likely to contain the answer)
            if msg_history:
                last_msg = msg_history[-1]
                if isinstance(last_msg, dict):
                    text = last_msg.get("text") or last_msg.get("content", "")
                    if text:
                        all_texts.append(text)
            
            # Then try all messages in reverse order (newest first)
            for msg in reversed(msg_history):
                if isinstance(msg, dict):
                    text = msg.get("text") or msg.get("content", "")
                    if text and text not in all_texts:
                        all_texts.append(text)
            
            # Search for JSON in all collected texts
            extracted = None
            for text in all_texts:
                extracted = _extract_jsons(text)
                if extracted:
                    break
            
            if extracted and isinstance(extracted, list) and len(extracted) > 0:
                # Use the last valid JSON object found (most likely the final answer)
                result = extracted[-1]
                if isinstance(result, dict):
                    # Try multiple possible keys for the response (ordered by priority)
                    response_keys = [
                        "response", "grade", "answer", "result", "assessment", 
                        "evaluation", "score", "verdict", "grading", "mark",
                        "prediction", "output", "decision", "conclusion"
                    ]
                    for key in response_keys:
                        if key in result and result[key] is not None:
                            val = result[key]
                            # Handle various types
                            if isinstance(val, (str, int, float, bool)):
                                prediction = str(val).strip()
                            elif isinstance(val, list) and len(val) > 0:
                                prediction = str(val[0]).strip()
                            else:
                                prediction = str(val).strip()
                            break
                    
                    # Extract reasoning if available
                    reasoning_keys = [
                        "reasoning", "analysis", "thought", "explanation", 
                        "rationale", "thinking", "evaluation", "assessment",
                        "justification", "commentary", "notes"
                    ]
                    for key in reasoning_keys:
                        if key in result and result[key] is not None:
                            val = result[key]
                            if isinstance(val, str):
                                reasoning = val.strip()
                            else:
                                reasoning = str(val).strip()
                            break
                    
                    # Extract confidence if available (new feature)
                    confidence_keys = ["confidence", "certainty", "confidence_score"]
                    for key in confidence_keys:
                        if key in result and result[key] is not None:
                            try:
                                confidence = float(result[key])
                                if 0 <= confidence <= 1:
                                    break
                            except (ValueError, TypeError):
                                continue
                    
                    self.log_fn(f"Extracted prediction: {prediction} (confidence: {confidence:.2f})")
                    if reasoning:
                        self.log_fn(f"Extracted reasoning length: {len(reasoning)} chars")
            
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {type(e).__name__}: {e}")
        
        return prediction, reasoning
