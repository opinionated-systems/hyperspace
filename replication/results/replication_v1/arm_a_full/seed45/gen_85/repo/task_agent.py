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
    
    Args:
        text: The text containing <json> tags to parse.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with multiple fallback strategies.
    
    Tries multiple patterns in order of reliability.
    Enhanced to handle more edge cases and malformed JSON.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None
        
    results = []
    
    # Strategy 1: <json> tags (highest priority - exact format)
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
            # Try to fix common JSON issues
            try:
                # Fix trailing commas
                fixed = re.sub(r',\s*}', '}', inner)
                fixed = re.sub(r',\s*]', ']', fixed)
                # Fix single quotes
                fixed = fixed.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    if results:
        return results
    
    # Strategy 2: ```json code blocks
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',\s*}', '}', match.strip())
                fixed = re.sub(r',\s*]', ']', fixed)
                fixed = fixed.replace("'", '"')
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    if results:
        return results
    
    # Strategy 3: Look for JSON objects directly using brace matching
    start_indices = [m.start() for m in re.finditer(r'\{', text)]
    
    for start in start_indices:
        try:
            brace_count = 0
            end = -1
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[start:]):
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
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = start + i + 1
                            break
            
            if end > start:
                obj_str = text[start:end]
                try:
                    results.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    try:
                        fixed = re.sub(r',\s*}', '}', obj_str)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        fixed = fixed.replace("'", '"')
                        results.append(json.loads(fixed))
                    except json.JSONDecodeError:
                        continue
        except (json.JSONDecodeError, ValueError):
            continue
    
    return results or None


def _extract_json_robust(text: str) -> list[dict] | None:
    """Most robust JSON extraction with multiple fallback strategies.
    
    This is the primary extraction function that tries multiple
    strategies in order of reliability.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        A list of parsed JSON objects, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: <json> tags (original format)
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Flexible extraction with enhanced error recovery
    results = _extract_json_flexible(text)
    if results:
        return results
    
    # Strategy 3: Try to extract from markdown-style responses
    # Look for patterns like "response": "correct" or "grade": "partial"
    markdown_patterns = [
        r'response[:\s]+["\']?(correct|incorrect|partial)["\']?',
        r'grade[:\s]+["\']?(correct|incorrect|partial)["\']?',
        r'evaluation[:\s]+["\']?(correct|incorrect|partial)["\']?',
    ]
    for pattern in markdown_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [{"response": match.group(1)}]
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Truncate very long inputs to prevent context overflow
        max_len = 8000
        problem = problem[:max_len] if len(problem) > max_len else problem
        solution = solution[:max_len] if len(solution) > max_len else solution
        student_answer = student_answer[:max_len] if len(student_answer) > max_len else student_answer
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions to International Mathematical Olympiad (IMO) style competition problems.

Your task is to carefully analyze the student's answer and provide a rigorous evaluation according to the official solution and grading guidelines.

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

Think through this step-by-step:

1. **Understand the Problem**: What is being asked? What are the key concepts and theorems involved?

2. **Analyze the Official Solution**: What is the correct approach? What is the final answer? What are the critical steps that must be present?

3. **Review the Student's Answer**: What approach did the student take? What is their final answer? Did they show all necessary work?

4. **Compare and Evaluate**: Does the student's answer match the official solution? Consider:
   - Is the final answer numerically/algebraically equivalent to the official solution?
   - Did the student demonstrate correct mathematical reasoning?
   - Are there any logical gaps or errors in the student's work?
   - Did the student use appropriate methods and theorems?
   - Is the solution complete or partial?
   - Did the student justify their steps with clear reasoning?
   - For partial credit: identify which specific steps were correct vs incorrect

5. **Assign Grade**: Based on your analysis, provide your evaluation using the rubric below.

## Grading Rubric

- **Correct**: Fully correct with proper reasoning, all steps justified, final answer matches official solution.
- **Incorrect**: Wrong answer, critical errors, incorrect methods, or final answer doesn't match.
- **Partial**: Some correct elements but incomplete, minor errors, lacks justification, or only partially matches.

## Response Format

Respond ONLY in this exact JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis here. Be thorough and mention specific mathematical steps, theorems used, and any errors found.",
    "response": "correct" | "incorrect" | "partial"
}}
</json>

CRITICAL RULES:
1. The "response" field must contain ONLY one of these three exact lowercase values: "correct", "incorrect", or "partial"
2. No other text, explanations, or formatting allowed in the "response" field
3. Use "partial" when the student has some correct work but not a complete solution
4. Use "incorrect" when the answer is fundamentally wrong or missing critical components
5. Use "correct" only when the solution is fully correct with proper justification"""
        
        return prompt

    def _extract_prediction(self, msg_history: list[dict]) -> str:
        """Extract prediction from message history.
        
        Tries JSON extraction first, then falls back to regex patterns.
        Enhanced to handle more edge cases and structured outputs.
        """
        if not msg_history:
            return "None"
        
        last_text = msg_history[-1].get("text", "")
        if not last_text:
            return "None"
        
        # Try JSON extraction first
        extracted = _extract_json_robust(last_text)
        if extracted:
            last_obj = extracted[-1]
            
            # Check for common response fields in priority order
            priority_keys = ["response", "grade", "evaluation", "score", "answer", "result", "verdict"]
            for key in priority_keys:
                if key in last_obj:
                    value = last_obj[key]
                    if isinstance(value, (str, int, float, bool)):
                        return str(value)
            
            # Check for correctness boolean
            if "correct" in last_obj:
                correct_val = last_obj["correct"]
                if isinstance(correct_val, bool):
                    return "correct" if correct_val else "incorrect"
                return str(correct_val)
            
            # Check for nested grading objects
            if "grading" in last_obj and isinstance(last_obj["grading"], dict):
                grading = last_obj["grading"]
                for key in ["grade", "evaluation", "verdict", "result"]:
                    if key in grading:
                        value = grading[key]
                        if isinstance(value, (str, int, float, bool)):
                            return str(value)
        
        # Fallback: regex patterns for common fields (more comprehensive)
        patterns = [
            (r'"response"\s*:\s*"([^"]+)"', 1),
            (r'"grade"\s*:\s*"([^"]+)"', 1),
            (r'"evaluation"\s*:\s*"([^"]+)"', 1),
            (r'"verdict"\s*:\s*"([^"]+)"', 1),
            (r'"correct"\s*:\s*(true|false)', 1),
            (r'Grade:\s*(correct|incorrect|partial)', 1),
            (r'Evaluation:\s*(correct|incorrect|partial)', 1),
            (r'Verdict:\s*(correct|incorrect|partial)', 1),
        ]
        for pattern, group in patterns:
            match = re.search(pattern, last_text, re.IGNORECASE)
            if match:
                return match.group(group).lower()
        
        # Text-based extraction with improved patterns
        text_lower = last_text.lower()
        
        # Check for explicit grading statements
        grading_patterns = [
            (r'the\s+answer\s+is\s+(correct|incorrect|partial)', 1),
            (r'this\s+is\s+(correct|incorrect|partial)', 1),
            (r'grade[\s:]+(correct|incorrect|partial)', 1),
            (r'evaluation[\s:]+(correct|incorrect|partial)', 1),
            (r'verdict[\s:]+(correct|incorrect|partial)', 1),
        ]
        for pattern, group in grading_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(group)
        
        # Check for incorrect first (to catch "not correct")
        if re.search(r'\b(incorrect|wrong|false|not correct|not valid)\b', text_lower):
            return "incorrect"
        
        if re.search(r'\b(correct|right|true|valid|accepted|accurate)\b', text_lower):
            return "correct"
        
        if re.search(r'\b(partial|partially|incomplete|partial credit)\b', text_lower):
            return "partial"
        
        # Return first 200 chars as fallback
        return last_text.strip()[:200] if last_text.strip() else "None"

    def _normalize_prediction(self, prediction: str) -> str:
        """Normalize prediction to standard format for evaluation.
        
        Enhanced to handle more edge cases and provide robust normalization.
        """
        if not prediction or prediction == "None":
            return "None"
        
        pred_lower = prediction.lower().strip()
        pred_clean = pred_lower.strip('"\'.,;: ')
        
        # Exact matches (expanded)
        exact_matches = {
            "correct": "correct",
            "incorrect": "incorrect", 
            "partial": "partial",
            "true": "correct",
            "false": "incorrect",
            "right": "correct",
            "wrong": "incorrect",
            "valid": "correct",
            "invalid": "incorrect",
            "accepted": "correct",
            "rejected": "incorrect",
            "accurate": "correct",
            "inaccurate": "incorrect",
            "full": "correct",
            "full credit": "correct",
            "no credit": "incorrect",
            "partial credit": "partial",
            "some credit": "partial",
            "half": "partial",
            "half credit": "partial",
        }
        
        if pred_clean in exact_matches:
            return exact_matches[pred_clean]
        
        # Handle score format
        if pred_lower.startswith("score:") or "/" in pred_lower:
            return prediction
        
        # Check for numeric patterns (e.g., "7/7", "3/7", "0/7")
        numeric_match = re.search(r'(\d+)\s*/\s*(\d+)', pred_lower)
        if numeric_match:
            num, denom = int(numeric_match.group(1)), int(numeric_match.group(2))
            if denom == 0:  # Avoid division by zero
                return prediction
            if num == denom:
                return "correct"
            elif num == 0:
                return "incorrect"
            else:
                return "partial"
        
        # Check for decimal scores (e.g., "0.5", "1.0", "0.0")
        decimal_match = re.search(r'^(\d+\.\d+|\d+)$', pred_clean)
        if decimal_match:
            try:
                val = float(decimal_match.group(1))
                if val >= 0.9:
                    return "correct"
                elif val <= 0.1:
                    return "incorrect"
                else:
                    return "partial"
            except ValueError:
                pass
        
        # Keyword detection - check incorrect first (more comprehensive)
        incorrect_patterns = [
            r'\b(incorrect|wrong|false|not correct|not valid|invalid|rejected|inaccurate)\b',
            r'\b(not right|not accurate|not accepted)\b',
            r'\b(error|mistake|faulty)\b',
        ]
        for pattern in incorrect_patterns:
            if re.search(pattern, pred_lower):
                return "incorrect"
        
        # Check for partial patterns
        partial_patterns = [
            r'\b(partial|partially|incomplete|partial credit|some credit|half|mostly)\b',
            r'\b(partly|somewhat|partially correct)\b',
        ]
        for pattern in partial_patterns:
            if re.search(pattern, pred_lower):
                return "partial"
        
        # Check for correct patterns (more comprehensive)
        correct_patterns = [
            r'\b(correct|right|true|valid|accepted|accurate|perfect|complete)\b',
            r'\b(fully correct|entirely correct|completely correct)\b',
            r'\b(full credit|full marks|all correct)\b',
        ]
        for pattern in correct_patterns:
            if re.search(pattern, pred_lower):
                return "correct"
        
        return prediction

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate required inputs
        required_keys = ["problem", "solution", "student_answer"]
        missing_keys = [k for k in required_keys if k not in inputs or not inputs[k]]
        if missing_keys:
            error_msg = f"Error: Missing required inputs: {missing_keys}"
            return error_msg, [{"role": "assistant", "text": error_msg}]
        
        instruction = self._build_prompt(inputs)
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            error_msg = f"Error: LLM call failed: {e}"
            return error_msg, [{"role": "assistant", "text": error_msg}]

        # Extract and normalize prediction
        raw_prediction = self._extract_prediction(msg_history)
        prediction = self._normalize_prediction(raw_prediction)

        return str(prediction), msg_history
