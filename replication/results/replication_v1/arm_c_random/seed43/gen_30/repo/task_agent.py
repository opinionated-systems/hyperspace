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
    Also handles markdown code blocks and raw JSON objects.
    Includes advanced cleanup for common LLM output issues.
    """
    results = []
    search_from = 0
    
    # First, try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        # Try to parse with progressive cleanup
        parsed = _try_parse_json_with_cleanup(inner)
        if parsed is not None:
            results.append(parsed)
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        md_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            parsed = _try_parse_json_with_cleanup(match.group(1).strip())
            if parsed is not None:
                results.append(parsed)
    
    # Last resort: try to find any JSON-like structure with "response" field
    if not results:
        # Look for objects with "response" key - use a more flexible pattern
        json_pattern = r'\{[^{}]*"response"[^{}]*(?:\}[^{}]*\})*\}'
        for match in re.finditer(json_pattern, text):
            parsed = _try_parse_json_with_cleanup(match.group())
            if parsed is not None:
                results.append(parsed)
    
    return results or None


def _try_parse_json_with_cleanup(text: str) -> dict | None:
    """Try to parse JSON with progressive cleanup strategies.
    
    Attempts multiple cleanup strategies to handle common LLM output issues:
    - Trailing commas
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Comments
    """
    cleanup_strategies = [
        lambda x: x,  # Try raw first
        lambda x: re.sub(r',(\s*[}\]])', r'\1', x),  # Remove trailing commas
        lambda x: x.replace("'", '"'),  # Replace single quotes with double
        lambda x: re.sub(r'\n\s*', ' ', x),  # Normalize newlines
        lambda x: re.sub(r',(\s*[}\]])', r'\1', x.replace("'", '"')),  # Combined
        lambda x: re.sub(r'//[^\n]*', '', x),  # Remove single-line comments
        lambda x: re.sub(r'/\*.*?\*/', '', x, flags=re.DOTALL),  # Remove multi-line comments
    ]
    
    for strategy in cleanup_strategies:
        try:
            cleaned = strategy(text)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning."""

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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions

1. **Initial Assessment**: Read the student's answer completely before forming any judgment.
2. **Step-by-Step Analysis**: Compare each step of the student's solution against the official solution.
   - Identify correct steps and valid reasoning
   - Note any errors, gaps, or logical flaws
   - Recognize creative alternative approaches that may be valid
3. **Partial Credit Evaluation**: Consider what mathematical understanding the student has demonstrated, even if the final answer is incorrect.
4. **Consistency Check**: Ensure your grading aligns with the provided guidelines and typical IMO standards.
5. **Final Grade Determination**: Assign a grade that reflects the student's demonstrated understanding and correctness.

## Grading Scale Reference
- **7 points**: Complete, correct solution with clear reasoning
- **6 points**: Minor flaw or omission in an otherwise correct solution
- **5 points**: Significant progress with one major gap or error
- **3-4 points**: Partial solution with some correct key ideas
- **1-2 points**: Minimal progress or significant misunderstanding
- **0 points**: No meaningful progress or completely wrong approach

## Response Format

Respond ONLY in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed analysis of the student's answer. Include: (1) What the student did correctly, (2) Any errors or gaps, (3) Why you assigned the specific grade",
    "response": "Your final grade as a number (0-7) or brief evaluation (e.g., '7', 'Partial credit: 3')"
}}
</json>

Important: The "response" field must contain ONLY the final grade. All explanation belongs in the "reasoning" field."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        
        # Log problem identifier for debugging
        problem_preview = inputs.get("problem", "")[:50] + "..." if inputs.get("problem") else "N/A"
        self.log_fn(f"Processing problem: {problem_preview}")

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with comprehensive error handling
        prediction = "None"
        reasoning = ""
        extraction_method = "none"
        
        try:
            # Try to extract from the last assistant message
            last_msg = msg_history[-1]["text"] if msg_history else ""
            
            if not last_msg:
                self.log_fn("Warning: Empty response from LLM")
                return str(prediction), msg_history
            
            # Primary extraction method
            extracted = _extract_jsons(last_msg)
            
            if extracted:
                last_json = extracted[-1]
                extraction_method = "json_extraction"
                
                if "response" in last_json:
                    prediction = last_json["response"]
                    # Clean up the prediction - extract just the numeric grade if possible
                    prediction = self._normalize_grade(prediction)
                if "reasoning" in last_json:
                    reasoning = last_json["reasoning"]
                
                # Log the reasoning for debugging
                if reasoning:
                    self.log_fn(f"Reasoning: {reasoning[:200]}...")
            else:
                # Fallback: try to extract grade directly from text
                extraction_method = "text_fallback"
                prediction = self._extract_grade_from_text(last_msg)
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            extraction_method = "error"
        
        self.log_fn(f"Extraction method: {extraction_method}, Prediction: {prediction}")

        return str(prediction), msg_history
    
    def _normalize_grade(self, grade: str) -> str:
        """Normalize grade to a standard format.
        
        Extracts numeric grades from various formats like:
        - "7 points" -> "7"
        - "Grade: 5" -> "5"
        - "Partial credit: 3" -> "3"
        """
        import re
        
        # Look for numeric grades (0-7)
        match = re.search(r'\b([0-7])\b', str(grade))
        if match:
            return match.group(1)
        
        # Return original if no normalization possible
        return str(grade).strip()
    
    def _extract_grade_from_text(self, text: str) -> str:
        """Extract grade from raw text when JSON parsing fails.
        
        Looks for common patterns indicating a grade.
        """
        import re
        
        # Look for explicit grade statements
        patterns = [
            r'grade[\s:]+([0-7])',
            r'score[\s:]+([0-7])',
            r'points?[\s:]+([0-7])',
            r'final grade[\s:]+([0-7])',
            r'\*\*([0-7])\s*points?\*\*',
            r'\*\*Grade:\s*([0-7])\*\*',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for standalone numbers 0-7 that might be grades
        # Avoid matching numbers that are part of larger numbers or dates
        match = re.search(r'(?:^|\s|\()([0-7])(?:\s*$|\s|\.|\))', text)
        if match:
            return match.group(1)
        
        return "None"
