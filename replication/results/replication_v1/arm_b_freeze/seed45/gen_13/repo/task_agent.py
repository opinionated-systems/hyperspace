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
        except json.JSONDecodeError:
            # Try to extract JSON from within the content
            try:
                # Find first { and last }
                json_start = inner.find("{")
                json_end = inner.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    results.append(json.loads(inner[json_start:json_end + 1]))
            except json.JSONDecodeError:
                continue
    
    # If no <json> blocks found, try markdown code blocks
    if not results:
        # Try ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*(.*?)```', text, re.DOTALL)
        for block in json_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                continue
        
        # Try ``` ... ``` blocks without language specifier
        if not results:
            code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
            for block in code_blocks:
                try:
                    results.append(json.loads(block.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Try bare JSON objects as fallback - improved pattern for nested structures
        if not results:
            # Find JSON-like structures with balanced braces
            potential_jsons = _extract_balanced_json(text)
            for pj in potential_jsons:
                try:
                    results.append(json.loads(pj))
                except json.JSONDecodeError:
                    continue
    
    return results or None


def _extract_balanced_json(text: str) -> list[str]:
    """Extract JSON objects with balanced braces from text.
    
    This handles nested JSON structures better than simple regex.
    """
    results = []
    i = 0
    while i < len(text):
        # Find the start of a potential JSON object
        if text[i] == '{':
            start = i
            brace_count = 1
            i += 1
            in_string = False
            escape_next = False
            
            while i < len(text) and brace_count > 0:
                char = text[i]
                
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                i += 1
            
            if brace_count == 0:
                # Found a balanced JSON object
                json_str = text[start:i]
                # Basic validation: must contain at least one key-value pair
                if '"' in json_str or "'" in json_str:
                    results.append(json_str)
        else:
            i += 1
    
    return results


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

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

        # Build a structured, comprehensive prompt
        instruction = self._build_grading_prompt(
            domain=domain,
            problem=problem,
            solution=solution,
            grading_guidelines=grading_guidelines,
            student_answer=student_answer,
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with improved logic
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Log reasoning if available
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")

        return str(prediction), msg_history

    def _build_grading_prompt(
        self,
        domain: str,
        problem: str,
        solution: str,
        grading_guidelines: str,
        student_answer: str,
    ) -> str:
        """Build a comprehensive grading prompt with clear structure."""
        
        # Determine expected response format from grading guidelines
        expected_format = self._infer_response_format(grading_guidelines)
        
        prompt = f"""You are an expert {domain} grader evaluating student solutions with precision and consistency.

Your task is to grade a student's answer by comparing it against the official solution and strictly following the grading guidelines.

=== PROBLEM STATEMENT ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING PROCESS ===
Follow these steps in your analysis:

1. UNDERSTAND: Identify the key concepts, theorems, and techniques required to solve this problem.

2. ANALYZE OFFICIAL SOLUTION: Break down the official solution into key steps and identify what constitutes a complete, correct answer.

3. EVALUATE STUDENT WORK: 
   - Check if the student's final answer matches the official solution
   - Analyze their reasoning process and logical steps
   - Identify any errors, omissions, or misconceptions
   - Note any alternative valid approaches

4. APPLY GRADING CRITERIA: Map the student's work against the grading guidelines explicitly.

5. DETERMINE GRADE: Based on your analysis, assign the appropriate grade.

=== RESPONSE FORMAT ===
You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed step-by-step analysis covering: (1) understanding of the problem, (2) evaluation of the student's approach, (3) specific errors or correct steps identified, (4) how grading guidelines were applied",
    "response": "{expected_format}"
}}
</json>

Important:
- The "response" field must contain ONLY the final grade assessment
- Be objective and consistent with the grading guidelines
- If partial credit is applicable, explain the breakdown in reasoning"""

        return prompt

    def _infer_response_format(self, grading_guidelines: str) -> str:
        """Infer the expected response format from grading guidelines."""
        guidelines_lower = grading_guidelines.lower()
        
        # Check for numeric score patterns
        if any(pattern in guidelines_lower for pattern in ["0-", "1-", "2-", "3-", "4-", "5-", "6-", "7-", "8-", "9-", "10-", "out of", "points", "score"]):
            return "Numeric score (e.g., '7/10' or '5')"
        
        # Check for letter grade patterns
        if any(grade in guidelines_lower for grade in ["a+", "a-", "b+", "b-", "c+", "c-", "d+", "d-", "f grade"]):
            return "Letter grade (e.g., 'A', 'B+', 'C')"
        
        # Check for binary/categorical patterns
        if any(word in guidelines_lower for word in ["correct", "incorrect", "true", "false", "pass", "fail"]):
            return "Binary assessment (e.g., 'Correct' or 'Incorrect')"
        
        # Default to flexible format
        return "The final grade/assessment (e.g., 'Correct', 'Partially Correct', 'Incorrect', or a numeric score)"

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with robust fallback logic."""
        prediction = "None"
        reasoning = ""
        
        if not msg_history:
            return prediction, reasoning
        
        # Collect all potential JSON objects from all messages
        all_extracted = []
        for msg in msg_history:
            text = msg.get("text", "")
            if text:
                extracted = _extract_jsons(text)
                if extracted:
                    all_extracted.extend(extracted)
        
        # Also try the raw response content if available
        for msg in msg_history:
            content = msg.get("content", "")
            if content and isinstance(content, str):
                extracted = _extract_jsons(content)
                if extracted:
                    all_extracted.extend(extracted)
        
        if not all_extracted:
            # Fallback: try to extract from the last message text directly
            last_text = msg_history[-1].get("text", "") if msg_history else ""
            if last_text:
                # Try to find any quoted strings that might be grades
                grade_patterns = [
                    r'"response"\s*:\s*"([^"]+)"',
                    r'"grade"\s*:\s*"([^"]+)"',
                    r'"answer"\s*:\s*"([^"]+)"',
                    r'"result"\s*:\s*"([^"]+)"',
                    r'"assessment"\s*:\s*"([^"]+)"',
                    r'"evaluation"\s*:\s*"([^"]+)"',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_text)
                    if match:
                        prediction = match.group(1)
                        break
            return prediction, reasoning
        
        # Use the last valid JSON object (most recent response)
        result = all_extracted[-1]
        
        # Extract prediction with priority order
        response_keys = ["response", "grade", "answer", "result", "assessment", "evaluation", "score"]
        for key in response_keys:
            if key in result:
                prediction = result[key]
                break
        
        # Extract reasoning with priority order
        reasoning_keys = ["reasoning", "analysis", "thought", "explanation", "rationale", "thinking"]
        for key in reasoning_keys:
            if key in result:
                reasoning = result[key]
                break
        
        return prediction, reasoning
