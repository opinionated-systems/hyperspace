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


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple strategies.
    
    Tries multiple methods in order of preference:
    1. Extract from <json>...</json> tags
    2. Extract from ```json...``` code blocks
    3. Find JSON objects directly in text
    """
    # Strategy 1: <json> tags
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks
    json_code_pattern = r'```json\s*(.*?)```'
    matches = re.findall(json_code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: ``` code blocks (any language)
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Find JSON objects directly (curly braces)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    continue
    
    return None


def _normalize_prediction(raw_value) -> str:
    """Normalize a raw prediction value to one of the valid labels.
    
    Valid labels: "correct", "incorrect", "partial", "almost"
    """
    if raw_value is None:
        return "unknown"
    
    raw_str = str(raw_value).lower().strip()
    
    # Direct matches
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Check for exact matches with quotes removed
    clean_str = raw_str.strip('"\'')
    if clean_str in ["correct", "incorrect", "partial", "almost"]:
        return clean_str
    
    # Handle common variations
    if raw_str in ["true", "yes", "right", "valid", "1", "full", "complete"]:
        return "correct"
    if raw_str in ["false", "no", "wrong", "invalid", "0", "none", "error"]:
        return "incorrect"
    if raw_str in ["part", "partially", "incomplete", "half", "some"]:
        return "partial"
    if raw_str in ["almost correct", "close", "minor errors", "nearly"]:
        return "almost"
    
    # Check for substring matches (more specific first)
    if "almost" in raw_str:
        return "almost"
    if "partial" in raw_str:
        return "partial"
    # Be careful with "correct" - "incorrect" contains "correct"
    if "incorrect" in raw_str or "wrong" in raw_str:
        return "incorrect"
    if "correct" in raw_str:
        return "correct"
    
    return "unknown"


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to understand the rubric.
    
    The guidelines often contain markers like:
    - (Partial) - indicates what constitutes partial credit
    - (Almost) - indicates what constitutes almost correct
    - (Correct) - indicates what constitutes correct
    - (Incorrect) - indicates what constitutes incorrect
    """
    result = {
        "partial_indicators": [],
        "almost_indicators": [],
        "correct_indicators": [],
        "incorrect_indicators": []
    }
    
    if not guidelines:
        return result
    
    # Look for section markers
    partial_match = re.search(r'\(Partial\)(.*?)(?=\(Almost\)|\(Correct\)|\(Incorrect\)|$)', guidelines, re.DOTALL | re.IGNORECASE)
    if partial_match:
        result["partial_indicators"] = [line.strip() for line in partial_match.group(1).strip().split('\n') if line.strip()]
    
    almost_match = re.search(r'\(Almost\)(.*?)(?=\(Partial\)|\(Correct\)|\(Incorrect\)|$)', guidelines, re.DOTALL | re.IGNORECASE)
    if almost_match:
        result["almost_indicators"] = [line.strip() for line in almost_match.group(1).strip().split('\n') if line.strip()]
    
    correct_match = re.search(r'\(Correct\)(.*?)(?=\(Partial\)|\(Almost\)|\(Incorrect\)|$)', guidelines, re.DOTALL | re.IGNORECASE)
    if correct_match:
        result["correct_indicators"] = [line.strip() for line in correct_match.group(1).strip().split('\n') if line.strip()]
    
    incorrect_match = re.search(r'\(Incorrect\)(.*?)(?=\(Partial\)|\(Almost\)|\(Correct\)|$)', guidelines, re.DOTALL | re.IGNORECASE)
    if incorrect_match:
        result["incorrect_indicators"] = [line.strip() for line in incorrect_match.group(1).strip().split('\n') if line.strip()]
    
    return result


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
        # Extract key information from inputs for better prompting
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        # Parse grading guidelines to understand the rubric
        rubric = _parse_grading_guidelines(grading_guidelines)
        
        # Build detailed rubric context
        rubric_context = ""
        if rubric["correct_indicators"]:
            rubric_context += "\n\nSPECIFIC INDICATORS for CORRECT grade:\n" + "\n".join(f"  • {ind}" for ind in rubric["correct_indicators"])
        if rubric["almost_indicators"]:
            rubric_context += "\n\nSPECIFIC INDICATORS for ALMOST grade:\n" + "\n".join(f"  • {ind}" for ind in rubric["almost_indicators"])
        if rubric["partial_indicators"]:
            rubric_context += "\n\nSPECIFIC INDICATORS for PARTIAL grade:\n" + "\n".join(f"  • {ind}" for ind in rubric["partial_indicators"])
        if rubric["incorrect_indicators"]:
            rubric_context += "\n\nSPECIFIC INDICATORS for INCORRECT grade:\n" + "\n".join(f"  • {ind}" for ind in rubric["incorrect_indicators"])
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (IMO, Putnam, etc.).

Your task is to classify the student's answer into EXACTLY ONE of these four categories:

GRADE DEFINITIONS:
• "correct" - The answer is fully correct, complete, and rigorous. All key steps are present and correct.
• "almost" - The answer is nearly correct with only minor errors, typos, or small omissions that don't affect overall correctness.
• "partial" - The answer has some correct elements and shows meaningful progress, but is incomplete or has significant issues.
• "incorrect" - The answer is fundamentally wrong, contains major errors, or shows no meaningful progress.

=== PROBLEM STATEMENT ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== GRADING GUIDELINES ===
{grading_guidelines}{rubric_context}

=== STUDENT'S ANSWER ===
{student_answer}

=== GRADING PROCESS ===
Step 1: Identify what the problem asks and what the official solution states.
Step 2: Compare the student's answer to the official solution.
Step 3: Check against the specific indicators in the grading guidelines above.
Step 4: Select the grade that best matches based on the indicators.

IMPORTANT: Respond ONLY with this exact JSON format:
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The response field must contain EXACTLY one lowercase word: correct, almost, partial, or incorrect."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = "unknown"
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Try flexible JSON extraction
            extracted = _extract_json_flexible(last_message)
            if extracted:
                # Try to get the response value
                if isinstance(extracted, dict):
                    if "response" in extracted:
                        prediction = _normalize_prediction(extracted["response"])
                    elif len(extracted) == 1:
                        # If only one key, use its value
                        prediction = _normalize_prediction(list(extracted.values())[0])
                    else:
                        # Try to find a value that looks like a grade
                        for key, value in extracted.items():
                            normalized = _normalize_prediction(value)
                            if normalized in ["correct", "almost", "partial", "incorrect"]:
                                prediction = normalized
                                break
            
            # If still unknown, try direct text extraction with more patterns
            if prediction == "unknown":
                text_lower = last_message.lower()
                # Look for quoted labels
                if '"correct"' in text_lower or "'correct'" in text_lower:
                    prediction = "correct"
                elif '"almost"' in text_lower or "'almost'" in text_lower:
                    prediction = "almost"
                elif '"partial"' in text_lower or "'partial'" in text_lower:
                    prediction = "partial"
                elif '"incorrect"' in text_lower or "'incorrect'" in text_lower:
                    prediction = "incorrect"
                # Also check for standalone words at word boundaries
                elif re.search(r'\bcorrect\b', text_lower) and not re.search(r'\bincorrect\b', text_lower):
                    prediction = "correct"
                elif re.search(r'\balmost\b', text_lower):
                    prediction = "almost"
                elif re.search(r'\bpartial\b', text_lower):
                    prediction = "partial"
                elif re.search(r'\bincorrect\b', text_lower) or re.search(r'\bwrong\b', text_lower):
                    prediction = "incorrect"
            
            self.log_fn(f"Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
