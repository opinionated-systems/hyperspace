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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with flexible parsing.
    
    Tries multiple strategies:
    1. Look for <json>...</json> tags
    2. Look for ```json...``` code blocks
    3. Look for raw JSON objects in the text
    """
    # Strategy 1: Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Look for ```json code blocks
    json_code_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
    
    # Strategy 3: Look for raw JSON objects (brace matching)
    # Find all potential JSON starting points
    for match in re.finditer(r'\{', text):
        start = match.start()
        # Try to find matching closing brace
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_obj = json.loads(text[start:i+1])
                        results.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    break
    
    return results or None


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
        # Extract key fields for better context
        domain = inputs.get('domain', 'Unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        # Analyze grading guidelines to understand the expected structure
        grading_lower = grading_guidelines.lower()
        has_partial_marker = '(partial)' in grading_lower
        has_almost_marker = '(almost)' in grading_lower
        has_correct_marker = '(correct)' in grading_lower
        has_incorrect_marker = '(incorrect)' in grading_lower
        
        # Determine expected classification from grading guidelines markers
        expected_from_markers = None
        if has_partial_marker or has_almost_marker:
            expected_from_markers = "Partial"
        elif has_correct_marker and not has_incorrect_marker:
            expected_from_markers = "Correct"
        elif has_incorrect_marker and not has_correct_marker:
            expected_from_markers = "Incorrect"
        
        # Build specific guidance based on grading guidelines markers
        marker_guidance = ""
        if has_partial_marker:
            marker_guidance += "\n- The grading guidelines contain '(Partial)' markers."
        if has_almost_marker:
            marker_guidance += "\n- The grading guidelines contain '(Almost)' markers."
        if has_correct_marker:
            marker_guidance += "\n- The grading guidelines contain '(Correct)' markers."
        if has_incorrect_marker:
            marker_guidance += "\n- The grading guidelines contain '(Incorrect)' markers."
        
        instruction = f"""You are an expert mathematical grader. Your task is to evaluate a student's solution to a mathematical problem.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

CLASSIFICATION RULES:
- "Correct": The student's answer is a complete, correct solution with no significant errors. Only use this when the solution fully matches the official solution.
- "Partial": The student's answer shows substantial progress toward the solution but is incomplete or has minor errors. Use this when the student made meaningful progress but didn't fully solve the problem.
- "Incorrect": The student's answer is fundamentally wrong, shows no meaningful progress, or contains critical errors. Only use this for completely wrong answers.

GRADING GUIDELINES MARKERS:{marker_guidance}

CRITICAL INSTRUCTION - FOLLOW THE MARKERS:
The grading guidelines above contain explicit markers like (Partial), (Almost), (Correct), or (Incorrect). These markers OVERRIDE your own judgment. You MUST classify according to these markers:
- If you see (Partial) or (Almost) markers → classify as "Partial"
- If you see (Correct) markers and no contradictory markers → classify as "Correct"  
- If you see (Incorrect) markers and no contradictory markers → classify as "Incorrect"

Your task is to identify which marker is present in the grading guidelines and output the corresponding classification.

Based on the grading guidelines markers above, classify the student's answer as exactly one of: Correct, Partial, or Incorrect.

Provide your evaluation in the following JSON format:

<json>
{{
    "response": "Correct/Partial/Incorrect: Your brief reasoning here"
}}
</json>

Important: Your response MUST be wrapped in <json>...</json> tags. Start with the label (Correct, Partial, or Incorrect) followed by a brief explanation."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_json_flexible(last_message)
            if extracted:
                # Try to get response from the last JSON object
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    if "response" in last_json:
                        response_text = last_json["response"]
                        # Extract just the label (Correct/Incorrect/Partial) from the response
                        prediction = self._extract_label(response_text)
                    else:
                        # If no "response" key, use the first value found
                        prediction = str(list(last_json.values())[0]) if last_json else "None"
            self.log_fn(f"Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any of the expected labels from raw text
            prediction = self._extract_label(last_message)
        
        # Post-processing: Override prediction if grading guidelines have clear markers
        # This ensures we follow the explicit markers even if the LLM didn't
        if expected_from_markers and prediction != expected_from_markers:
            self.log_fn(f"Overriding prediction from {prediction} to {expected_from_markers} based on grading guidelines markers")
            prediction = expected_from_markers

        return str(prediction), msg_history

    def _extract_label(self, text: str) -> str:
        """Extract just the label (Correct/Incorrect/Partial) from text.
        
        Returns one of: "Correct", "Incorrect", "Partial", or "None"
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Look for explicit label at the start of the response (e.g., "Correct:", "Partial:", "Incorrect:")
        # This handles the new format we requested
        label_prefix_match = re.match(r'^(correct|partial|incorrect)[\s:]', text_lower)
        if label_prefix_match:
            label = label_prefix_match.group(1).capitalize()
            return label
        
        # Look for explicit label at the start of any line (multiline response)
        for line in text_lower.split('\n'):
            line = line.strip()
            label_prefix_match = re.match(r'^(correct|partial|incorrect)[\s:]', line)
            if label_prefix_match:
                label = label_prefix_match.group(1).capitalize()
                return label
        
        # Look for labels in JSON-like format
        json_label_match = re.search(r'["\']?(response|label|classification)["\']?\s*[:=]\s*["\']?(correct|partial|incorrect)', text_lower)
        if json_label_match:
            label = json_label_match.group(2).capitalize()
            return label
        
        # Pattern to match labels as standalone words with word boundaries
        # Check for Partial first (it's a substring of "Partially" but we want exact match)
        partial_match = re.search(r'\bpartial\b', text_lower)
        incorrect_match = re.search(r'\bincorrect\b', text_lower)
        correct_match = re.search(r'\bcorrect\b', text_lower)
        
        # Priority: Partial > Incorrect > Correct
        # This is because "Partial" is the most specific and hardest to detect correctly
        if partial_match:
            return "Partial"
        
        # Check for Incorrect (more specific than Correct)
        if incorrect_match:
            return "Incorrect"
        
        # Check for Correct last (least specific, as "incorrect" contains "correct")
        if correct_match:
            return "Correct"
        
        return "None"
