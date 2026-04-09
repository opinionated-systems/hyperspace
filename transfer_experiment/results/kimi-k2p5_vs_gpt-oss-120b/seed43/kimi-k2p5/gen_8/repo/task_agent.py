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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks like ```json...```.
    
    Also handles plain JSON objects that might not be wrapped in tags.
    """
    results = []
    
    # Try to find JSON in markdown code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    
    # Try to find plain JSON objects (looking for {"response": ...} pattern)
    # This handles cases where the LLM outputs JSON without proper tags
    plain_json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    plain_matches = re.findall(plain_json_pattern, text, re.DOTALL)
    for match in plain_matches:
        try:
            # Reconstruct the JSON object
            results.append({"response": match})
        except Exception:
            continue
    
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with flexible parsing.
    
    Tries multiple strategies:
    1. Look for <json>...</json> tags
    2. Look for ```json...``` code blocks
    3. Look for plain JSON objects with "response" field
    4. Look for raw JSON objects in the text
    """
    results = []
    
    # Strategy 1: Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Try markdown code blocks and plain JSON
    results = _extract_json_from_markdown(text)
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
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer, points

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better context
        domain = inputs.get('domain', 'Unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
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
- "Correct": The student's answer is a complete, correct solution with no significant errors. The solution fully proves all claims, handles all cases, and matches the official solution's conclusion. Use this ONLY when the student has solved the problem completely.
- "Partial": The student's answer shows substantial progress toward the solution but is incomplete, has gaps in reasoning, or contains minor errors. The student must have made meaningful mathematical progress - found key invariants, proved significant lemmas, or developed a viable approach. Use this when the student was on the right track but didn't complete the proof.
- "Incorrect": The student's answer is fundamentally wrong, shows no meaningful progress, contains critical errors, or completely misses the key insights needed to solve the problem. Use this when the student's approach is flawed or they made no substantial progress.
- "Almost": The student's answer is very close to correct but has minor verification mistakes or small gaps that prevent it from being fully correct.

Carefully analyze the student's answer against the official solution and grading guidelines. Consider:
1. Did the student identify the key mathematical concepts?
2. Is the reasoning logically sound?
3. Are there gaps or errors in the proof?
4. Does the solution handle all cases?

Provide your evaluation in the following JSON format:

<json>
{{
    "response": "Correct: Your brief reasoning here"
}}
</json>

OR

<json>
{{
    "response": "Partial: Your brief reasoning here"
}}
</json>

OR

<json>
{{
    "response": "Incorrect: Your brief reasoning here"
}}
</json>

OR

<json>
{{
    "response": "Almost: Your brief reasoning here"
}}
</json>

CRITICAL: Your response MUST start with the exact label (Correct, Partial, Incorrect, or Almost) followed by a colon and then your reasoning. The entire response must be wrapped in <json>...</json> tags. Do not include any text outside the JSON tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            
            # First try to extract label directly from the text (handles cases where LLM doesn't use proper JSON)
            direct_label = self._extract_label(last_message)
            if direct_label != "None":
                prediction = direct_label
                self.log_fn(f"Directly extracted prediction: {prediction}")
            else:
                # Try JSON extraction
                extracted = _extract_json_flexible(last_message)
                if extracted:
                    # Try to get response from the last JSON object
                    last_json = extracted[-1]
                    if isinstance(last_json, dict):
                        if "response" in last_json:
                            response_text = last_json["response"]
                            # Extract just the label (Correct/Incorrect/Partial/Almost) from the response
                            prediction = self._extract_label(response_text)
                        else:
                            # If no "response" key, try to find any string value that looks like a label
                            for key, value in last_json.items():
                                if isinstance(value, str):
                                    label = self._extract_label(value)
                                    if label != "None":
                                        prediction = label
                                        break
                            else:
                                # If no label found in values, use the first value as string
                                prediction = str(list(last_json.values())[0]) if last_json else "None"
                self.log_fn(f"Extracted prediction from JSON: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any of the expected labels from raw text
            try:
                last_message = msg_history[-1]["text"] if msg_history else ""
                prediction = self._extract_label(last_message)
            except Exception:
                prediction = "None"

        return str(prediction), msg_history

    def _extract_label(self, text: str) -> str:
        """Extract just the label (Correct/Incorrect/Partial/Almost) from text.
        
        Returns one of: "Correct", "Incorrect", "Partial", "Almost", or "None"
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Look for explicit label at the start of the response (e.g., "Correct:", "Partial:", "Incorrect:", "Almost:")
        # This handles the new format we requested
        label_prefix_match = re.match(r'^(correct|partial|incorrect|almost)[\s:]', text_lower)
        if label_prefix_match:
            label = label_prefix_match.group(1).capitalize()
            return label
        
        # Look for explicit label at the start of any line (multiline response)
        for line in text_lower.split('\n'):
            line = line.strip()
            label_prefix_match = re.match(r'^(correct|partial|incorrect|almost)[\s:]', line)
            if label_prefix_match:
                label = label_prefix_match.group(1).capitalize()
                return label
        
        # Look for labels in JSON-like format with quotes
        json_label_match = re.search(r'["\']?(response|label|classification|grade|evaluation)["\']?\s*[:=]\s*["\']?(correct|partial|incorrect|almost)', text_lower)
        if json_label_match:
            label = json_label_match.group(2).capitalize()
            return label
        
        # Look for labels in parentheses like "(Correct)" or "[Partial]"
        paren_match = re.search(r'[\(\[](correct|partial|incorrect|almost)[\)\]]', text_lower)
        if paren_match:
            label = paren_match.group(1).capitalize()
            return label
        
        # Look for labels followed by reasoning indicators
        reasoning_match = re.search(r'\b(correct|partial|incorrect|almost)\s*[-:]\s*', text_lower)
        if reasoning_match:
            label = reasoning_match.group(1).capitalize()
            return label
        
        # Pattern to match labels as standalone words with word boundaries
        # Check for Almost first (most specific)
        almost_match = re.search(r'\balmost\b', text_lower)
        # Check for Partial (it's a substring of "Partially" but we want exact match)
        partial_match = re.search(r'\bpartial\b', text_lower)
        # Check for Incorrect (more specific than Correct)
        incorrect_match = re.search(r'\bincorrect\b', text_lower)
        # Check for Correct last (least specific, as "incorrect" contains "correct")
        correct_match = re.search(r'\bcorrect\b', text_lower)
        
        # Priority: Almost > Partial > Incorrect > Correct
        # This is because "Almost" and "Partial" are the most specific
        if almost_match:
            return "Almost"
        
        if partial_match:
            return "Partial"
        
        # Check for Incorrect (more specific than Correct)
        if incorrect_match:
            return "Incorrect"
        
        # Check for Correct last (least specific, as "incorrect" contains "correct")
        if correct_match:
            return "Correct"
        
        return "None"
