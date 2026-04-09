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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
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
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                continue
    
    # Try to find plain JSON objects (looking for {"response": ...} pattern)
    # This handles cases where the LLM outputs JSON without proper tags
    plain_json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    plain_matches = re.findall(plain_json_pattern, text, re.DOTALL)
    for match in plain_matches:
        try:
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
        
        instruction = f"""You are an expert mathematical grader evaluating IMO-level solutions. Classify the student's solution into exactly ONE of four categories.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

=== CLASSIFICATION CRITERIA ===

Choose exactly ONE label:

**Correct**: Complete, rigorous, correct solution. ALL claims proven, ALL cases handled, NO gaps.

**Almost**: Structurally complete solution with only MINOR issues. 90-95% done. Main proof framework is sound, just needs small fixes (calculation error, one trivial case missing, minor technical flaw). The hard work is done.

**Partial**: Shows SUBSTANTIAL progress with KEY mathematical insights. Student found KEY invariants/lemmas central to the problem. Has VIABLE approach with MEANINGFUL progress beyond restating. An expert would say "they found the key idea, just need to finish."

**Incorrect**: NO meaningful mathematical progress. Just restating problem, trivial observations, wrong approach, no valuable insights. When in doubt, choose Incorrect.

=== CRITICAL DISTINCTIONS ===

**Almost vs Partial** (MOST IMPORTANT):
- "Almost" = Structurally complete, minor fixes only. Proof framework is done.
- "Partial" = Found key idea but significant work remains. Not structurally complete.

**Partial vs Incorrect**:
- "Partial" = Must have SUBSTANTIAL key insights (lemma, invariant, viable approach)
- "Incorrect" = No valuable insight. Be CONSERVATIVE - when in doubt, choose Incorrect.

=== DECISION PROCESS ===

1. Is it complete and correct? → "Correct"
2. Is it structurally complete with only minor issues? → "Almost"  
3. Does it show substantial progress with KEY insights? → "Partial"
4. Otherwise → "Incorrect"

=== EXAMPLES ===

- Only restating problem → "Incorrect"
- Found key invariant but didn't complete proof → "Partial"
- Complete proof with one small calculation error → "Almost"
- Complete rigorous proof with all cases → "Correct"
- Random attempts, no insight → "Incorrect"

Provide your evaluation in this exact JSON format:

<json>
{{
    "response": "LABEL: Your brief reasoning here"
}}
</json>

Where LABEL is one of: Correct, Almost, Partial, Incorrect

CRITICAL RULES:
1. Start with the exact label followed by a colon
2. Put your reasoning after the colon
3. Wrap everything in <json>...</json> tags
4. No text outside the JSON tags"""

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
        """Extract just the label (Correct/Almost/Partial/Incorrect) from text.
        
        Returns one of: "Correct", "Almost", "Partial", "Incorrect", or "None"
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Strategy 1: Look for explicit label at the start of the response
        label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)[\s:]', text_lower)
        if label_prefix_match:
            return label_prefix_match.group(1).capitalize()
        
        # Strategy 2: Look for explicit label at the start of any line
        for line in text_lower.split('\n'):
            line = line.strip()
            label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)[\s:]', line)
            if label_prefix_match:
                return label_prefix_match.group(1).capitalize()
        
        # Strategy 3: Look for labels in JSON-like format
        json_label_match = re.search(r'["\']?(response|label|classification|grade|evaluation|result)["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)', text_lower)
        if json_label_match:
            return json_label_match.group(2).capitalize()
        
        # Strategy 4: Look for labels in parentheses
        paren_match = re.search(r'[\(\[](correct|almost|partial|incorrect)[\)\]]', text_lower)
        if paren_match:
            return paren_match.group(1).capitalize()
        
        # Strategy 5: Look for labels followed by reasoning indicators
        reasoning_match = re.search(r'\b(correct|almost|partial|incorrect)\s*[-:]\s*', text_lower)
        if reasoning_match:
            return reasoning_match.group(1).capitalize()
        
        # Strategy 6: Look for labels in quotes
        quoted_match = re.search(r'["\'](correct|almost|partial|incorrect)["\']', text_lower)
        if quoted_match:
            return quoted_match.group(1).capitalize()
        
        # Strategy 7: Look for labels as standalone words
        # Check for "Almost" first (to avoid "Correct" matching inside words)
        almost_match = re.search(r'\balmost\b', text_lower)
        if almost_match:
            return "Almost"
        
        # Check for "Correct" (but not when preceded by "in")
        correct_match = re.search(r'(?<!in)\bcorrect\b', text_lower)
        if correct_match:
            return "Correct"
        
        # Check for "Partial"
        partial_match = re.search(r'\bpartial\b', text_lower)
        if partial_match:
            return "Partial"
        
        # Check for "Incorrect" last
        incorrect_match = re.search(r'\bincorrect\b', text_lower)
        if incorrect_match:
            return "Incorrect"
        
        return "None"
