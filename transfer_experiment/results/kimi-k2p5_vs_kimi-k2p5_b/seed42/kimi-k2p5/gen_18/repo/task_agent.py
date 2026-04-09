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
            # Handle potential double curly braces from f-string formatting
            if inner.startswith('{{') and inner.endswith('}}'):
                inner = inner[1:-1]
            # Also handle escaped braces
            inner_clean = inner.replace('{{', '{').replace('}}', '}')
            results.append(json.loads(inner_clean))
        except json.JSONDecodeError:
            # Try with more aggressive cleaning
            try:
                # Remove any leading/trailing whitespace and newlines
                inner_clean = inner.strip().replace('\n', ' ').replace('\r', '')
                # Try to find JSON object within the text
                json_start = inner_clean.find('{')
                json_end = inner_clean.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    inner_clean = inner_clean[json_start:json_end+1]
                    results.append(json.loads(inner_clean))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON objects from ```json...``` code blocks."""
    results = []
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try with more aggressive cleaning
            try:
                inner_clean = match.strip().replace('\n', ' ').replace('\r', '')
                json_start = inner_clean.find('{')
                json_end = inner_clean.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    inner_clean = inner_clean[json_start:json_end+1]
                    results.append(json.loads(inner_clean))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_raw(text: str) -> list[dict] | None:
    """Extract raw JSON objects from text (objects wrapped in {})."""
    results = []
    # Find all JSON-like structures with improved pattern
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            # Try with cleaning
            try:
                cleaned = match.strip().replace('\n', ' ').replace('\r', '')
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
    
    # Also try a simpler approach: look for {"response": ...} patterns
    simple_pattern = r'\{\s*"response"\s*:\s*"[^"]+"\s*\}'
    simple_matches = re.findall(simple_pattern, text, re.DOTALL)
    for match in simple_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and "response" in parsed:
                # Avoid duplicates
                if parsed not in results:
                    results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return results or None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects
    4. Direct text search for category labels
    
    Returns the "response" field value, or None if extraction fails.
    """
    valid_categories = {"correct", "incorrect", "partial", "almost"}
    text_lower = text.lower()
    
    # Try <json> tags first
    extracted = _extract_jsons(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                val = str(item["response"]).strip().lower()
                if val in valid_categories:
                    return val.capitalize()
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                val = str(item["response"]).strip().lower()
                if val in valid_categories:
                    return val.capitalize()
    
    # Try raw JSON
    extracted = _extract_json_raw(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                val = str(item["response"]).strip().lower()
                if val in valid_categories:
                    return val.capitalize()
    
    # Look for explicit classification patterns (highest priority patterns)
    classification_patterns = [
        # JSON-like response field patterns
        r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'["\']?response["\']?\s*:\s*["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        # Explicit declarations
        r'classification\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'grade\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'category\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'label\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'final\s*(?:answer|classification|grade|decision)\s*[:\-]?\s*(correct|incorrect|partial|almost)',
        r'(?:the\s+)?(?:student\s+)?(?:answer|solution|work)\s+(?:is\s+)?(correct|incorrect|partial|almost)',
        r'(?:i\s+)?(?:would\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:this\s+is\s+)?["\']?(correct|incorrect|partial|almost)["\']?(?:\s+classification)?',
        r'\*\*(correct|incorrect|partial|almost)\*\*',
        r'\bverdict\s*[:=]?\s*(correct|incorrect|partial|almost)\b',
        r'\b(outcome|result)\s*[:=]?\s*(correct|incorrect|partial|almost)\b',
    ]
    
    for pattern in classification_patterns:
        match = re.search(pattern, text_lower)
        if match:
            category = match.group(1).lower()
            if category in valid_categories:
                return category.capitalize()
    
    # Look for quoted category labels
    for category in valid_categories:
        patterns = [
            rf'"{category}"',
            rf"'{category}'",
            rf'`{category}`',
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category.capitalize()
    
    # Check last 50 lines for standalone category words
    lines = text_lower.split('\n')
    for line in reversed(lines[-50:]):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//') or line.startswith('*'):
            continue
        for category in valid_categories:
            if re.search(rf'\b{category}\b', line):
                # Verify it's not part of a larger word
                if not re.search(rf'{category}[a-z]', line) and not re.search(rf'[a-z]{category}', line):
                    return category.capitalize()
    
    return None


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
        # Extract fields from inputs for better prompt construction
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Classify the student's answer into exactly one of four categories.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## CLASSIFICATION CATEGORIES:

**Correct**: Complete, flawless, rigorous solution. 100% perfect with zero issues. Full marks.

**Almost**: Nearly complete solution with ONLY minor issues. Student FINISHED the main work but made small errors (calculation, notation, trivial case). 85-99% complete.

**Partial**: Significant progress but INCOMPLETE. Student found key insights but STOPPED EARLY or missed major components. Has right approach but didn't finish. 3-6/10 marks.

**Incorrect**: No meaningful progress. Wrong approach, fundamental flaws, or only trivial observations. 0-2/10 marks.

## DECISION WORKFLOW (Follow Exactly):

1. **Did student make meaningful progress?** (found key insight, right approach)
   - NO → **Incorrect**
   - YES → Continue

2. **Did student FINISH the solution?** (reached conclusion, main proof complete)
   - NO (stopped early, major gaps) → **Partial**
   - YES → Continue

3. **Are there ANY errors/issues?** (even tiny ones)
   - YES → **Almost**
   - NO → **Correct**

## KEY DISTINCTION: Partial vs Almost
- **Partial** = INCOMPLETE (student stopped, didn't finish main proof)
- **Almost** = FINISHED with minor errors (done but has small mistakes)
- Test: Would it be Correct if student just fixed small errors? Yes → Almost, No → Partial

## OUTPUT FORMAT - ONLY JSON:

<json>
{{
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "response": "Almost"
}}
</json>

OR

<json>
{{
    "response": "Partial"
}}
</json>

OR

<json>
{{
    "response": "Incorrect"
}}
</json>

**RULES:**
1. Output ONLY the JSON block - no other text
2. Use ONLY: "Correct", "Almost", "Partial", "Incorrect"
3. "Correct" only for 100% flawless solutions
4. "Partial" = incomplete (stopped early)
5. "Almost" = finished with minor errors"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = None
        try:
            # Try to get text from the last message
            last_message = ""
            if msg_history and len(msg_history) > 0:
                last_message = msg_history[-1].get("text", "")
                if not last_message and "content" in msg_history[-1]:
                    last_message = msg_history[-1].get("content", "")
            
            if last_message:
                extracted_prediction = extract_prediction(last_message)
                if extracted_prediction is not None:
                    prediction = extracted_prediction
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                else:
                    self.log_fn(f"Failed to extract prediction from response: {last_message[:200]}...")
            else:
                self.log_fn("No message text found in history")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate the prediction is one of the allowed categories
        valid_categories = ["Correct", "Incorrect", "Partial", "Almost"]
        
        # Normalize prediction: strip whitespace and capitalize
        if prediction is not None:
            prediction = prediction.strip().capitalize()
        
        if prediction not in valid_categories:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid category. Defaulting to 'None'.")
            prediction = "None"
        
        return str(prediction), msg_history
