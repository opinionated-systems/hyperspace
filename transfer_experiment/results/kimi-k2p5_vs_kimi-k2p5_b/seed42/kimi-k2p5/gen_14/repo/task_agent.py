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
    4. Direct text search for category labels with context analysis
    
    Returns the "response" field value, or None if extraction fails.
    """
    # Try <json> tags first
    extracted = _extract_jsons(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                return str(item["response"])
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                return str(item["response"])
    
    # Try raw JSON
    extracted = _extract_json_raw(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                return str(item["response"])
    
    # Fallback: search for explicit category mentions in the text
    text_lower = text.lower()
    
    # Look for explicit classification statements with higher priority
    classification_patterns = [
        r'classification\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'grade\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'category\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'label\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'response\s*[:=]\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'final\s*(?:answer|classification|grade|decision)\s*[:\-]?\s*(correct|incorrect|partial|almost)',
        r'(?:the\s+)?(?:student\s+)?(?:answer|solution|work)\s+(?:is\s+)?(correct|incorrect|partial|almost)',
        r'(?:i\s+)?(?:would\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:this\s+is\s+)?["\']?(correct|incorrect|partial|almost)["\']?(?:\s+classification)?',
        r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'\*\*(correct|incorrect|partial|almost)\*\*',  # Markdown bold
        r'\bverdict\s*[:=]?\s*(correct|incorrect|partial|almost)\b',
        r'\b(outcome|result)\s*[:=]?\s*(correct|incorrect|partial|almost)\b',
    ]
    
    for pattern in classification_patterns:
        match = re.search(pattern, text_lower)
        if match:
            category = match.group(1).lower()
            return category.capitalize()
    
    # Look for the exact category labels with word boundaries
    categories = ["correct", "incorrect", "partial", "almost"]
    for category in categories:
        # Check for quoted versions first (more reliable)
        patterns = [
            rf'"{category}"',
            rf"'{category}'",
            rf'`{category}`',
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category.capitalize()
    
    # Check for standalone words in the last 150 lines (increased for better coverage)
    lines = text_lower.split('\n')
    for line in reversed(lines[-150:]):  # Check last 150 lines
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//') or line.startswith('*'):
            continue
        for category in categories:
            # Match as a whole word with word boundaries
            if re.search(rf'\b{category}\b', line):
                # Make sure it's not part of a larger word
                if not re.search(rf'{category}[a-z]', line) and not re.search(rf'[a-z]{category}', line):
                    return category.capitalize()
    
    # Very last resort: check anywhere in the text for standalone words
    for category in categories:
        if re.search(rf'\b{category}\b', text_lower):
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
        
        instruction = f"""You are an expert mathematical problem solver and grader for IMO (International Mathematical Olympiad) problems.

Your task is to grade a student's answer to a mathematical problem based on the provided solution and grading guidelines.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Guide with EXAMPLES:

**Correct**: The student's answer is FULLY CORRECT, COMPLETE, and RIGOROUS.
- All key steps from the official solution are present
- The logic is sound with no gaps or errors
- The proof/solution is complete and would receive full marks
- Use this ONLY when you are confident the solution is complete and flawless
- Example: Student proves all required cases with correct logic

**Incorrect**: The student's answer is FUNDAMENTALLY FLAWED or shows MINIMAL PROGRESS.
- The approach is completely wrong or misguided
- Critical logical errors invalidate the main argument
- The solution fails to make meaningful progress (just restates problem or has trivial observations)
- The conclusion is wrong or not reached
- Use this when the student has NOT demonstrated understanding of the key insights
- Example: Student uses completely wrong method or makes no real progress

**Partial**: The student has made SIGNIFICANT PROGRESS but the solution is INCOMPLETE.
- Found key invariants, lemmas, or insights from the official solution
- Has the right approach but missing crucial steps or cases
- Made substantial progress but proof is unfinished
- Demonstrates understanding but solution has major gaps remaining
- Use this when student found important pieces but didn't complete the proof
- Example: Found the right invariant but didn't complete the proof; identified key lemma but didn't prove it; has right strategy but missing multiple steps

**Almost**: The solution is NEARLY COMPLETE with ONLY MINOR issues.
- The main proof structure is correct and complete
- Only small technical errors, calculation mistakes, or notation issues
- Missing only trivial cases that don't affect the main argument
- Core logic is sound, just needs minor fixes
- The solution would be "Correct" if tiny errors were fixed
- Use this when the solution is essentially done but has minor flaws
- Example: Small arithmetic error in final calculation; missing a trivial base case; minor notation inconsistency; one small logical gap that doesn't affect main result

## QUICK DECISION TREE (Use this!):

1. Does the solution have the RIGHT APPROACH? 
   - No → "Incorrect"
   - Yes → Continue to step 2

2. Is the proof COMPLETE (all cases covered, all steps justified)?
   - No → "Partial" (they had the right idea but didn't finish)
   - Yes → Continue to step 3

3. Are there ANY errors (even tiny ones)?
   - Yes → "Almost" (finished but with minor flaws)
   - No → "Correct" (flawless)

## KEY DISTINCTIONS (CRITICAL - Focus on these!):

**"Partial" vs "Almost" (MOST IMPORTANT - This is where errors happen!):**
- "Partial" = proof is INCOMPLETE, major work still needed (student stopped halfway through)
- "Almost" = proof is ESSENTIALLY DONE, only tiny fixes needed (student finished but made small mistakes)
- KEY TEST: Count the amount of work remaining:
  - If more than 10-20% of the proof is missing → "Partial"
  - If less than 10% is missing (just small fixes) → "Almost"
- "Partial" means the student is still in the middle of solving
- "Almost" means the student has essentially solved it but made minor errors
- BE GENEROUS with "Almost" - when in doubt between Partial and Almost, choose "Almost"

**"Partial" vs "Incorrect":**
- "Partial" = student found key insights, made real progress on right track
- "Incorrect" = wrong approach or minimal meaningful progress
- KEY TEST: Did student identify the right strategy? Yes → "Partial", No → "Incorrect"

**"Almost" vs "Correct":**
- "Almost" = has minor flaws (small error in one step)
- "Correct" = completely flawless
- KEY TEST: Are there ANY errors at all? Yes → "Almost", No → "Correct"

## GRADING GUIDELINES HINTS:
- If guidelines mention "(Partial)" criteria → likely "Partial"
- If guidelines mention "(Almost)" criteria → likely "Almost"
- Phrases like "minor error", "small mistake", "trivial case" → suggests "Almost"
- Phrases like "incomplete", "missing steps", "unfinished" → suggests "Partial"
- If the solution is 80%+ complete → consider "Almost" over "Partial"

## YOUR TASK:

Step 1: Identify KEY INSIGHTS from the official solution.

Step 2: Analyze what the student actually did:
- Did they find the right approach?
- What did they prove correctly?
- What is missing or wrong?
- Estimate the percentage of the solution that is complete (0%, 25%, 50%, 75%, 90%, 100%)

Step 3: Apply the decision tree above to classify.

Step 4: CRITICAL - Before finalizing, ask yourself:
- If the solution is 80-95% complete with minor errors, did I choose "Almost" (not "Partial")?
- If the student found the right approach but stopped early, did I choose "Partial" (not "Incorrect")?
- If the solution is flawless, did I choose "Correct"?
- If the approach is fundamentally wrong, did I choose "Incorrect"?

Step 5: Double-check your classification against the examples and distinctions.

You must classify the student's answer into EXACTLY ONE of these four categories: "Correct", "Incorrect", "Partial", or "Almost".

## CRITICAL: Your ENTIRE response must be ONLY the JSON block below. Do not include any other text, explanation, or formatting outside the JSON tags.

<json>
{{
    "response": "Correct"
}}
</json>

OR

<json>
{{
    "response": "Incorrect"
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
    "response": "Almost"
}}
</json>

IMPORTANT RULES:
1. Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost"
2. Do not add any explanation in the response field - only the label
3. Make sure to include the <json> and </json> tags around your JSON response
4. Be CONSERVATIVE about "Correct" - only use for truly complete, flawless solutions
5. Be GENEROUS about "Partial" - use when student shows genuine understanding
6. Be VERY GENEROUS about "Almost" - this is the most underused category! Use when solution is 80%+ complete with minor issues
7. Use "Incorrect" only when the approach is fundamentally wrong or progress is minimal
8. Your ENTIRE output must be ONLY the JSON block - no other text allowed

## COMMON MISTAKES TO AVOID:
- Don't default to "Partial" for solutions that are nearly complete - use "Almost" instead
- "Almost" should be used for solutions that would get 7-8/10 points, not "Partial"
- "Partial" is for solutions that would get 3-5/10 points (significant progress but major gaps)
- When the student has the right approach and is 80%+ done, always choose "Almost" over "Partial"
"""

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
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid category. Defaulting to 'Partial'.")
            prediction = "Partial"  # Default to Partial as a balanced choice when uncertain
        
        return str(prediction), msg_history
