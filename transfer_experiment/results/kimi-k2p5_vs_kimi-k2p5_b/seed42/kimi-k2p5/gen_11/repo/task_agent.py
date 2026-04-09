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
            rf'"{category}"',
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category.capitalize()
    
    # Check for standalone words in the last 100 lines (increased for better coverage)
    lines = text_lower.split('\n')
    for line in reversed(lines[-100:]):  # Check last 100 lines
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

## Classification Guide - READ CAREFULLY:

**Correct**: The student's answer is FULLY CORRECT, COMPLETE, and RIGOROUS.
- All key steps from the official solution are present
- The logic is sound with no gaps or errors
- The proof/solution is complete and would receive full marks
- The conclusion is correct and properly justified
- Use this ONLY when you are confident the solution is complete

**Incorrect**: The student's answer is FUNDAMENTALLY FLAWED or shows MINIMAL PROGRESS.
- The approach is completely wrong or misguided
- Critical logical errors invalidate the main argument
- The solution fails to make meaningful progress (just restates problem or has trivial observations)
- The conclusion is wrong or not reached
- Use this when the student has NOT demonstrated understanding of the key insights

**Partial**: The student has made SIGNIFICANT PROGRESS but the solution is INCOMPLETE.
- Found key invariants, lemmas, or insights from the official solution
- Has the right approach but missing crucial steps or cases
- Made substantial progress but proof is unfinished
- Demonstrates understanding but solution has major gaps
- Use this when student found important pieces but didn't complete the proof
- This is for work that shows genuine understanding but is clearly incomplete
- EXAMPLES: Found the right invariant but didn't complete the proof; identified key lemma but didn't prove it; has right strategy but missing multiple steps

**Almost**: The solution is NEARLY COMPLETE with ONLY MINOR issues.
- The main proof structure is correct and complete
- Only small technical errors, calculation mistakes, or notation issues
- Missing only trivial cases that don't affect the main argument
- Core logic is sound, just needs minor fixes
- The solution would be "Correct" if tiny errors were fixed
- Use this when the solution is essentially done but has minor flaws
- EXAMPLES: Small arithmetic error in final calculation; missing a trivial base case; minor notation inconsistency; one small logical gap that doesn't affect main result

## Critical Distinctions - READ THIS CAREFULLY:

1. "Partial" vs "Almost": 
   - "Partial" = major gaps remain, proof is unfinished (e.g., found invariant but didn't complete proof)
   - "Almost" = proof is essentially done, only tiny issues remain (e.g., minor calculation error in final step)
   - KEY TEST: If removing the errors would make it "Correct", use "Almost". If major work is still needed, use "Partial".
   - "Partial" means the student stopped halfway; "Almost" means they finished but made small mistakes

2. "Partial" vs "Incorrect":
   - "Partial" = found key insights, made real progress (e.g., identified the right invariant/approach)
   - "Incorrect" = wrong approach or minimal meaningful progress (e.g., completely wrong method)
   - KEY TEST: Did the student identify the right strategy? If yes → "Partial", if no → "Incorrect".
   - IMPORTANT: Give benefit of doubt to "Partial" when student shows genuine understanding

3. "Almost" vs "Correct":
   - "Almost" = has minor flaws (e.g., small error in one step)
   - "Correct" = completely flawless
   - KEY TEST: Are there ANY errors at all? If yes → "Almost", if no → "Correct".

## Your Task - STEP BY STEP:

Step 1: Identify the KEY INSIGHTS and MAIN PROOF STEPS from the official solution.

Step 2: Analyze what the student has actually proven:
- What lemmas/invariants did they find?
- What proof techniques did they use correctly?
- What is missing or wrong?

Step 3: Check the grading guidelines - they indicate what partial credit was awarded.
- If guidelines mention "(Partial)" criteria, the ground truth is likely Partial
- If guidelines mention "(Almost)" criteria, the ground truth is likely Almost
- Pay special attention to phrases like "minor error", "small mistake", "trivial case" → suggests "Almost"
- Look for phrases like "incomplete", "missing steps", "unfinished" → suggests "Partial"

Step 4: Apply these rules:
- If student found key insights but proof is incomplete → "Partial"
- If solution is nearly perfect with tiny errors → "Almost"  
- If solution is completely rigorous → "Correct"
- If approach is wrong or minimal progress → "Incorrect"

Step 5: Be BALANCED in your classification:
- When in doubt between "Correct" and "Almost", choose "Almost" (be conservative about "Correct")
- When in doubt between "Partial" and "Incorrect", prefer "Partial" if key insights are demonstrated
- Use "Almost" when the solution is clearly nearly complete with only minor issues
- Look for explicit "(Almost)" or "(Partial)" markers in the grading guidelines

Step 6: FINAL CHECK - Verify your classification:
- Re-read the student's answer one more time
- Confirm your classification matches the evidence
- Make sure you haven't missed any critical errors or gaps
- Check if the grading guidelines explicitly mention "(Almost)" or "(Partial)"

Step 7: DECISION FRAMEWORK - Use this to decide:
- Does the solution have the RIGHT APPROACH? (Yes → Partial/Almost/Correct, No → Incorrect)
- Is the proof COMPLETE? (No → Partial, Yes → check for errors)
- Are there ANY ERRORS? (Yes → Almost, No → Correct)
- Are there MAJOR GAPS? (Yes → Incorrect or Partial, depending on insights found)

Step 8: SPECIAL GUIDANCE for "Almost" and "Partial":
- "Almost": Look for solutions that are 90-99% complete with only minor issues
- "Partial": Look for solutions that are 30-70% complete with significant progress but major gaps remain
- If you see a solution that found the key insight but didn't finish the proof → "Partial"
- If you see a solution that finished the proof but has a small calculation error → "Almost"

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
5. Be GENEROUS about "Partial" - use when student shows genuine understanding and key insights
6. Be ACTIVE about "Almost" - use when solution is 90%+ complete with only minor issues
7. Use "Incorrect" only when the approach is fundamentally wrong or progress is minimal
8. Your ENTIRE output must be ONLY the JSON block - no other text allowed"""

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
