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


def _extract_json_from_code_blocks(text: str) -> list[dict] | None:
    """Extract JSON objects from ```json...``` code blocks."""
    results = []
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_raw(text: str) -> list[dict] | None:
    """Extract raw JSON objects from text (objects wrapped in {})."""
    results = []
    # Find all JSON-like structures
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
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
    # Use a more sophisticated approach to find the final classification
    text_lower = text.lower()
    
    # Look for explicit classification statements
    classification_patterns = [
        r'classification\s*:\s*(correct|incorrect|partial|almost)',
        r'grade\s*:\s*(correct|incorrect|partial|almost)',
        r'category\s*:\s*(correct|incorrect|partial|almost)',
        r'label\s*:\s*(correct|incorrect|partial|almost)',
        r'response\s*:\s*(correct|incorrect|partial|almost)',
        r'final\s*(?:answer|classification|grade)\s*[:\-]?\s*(correct|incorrect|partial|almost)',
        r'(?:the\s+)?(?:student\s+)?(?:answer|solution|work)\s+(?:is\s+)?(correct|incorrect|partial|almost)',
        r'(?:i\s+)?(?:would\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?(correct|incorrect|partial|almost)',
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
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category.capitalize()
    
    # Last resort: check for standalone words, but be more careful
    # Only match if the word appears near the end of the text (final conclusion)
    lines = text_lower.split('\n')
    for line in reversed(lines[-20:]):  # Check last 20 lines
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        for category in categories:
            # Match as a whole word, not part of another word
            if re.search(rf'\b{category}\b', line):
                # Make sure it's not part of a larger word
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

**Almost**: The solution is NEARLY COMPLETE with ONLY MINOR issues.
- The main proof structure is correct and complete
- Only small technical errors, calculation mistakes, or notation issues
- Missing only trivial cases that don't affect the main argument
- Core logic is sound, just needs minor fixes
- Use this SPARINGLY - only when the solution is essentially correct

## Critical Distinctions:

1. "Partial" vs "Almost": 
   - "Partial" = major gaps remain, proof is unfinished (e.g., found invariant but didn't complete proof)
   - "Almost" = proof is essentially done, only tiny issues remain (e.g., minor calculation error in final step)

2. "Partial" vs "Incorrect":
   - "Partial" = found key insights, made real progress (e.g., identified the right invariant/approach)
   - "Incorrect" = wrong approach or minimal meaningful progress (e.g., completely wrong method)

3. "Almost" vs "Correct":
   - "Almost" = has minor flaws (e.g., small error in one step)
   - "Correct" = completely flawless

## Your Task - STEP BY STEP:

Step 1: Identify the KEY INSIGHTS and MAIN PROOF STEPS from the official solution.

Step 2: Analyze what the student has actually proven:
- What lemmas/invariants did they find?
- What proof techniques did they use correctly?
- What is missing or wrong?

Step 3: Check the grading guidelines - they indicate what partial credit was awarded.

Step 4: Apply these STRICT rules:
- If student found key insights but proof is incomplete → "Partial"
- If solution is nearly perfect with tiny errors → "Almost"  
- If solution is completely rigorous → "Correct"
- If approach is wrong or minimal progress → "Incorrect"

Step 5: Be CONSERVATIVE:
- When in doubt between "Correct" and "Partial", choose "Partial"
- When in doubt between "Partial" and "Incorrect", choose based on whether key insights were found
- Only use "Almost" if the solution is clearly nearly complete

Step 6: FINAL CHECK - Verify your classification:
- Re-read the student's answer one more time
- Confirm your classification matches the evidence
- Make sure you haven't missed any critical errors or gaps

You must classify the student's answer into EXACTLY ONE of these four categories: "Correct", "Incorrect", "Partial", or "Almost".

Respond in this exact JSON format:
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
4. Be STRICT about "Correct" - only use for truly complete solutions
5. Be GENEROUS about "Partial" - recognize when student found key insights
6. Use "Almost" SPARINGLY - only for solutions that are essentially correct
7. Always provide your final classification in the JSON format above"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            extracted_prediction = extract_prediction(last_message)
            if extracted_prediction is not None:
                prediction = extracted_prediction
                self.log_fn(f"Successfully extracted prediction: {prediction}")
            else:
                self.log_fn(f"Failed to extract prediction from response: {last_message[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate the prediction is one of the allowed categories
        valid_categories = ["Correct", "Incorrect", "Partial", "Almost"]
        if prediction not in valid_categories:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid category. Defaulting to 'Partial'.")
            prediction = "Partial"  # Default to Partial as a conservative choice

        return str(prediction), msg_history
