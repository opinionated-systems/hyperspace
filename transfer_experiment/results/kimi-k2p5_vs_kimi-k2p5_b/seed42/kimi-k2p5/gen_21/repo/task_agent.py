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
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to classify the student's answer into exactly one of four categories: Correct, Almost, Partial, or Incorrect.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## CLASSIFICATION CATEGORIES (with detailed criteria):

**Correct** (10/10 points):
- Complete, rigorous solution with ALL steps correct
- No errors, gaps, or missing cases
- Proof is fully valid and would receive full marks
- ONLY use this if the solution is 100% flawless

**Almost** (7-9/10 points):
- Solution is SUBSTANTIALLY COMPLETE (main proof structure is done)
- Student reached the conclusion but made MINOR errors
- Examples of minor errors: small calculation mistakes, missing trivial case, notation issues, minor logical gaps that don't affect main result
- KEY TEST: If the student fixed only small errors, would it be Correct? If YES → Almost
- The solution shows the student understood the problem and solved the core challenge
- CRITICAL: The student MUST have reached the final conclusion/answer

**Partial** (3-6/10 points):
- Student made MEANINGFUL PROGRESS but solution is INCOMPLETE
- Found key insight or right approach but STOPPED before finishing
- Missing major proof components or final conclusion
- Has good ideas but didn't complete the main argument
- KEY TEST: Did the student finish the main proof? If NO → Partial
- Examples: Found invariant but didn't prove it works, set up approach but didn't execute, proved lemma but not main result, stopped at key insight without completing
- CRITICAL: The student did NOT reach the final conclusion/answer

**Incorrect** (0-2/10 points):
- NO meaningful progress toward solution
- Wrong approach, fundamental misunderstanding, or trivial observations only
- Solution is essentially empty, irrelevant, or completely flawed
- No key insights or correct approach identified
- Examples: Only restated problem, made false claims, used wrong method, no useful work

## CRITICAL DISTINCTIONS (PAY CLOSE ATTENTION):

**Partial vs Almost** (THIS IS THE MOST COMMON ERROR - BE CAREFUL!):
- Partial = INCOMPLETE (student stopped early, major gaps remain, NO final answer)
- Almost = FINISHED but has minor flaws (main work IS done, HAS final answer)
- DECISION TREE:
  1. Does the solution have a final answer/conclusion? 
     - NO → Partial (or Incorrect if no progress)
     - YES → Continue to question 2
  2. Are there errors in the solution?
     - YES (any errors) → Almost
     - NO (perfect) → Correct

**Almost vs Correct**:
- Correct = ZERO errors (absolutely perfect)
- Almost = ANY errors exist, even tiny ones
- Be STRICT: If you see ANY mistake, use Almost not Correct

**Incorrect vs Partial**:
- Incorrect = No meaningful insight or approach (just restating problem, random guesses)
- Partial = Has correct approach/insight but incomplete execution (found key idea, proved lemma, set up framework)
- Ask: "Did the student find a key insight or correct approach?"
  - If NO real insight → Incorrect
  - If YES but incomplete → Partial

## MANDATORY STEP-BY-STEP ANALYSIS:

You MUST follow this exact process before giving your answer:

**Step 1: Check for Final Answer/Conclusion**
- Does the student's solution end with a clear final answer or conclusion?
- Does it look like they stopped mid-proof or at a lemma?
- If NO final answer → Consider Partial or Incorrect
- If YES final answer → Consider Almost or Correct

**Step 2: Assess Quality of Work**
- Did they find a key insight/approach? (If NO → Incorrect)
- Is the main proof structure complete? (If NO → Partial)
- Are there ANY errors, even small ones? (If YES → Almost; If NO → Correct)

**Step 3: Verify Your Choice**
- If you chose Almost: Confirm they reached the conclusion AND have minor errors
- If you chose Partial: Confirm they have good ideas but NO final conclusion
- If you chose Incorrect: Confirm they have no real insight
- If you chose Correct: Confirm the solution is 100% perfect

## COMMON MISTAKES TO AVOID:

1. **Don't confuse Partial with Almost**: Partial means incomplete (no final answer). Almost means complete but flawed.

2. **Don't be too generous with Correct**: Only use Correct if you cannot find ANY flaw, no matter how small.

3. **Don't be too harsh with Partial**: If they found a key insight or made meaningful progress, it's Partial not Incorrect.

4. **Check the ending carefully**: Many solutions that look "Almost" are actually "Partial" because they stop before the final conclusion.

## EXAMPLES:

**Correct**: Complete proof, all steps valid, no gaps, full rigor, perfect final answer.

**Almost**: Student proved main theorem, reached correct final answer, but made a calculation error in one step, or missed checking one trivial case. The core solution is there but has minor flaws.

**Partial**: Student identified the right invariant and proved it exists (key insight!), but didn't show it leads to the solution. Or set up the induction framework but didn't complete the inductive step. Has good ideas but NO final answer.

**Incorrect**: Student only restated the problem, or made false claims without proof, or used a completely wrong method. No useful mathematical progress.

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
3. "Correct" is EXTREMELY RARE - only for 100% flawless solutions
4. "Almost" requires: (a) final answer reached, AND (b) minor errors present
5. "Partial" requires: (a) meaningful progress/insight, AND (b) NO final answer
6. "Incorrect" requires: no meaningful insight or approach"""

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
