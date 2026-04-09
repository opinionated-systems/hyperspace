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
        r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?',
        r'classification\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'grade\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'category\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'label\s*[:=]\s*(correct|incorrect|partial|almost)',
        r'final\s*(?:answer|classification|grade|decision)\s*[:\-]?\s*(correct|incorrect|partial|almost)',
        r'(?:the\s+)?(?:student\s+)?(?:answer|solution|work)\s+(?:is\s+)?(correct|incorrect|partial|almost)',
        r'(?:i\s+)?(?:would\s+)?(?:classify|grade|label)\s+(?:this\s+)?(?:as\s+)?["\']?(correct|incorrect|partial|almost)["\']?',
        r'(?:this\s+is\s+)?["\']?(correct|incorrect|partial|almost)["\']?(?:\s+classification)?',
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
    
    # Check for standalone words in the last 200 lines (increased for better coverage)
    lines = text_lower.split('\n')
    for line in reversed(lines[-200:]):  # Check last 200 lines
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

## CLASSIFICATION RUBRIC - READ CAREFULLY:

### **Correct** (Full Marks - 10/10)
- The solution is COMPLETE, FLAWLESS, and RIGOROUS
- All required cases are covered
- All logical steps are justified
- No gaps, no errors, no missing pieces
- Use ONLY when you would give full marks

### **Almost** (High Marks - 7-9/10)  
- The solution is NEARLY COMPLETE with ONLY MINOR issues
- Main proof structure is correct and essentially done
- Small technical errors, calculation mistakes, or notation issues
- Missing only trivial cases that don't affect the main argument
- Core logic is sound, just needs minor fixes
- **WHEN TO USE**: Solution is 80-95% complete with minor flaws
- **EXAMPLES**: Small arithmetic error; missing trivial base case; minor notation issue; one small logical gap

### **Partial** (Medium Marks - 3-6/10)
- The student made SIGNIFICANT PROGRESS but solution is INCOMPLETE
- Found key invariants, lemmas, or insights from official solution
- Has the RIGHT APPROACH but missing crucial steps or cases
- Demonstrates genuine understanding but major gaps remain
- **WHEN TO USE**: Student found the right strategy but stopped halfway through
- **EXAMPLES**: Found right invariant but didn't complete proof; identified key lemma but didn't prove it; has right idea but missing multiple steps

### **Incorrect** (Low/No Marks - 0-2/10)
- The approach is FUNDAMENTALLY FLAWED or shows MINIMAL PROGRESS
- Wrong method or misguided strategy
- Critical logical errors invalidate the argument
- No meaningful progress (just restates problem or trivial observations)
- **WHEN TO USE**: Student did NOT demonstrate understanding of key insights
- **EXAMPLES**: Completely wrong method; no real progress; conclusion is wrong

## DECISION WORKFLOW - FOLLOW THIS EXACTLY:

**STEP 1: Check for Correct Approach**
- Does the student understand what the problem is asking?
- Did they identify the right strategy/method?
- If NO → classify as **"Incorrect"** and STOP
- If YES → continue to Step 2

**STEP 2: Estimate Completion Percentage**
- What percentage of the solution is complete? (0%, 25%, 50%, 75%, 90%, 100%)
- If less than 70% complete → classify as **"Partial"** and STOP
- If 70% or more complete → continue to Step 3

**STEP 3: Check for Errors**
- Are there ANY errors, even tiny ones?
- Are all cases covered completely?
- If ANY errors exist → classify as **"Almost"** and STOP
- If completely flawless → classify as **"Correct"** and STOP

## CRITICAL DISTINCTIONS:

**"Partial" vs "Almost" - THIS IS THE HARDEST DISTINCTION:**
- "Partial" = student STOPPED EARLY (incomplete work, major gaps remain)
- "Almost" = student FINISHED but made SMALL MISTAKES (essentially done)
- **KEY TEST**: Would the solution be "Correct" if the student just fixed small errors?
  - If YES → "Almost"
  - If NO (needs significant new work) → "Partial"
- **BE GENEROUS WITH "ALMOST"** - When in doubt, choose "Almost" over "Partial"

**"Partial" vs "Incorrect":**
- Did the student find ANY key insight from the official solution?
- If YES → "Partial"
- If NO (completely wrong approach) → "Incorrect"

**"Almost" vs "Correct":**
- Is the solution 100% flawless?
- If YES → "Correct"
- If ANY issues at all → "Almost"

## GRADING GUIDELINES ANALYSIS:
- Look for "(Partial)" markers in guidelines → suggests "Partial"
- Look for "(Almost)" markers in guidelines → suggests "Almost"
- Phrases like "minor error", "small mistake" → "Almost"
- Phrases like "incomplete", "missing steps" → "Partial"

## YOUR ANALYSIS PROCESS:

1. **Identify Key Insights**: What are the main ideas in the official solution?

2. **Analyze Student Work**: 
   - Did they find the right approach?
   - What percentage is complete?
   - What errors exist (if any)?

3. **Apply Decision Workflow** above

4. **Final Verification** (ASK YOURSELF):
   - If I think "Partial": Is this truly incomplete, or just has minor errors? (If minor errors → change to "Almost")
   - If I think "Almost": Is this truly 80%+ complete with only small issues?
   - If I think "Incorrect": Did the student really make NO meaningful progress?
   - If I think "Correct": Is this TRULY flawless?

## OUTPUT FORMAT:

Your ENTIRE response must be ONLY a JSON block. No other text.

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

## IMPORTANT RULES:
1. Use ONLY these four exact labels: "Correct", "Almost", "Partial", "Incorrect"
2. ONLY output the JSON block - no explanations, no other text
3. Be CONSERVATIVE with "Correct" (only truly flawless)
4. Be GENEROUS with "Almost" (most underused category - use for 80%+ solutions with minor issues)
5. "Partial" is for incomplete work, not for nearly-finished work with errors
6. "Incorrect" is only for fundamentally wrong approaches or minimal progress"""

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
        
        # Additional validation: Check if the prediction makes sense given the grading guidelines
        # If guidelines explicitly mention a category, we should trust that more
        grading_lower = grading_guidelines.lower()
        
        # If prediction is "Partial" but guidelines strongly suggest "Almost", reconsider
        if prediction == "Partial":
            almost_indicators = ["(almost)", "minor error", "small mistake", "trivial case", 
                                "essentially correct", "nearly complete", "minor flaw"]
            if any(indicator in grading_lower for indicator in almost_indicators):
                self.log_fn(f"Guidelines suggest 'Almost' - upgrading from 'Partial'")
                prediction = "Almost"
        
        # If prediction is "Incorrect" but guidelines suggest partial credit
        if prediction == "Incorrect":
            partial_indicators = ["(partial)", "significant progress", "right approach", 
                                 "found key insight", "correct strategy"]
            if any(indicator in grading_lower for indicator in partial_indicators):
                self.log_fn(f"Guidelines suggest 'Partial' - upgrading from 'Incorrect'")
                prediction = "Partial"
        
        return str(prediction), msg_history
