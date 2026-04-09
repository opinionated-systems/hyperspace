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

# Valid prediction labels
VALID_LABELS = {"Correct", "Incorrect", "Partial", "Almost"}


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
    # Find all JSON-like structures - improved pattern for nested braces
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    # Additional fallback: try to find any text between curly braces
    if not results:
        # Look for patterns like {"response": "..."} or { "response" : "..." }
        response_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    return results or None


def _extract_direct_label(text: str) -> str | None:
    """Extract label directly from text by looking for valid labels."""
    # First, try to find explicit label declarations (highest priority)
    label_patterns = [
        r'["\']?(?:response|label|grade|classification|prediction)["\']?\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:is|would be|should be|classified as)\s+["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:grade|classification)\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'\*\s*(Correct|Incorrect|Partial|Almost)\s*\*',
        r'\b\*\*(Correct|Incorrect|Partial|Almost)\*\*\b',
        r'`(Correct|Incorrect|Partial|Almost)`',
        r'\bThe\s+(?:answer|classification|grade)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bI\s+(?:would|will)\s+(?:classify|grade|label)\s+(?:this|it|the\s+answer)\s+as\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1)
            for valid_label in VALID_LABELS:
                if label.lower() == valid_label.lower():
                    return valid_label
    
    # Look for labels in the last few lines (often where the conclusion is)
    lines = text.split('\n')
    last_lines = lines[-10:] if len(lines) > 10 else lines
    for line in reversed(last_lines):
        line = line.strip()
        for label in VALID_LABELS:
            # Look for label as a whole word with word boundaries
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                return label
    
    # Clean up the text - remove punctuation that might interfere
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}]', ' ', text)
    words = cleaned_text.split()
    
    # Check each word (case-insensitive)
    for word in words:
        word_stripped = word.strip('.,;:!?')
        for label in VALID_LABELS:
            if word_stripped.lower() == label.lower():
                return label
    
    # Also check for labels as substrings in all lines
    for line in lines:
        line = line.strip()
        for label in VALID_LABELS:
            # Look for label as a whole word
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                return label
    
    return None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects
    4. Direct label extraction from text
    
    Returns the "response" field value, or None if extraction fails.
    """
    # Try <json> tags first
    extracted = _extract_jsons(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                # Validate the response is one of the expected labels
                if response in VALID_LABELS:
                    return response
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Try raw JSON
    extracted = _extract_json_raw(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                if response in VALID_LABELS:
                    return response
    
    # Fallback: direct label extraction from text
    return _extract_direct_label(text)


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

## Classification Criteria - READ CAREFULLY:

**"Correct"** - Use ONLY when ALL of these are true:
- The student provides a complete, rigorous proof/solution from start to finish
- All key steps are present and logically sound with no gaps
- The answer matches the official solution's conclusion with proper justification
- No significant errors, missing cases, or logical gaps
- The proof would receive full marks in an IMO competition
- **CRITICAL**: If there is ANY doubt about completeness, use "Almost" or "Partial" instead

**"Incorrect"** - Use when ANY of these are true:
- The student's answer is fundamentally wrong or the approach is completely misguided
- The solution contains critical logical flaws that invalidate the entire argument
- The answer does not address the problem requirements or makes no meaningful progress
- The student misunderstood the problem statement
- The solution uses an approach that cannot possibly work
- **CRITICAL**: If the student made NO meaningful progress toward the correct solution, it is "Incorrect", NOT "Partial"

**"Partial"** - Use when the student made SIGNIFICANT PROGRESS but the solution is INCOMPLETE:
- The student found a correct key insight, invariant, or strategy but didn't complete the proof
- Significant progress was made (e.g., proved one direction, found the right approach) but major gaps remain
- The approach is correct but missing crucial steps, cases, or the conclusion
- The student identified the right strategy but failed to execute it fully or rigorously
- The solution has the main idea but lacks formal proof or verification
- **KEY DISTINCTION**: The student is missing substantial parts of the solution (30-70% complete)
- **CRITICAL**: Do NOT use "Partial" for solutions that make no meaningful progress - those are "Incorrect"

**"Almost"** - Use when the solution is NEARLY COMPLETE with only MINOR issues:
- The solution is nearly complete and would be "Correct" except for minor technical errors
- The main proof structure is valid but there's a small gap in reasoning
- The proof is valid except for a minor case, edge condition, or small calculation error
- The student demonstrated deep understanding but made one small mistake that doesn't invalidate the main argument
- The error is fixable with a minor adjustment
- **KEY DISTINCTION**: The student essentially solved the problem - the structure is there, just needs a small fix (80-95% complete)
- **CRITICAL**: If the solution has multiple gaps or missing cases, it is "Partial", not "Almost"

## CRITICAL DISTINCTIONS:

### "Almost" vs "Partial"

**Use "Almost" when:**
- The student has 80-95% of a complete solution
- The main proof structure is intact
- Only minor technical details are missing or wrong
- The solution could be fixed by an expert in 5-10 minutes
- Example: Correct approach, correct main lemma, but a small calculation error or missing one trivial case

**Use "Partial" when:**
- The student has 30-70% of a complete solution  
- Major parts of the proof are missing
- The main idea is there but significant work remains
- The solution would require substantial effort to complete
- Example: Found the right invariant but couldn't prove the key property; proved one direction of an iff but not the other

### "Partial" vs "Incorrect"

**Use "Partial" when:**
- The student found a correct key insight or approach
- There is meaningful progress toward the solution (at least 30% complete)
- The approach, if completed, would lead to a correct solution

**Use "Incorrect" when:**
- The student made NO meaningful progress
- The approach is fundamentally flawed and cannot work
- The solution is completely off-track
- The student misunderstood the problem

## CONCRETE EXAMPLES:

**"Correct" examples:**
- Student provides a complete, rigorous proof with all steps justified
- Student's solution matches the official solution in approach and conclusion
- All cases are handled, all lemmas are proved, no logical gaps
- Would receive full marks in competition

**"Almost" examples:**
- Student has a complete proof but makes a small calculation error in one step
- Student proves everything correctly but misses one trivial edge case (e.g., n=1 case)
- Student's proof structure is valid but has a minor gap that could be fixed in minutes
- Student demonstrates full understanding but has a small technical oversight
- Student gets the right answer and main proof is correct, but one small lemma is stated without proof
- Student's solution is 90% complete, missing only a minor technical detail

**"Partial" examples:**
- Student proves only one direction of an "if and only if" statement
- Student finds the right invariant but fails to prove sufficiency
- Student has the main idea but the proof is missing multiple key steps
- Student correctly identifies the approach but execution has major gaps
- Student makes significant progress (50-70%) but the solution is incomplete
- Student proves a key lemma but doesn't connect it to the main result
- Student has the right idea but the proof is incomplete in several places

**"Incorrect" examples:**
- Student's approach is fundamentally wrong or misguided
- Student misunderstands the problem statement
- Student makes no meaningful progress toward the solution
- Student's solution contains critical logical flaws that invalidate the entire argument
- Student uses an approach that cannot possibly solve the problem
- Student's solution is completely unrelated to the actual problem

## Your Task - STEP BY STEP ANALYSIS REQUIRED:

Step 1: Analyze what the problem asks for and what the official solution provides.
- What is the core question or task?
- What is the expected answer format?
- What key theorems or techniques does the official solution use?

Step 2: Analyze the student's answer in detail:
- Did they understand the problem correctly?
- What key insights did they identify?
- Are there logical gaps or errors? (Identify specific line numbers if possible)
- How complete is the solution? (Estimate percentage: 0%, 25%, 50%, 75%, 90%, 100%)
- Does the student's final answer match the official solution's conclusion?
- **CRITICAL**: Did the student make MEANINGFUL progress or is the approach fundamentally flawed?

Step 3: Check against grading guidelines (if provided):
- What specific criteria are mentioned?
- Does the student's answer meet these criteria?
- Are there point allocations mentioned that indicate what constitutes partial credit?
- **CRITICAL**: The grading guidelines often explicitly state what constitutes "Partial" vs "Almost"

Step 4: Make your classification decision using this decision tree:
1. Is the approach fundamentally wrong or making no progress? → "Incorrect"
2. Is the solution complete and rigorous with no gaps? → "Correct"
3. Is the solution 80-95% complete with only minor fixable errors? → "Almost"
4. Is the solution 30-70% complete with major gaps but correct approach? → "Partial"
5. When in doubt between categories, be CONSERVATIVE (lean toward the lower grade)

Step 5: VERIFY your decision against the grading guidelines:
- Does your classification match what the grading guidelines indicate?
- If the guidelines say "(Partial)" for certain criteria and "(Almost)" for others, which does the student's answer satisfy?
- **CRITICAL**: If the grading guidelines explicitly state that a solution with certain characteristics is "Partial" or "Almost", FOLLOW those guidelines
- Re-evaluate: Are you being too generous or too harsh?
- **FINAL CHECK**: If you classified as "Partial" but the solution makes no meaningful progress, change to "Incorrect"
- **FINAL CHECK**: If you classified as "Correct" but there are gaps or missing cases, change to "Almost" or "Partial"
- **FINAL CHECK**: If you classified as "Almost" but there are multiple gaps or missing critical cases, change to "Partial"

## IMPORTANT DECISION RULES:
1. **Be EXTREMELY CONSERVATIVE with "Correct"** - only use if you're 100% confident it's complete with NO gaps
2. **"Almost" means ESSENTIALLY SOLVED** - just needs a small fix (80-95% complete, fixable in 5-10 minutes)
3. **"Partial" means SIGNIFICANT PROGRESS but INCOMPLETE** - major work remains (30-70% complete)
4. **"Incorrect" means NO MEANINGFUL PROGRESS** - the approach is wrong or no progress was made
5. If the student found the right invariant/approach but didn't prove sufficiency → "Partial" (not Almost)
6. If the proof structure is complete but has a minor gap → "Almost" (not Partial)
7. If the main idea is wrong or no progress → "Incorrect" (not Partial)
8. When in doubt between "Almost" and "Partial", ask: "Could this solution be fixed in 5-10 minutes by an expert?" If yes → "Almost", if no → "Partial"
9. When in doubt between "Partial" and "Incorrect", ask: "Did the student find a correct key insight?" If yes → "Partial", if no → "Incorrect"
10. **Pay special attention to the grading guidelines** - they often specify what constitutes each category
11. If the student has the right answer but the proof has a small gap → "Almost" (not Partial)
12. If the student proves most of the solution but misses a key lemma → "Partial"
13. **If the solution has multiple gaps or missing cases, it is "Partial" not "Almost"**
14. **If the solution omits a critical case that affects the validity, it is "Partial" not "Almost"**

## RESPONSE FORMAT - CRITICAL:
You MUST respond with ONLY a JSON object in this exact format:
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

CRITICAL INSTRUCTIONS:
1. Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost"
2. The "response" field must contain ONLY the label - no extra text, no explanation
3. You MUST include the <json> and </json> tags around your JSON response
4. Do not include any text after the JSON block
5. The JSON must be valid - use double quotes around the label
6. DO NOT include markdown formatting like ```json inside the <json> tags
7. The response should look EXACTLY like: <json>{{"response": "Correct"}}</json>

## BEFORE YOU RESPOND - FINAL VERIFICATION:
1. Re-read the grading guidelines one more time - does your classification match?
2. If the guidelines indicate "(Partial)" for the criteria the student met, did you classify as "Partial"?
3. If the guidelines indicate "(Almost)" for the criteria the student met, did you classify as "Almost"?
4. If you were about to say "Partial" but the student made no meaningful progress, change to "Incorrect"
5. If you were about to say "Correct" but there are ANY gaps, change to "Almost" or "Partial"
6. Be CONSERVATIVE - when in doubt, choose the lower classification

Now provide your final classification in the required JSON format."""

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
                self.log_fn(f"Failed to extract prediction from response: {last_message[:500]}...")
                # Try to extract any valid label from the entire message history
                for msg in reversed(msg_history):
                    if "text" in msg:
                        extracted = extract_prediction(msg["text"])
                        if extracted is not None:
                            prediction = extracted
                            self.log_fn(f"Extracted prediction from earlier message: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate the final prediction
        if prediction not in VALID_LABELS:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid label. Defaulting to 'None'.")
            prediction = "None"
        
        # Log detailed information for debugging
        self.log_fn(f"Final prediction: {prediction}")
        self.log_fn(f"Problem type: {inputs.get('domain', 'unknown')}")
        self.log_fn(f"Student answer length: {len(student_answer)} chars")

        return str(prediction), msg_history
