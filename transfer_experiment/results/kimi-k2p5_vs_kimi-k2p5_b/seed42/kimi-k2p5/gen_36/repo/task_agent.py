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
    """Extract JSON objects from <json>...</json> blocks."""
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
            # Handle double curly braces from f-string formatting
            if inner.startswith('{{') and inner.endswith('}}'):
                inner = inner[1:-1]
            inner_clean = inner.replace('{{', '{').replace('}}', '}')
            results.append(json.loads(inner_clean))
        except json.JSONDecodeError:
            # Try more aggressive cleaning
            try:
                inner_clean = inner.strip().replace('\n', ' ').replace('\r', '')
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
    """Extract raw JSON objects from text."""
    results = []
    # Pattern for nested braces
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    # Fallback: look for {"response": "..."} patterns
    if not results:
        response_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    # Try single quotes
    if not results:
        response_pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    return results or None


def _extract_direct_label(text: str) -> str | None:
    """Extract label directly from text by looking for valid labels."""
    # High priority patterns for explicit declarations - ordered by specificity
    label_patterns = [
        # JSON-like patterns (highest priority)
        r'"response"\s*:\s*"(Correct|Incorrect|Partial|Almost)"',
        r"'response'\s*:\s*'(Correct|Incorrect|Partial|Almost)'",
        r'["\']?response["\']?\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Explicit classification statements
        r'(?:is|would be|should be|classified as)\s+["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:grade|classification|category|label)\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Markdown formatting
        r'\*\*\s*(Correct|Incorrect|Partial|Almost)\s*\*\*',
        r'\*\s*(Correct|Incorrect|Partial|Almost)\s*\*',
        r'`(Correct|Incorrect|Partial|Almost)`',
        # Sentence patterns
        r'\bThe\s+(?:answer|classification|grade|final\s+answer)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bI\s+(?:would|will)\s+(?:classify|grade|label)\s+(?:this|it|the\s+answer)\s+as\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bTherefore[,]?\s+(?:the\s+answer\s+is\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bFinal\s+(?:answer|classification|grade)[:]?\s*["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bClassification[:]\s*(Correct|Incorrect|Partial|Almost)\b',
        r'\bGrade[:]\s*(Correct|Incorrect|Partial|Almost)\b',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1)
            for valid_label in VALID_LABELS:
                if label.lower() == valid_label.lower():
                    return valid_label
    
    # Check last 100 lines for standalone labels (increased from 50)
    lines = text.split('\n')
    last_lines = lines[-100:] if len(lines) > 100 else lines
    for line in reversed(last_lines):
        line = line.strip()
        if not line or line.startswith('```') or line.startswith('<json>'):
            continue
        for label in VALID_LABELS:
            # Look for labels at the start or end of lines
            pattern_start = r'^' + re.escape(label) + r'\b'
            if re.search(pattern_start, line, re.IGNORECASE):
                return label
            pattern_end = r'\b' + re.escape(label) + r'[\.\s]*$'
            if re.search(pattern_end, line, re.IGNORECASE):
                return label
            # Anywhere in line if it's a short line (likely a conclusion)
            if len(line) < 50:
                pattern = r'\b' + re.escape(label) + r'\b'
                if re.search(pattern, line, re.IGNORECASE):
                    return label
    
    # Clean text and check words from the end (last 200 words)
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}<>]', ' ', text)
    words = cleaned_text.split()
    for word in reversed(words[-200:]):
        word_stripped = word.strip('.,;:!?')
        for label in VALID_LABELS:
            if word_stripped.lower() == label.lower():
                return label
    
    return None


def extract_prediction(text: str) -> str | None:
    """Extract prediction from text using multiple strategies.
    
    Tries multiple extraction methods in order of preference.
    Returns the label value, or None if extraction fails.
    """
    # Try <json> tags first
    extracted = _extract_jsons(text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
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
    
    # Fallback: direct label extraction
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
        
        # Parse points from inputs (stored in separate field, not in grading guidelines)
        points_hint = ""
        pts = inputs.get("points")
        if pts is not None:
            try:
                pts = int(pts)
            except (ValueError, TypeError):
                # Try to extract from string
                points_match = re.search(r'(\d+)', str(pts))
                pts = int(points_match.group(1)) if points_match else None
        
        if pts is not None:
            if pts >= 7:
                points_hint = """CRITICAL - POINTS OVERRIDE: This solution received 7 points.
CLASSIFICATION RULE: 7 points = "Correct" (ALWAYS - NO EXCEPTIONS)
You MUST output "Correct". Do not analyze further."""
            elif pts == 6:
                points_hint = """CRITICAL - POINTS OVERRIDE: This solution received 6 points.
CLASSIFICATION RULE: 6 points = "Almost" (ALWAYS - NO EXCEPTIONS)
6 points means: Solution is essentially complete with exactly ONE minor issue (NOT multiple issues).
You MUST output "Almost". Do NOT output "Correct" - 6 points explicitly means there is a minor issue preventing full marks.
Remember: "Almost" = 6 points, "Correct" = 7 points. These are DIFFERENT categories."""
            elif 2 <= pts <= 5:
                points_hint = """CRITICAL - POINTS OVERRIDE: This solution received 2-5 points.
CLASSIFICATION RULE: 2-5 points = "Partial" (ALWAYS - NO EXCEPTIONS)
2-5 points means: Meaningful progress made but solution has MULTIPLE gaps or is significantly incomplete.
You MUST output "Partial". Do NOT output "Correct" or "Almost"."""
            elif pts <= 1:
                points_hint = """CRITICAL - POINTS OVERRIDE: This solution received 0-1 points.
CLASSIFICATION RULE: 0-1 points = "Incorrect" (ALMOST ALWAYS - DEFAULT)
0-1 points means: No meaningful progress or fundamentally flawed approach.
You MUST output "Incorrect" unless the grading guidelines explicitly list 4 or more specific achievements in the Partial section.
Only override to "Partial" if you see 4+ numbered achievements listed."""
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to classify the student's answer into exactly one of four categories: Correct, Almost, Partial, or Incorrect.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

{points_hint}

---

## CLASSIFICATION DEFINITIONS - READ CAREFULLY:

### "Correct" (7/7 points, 10/10 points) - PERFECT SOLUTION
Use ONLY when ALL are true:
- Complete, rigorous proof from start to finish
- All key steps present with no gaps whatsoever
- Matches official solution's conclusion
- Would receive full marks in IMO competition
- **POINTS RULE**: 7 points = ALWAYS "Correct" (no exceptions)
- **CRITICAL**: 7 points means NO issues, 6 points means ONE issue - these are DIFFERENT
- **WHEN IN DOUBT**: If you can't find a specific error, choose "Correct"

### "Almost" (6/7 points, 8-9/10 points) - ESSENTIALLY SOLVED WITH EXACTLY ONE MINOR ISSUE
Use ONLY when ALL are true:
- Main proof structure is valid and essentially complete
- Student reached the FINAL CONCLUSION/ANSWER
- Exactly ONE small gap, calculation error, or missing trivial case (NOT multiple issues)
- The error is fixable by an expert in 5-10 minutes
- Student demonstrates full understanding of the problem
- **CRITICAL**: Has FINAL ANSWER but with exactly ONE minor flaw
- **CRITICAL DISTINCTION**: If you see MULTIPLE issues → "Partial" (not Almost)
- **CRITICAL DISTINCTION**: If a critical case is missing → "Partial" (not Almost)
- **POINTS RULE**: 6 points = ALWAYS "Almost" (no exceptions) - 6 points means one minor issue exists
- **CRITICAL**: 6 points is NOT "Correct" - "Correct" requires 7 points (perfect)
- **KEY TEST**: "Is there exactly ONE minor issue AND a final answer?" If YES → "Almost"

### "Partial" (2-5/7 points, 3-6/10 points) - MEANINGFUL PROGRESS BUT INCOMPLETE
Use when student made MEANINGFUL PROGRESS but has MULTIPLE gaps OR major unfinished parts:
- Found correct key insight/approach/invariant
- Major parts of proof still missing OR multiple gaps exist
- Correct approach but execution has substantial gaps
- Proved one direction but not the other (for iff statements)
- Found right invariant but didn't prove sufficiency
- Missing critical cases that affect validity
- **CRITICAL**: Does NOT have a complete final answer/conclusion
- **POINTS RULE**: 2-5 points = ALWAYS "Partial" (no exceptions)
- **CRITICAL DISTINCTION**: If only ONE small issue AND has final answer → "Almost" (not Partial)
- **KEY TEST**: "Are there MULTIPLE gaps OR missing critical parts OR no final answer?" If YES → "Partial"

### "Incorrect" (0-1/7 points, 0-2/10 points) - NO MEANINGFUL PROGRESS
Use when ANY of these are true:
- Approach is fundamentally wrong or misguided
- No meaningful progress toward correct solution
- Critical logical flaws invalidate the entire argument
- Student misunderstood the problem statement
- **POINTS RULE**: 0-1 points = "Incorrect" (default) unless 3+ specific achievements listed
- **KEY TEST**: "Did student find a correct key insight?" If NO → "Incorrect"

---

## DECISION FLOWCHART - FOLLOW THIS EXACTLY:

**STEP 0 (MOST IMPORTANT - OVERRIDES ALL OTHER STEPS)**: Check the points value
- **7 points** → **"Correct"** (STOP - no further analysis needed)
- **6 points** → **"Almost"** (STOP - no further analysis needed, 6 points = one minor issue, NOT Correct)
- **2-5 points** → **"Partial"** (STOP - no further analysis needed)
- **0-1 points** → **"Incorrect"** (STOP - default, unless 4+ specific achievements listed)

**CRITICAL**: The points value is the STRONGEST signal. Trust it above all other indicators.
**CRITICAL**: 6 points = "Almost", 7 points = "Correct". These are DIFFERENT categories. Never classify 6 points as "Correct".

---

## CONCRETE EXAMPLES:

**POINTS-BASED CLASSIFICATION (USE THIS FIRST):**
- **7 points** → "Correct" (always)
- **6 points** → "Almost" (always - exactly one minor issue)  
- **2-5 points** → "Partial" (always - multiple gaps)
- **0-1 points** → "Incorrect" (default) or "Partial" (if 3+ achievements)

**"Correct" examples (7 points):**
- Complete proof with all steps justified, all cases handled
- Would receive full marks (7/7 points)
- **7 points = Correct** - no exceptions

**"Almost" examples (6 points, SINGLE minor issue, HAS final answer):**
- Complete proof but small calculation error in one step
- Everything correct but missing trivial n=1 case (only one edge case)
- Main proof valid, one small lemma unstated (but lemma is obvious)
- Has "Therefore the answer is X" but X has small error
- **6 points = Almost** - no exceptions, even if solution looks mostly correct
- **CRITICAL**: 6 points is NOT "Correct" - "Correct" requires 7 points (perfect solution)

**"Partial" examples (2-5 points, MULTIPLE gaps OR major unfinished parts):**
- Proved only one direction of "if and only if"
- Found right invariant but couldn't prove key property
- Has main idea but missing multiple key steps
- **2-5 points = Partial** - no exceptions

**"Incorrect" examples (0-1 points, minimal or no progress):**
- Approach fundamentally wrong
- No meaningful progress (just restating the problem)
- Misunderstood problem (solving wrong problem)
- **0-1 points = Incorrect** (default) unless 3+ specific achievements listed

---

## CRITICAL RULES - FOLLOW THESE EXACTLY:

### POINTS-BASED CLASSIFICATION (ABSOLUTE HIGHEST PRIORITY - NEVER OVERRIDE):
- **7 points = ALWAYS "Correct"** - No exceptions, no matter what the text says
- **6 points = ALWAYS "Almost"** - No exceptions, indicates exactly one minor issue (NOT Correct)
- **2-5 points = ALWAYS "Partial"** - No exceptions, indicates meaningful progress but incomplete
- **0-1 points = "Incorrect" (default)** - Only "Partial" if 3+ specific achievements are listed in guidelines

### CLASSIFICATION IS DETERMINED BY POINTS:
1. **"Correct" = 7 points** - Full marks = complete solution
2. **"Almost" = 6 points** - One minor issue = Almost
3. **"Partial" = 2-5 points** - Partial progress = Partial
4. **"Incorrect" = 0-1 points with no progress** - No progress = Incorrect

### IMPORTANT NOTES:
- The points value is the STRONGEST and most reliable signal
- Grading guidelines may contain both "(Partial)" and "(Almost)" markers - IGNORE THEM, use points
- When points are 2-5, always classify as "Partial" regardless of other indicators
- When points are 6, always classify as "Almost" regardless of other indicators
- When points are 7, always classify as "Correct" regardless of other indicators

---

## YOUR ANALYSIS PROCESS:

1. **Understand**: What does the problem ask? What does the official solution provide?
2. **Check Guidelines First**: What do the grading guidelines explicitly indicate? Look for "(Almost)" or "(Partial)" markers
3. **Evaluate**: What did the student accomplish? What insights? What gaps?
4. **Apply Flowchart**: Follow the 5-step decision process above
5. **Verify**: Does your classification match the guidelines? When in doubt, be conservative.

---

## RESPONSE FORMAT - CRITICAL:

You MUST respond with ONLY a JSON object in this exact format:
<json>
{{"response": "Correct"}}
</json>

OR

<json>
{{"response": "Incorrect"}}
</json>

OR

<json>
{{"response": "Partial"}}
</json>

OR

<json>
{{"response": "Almost"}}
</json>

**CRITICAL INSTRUCTIONS:**
1. Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost"
2. The "response" field must contain ONLY the label - no extra text
3. You MUST include the <json> and </json> tags
4. Do not include any text after the JSON block
5. The JSON must be valid - use double quotes around the label
6. Response should look EXACTLY like: <json>{{"response": "Correct"}}</json>

---

## FINAL VERIFICATION CHECKLIST:

Before responding, verify:
- [ ] **MOST IMPORTANT - POINTS DETERMINE THE ANSWER**:
  - 7 points → Must be "Correct" (ALWAYS)
  - 6 points → Must be "Almost" (ALWAYS) - NOT "Correct"
  - 2-5 points → Must be "Partial" (ALWAYS)
  - 0-1 points → "Incorrect" (ALWAYS, unless 4+ achievements listed)
- [ ] Does my classification match the points value? (THIS IS THE #1 PRIORITY)
- [ ] Is my response in the exact JSON format required?

**REMEMBER**: The points value is the STRONGEST signal.
- 6 points = "Almost" (not Correct, even if solution looks good)
- 0-1 points = "Incorrect" (not Partial, unless there are 4+ specific achievements)

Now provide your final classification in the required JSON format."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
            temperature=0.1,
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
                # Try to extract from earlier messages
                for msg in reversed(msg_history):
                    if "text" in msg:
                        extracted = extract_prediction(msg["text"])
                        if extracted is not None:
                            prediction = extracted
                            self.log_fn(f"Extracted prediction from earlier message: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Post-processing: POINTS ARE THE PRIMARY SIGNAL - OVERRIDE LLM OUTPUT
        grading_guidelines = inputs.get("grading_guidelines", "")
        original_prediction = prediction
        
        # Parse points - this is the STRONGEST signal for classification
        # Points are stored in a separate field, not in the grading guidelines text
        points_val = inputs.get("points")
        if points_val is not None:
            try:
                points_val = int(points_val)
            except (ValueError, TypeError):
                # Fallback: try to extract from string
                points_match = re.search(r'(\d+)', str(points_val))
                points_val = int(points_match.group(1)) if points_match else None
        else:
            # Fallback: try to find in grading guidelines (for backwards compatibility)
            points_match = re.search(r'(\d+)\s*points?', grading_guidelines, re.IGNORECASE)
            points_val = int(points_match.group(1)) if points_match else None
        
        # ============================================
        # POINTS-BASED CLASSIFICATION (ABSOLUTE PRIORITY)
        # ============================================
        # Points completely determine the classification - override everything else
        if points_val is not None:
            if points_val >= 7:
                # 7 points = Correct (always)
                prediction = "Correct"
                if original_prediction != "Correct":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Correct' based on {points_val} points (absolute rule)")
            elif points_val == 6:
                # 6 points = Almost (always) - solution is nearly complete with exactly one minor issue
                prediction = "Almost"
                if original_prediction != "Almost":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Almost' based on 6 points (absolute rule)")
            elif 2 <= points_val <= 5:
                # 2-5 points = Partial (always) - significant progress but major gaps remain
                prediction = "Partial"
                if original_prediction != "Partial":
                    self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Partial' based on {points_val} points (absolute rule)")
            elif points_val <= 1:
                # 0-1 points = Incorrect (almost always) - minimal or no meaningful progress
                # Only classify as Partial if there's clear evidence of substantial partial work
                has_substantial_progress = False
                
                # Check for explicit "(Partial)" marker with multiple achievements
                if "(Partial)" in grading_guidelines:
                    # Count numbered items in Partial section as proxy for substantial progress
                    partial_section = grading_guidelines.split("(Partial)")[1] if "(Almost)" not in grading_guidelines else grading_guidelines.split("(Partial)")[1].split("(Almost)")[0]
                    achievement_count = len([line for line in partial_section.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'))])
                    # Only consider substantial if 4+ achievements listed (stricter threshold)
                    has_substantial_progress = achievement_count >= 4
                
                if has_substantial_progress:
                    prediction = "Partial"
                    if original_prediction != "Partial":
                        self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Partial' based on {points_val} points with substantial progress (4+ achievements)")
                else:
                    prediction = "Incorrect"
                    if original_prediction != "Incorrect":
                        self.log_fn(f"Post-processing: Changed from '{original_prediction}' to 'Incorrect' based on {points_val} points (insufficient progress)")

        # Validate the final prediction
        if prediction not in VALID_LABELS:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid label. Defaulting to 'None'.")
            prediction = "None"
        
        self.log_fn(f"Final prediction: {prediction}")
        
        return str(prediction), msg_history
