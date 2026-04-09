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
    # High priority patterns for explicit declarations
    label_patterns = [
        # JSON-like patterns
        r'"response"\s*:\s*"(Correct|Incorrect|Partial|Almost)"',
        r"'response'\s*:\s*'(Correct|Incorrect|Partial|Almost)'",
        r'["\']?response["\']?\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Explicit declarations
        r'(?:is|would be|should be|classified as)\s+["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:grade|classification|category|label)\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Markdown formatting
        r'\*\s*(Correct|Incorrect|Partial|Almost)\s*\*',
        r'\*\*(Correct|Incorrect|Partial|Almost)\*\*',
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
    
    # Check last 30 lines for standalone labels
    lines = text.split('\n')
    last_lines = lines[-30:] if len(lines) > 30 else lines
    for line in reversed(last_lines):
        line = line.strip()
        if not line or line.startswith('```') or line.startswith('<json>'):
            continue
        for label in VALID_LABELS:
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                if not line.startswith('```') and not line.startswith('<'):
                    return label
    
    # Clean text and check words
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}<>]', ' ', text)
    words = cleaned_text.split()
    for word in reversed(words):
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
        
        instruction = f"""You are an expert IMO (International Mathematical Olympiad) grader. Your task is to classify the student's answer into exactly one of four categories: Correct, Almost, Partial, or Incorrect.

## Problem:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

---

## CLASSIFICATION DEFINITIONS - READ CAREFULLY:

### "Correct" (100% complete, 10/10 points)
Use ONLY when ALL are true:
- Complete, rigorous proof from start to finish
- All key steps present with no gaps whatsoever
- Matches official solution's conclusion
- Would receive full marks in IMO competition
- **If ANY doubt about completeness, use "Almost" or "Partial"**

### "Almost" (80-95% complete, 7-9/10 points) - ESSENTIALLY SOLVED
Use when the solution is NEARLY CORRECT with ONLY ONE minor fixable issue:
- Main proof structure is valid and essentially complete
- Student reached the FINAL CONCLUSION/ANSWER
- Exactly ONE small gap, calculation error, or missing trivial case
- The error is fixable by an expert in 5-10 minutes
- Student demonstrates full understanding of the problem
- **CRITICAL**: Has FINAL ANSWER but with minor flaw(s)
- **CRITICAL DISTINCTION**: If you see MULTIPLE issues → "Partial" (not Almost)
- **CRITICAL DISTINCTION**: If a critical case is missing → "Partial" (not Almost)
- **KEY TEST**: "Is there exactly ONE minor issue AND a final answer?" If YES → "Almost"

### "Partial" (30-70% complete, 3-6/10 points) - MEANINGFUL PROGRESS BUT INCOMPLETE
Use when student made MEANINGFUL PROGRESS but has MULTIPLE gaps OR major unfinished parts:
- Found correct key insight/approach/invariant
- Major parts of proof still missing OR multiple gaps exist
- Correct approach but execution has substantial gaps
- Proved one direction but not the other (for iff statements)
- Found right invariant but didn't prove sufficiency
- Missing critical cases that affect validity
- **CRITICAL**: Does NOT have a complete final answer/conclusion
- **CRITICAL DISTINCTION**: If only ONE small issue AND has final answer → "Almost" (not Partial)
- **KEY TEST**: "Are there MULTIPLE gaps OR missing critical parts OR no final answer?" If YES → "Partial"

### "Incorrect" (0-25% complete, 0-2/10 points) - NO MEANINGFUL PROGRESS
Use when ANY of these are true:
- Approach is fundamentally wrong or misguided
- No meaningful progress toward correct solution
- Critical logical flaws invalidate the entire argument
- Student misunderstood the problem statement
- **KEY TEST**: "Did student find a correct key insight?" If NO → "Incorrect"

---

## DECISION FLOWCHART - FOLLOW THIS EXACTLY:

**STEP 1**: Did the student make ANY meaningful progress toward the correct solution?
- NO progress / wrong approach / misunderstood problem → **"Incorrect"** (STOP)
- YES, some progress found → Continue to Step 2

**STEP 2**: Does the student have a FINAL ANSWER/CONCLUSION?
- Look for: "Therefore...", "Thus...", "The answer is...", "QED", "∎", boxed answers
- NO final answer (stopped early, incomplete) → **"Partial"** (STOP - has progress but incomplete)
- YES, has final answer → Continue to Step 3

**STEP 3**: Is the solution 100% complete and rigorous with NO gaps whatsoever?
- YES, 100% complete → **"Correct"** (STOP)
- Has some gaps → Continue to Step 4

**STEP 4**: Count the number of issues. Is there EXACTLY ONE minor issue?
- Exactly ONE small calculation error? → **"Almost"**
- Exactly ONE missing trivial edge case (like n=1)? → **"Almost"**
- Exactly ONE small lemma stated without proof? → **"Almost"**
- Main proof structure intact, just needs minor polish? → **"Almost"**
- **"Almost" = SINGLE issue, 80-95% complete, HAS final answer**
- MULTIPLE issues OR missing critical case? → **"Partial"**

---

## CONCRETE EXAMPLES:

**"Correct" examples:**
- Complete proof with all steps justified, all cases handled
- Would receive full marks
- Example: Full proof of AM-GM inequality with all cases verified

**"Almost" examples (SINGLE minor issue, HAS final answer):**
- Complete proof but small calculation error in one step (e.g., arithmetic mistake in final line)
- Everything correct but missing trivial n=1 case (only one edge case)
- Main proof valid, one small lemma unstated (but lemma is obvious)
- 90% complete, needs only minor technical fix
- Has "Therefore the answer is X" but X has small error
- **Counter-example**: If missing n=1 AND n=2 cases → "Partial" (multiple issues)
- **Counter-example**: If calculation error AND missing case → "Partial" (multiple issues)

**"Partial" examples (MULTIPLE gaps OR major unfinished parts, NO final answer):**
- Proved only one direction of "if and only if" (missing entire other direction)
- Found right invariant but couldn't prove key property (two gaps: invariant found, proof missing)
- Has main idea but missing multiple key steps (multiple gaps)
- Proved key lemma but didn't connect to main result (two parts: lemma done, connection missing)
- 50-70% complete, significant work remains
- Stops at "We need to show..." or "It remains to prove..."
- **Counter-example**: If only one small lemma missing AND has final answer → "Almost" (single issue)

**"Incorrect" examples:**
- Approach fundamentally wrong (e.g., using induction where it doesn't apply)
- No meaningful progress (just restating the problem)
- Misunderstood problem (solving wrong problem)
- Solution completely off-track

---

## CRITICAL RULES:

1. **"Almost" = EXACTLY ONE minor fixable issue AND has FINAL ANSWER** (80-95% complete)
2. **"Partial" = MULTIPLE gaps OR major unfinished parts OR NO final answer** (30-70% complete)
3. **"Incorrect" = NO meaningful progress** (0-25% complete)
4. **When in doubt, be CONSERVATIVE** (choose lower grade)
5. **Pay special attention to grading guidelines** - they often explicitly state "(Partial)" or "(Almost)"
6. **COUNT THE ISSUES**: If you can identify more than one problem → "Partial"
7. **CRITICAL vs TRIVIAL**: Missing a critical case (like n=0 for a counting problem) → "Partial"
8. **DIRECTIONS**: Proving only one direction of "if and only if" → "Partial" (major part missing)
9. **FINAL ANSWER TEST**: If no "Therefore..." or "The answer is..." → "Partial" (not Almost)

---

## YOUR ANALYSIS PROCESS:

1. **Understand**: What does the problem ask? What does the official solution provide?
2. **Evaluate**: What did the student accomplish? What insights? What gaps?
3. **Check Guidelines**: What do the grading guidelines explicitly indicate?
4. **Apply Flowchart**: Follow the 4-step decision process above
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
- [ ] Did I follow the 4-step decision flowchart?
- [ ] Did I check for a FINAL ANSWER? (No final answer → Partial, not Almost)
- [ ] Did I COUNT the number of issues? (Exactly ONE = Almost, MULTIPLE = Partial)
- [ ] Does my classification match the grading guidelines?
- [ ] If guidelines say "(Partial)", did I classify as "Partial"?
- [ ] If guidelines say "(Almost)", did I classify as "Almost"?
- [ ] Is there a critical case missing? If yes → "Partial"
- [ ] Is there only one direction of "if and only if"? If yes → "Partial"
- [ ] Am I being conservative when in doubt?
- [ ] Is my response in the exact JSON format required?

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

        # Validate the final prediction
        if prediction not in VALID_LABELS:
            self.log_fn(f"Warning: Extracted prediction '{prediction}' is not a valid label. Defaulting to 'None'.")
            prediction = "None"
        
        self.log_fn(f"Final prediction: {prediction}")
        
        return str(prediction), msg_history
