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
    
    # Try to find response field with single quotes
    if not results:
        response_pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    # Try to find response field with any quotes and spacing variations
    if not results:
        response_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(Correct|Incorrect|Partial|Almost)["\']?\s*\}'
        matches = re.findall(response_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            results.append({"response": match.strip()})
    
    return results or None


def _extract_direct_label(text: str) -> str | None:
    """Extract label directly from text by looking for valid labels."""
    # First, try to find explicit label declarations (highest priority)
    label_patterns = [
        # JSON-like patterns with response field (most specific first)
        r'"response"\s*:\s*"(Correct|Incorrect|Partial|Almost)"',
        r"'response'\s*:\s*'(Correct|Incorrect|Partial|Almost)'",
        r'"response"\s*:\s*"(Correct|Incorrect|Partial|Almost)"',
        # Explicit declarations with various formats
        r'["\']?(?:response|label|grade|classification|prediction)["\']?\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:is|would be|should be|classified as)\s+["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        r'(?:grade|classification)\s*[:=]\s*["\']?\s*(Correct|Incorrect|Partial|Almost)\s*["\']?',
        # Markdown formatting
        r'\*\s*(Correct|Incorrect|Partial|Almost)\s*\*',
        r'\b\*\*(Correct|Incorrect|Partial|Almost)\*\*\b',
        r'`(Correct|Incorrect|Partial|Almost)`',
        # Sentence patterns
        r'\bThe\s+(?:answer|classification|grade|final\s+answer)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bI\s+(?:would|will)\s+(?:classify|grade|label)\s+(?:this|it|the\s+answer)\s+as\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bTherefore[,]?\s+(?:the\s+answer\s+is\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bFinal\s+(?:answer|classification|grade)[:]?\s*["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bClassification[:]\s*(Correct|Incorrect|Partial|Almost)\b',
        r'\bGrade[:]\s*(Correct|Incorrect|Partial|Almost)\b',
        # Additional patterns for issue count statements
        r'I count \d+ issue\(s\).*\b(Correct|Incorrect|Partial|Almost)\b',
        r'Classification:\s*(Correct|Incorrect|Partial|Almost)',
        r'Grade:\s*(Correct|Incorrect|Partial|Almost)',
        # Classification at end of reasoning
        r'\bclassify\s+this\s+as\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bthis\s+(?:solution|answer)\s+(?:is|should\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bthe\s+(?:solution|answer)\s+(?:is|should\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
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
    last_lines = lines[-50:] if len(lines) > 50 else lines
    for line in reversed(last_lines):
        line = line.strip()
        # Skip empty lines and common non-content lines
        if not line or line.startswith('```') or line.startswith('<json>'):
            continue
        for label in VALID_LABELS:
            # Look for label as a whole word with word boundaries
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                # Additional check: make sure it's not part of a larger word
                # and not in a code block or tag
                if not line.startswith('```') and not line.startswith('<'):
                    return label
    
    # Clean up the text - remove punctuation that might interfere
    cleaned_text = re.sub(r'["\'\(\)\[\]\{\}<>`]', ' ', text)
    words = cleaned_text.split()
    
    # Check each word (case-insensitive), prioritizing later words
    for word in reversed(words):
        word_stripped = word.strip('.,;:!?')
        for label in VALID_LABELS:
            if word_stripped.lower() == label.lower():
                return label
    
    # Also check for labels as substrings in all lines (last resort)
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith('```') or line.startswith('<'):
            continue
        for label in VALID_LABELS:
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

---

## OUTPUT FORMAT EXAMPLES:

Here are examples of correct responses:

Example 1 - Correct solution:
<json>
{{"response": "Correct"}}
</json>

Example 2 - Solution with exactly one minor issue:
<json>
{{"response": "Almost"}}
</json>

Example 3 - Solution with multiple gaps:
<json>
{{"response": "Partial"}}
</json>

Example 4 - No meaningful progress:
<json>
{{"response": "Incorrect"}}
</json>

---

## CLASSIFICATION DEFINITIONS - READ CAREFULLY:

### "Correct" (100% complete)
Use ONLY when ALL are true:
- Complete, rigorous proof from start to finish
- All key steps present with no gaps
- Matches official solution's conclusion
- Would receive full marks in IMO competition
- **If ANY doubt about completeness, use "Almost" or "Partial"**

### "Almost" (80-95% complete, essentially solved, SINGLE minor issue)
Use when the solution is NEARLY CORRECT with ONLY ONE minor fixable issue:
- Main proof structure is valid and essentially complete
- Exactly ONE small gap, calculation error, or missing trivial case
- The error is fixable by an expert in 5-10 minutes
- Student demonstrates full understanding of the problem
- **CRITICAL DISTINCTION**: If you see MULTIPLE issues → "Partial" (not Almost)
- **CRITICAL DISTINCTION**: If a critical case is missing → "Partial" (not Almost)
- **KEY TEST**: "Is there exactly ONE minor issue?" If YES → "Almost"
- **IMPORTANT**: Having correct key insights alone is NOT enough for "Almost" - the solution must be 80-95% complete with only minor cleanup needed

### "Partial" (30-70% complete, significant progress, MULTIPLE gaps)
Use when student made MEANINGFUL PROGRESS but has MULTIPLE gaps OR major unfinished parts:
- Found correct key insight/approach/invariant
- Major parts of proof still missing OR multiple gaps exist
- Correct approach but execution has substantial gaps
- Proved one direction but not the other (for iff statements)
- Found right invariant but didn't prove sufficiency
- Missing critical cases that affect validity
- **CRITICAL DISTINCTION**: If only ONE small issue → "Almost" (not Partial)
- **KEY TEST**: "Are there MULTIPLE gaps OR missing critical parts?" If YES → "Partial"
- **IMPORTANT**: "Partial" requires BOTH meaningful progress AND significant gaps remaining

### "Incorrect" (0-25% complete, no meaningful progress)
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

**STEP 2**: Is the solution complete and rigorous with NO gaps whatsoever?
- YES, 100% complete → **"Correct"** (STOP)
- Has some gaps → Continue to Step 3

**STEP 3**: Assess completion level and count issues
- Is the solution 80-95% complete with EXACTLY ONE minor issue?
  - Exactly ONE small calculation error? → **"Almost"**
  - Exactly ONE missing trivial edge case (like n=1)? → **"Almost"**
  - Exactly ONE small lemma stated without proof? → **"Almost"**
  - Main proof structure intact, just needs minor polish? → **"Almost"**
- Is the solution 30-70% complete OR has MULTIPLE issues? → Continue to Step 4

**STEP 4**: Are there MULTIPLE gaps OR missing critical cases OR major parts unfinished?
- Multiple gaps in reasoning? → **"Partial"**
- Missing critical case affecting validity? → **"Partial"**
- Proved only one direction of iff? → **"Partial"**
- Found right approach but couldn't execute fully? → **"Partial"**
- Has correct insights but significant work remains? → **"Partial"**
- **"Partial" = MULTIPLE gaps OR 30-70% complete**

**IMPORTANT DISTINCTION**: 
- "Almost" = 80-95% complete, essentially solved, needs only minor fix
- "Partial" = 30-70% complete, meaningful progress but significant gaps remain
- Having correct insights alone does NOT make a solution "Almost"

---

## CONCRETE EXAMPLES:

**"Correct" examples:**
- Complete proof with all steps justified, all cases handled
- Would receive full marks
- Example: Full proof of AM-GM inequality with all cases verified

**"Almost" examples (SINGLE minor issue, 80-95% complete):**
- Complete proof but small calculation error in one step (e.g., arithmetic mistake in final line)
- Everything correct but missing trivial n=1 case (only one edge case)
- Main proof valid, one small lemma unstated (but lemma is obvious)
- 90% complete, needs only minor technical fix
- Solution is essentially correct, just needs polishing
- **Counter-example**: If missing n=1 AND n=2 cases → "Partial" (multiple issues)
- **Counter-example**: If calculation error AND missing case → "Partial" (multiple issues)
- **Counter-example**: If only have key insights but proof is incomplete → "Partial" (not Almost)

**"Partial" examples (MULTIPLE gaps OR major unfinished parts, 30-70% complete):**
- Proved only one direction of "if and only if" (missing entire other direction)
- Found right invariant but couldn't prove key property (two gaps: invariant found, proof missing)
- Has main idea but missing multiple key steps (multiple gaps)
- Proved key lemma but didn't connect to main result (two parts: lemma done, connection missing)
- 50-70% complete, significant work remains
- Made meaningful progress but solution is far from complete
- **Counter-example**: If only one small lemma missing → "Almost" (single issue)
- **Counter-example**: If solution is essentially complete with minor error → "Almost" (not Partial)

**"Incorrect" examples:**
- Approach fundamentally wrong (e.g., using induction where it doesn't apply)
- No meaningful progress (just restating the problem)
- Misunderstood problem (solving wrong problem)
- Solution completely off-track

---

## CRITICAL RULES:

1. **"Almost" = EXACTLY ONE minor fixable issue** (80-95% complete)
2. **"Partial" = MULTIPLE gaps OR major unfinished parts** (30-70% complete)
3. **"Incorrect" = NO meaningful progress** (0-25% complete)
4. **When in doubt, be CONSERVATIVE** (choose lower grade)
5. **Pay special attention to grading guidelines** - they often explicitly state "(Partial)" or "(Almost)"
6. **COUNT THE ISSUES**: If you can identify more than one problem → "Partial"
7. **CRITICAL vs TRIVIAL**: Missing a critical case (like n=0 for a counting problem) → "Partial"
8. **DIRECTIONS**: Proving only one direction of "if and only if" → "Partial" (major part missing)
9. **KEY INSIGHTS ALONE ≠ "Almost"**: Having correct ideas but incomplete proof → "Partial" (not Almost)
10. **COMPLETION LEVEL**: "Almost" requires solution to be essentially complete (80-95%), not just on the right track

---

## YOUR ANALYSIS PROCESS:

### Step 1: Understand
- Read the problem carefully - understand what is being asked
- Study the official solution - understand the complete correct approach
- Review the grading guidelines - note any explicit "(Partial)" or "(Almost)" markers

### Step 2: Analyze the Student's Answer
- Identify what the student got right (correct insights, approaches, partial results)
- Identify what the student got wrong (errors, gaps, missing parts)
- Count the number of distinct issues/gaps
- Assess the completion percentage (0-25%, 30-70%, 80-95%, or 100%)

### Step 3: Apply Classification Criteria
- **If 0 issues and 100% complete** → "Correct"
- **If 1 minor issue and 80-95% complete** → "Almost"
- **If 2+ issues OR major unfinished parts OR 30-70% complete** → "Partial"
- **If no meaningful progress OR fundamentally wrong approach** → "Incorrect"

### Step 4: Verify Against Guidelines
- Check if grading guidelines explicitly state "(Partial)" or "(Almost)"
- If guidelines conflict with your assessment, prioritize the guidelines
- Be conservative when in doubt (choose the lower grade)

---

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
- [ ] Did I COUNT the number of issues? (Exactly ONE = Almost, MULTIPLE = Partial)
- [ ] Did I assess completion level? (80-95% for Almost, 30-70% for Partial)
- [ ] Does my classification match the grading guidelines?
- [ ] If guidelines say "(Partial)", did I classify as "Partial"?
- [ ] If guidelines say "(Almost)", did I classify as "Almost"?
- [ ] Is there a critical case missing? If yes → "Partial"
- [ ] Is there only one direction of "if and only if"? If yes → "Partial"
- [ ] Does the student have key insights but incomplete proof? If yes → "Partial" (not Almost)
- [ ] Am I being conservative when in doubt?
- [ ] Is my response in the exact JSON format required?

## FINAL STEP: EXPLICIT ISSUE COUNT AND CLASSIFICATION
State explicitly in your reasoning:
1. "Completion level: [X]%" (estimate: 0-25%, 30-70%, 80-95%, or 100%)
2. "Number of issues: [N]" (count all gaps, errors, missing parts)
3. "Key insights present: [yes/no]" (did they find the right approach?)
4. "Major gaps remaining: [yes/no]" (are there significant unfinished parts?)

Then apply this classification:
- If N=0 and 100% complete → "Correct"
- If N=1, minor, and 80-95% complete → "Almost"
- If N≥2 OR any issue is major OR 30-70% complete → "Partial"
- If N=0 and no progress (0-25%) OR fundamentally wrong → "Incorrect"

**Remember**: "Almost" requires the solution to be essentially complete (80-95%) with only ONE minor fixable issue. If the solution has correct insights but significant gaps remain, it is "Partial" not "Almost".

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


class TaskAgentWithRetry(TaskAgent):
    """Task agent with retry mechanism for better reliability."""

    def forward(self, inputs: dict, max_retries: int = 2) -> tuple[str, list[dict]]:
        """Run the task agent with retry logic.
        
        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer
            max_retries: maximum number of retries if extraction fails

        Returns:
            (prediction, msg_history)
        """
        prediction, msg_history = super().forward(inputs)
        
        # If prediction is valid, return it
        if prediction in VALID_LABELS:
            return prediction, msg_history
        
        # Try retries with a reminder about the format
        for attempt in range(max_retries):
            self.log_fn(f"Retry attempt {attempt + 1}/{max_retries} due to invalid prediction: {prediction}")
            
            # Add a reminder message to the history
            retry_msg = """Your previous response did not follow the required format. 
Please provide ONLY a JSON object in this exact format:
<json>
{"response": "Correct"}
</json>

OR

<json>
{"response": "Incorrect"}
</json>

OR

<json>
{"response": "Partial"}
</json>

OR

<json>
{"response": "Almost"}
</json>

Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost".
Do not include any other text."""
            
            try:
                response, new_msg_history, info = get_response_from_llm(
                    msg=retry_msg,
                    model=self.model,
                    msg_history=msg_history,
                    temperature=0.1,
                )
                
                # Combine message histories
                msg_history = msg_history + new_msg_history
                
                # Try to extract prediction
                last_message = msg_history[-1]["text"] if msg_history else ""
                extracted_prediction = extract_prediction(last_message)
                
                if extracted_prediction is not None and extracted_prediction in VALID_LABELS:
                    prediction = extracted_prediction
                    self.log_fn(f"Successfully extracted prediction on retry: {prediction}")
                    break
            except Exception as e:
                self.log_fn(f"Error during retry: {e}")
                continue
        
        return str(prediction), msg_history
