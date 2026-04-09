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
        # Final classification statement patterns
        r'Classification:\s*(Correct|Incorrect|Partial|Almost)',
        r'Final classification:\s*(Correct|Incorrect|Partial|Almost)',
        r'"Classification":\s*"(Correct|Incorrect|Partial|Almost)"',
        r"'Classification':\s*'(Correct|Incorrect|Partial|Almost)'",
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
    last_lines = lines[-30:] if len(lines) > 30 else lines
    for line in reversed(last_lines):
        line = line.strip()
        # Skip empty lines and common non-content lines
        if not line or line.startswith('```') or line.startswith('<json>') or line.startswith('</json>'):
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

## OUTPUT FORMAT - RESPOND WITH EXACTLY ONE OF THESE:

<json>
{{"response": "Correct"}}
</json>

<json>
{{"response": "Almost"}}
</json>

<json>
{{"response": "Partial"}}
</json>

<json>
{{"response": "Incorrect"}}
</json>

---

## CLASSIFICATION DEFINITIONS

### "Correct" (100% complete)
Use ONLY when:
- Complete, rigorous proof from start to finish
- All key steps present with no gaps
- Would receive full marks in IMO competition

### "Almost" (80-95% complete, SINGLE minor issue)
Use ONLY when ALL are true:
- Solution is essentially complete (80-95% done)
- EXACTLY ONE minor, fixable issue
- Main proof structure is valid and complete
- Examples: one small calculation error, one trivial edge case missing, one obvious lemma unstated
- **CRITICAL**: If you see 2+ issues → "Partial" (not Almost)
- **CRITICAL**: If solution is <80% complete → "Partial" (not Almost)
- **CRITICAL**: If grading guidelines say "(Almost)" and student matches that description → "Almost"

### "Partial" (30-70% complete, meaningful progress but gaps remain)
Use when ANY of these are true:
- Found correct key insight/approach but execution has substantial gaps
- MULTIPLE gaps in reasoning exist
- Missing critical cases that affect validity
- Proved only one direction of "if and only if"
- Has correct ideas but significant work remains (30-70% complete)
- **KEY TEST**: "Did they find the right approach but couldn't finish?" → "Partial"
- **CRITICAL**: If grading guidelines say "(Partial)" and student matches that description → "Partial"

### "Incorrect" (0-25% complete, no meaningful progress)
Use when ANY of these are true:
- Approach is fundamentally wrong
- No meaningful progress toward correct solution
- Critical logical flaws invalidate the argument
- Student misunderstood the problem

---

## INTERPRETING GRADING GUIDELINES

The grading guidelines often contain BOTH "(Partial)" and "(Almost)" sections with different criteria.

**How to use the guidelines:**
1. Read the "(Partial)" section - this describes what earns PARTIAL credit
2. Read the "(Almost)" section - this describes what earns ALMOST FULL credit
3. Compare the student's answer to BOTH sections
4. Choose the classification that matches where the student's answer falls

**Key insight**: The guidelines are hierarchical:
- "(Partial)" = achieved basic insights but incomplete
- "(Almost)" = nearly complete, just needs minor fixes
- If student matches "(Almost)" criteria → classify as "Almost"
- If student only matches "(Partial)" criteria → classify as "Partial"

---

## DECISION FLOWCHART

**STEP 1**: Did student make ANY meaningful progress?
- NO → **"Incorrect"** (STOP)
- YES → Continue

**STEP 2**: Is solution 100% complete with NO gaps?
- YES → **"Correct"** (STOP)
- NO → Continue

**STEP 3**: Check grading guidelines - does student match "(Almost)" criteria?
- YES → **"Almost"** (STOP)
- NO → Continue

**STEP 4**: Is solution 80-95% complete with EXACTLY ONE minor issue?
- YES → **"Almost"** (STOP)
- NO (has 2+ issues OR <80% complete) → Continue

**STEP 5**: Did student find correct approach but has significant gaps?
- YES → **"Partial"** (STOP)

---

## DETAILED EXAMPLES

**"Correct" examples:**
- Complete proof with all steps justified
- All cases handled properly
- Would receive full marks

**"Almost" examples (SINGLE minor issue only):**
- Complete proof but arithmetic error in final calculation
- Everything correct but missing trivial n=1 case
- Main proof valid, one small obvious lemma unstated
- 90% complete, needs only minor technical fix
- Solution is essentially done, just needs one small correction
- **NOT Almost**: Missing n=1 AND n=2 → "Partial" (multiple issues)
- **NOT Almost**: Only has key insights but proof incomplete → "Partial"
- **NOT Almost**: Missing critical case → "Partial"

**"Partial" examples (meaningful progress but incomplete):**
- Found right invariant but couldn't prove key property
- Has main idea but missing multiple key steps
- Proved key lemma but didn't connect to main result
- 50-70% complete, significant work remains
- Proved only one direction of "if and only if"
- Has correct approach but execution has substantial gaps
- **NOT Partial**: Only one small issue → "Almost"

**"Almost" vs "Partial" - How to Decide:**
- "Almost" = Solution is NEARLY COMPLETE (can see the finish line, just one small step away)
- "Partial" = Solution has SIGNIFICANT GAPS (right direction but lots of work still needed)
- When grading guidelines have BOTH sections, ask: "Does the student's answer match the (Almost) criteria or just the (Partial) criteria?"
- If student has what the guidelines call "(Almost)" achievement → classify as "Almost"
- If student only has what the guidelines call "(Partial)" achievement → classify as "Partial"

**"Incorrect" examples:**
- Approach fundamentally wrong
- No meaningful progress (just restating problem)
- Misunderstood what problem is asking

---

## CRITICAL RULES

1. **"Almost" = EXACTLY ONE minor issue AND 80-95% complete**
2. **"Partial" = MULTIPLE gaps OR 30-70% complete**
3. **"Incorrect" = NO meaningful progress**
4. **COUNT ISSUES**: 1 issue = Almost, 2+ issues = Partial
5. **Pay attention to grading guidelines** - they often mark "(Partial)" or "(Almost)"
   - If guidelines say "(Almost)" and student matches → classify as "Almost"
   - If guidelines say "(Partial)" and student matches → classify as "Partial"
   - Guidelines are the PRIMARY source for determining the classification
6. **Key insights alone ≠ Almost**: Must be 80-95% complete
7. **Missing critical case = Partial** (not Almost)
8. **One direction of iff = Partial** (major part missing)
9. **When in doubt between Partial and Almost**: 
   - If solution is 80%+ complete with 1 issue → "Almost"
   - If solution is <80% complete OR has 2+ issues → "Partial"
   - Look at what the grading guidelines emphasize

---

## YOUR ANALYSIS PROCESS

1. **Understand**: Read problem, study official solution, note grading guidelines
2. **Check Guidelines First**: Look for "(Partial)" and "(Almost)" sections in grading guidelines
   - These sections describe what achievements earn each classification
   - Compare student answer to BOTH sections
3. **Analyze**: Identify what student got right and wrong
4. **Count**: Count distinct issues/gaps (1 = Almost, 2+ = Partial)
5. **Assess**: Estimate completion percentage
6. **Classify**: Apply decision flowchart
7. **Verify**: Check against grading guidelines - does your classification match the appropriate section?

---

## BEFORE RESPONDING - CHECK:

- [ ] Did I check the grading guidelines for "(Partial)" and "(Almost)" sections?
- [ ] Does the student's answer match the "(Almost)" criteria in the guidelines?
- [ ] Did I count the issues? (1 = Almost, 2+ = Partial)
- [ ] Did I assess completion level? (80-95% = Almost, 30-70% = Partial)
- [ ] Is there only one direction of "if and only if"? → "Partial"
- [ ] Am I being conservative when in doubt between Partial and Almost?

## FINAL CLASSIFICATION STATEMENT

State explicitly:
- "Grading Guidelines Analysis: [Which section does student match - (Partial) or (Almost)?]"
- "Completion: [0-25%/30-70%/80-95%/100%]"
- "Issues found: [N]"
- "Key insight found: [yes/no]"
- "Classification: [Correct/Almost/Partial/Incorrect]"

Then provide ONLY the JSON response in the exact format shown above."""

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

You MUST provide ONLY a JSON object in this exact format (choose ONE):

<json>
{"response": "Correct"}
</json>

<json>
{"response": "Almost"}
</json>

<json>
{"response": "Partial"}
</json>

<json>
{"response": "Incorrect"}
</json>

**CRITICAL RULES:**
1. Use ONLY one of these four exact labels: "Correct", "Incorrect", "Partial", or "Almost"
2. The "response" field must contain ONLY the label - no extra text
3. You MUST include the <json> and </json> tags
4. Do not include any text before or after the JSON block
5. The JSON must be valid - use double quotes around the label

**CLASSIFICATION REMINDER:**
- "Correct" = 100% complete, no gaps
- "Almost" = 80-95% complete with EXACTLY ONE minor issue
  * Solution is nearly done, just needs one small fix
  * Check if grading guidelines have "(Almost)" section that matches
- "Partial" = 30-70% complete with multiple gaps OR missing critical parts
  * Has right approach but significant work remains
  * Check if grading guidelines have "(Partial)" section that matches
- "Incorrect" = no meaningful progress or wrong approach

**KEY DISTINCTION - Almost vs Partial:**
- "Almost" = ONE issue, 80-95% complete, nearly finished
- "Partial" = MULTIPLE issues OR <80% complete, substantial gaps remain
- Look at grading guidelines: which section does the student match?"""
            
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
