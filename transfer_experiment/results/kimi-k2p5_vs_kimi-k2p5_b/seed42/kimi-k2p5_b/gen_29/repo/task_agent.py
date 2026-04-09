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
        
        # Try to parse the JSON
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common issues
            # Remove any trailing commas before closing braces
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            # Try again with fixed version
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field using regex
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                if response_match:
                    results.append({"response": response_match.group(1)})
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
            # Try to fix common issues
            fixed = re.sub(r',\s*}', '}', match.strip())
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field using regex
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', match)
                if response_match:
                    results.append({"response": response_match.group(1)})
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
            # Try to fix common issues
            fixed = re.sub(r',\s*}', '}', match.strip())
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    results.append(parsed)
            except json.JSONDecodeError:
                # Try to extract just the response field using regex
                response_match = re.search(r'"response"\s*:\s*"([^"]+)"', match)
                if response_match:
                    results.append({"response": response_match.group(1)})
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
        # Additional patterns for reasoning-based conclusions
        r'\bso\s+the\s+(?:answer|classification|grade)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bhence[,]?\s+(?:the\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bthus[,]?\s+(?:the\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        # Additional patterns for explicit statements
        r'\bI\s+choose\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bMy\s+(?:classification|grade|answer)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bThe\s+(?:correct|appropriate)\s+(?:classification|grade)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bThis\s+(?:is|should\s+be)\s+(?:graded|classified)\s+as\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bI\s+(?:select|pick|choose)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
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
    
    # Final fallback: check the very last non-empty line for exact match
    for line in reversed(lines):
        line_clean = line.strip().strip('.,;:!?"\'`*[]{}()<>')
        if line_clean:
            for label in VALID_LABELS:
                if line_clean.lower() == label.lower():
                    return label
    
    # Ultra fallback: search entire text for any occurrence of valid labels
    # This is a last resort when all other methods fail
    # IMPORTANT: Check for longer labels first to avoid substring matches
    # e.g., "incorrect" contains "correct", so check "incorrect" first
    text_lower = text.lower()
    # Sort labels by length (longest first) to avoid substring issues
    sorted_labels = sorted(VALID_LABELS, key=len, reverse=True)
    
    # Find all occurrences and return the last one found
    last_pos = -1
    last_label = None
    for label in sorted_labels:
        # Use word boundary to avoid matching "correct" inside "incorrect"
        pattern = r'\b' + re.escape(label.lower()) + r'\b'
        for match in re.finditer(pattern, text_lower):
            if match.start() > last_pos:
                last_pos = match.start()
                last_label = label
    if last_label:
        return last_label
    
    # Final final fallback: look for labels in any context (very last resort)
    # Still use sorted labels to prefer longer matches
    for label in sorted_labels:
        if label.lower() in text_lower:
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
    # Clean the text first - remove common formatting issues
    cleaned_text = text.strip()
    
    # Try <json> tags first
    extracted = _extract_jsons(cleaned_text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                # Case-insensitive validation
                for label in VALID_LABELS:
                    if response.lower() == label.lower():
                        return label
    
    # Try ```json code blocks
    extracted = _extract_json_from_code_blocks(cleaned_text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                for label in VALID_LABELS:
                    if response.lower() == label.lower():
                        return label
    
    # Try raw JSON
    extracted = _extract_json_raw(cleaned_text)
    if extracted:
        for item in extracted:
            if isinstance(item, dict) and "response" in item:
                response = str(item["response"]).strip()
                for label in VALID_LABELS:
                    if response.lower() == label.lower():
                        return label
    
    # Fallback: direct label extraction from text
    return _extract_direct_label(cleaned_text)


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

## CLASSIFICATION RUBRIC - FOLLOW THESE RULES EXACTLY

### The Four Categories:

**Correct**: The answer is 100% complete and correct.
- Every step is fully worked out with no gaps
- No errors, typos, or missing cases
- Would receive full marks in a competition
- The proof is rigorous and complete

**Almost**: The answer is nearly perfect with exactly ONE tiny issue.
- 95-99% complete - essentially a full proof
- Only ONE minor error (e.g., arithmetic mistake, typo, trivial case omitted)
- Can be fixed to perfect in under 30 seconds
- **CRITICAL**: If there are 2+ issues, CANNOT be "Almost"
- **CRITICAL**: "Almost" is RARE - only use when the proof is essentially complete

**Partial**: The answer shows meaningful progress but has significant gaps.
- 30-84% complete - found key insight but couldn't finish
- Multiple issues OR one major gap in reasoning
- Missing critical cases or directions (e.g., only proved one direction of "if and only if")
- Good start, but needs substantial additional work
- **CRITICAL**: If student found a valid key insight, CANNOT be "Incorrect"
- **CRITICAL**: This is the MOST COMMON category for incomplete work

**Incorrect**: The answer shows little to no valid progress.
- 0-29% complete - no meaningful work toward solution
- Fundamental misunderstanding or completely wrong approach
- No key insight found
- Random guessing or completely irrelevant work

---

## DECISION WORKFLOW - FOLLOW STEP BY STEP

**Step 1**: Did the student find ANY valid key insight or make meaningful progress toward the solution?
- Look for: correct approach, valid lemma, key observation, partial proof
- If NO valid insight at all → **Incorrect**
- If YES, found something valid → Continue to Step 2

**Step 2**: Is the proof essentially complete (95%+)?
- Check: Are all major steps present? Is the logic mostly complete?
- If NO, significant gaps remain → **Partial**
- If YES, essentially a complete proof → Continue to Step 3

**Step 3**: Count the issues carefully. Is there exactly ONE minor issue?
- Minor issues: single arithmetic error, one typo, one trivial case omitted
- Major issues: missing proof direction, logical gap, multiple errors
- If exactly ONE minor issue → **Almost**
- If ZERO issues → **Correct**
- If TWO or more issues (even if minor) → **Partial**

---

## DETAILED EXAMPLES

**Example 1 - Correct:**
Problem: Prove that for all positive integers n, n³ - n is divisible by 6.
Student: Shows n³ - n = n(n-1)(n+1), explains this is product of 3 consecutive integers, 
so divisible by 2 (one even) and 3 (one multiple of 3), hence by 6. All cases covered.
→ Classification: Correct

**Example 2 - Almost:**
Problem: Find all primes p such that p² + 2 is also prime.
Student: Correctly shows p=3 works (gives 11), proves p=2 gives 6 (not prime), 
shows for p>3, p² ≡ 1 (mod 3) so p²+2 ≡ 0 (mod 3) and composite. 
**BUT** makes one arithmetic error: says 3²+2=10 instead of 11, though conclusion is correct.
→ Classification: Almost (one minor arithmetic error in an otherwise complete proof)

**Example 3 - Partial:**
Problem: Prove that if a² + b² + c² = ab + bc + ca for real numbers a, b, c, then a = b = c.
Student: Rearranges to (a-b)² + (b-c)² + (c-a)² = 0 correctly, but then says 
"this implies a=b=c" without explaining why squares summing to zero means each is zero.
Missing the key step justification.
→ Classification: Partial (found key insight but gap in reasoning)

**Example 4 - Partial (missing direction):**
Problem: Prove that triangle ABC is equilateral if and only if a² + b² + c² = ab + bc + ca.
Student: Only proves the forward direction (equilateral → equation holds).
Does not prove the converse (equation holds → equilateral).
→ Classification: Partial (only proved one direction of "if and only if")

**Example 5 - Incorrect:**
Problem: Prove there are infinitely many primes.
Student: Lists primes 2, 3, 5, 7, 11 and says "we can always find more by checking larger numbers."
No valid proof structure, no key insight about constructing new primes.
→ Classification: Incorrect (no meaningful progress toward actual proof)

**Example 6 - Incorrect:**
Problem: Find all functions f: R→R such that f(x+y) = f(x) + f(y).
Student: Guesses f(x) = x² and checks it doesn't work, then gives up.
No valid approach to finding linear solutions.
→ Classification: Incorrect (wrong approach, no key insight)

---

## COMMON MISTAKES TO AVOID

1. **DON'T overuse "Almost"**: "Almost" should be rare. Most incomplete work is "Partial".
2. **DON'T call work with multiple issues "Almost"**: Two typos = Partial, not Almost.
3. **DON'T call work with major gaps "Almost"**: Missing a proof direction = Partial.
4. **DON'T call work with valid insights "Incorrect"**: If they found something meaningful, it's at least Partial.
5. **DO use "Partial" for incomplete proofs with valid insights**: This is the most common category.

---

## OUTPUT FORMAT (MANDATORY)

You MUST end your response with EXACTLY ONE JSON block in this format:

<json>
{{"response": "Correct"}}
</json>

OR

<json>
{{"response": "Almost"}}
</json>

OR

<json>
{{"response": "Partial"}}
</json>

OR

<json>
{{"response": "Incorrect"}}
</json>

**CRITICAL RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect (exact spelling, case-sensitive)
2. The JSON block must be the LAST thing in your response
3. No text after the JSON block
4. Use double quotes in the JSON, not single quotes
5. Include the <json> tags exactly as shown

**Example of correct output:**
After analyzing the student's answer, I found they identified the key approach and proved the main direction, but did not prove the converse. This is significant incomplete work.

<json>
{{"response": "Partial"}}
</json>"""

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


# Default to using TaskAgentWithRetry for better reliability
class DefaultTaskAgent(TaskAgent):
    """Default task agent with retry mechanism."""
    
    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent with retry logic."""
        # First attempt
        prediction, msg_history = super().forward(inputs)
        
        # If prediction is valid, return it
        if prediction in VALID_LABELS:
            return prediction, msg_history
        
        # Try retries with a reminder about the format
        for attempt in range(2):
            self.log_fn(f"Retry attempt {attempt + 1}/2 due to invalid prediction: {prediction}")
            
            # Add a reminder message to the history
            retry_msg = """Your previous response did not follow the required format. 

You MUST end your response with EXACTLY ONE JSON block in this format:

<json>
{"response": "Correct"}
</json>

OR

<json>
{"response": "Almost"}
</json>

OR

<json>
{"response": "Partial"}
</json>

OR

<json>
{"response": "Incorrect"}
</json>

**CRITICAL RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect (exact spelling, case-sensitive)
2. Include the <json> tags exactly as shown
3. The JSON block must be the LAST thing in your response
4. No text after the JSON block
5. Use double quotes in the JSON, not single quotes

**Remember the classification rules:**
- Correct: 100% complete, no issues
- Almost: 95-99% complete, exactly ONE minor issue (RARE - use sparingly!)
- Partial: 30-84% complete, found key insight but significant gaps remain (MOST COMMON for incomplete work)
- Incorrect: 0-29% complete, no meaningful progress

**Key reminders:**
- "Almost" should be RARE - only when proof is essentially complete with exactly one tiny issue
- "Partial" is for work with valid insights but significant gaps (most common)
- "Incorrect" is only when NO valid insight was found

Now provide ONLY the JSON block with your classification."""
            
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

You MUST end your response with EXACTLY ONE JSON block in this format:

<json>
{"response": "Correct"}
</json>

OR

<json>
{"response": "Almost"}
</json>

OR

<json>
{"response": "Partial"}
</json>

OR

<json>
{"response": "Incorrect"}
</json>

**CRITICAL RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect (exact spelling, case-sensitive)
2. Include the <json> tags exactly as shown
3. The JSON block must be the LAST thing in your response
4. No text after the JSON block
5. Use double quotes in the JSON, not single quotes

**Remember the classification rules:**
- Correct: 100% complete, no issues
- Almost: 95-99% complete, exactly ONE minor issue (RARE - use sparingly!)
- Partial: 30-84% complete, found key insight but significant gaps remain (MOST COMMON for incomplete work)
- Incorrect: 0-29% complete, no meaningful progress

**Key reminders:**
- "Almost" should be RARE - only when proof is essentially complete with exactly one tiny issue
- "Partial" is for work with valid insights but significant gaps (most common)
- "Incorrect" is only when NO valid insight was found

Now provide ONLY the JSON block with your classification."""
            
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
