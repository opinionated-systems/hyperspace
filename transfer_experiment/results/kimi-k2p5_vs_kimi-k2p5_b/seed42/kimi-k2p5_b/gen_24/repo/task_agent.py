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
        # Additional patterns for reasoning-based conclusions
        r'\bso\s+the\s+(?:answer|classification|grade)\s+(?:is|would\s+be)\s+["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bhence[,]?\s+(?:the\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
        r'\bthus[,]?\s+(?:the\s+)?["\']?(Correct|Incorrect|Partial|Almost)["\']?',
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
    text_lower = text.lower()
    # Find all occurrences and return the last one found
    last_pos = -1
    last_label = None
    for label in VALID_LABELS:
        pos = text_lower.rfind(label.lower())
        if pos > last_pos:
            last_pos = pos
            last_label = label
    if last_label:
        return last_label
    
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

## CLASSIFICATION DEFINITIONS (STRICT INTERPRETATION)

**Correct**: Complete, rigorous proof with ZERO issues. Full marks. Use ONLY if perfect.
- Every step is justified
- No logical gaps
- No calculation errors
- All cases covered

**Almost**: 80-95% complete with EXACTLY ONE minor issue. Solution is nearly done, just needs one small fix.
- **CRITICAL**: The answer looks almost complete - like a "Correct" answer with one tiny flaw
- Examples: one small calculation error, one trivial case missing, one step needs slight clarification
- The student clearly understands the solution and is 90%+ done
- **NOT Almost if**: 2+ issues, major logical gap, missing critical case, less than 80% complete
- **Key test**: If you could fix the issue in 1-2 sentences, it's Almost

**Partial**: 30-70% complete with meaningful progress but significant gaps remain.
- Found correct approach/key insight but couldn't finish the proof
- Multiple gaps (2+ issues) OR missing critical cases
- Proved only one direction of "if and only if"
- Has the right idea but execution is incomplete
- **NOT Partial if**: only one small issue (→ Almost), no progress at all (→ Incorrect)
- **Key test**: Good start but needs substantial work to complete (more than just a quick fix)

**Incorrect**: 0-25% complete. No meaningful progress OR fundamentally wrong approach.
- No key insight found
- Approach is wrong or irrelevant
- **NOT Incorrect if they found the key insight** (even incomplete → Partial)

---

## DETAILED EXAMPLES: Almost vs Partial

### Example 1: ALMOST (One minor arithmetic error)
Problem: Prove that 2+2=4
Student: "We know 2+2 = 3+1 = 5-1 = 4. Therefore 2+2=5." 
Analysis: The reasoning is correct but there's ONE arithmetic error at the end (said 5 instead of 4). This is ALMOST - one tiny fix needed.

### Example 2: ALMOST (Missing one trivial case)
Problem: Prove for all positive integers n...
Student: Complete proof for n≥2, but forgot to verify n=1.
Analysis: The proof is complete for the main case, just missing one trivial base case. This is ALMOST.

### Example 3: PARTIAL (Multiple gaps)
Problem: Prove A iff B
Student: Proved A→B correctly. For B→A, started the approach but got stuck and didn't finish.
Analysis: Only proved one direction completely. The other direction has a start but is incomplete. This is PARTIAL - significant work remains.

### Example 4: PARTIAL (Right idea, incomplete execution)
Problem: Find all solutions to a complex equation
Student: Identified the correct substitution method and set up the equation correctly, but only solved for 2 of the 4 cases.
Analysis: Found the key insight (substitution) but execution is incomplete. This is PARTIAL.

### Example 5: INCORRECT (Wrong approach)
Problem: Prove a number theory result
Student: Tried to use calculus methods on a discrete problem. No valid number theory insight.
Analysis: Completely wrong approach, no key insight. This is INCORRECT.

### Example 6: INCORRECT (No progress)
Problem: Prove by contradiction that √2 is irrational
Student: "I think √2 might be rational because it's close to 1.414."
Analysis: No valid mathematical reasoning, no attempt at proof structure. This is INCORRECT.

---

## CRITICAL DECISION FRAMEWORK

### Step 1: Check Grading Guidelines
- Look for "(Partial)" and "(Almost)" sections in the guidelines
- These explicitly describe what constitutes each grade

### Step 2: Assess Completion Level
- 95-100% complete + 0 issues → **Correct**
- 80-95% complete + 1 minor issue → **Almost**
- 30-70% complete + significant gaps → **Partial**
- 0-25% complete OR wrong approach → **Incorrect**

### Step 3: Count Issues (Critical for Almost vs Partial)
- **Exactly 1 minor issue** → **Almost**
- **2+ issues OR 1 major issue** → **Partial**

### Step 4: The "Fix Test" (Almost vs Partial)
- Can the answer be made Correct with a 1-2 sentence fix? → **Almost**
- Would require significant additional work to complete? → **Partial**

### Step 5: The "Key Insight" Test (Partial vs Incorrect)
- Did they find the right approach/technique? → **Partial** (at minimum)
- No key insight, completely wrong direction? → **Incorrect**

---

## COMMON MISTAKES TO AVOID

1. **Don't confuse Almost with Correct**: If there's ANY issue, it's not Correct
2. **Don't confuse Almost with Partial**: 
   - Almost = nearly perfect, one tiny flaw
   - Partial = good start but incomplete
3. **Don't grade based on length**: A short correct proof is Correct; a long incomplete proof is Partial
4. **Don't be too harsh on Almost**: If it's 90% there with one small issue, it's Almost not Partial
5. **Don't be too lenient on Incorrect**: If they found the key insight, it's at least Partial

---

## YOUR TASK

1. Read the problem, solution, and grading guidelines carefully
2. Compare student answer to the "(Partial)" and "(Almost)" sections in guidelines
3. Apply the "Fix Test": Can this be fixed in 1-2 sentences?
4. Count issues and assess completion percentage
5. Choose ONE classification: Correct, Almost, Partial, or Incorrect

---

## OUTPUT FORMAT (REQUIRED)

You MUST end your response with EXACTLY ONE of these JSON blocks:

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

**IMPORTANT**: 
- Provide your reasoning first, then the JSON block
- The JSON block must be the LAST thing in your response
- Use ONLY the four labels above, exactly as spelled
- Do not add any text after the JSON block"""

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

**RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect
2. Include the <json> tags
3. The JSON block must be the LAST thing in your response
4. No text after the JSON block

Now provide your classification."""
            
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

**RULES:**
1. Use ONLY one of: Correct, Almost, Partial, Incorrect
2. Include the <json> tags
3. The JSON block must be the LAST thing in your response
4. No text after the JSON block

Now provide your classification."""
            
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
