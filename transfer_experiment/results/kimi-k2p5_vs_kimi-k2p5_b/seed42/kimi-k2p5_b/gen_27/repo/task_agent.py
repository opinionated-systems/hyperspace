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

## CLASSIFICATION RUBRIC (USE THIS EXACTLY)

### STEP 1: Assess Completion Level
First, estimate what percentage of the solution is complete:
- 95-100%: Every step is fully worked out, essentially a complete proof
- 70-94%: Most of the solution is there but with gaps
- 30-69%: Some key ideas present but significant work missing
- 0-29%: Little to no valid progress

### STEP 2: Count Issues
Count how many distinct issues/problems exist in the answer:
- 0 issues: No errors, gaps, or missing steps
- 1 minor issue: One small error that doesn't affect the main logic (e.g., typo, arithmetic mistake, missing trivial case)
- 2+ issues OR 1 major issue: Multiple errors OR one significant gap in reasoning

### STEP 3: Apply Classification Rules

**Correct**: Use ONLY when:
- Completion: 100%
- Issues: 0
- The proof is complete, rigorous, and would receive full marks

**Almost**: Use ONLY when:
- Completion: 85-99% (nearly complete)
- Issues: EXACTLY 1 minor issue
- The answer is essentially correct but needs one tiny fix (1-2 sentences)
- Examples: one arithmetic error, one typo in final answer, one trivial case omitted
- KEY TEST: Could a teacher fix this to be perfect in under 30 seconds?

**Partial**: Use when:
- Completion: 30-84% (meaningful progress but incomplete)
- Issues: 2+ issues OR 1 major gap OR missing critical cases
- The student found the right approach/key insight but couldn't complete the proof
- Examples: proved only one direction of "if and only if", missing multiple cases, right idea but execution incomplete
- KEY TEST: Good start found, but needs substantial additional work

**Incorrect**: Use when:
- Completion: 0-29% (no meaningful progress)
- Issues: Fundamental misunderstanding or wrong approach
- No key insight found, or approach is completely wrong
- KEY TEST: Did they find any valid key insight? If yes → at least Partial

### STEP 4: Critical Decision Rules

1. **If 2+ issues exist → CANNOT be Almost → must be Partial or lower**
2. **If completion < 85% → CANNOT be Almost → must be Partial or lower**
3. **If key insight found → CANNOT be Incorrect → at least Partial**
4. **If proof is 100% complete with 0 issues → MUST be Correct**

### STEP 5: Final Classification
Based on your analysis above, select ONE label.

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
After analyzing the student's answer, I found it contains the key insight and proves the main case, but is missing the second direction of the if-and-only-if proof. This represents significant incomplete work.

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
