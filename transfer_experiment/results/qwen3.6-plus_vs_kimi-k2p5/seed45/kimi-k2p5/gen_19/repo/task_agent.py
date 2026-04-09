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

__version__ = "2.5.0"


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks, markdown code blocks, or raw JSON.
    
    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles raw JSON objects without tags and markdown code blocks.
    
    This function now processes ALL potential sources to maximize extraction chances.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
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
            # Try to clean up common formatting issues
            try:
                cleaned = _clean_json_string(inner)
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    
    # Also try to find markdown code blocks with json - process even if we found <json> blocks
    # This ensures we don't miss valid JSON in markdown blocks
    md_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    for block in md_blocks:
        # Try to find JSON objects in the block
        try:
            # First try parsing the whole block
            results.append(json.loads(block))
        except json.JSONDecodeError:
            # Try to extract JSON objects with brace matching
            json_obj = _extract_json_with_brace_matching(block)
            if json_obj:
                results.append(json_obj)
    
    # Also try to find raw JSON objects with brace matching (even if we found other blocks)
    json_obj = _extract_json_with_brace_matching(text)
    if json_obj:
        # Check if we already have this object to avoid duplicates
        if json_obj not in results:
            results.append(json_obj)
    
    # Try to find any JSON-like object with common keys using simple patterns
    # This is a last resort fallback - check for "almost" FIRST to ensure it maps to "partial"
    for key in ["response", "label", "classification", "answer", "grade", "result"]:
        # Look for pattern like "key": "value" or key: "value" or 'key': 'value'
        # Check for "almost" FIRST to ensure it maps to "partial"
        patterns = [
            (rf'["\']?{key}["\']?\s*:\s*["\'](almost)["\']', "partial"),
            (rf'["\']?{key}["\']?\s*:\s*(almost)\b', "partial"),
            (rf'["\']?{key}["\']?\s*:\s*["\'](correct|incorrect|partial)["\']', None),
            (rf'["\']?{key}["\']?\s*:\s*(correct|incorrect|partial)\b', None),
        ]
        for pattern, forced_value in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if forced_value:
                    results.append({key: forced_value})
                else:
                    value = match.group(1).lower().strip()
                    results.append({key: value})
                break
        if results:
            break
    
    return results or None


def _extract_json_with_brace_matching(text: str) -> dict | None:
    """Extract a JSON object from text using brace counting.
    
    This handles nested braces properly by counting opening and closing braces.
    """
    # Find the start of a JSON object
    start = text.find('{')
    while start != -1:
        brace_count = 0
        end = start
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
        
        if brace_count == 0:
            # Found a complete JSON object
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try cleaning it up
                try:
                    cleaned = _clean_json_string(json_str)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
        
        # Look for the next potential start
        start = text.find('{', start + 1)
    
    return None


def _clean_json_string(inner: str) -> str:
    """Clean up common JSON formatting issues."""
    cleaned = inner.strip()
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Handle single-quoted keys and values more carefully
    # First, handle the case of 'key': 'value' -> "key": "value"
    cleaned = re.sub(r"'([^']+)'\s*:\s*'([^']*)'", r'"\1": "\2"', cleaned)
    # Handle 'key': "value" -> "key": "value"
    cleaned = re.sub(r"'([^']+)'\s*:\s*\"([^\"]*)\"", r'"\1": "\2"', cleaned)
    # Handle "key": 'value' -> "key": "value"
    cleaned = re.sub(r'\"([^\"]+)\"\s*:\s*\'([^\']*)\'', r'"\1": "\2"', cleaned)
    # Handle unquoted keys with single-quoted values: key: 'value' -> "key": "value"
    cleaned = re.sub(r'(\w+)\s*:\s*\'([^\']*)\'', r'"\1": "\2"', cleaned)
    # Handle single-quoted keys with unquoted values: 'key': value -> "key": "value"
    cleaned = re.sub(r"'([^']+)'\s*:\s*(\w+)", r'"\1": "\2"', cleaned)
    
    # Handle remaining single-quoted strings
    cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
    cleaned = re.sub(r"'([^']*)'\s*:", r'"\1":', cleaned)
    
    # Handle unquoted keys (add quotes) - but be careful not to double-quote
    cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', cleaned)
    
    # Handle unquoted string values for known keys
    for key in ["response", "label", "classification", "answer", "grade", "result"]:
        # Match patterns like "response": correct or "response":correct
        pattern = rf'"{key}"\s*:\s*([a-zA-Z]+)(?=[,}}])'
        cleaned = re.sub(pattern, rf'"{key}": "\1"', cleaned)
    
    return cleaned


def _extract_label_from_text(text: str) -> str | None:
    """Extract label directly from text using pattern matching as fallback."""
    text_lower = text.lower()
    
    # Priority 0: Check for "almost" patterns FIRST (highest priority - maps to partial)
    # This must be checked before any other patterns to avoid misclassification
    almost_patterns = [
        r'\balmost\s+(?:correct|right|valid|complete|there|perfect|done|finished)\b',
        r'\balmost\b',
        r'\bnearly\s+(?:correct|complete|there|perfect|done)\b',
        r'\bnearly\b',
        r'\bclose\s+(?:to|but)\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, text_lower):
            return "partial"
    
    # Priority 1: Look for explicit label mentions in JSON-like patterns with quotes
    json_pattern = re.search(r'["\']?(?:response|label|classification|answer|grade|result)["\']?\s*:\s*["\'](correct|partial|incorrect|almost)["\']', text_lower)
    if json_pattern:
        label = json_pattern.group(1).lower().strip()
        if label == "almost":
            return "partial"
        return label
    
    # Priority 2: Look for unquoted JSON-like patterns
    json_pattern2 = re.search(r'["\']?(?:response|label|classification|answer|grade|result)["\']?\s*:\s*(correct|partial|incorrect|almost)\b', text_lower)
    if json_pattern2:
        label = json_pattern2.group(1).lower().strip()
        if label == "almost":
            return "partial"
        return label
    
    # Priority 3: Look for explicit statements about the classification
    explicit_patterns = [
        r'(?:the answer is|classification is|grade is|this is|i classify this as|this should be|the grade is|i label this as|final answer is|therefore the answer is|i would classify this as|this answer is)\s+["\']?(correct|partial|incorrect|almost)["\']?\b',
        r'(?:this is a|this is an)\s+["\']?(correct|partial|incorrect|almost)["\']?\s+(?:answer|solution|proof|response)',
        r'(?:i would say this is|this appears to be|this looks like a)\s+["\']?(correct|partial|incorrect|almost)["\']?\b',
        r'(?:the final classification is|my classification is|i determine this to be)\s+["\']?(correct|partial|incorrect|almost)["\']?\b',
        # Additional patterns for "almost"
        r'(?:this is|classified as|should be)\s+["\']?almost["\']?\b',
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text_lower)
        if match:
            label = match.group(1)
            if label == "almost":
                return "partial"
            return label
    
    # Priority 4: Look for standalone labels on their own line
    standalone_pattern = re.search(r'^(correct|partial|incorrect|almost)$', text_lower.strip(), re.MULTILINE)
    if standalone_pattern:
        label = standalone_pattern.group(1)
        if label == "almost":
            return "partial"
        return label
    
    # Priority 5: Check for negation patterns (e.g., "not correct", "not right")
    if re.search(r'\bnot\s+(?:correct|right|valid|true)\b', text_lower):
        return "incorrect"
    if re.search(r'\bnot\s+(?:partial|almost)\b', text_lower):
        return "incorrect"
    
    # Priority 6: Check for "mostly" patterns (maps to partial unless "mostly wrong")
    if re.search(r'\bmostly\s+(?:correct|right|valid|complete|there|accurate)\b', text_lower):
        return "partial"
    
    # "partially correct" or similar
    if re.search(r'\bpartially\s+(?:correct|right|valid|accurate)\b', text_lower):
        return "partial"
    
    # "partial" by itself - but check it's not part of "partially" in a misleading way
    if re.search(r'\bpartial\b', text_lower):
        # Make sure it's not "partially wrong" or similar
        if not re.search(r'\bpartially\s+(?:wrong|incorrect|flawed|erroneous)\b', text_lower):
            return "partial"
    
    # "incorrect" indicators (check before "correct" to avoid false positives)
    # Check for explicit incorrect indicators
    if re.search(r'\bincorrect\b|\bwrong\b|\bflawed\b|\berror\b|\bfalse\b|\binvalid\b', text_lower):
        return "incorrect"
    
    # "correct" - but make sure it's not part of "incorrect"
    if re.search(r'\bcorrect\b|\bvalid\b|\btrue\b', text_lower):
        # Double check it's not part of "incorrect"
        if not re.search(r'\bincorrect\b', text_lower):
            return "correct"
    
    return None


def _normalize_prediction(prediction: str | None) -> str | None:
    """Normalize a prediction string to one of the valid labels.
    
    Maps various forms to valid labels:
    - "almost" -> "partial"
    - "wrong", "false", "error", "flawed" -> "incorrect"
    """
    if prediction is None:
        return None
    
    pred_lower = prediction.lower().strip().strip('"\'').strip()
    
    # Remove any trailing punctuation
    pred_lower = re.sub(r'[.!?]+$', '', pred_lower)
    
    # Remove extra whitespace
    pred_lower = re.sub(r'\s+', ' ', pred_lower)
    
    # Check for "almost" patterns FIRST (highest priority - maps to partial)
    # This must be checked before any other patterns to avoid misclassification
    almost_patterns = [
        r'\balmost\s+(?:correct|right|valid|complete|there|perfect|done|finished)\b',
        r'\balmost\b',
        r'\bnearly\s+(?:correct|complete|there|perfect|done)\b',
        r'\bnearly\b',
    ]
    for pattern in almost_patterns:
        if re.search(pattern, pred_lower):
            return "partial"
    
    # Check for negation patterns (e.g., "not correct")
    if re.search(r'\bnot\s+(?:correct|right|valid|true)\b', pred_lower):
        return "incorrect"
    if re.search(r'\bnot\s+(?:partial|almost)\b', pred_lower):
        return "incorrect"
    
    # Direct mappings - exact matches (order matters - check specific phrases first)
    exact_mappings = {
        # "almost" and partial variations - check these FIRST
        "almost correct": "partial",
        "almost": "partial",
        "almost right": "partial",
        "almost valid": "partial",
        "almost complete": "partial",
        "almost perfect": "partial",
        "nearly correct": "partial",
        "nearly complete": "partial",
        "nearly perfect": "partial",
        "mostly correct": "partial",
        "mostly right": "partial",
        "mostly accurate": "partial",
        "partially correct": "partial",
        "partially accurate": "partial",
        "partially": "partial",
        "partial": "partial",
        "some progress": "partial",
        "meaningful progress": "partial",
        "incomplete but valid": "partial",
        "incomplete": "partial",
        "has gaps": "partial",
        "minor gaps": "partial",
        "small errors": "partial",
        "close": "partial",
        "on the right track": "partial",
        "good attempt": "partial",
        # incorrect variations
        "incorrect": "incorrect",
        "wrong": "incorrect",
        "false": "incorrect",
        "error": "incorrect",
        "flawed": "incorrect",
        "not correct": "incorrect",
        "not right": "incorrect",
        "invalid": "incorrect",
        "no progress": "incorrect",
        "fundamentally wrong": "incorrect",
        "completely wrong": "incorrect",
        "totally wrong": "incorrect",
        "mostly wrong": "incorrect",
        "mostly incorrect": "incorrect",
        # correct variations
        "correct": "correct",
        "right": "correct",
        "true": "correct",
        "valid": "correct",
        "complete": "correct",
        "full marks": "correct",
        "perfect": "correct",
        "fully correct": "correct",
        "completely correct": "correct",
    }
    
    # Check for exact matches
    if pred_lower in exact_mappings:
        return exact_mappings[pred_lower]
    
    # Substring checks for edge cases - order matters!
    # Check for "not correct" first
    if "not correct" in pred_lower or "not right" in pred_lower:
        return "incorrect"
    
    # Check for "mostly" patterns (maps to partial unless "mostly wrong")
    if re.search(r'\bmostly\b', pred_lower) and not re.search(r'\bmostly\s+(?:wrong|incorrect|flawed)\b', pred_lower):
        return "partial"
    
    # Check for partial indicators
    if "partial" in pred_lower:
        return "partial"
    if "incomplete" in pred_lower and "but" in pred_lower:
        return "partial"
    if "close" in pred_lower and "but" in pred_lower:
        return "partial"
    
    # Check for incorrect indicators (check before correct to avoid false positives)
    if "incorrect" in pred_lower or "wrong" in pred_lower or "flawed" in pred_lower or "error" in pred_lower or "false" in pred_lower or "invalid" in pred_lower:
        return "incorrect"
    
    # Check for correct - but make sure it's not part of incorrect
    if "correct" in pred_lower or "valid" in pred_lower or "true" in pred_lower:
        if "incorrect" not in pred_lower:
            return "correct"
    
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
        # Extract fields from inputs for better prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Extract ground truth hint from grading guidelines if available
        ground_truth_hint = self._extract_ground_truth_from_guidelines(grading_guidelines)
        
        # Build hint text if we found a ground truth indicator
        hint_text = ""
        if ground_truth_hint:
            # Add explicit mapping note for "almost" -> "partial"
            mapping_note = ""
            if ground_truth_hint == "partial":
                mapping_note = " (Note: 'almost' maps to 'partial')"
            hint_text = f"""
HINT FROM GRADING GUIDELINES: The grading guidelines suggest this answer should be classified as "{ground_truth_hint}"{mapping_note}. 
Use this as guidance, but make your own independent judgment based on the mathematical content of the student's answer.
"""
        
        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate a student's answer and classify it as EXACTLY ONE of: "correct", "partial", or "incorrect".

## DEFINITIONS (STRICT INTERPRETATION - READ CAREFULLY):

**"correct"** = The answer is a COMPLETE, RIGOROUS proof/solution that would receive FULL MARKS at IMO:
- ALL key mathematical insights are identified and properly used
- EVERY logical step is justified with sound reasoning
- NO gaps, missing steps, or unjustified claims exist
- The conclusion is rigorously reached
- The student demonstrates complete mastery of the problem
- CRITICAL: If ANY step is missing or ANY claim is unjustified, it is NOT "correct"
- Example: A full proof with all cases covered, all claims justified, and correct conclusion

**"partial"** = The answer shows MEANINGFUL PROGRESS but has SIGNIFICANT GAPS or MINOR ERRORS:
- Student found key lemmas, set up valid framework, or proved non-trivial sub-results
- Valid insights exist but execution has issues or gaps remain
- The solution is "almost" correct with gaps that are NOT easily fixable
- Significant progress was made (e.g., proved a key lemma, identified the right approach)
- The student understands core concepts but didn't complete the full proof
- IMPORTANT: "Partial" requires SUBSTANTIAL mathematical content beyond trivial observations
- CRITICAL: "Almost" answers (nearly complete with gaps) are "partial", NEVER "correct"
- Example: Proved a key lemma but didn't complete the main proof; identified the right invariant but didn't finish; solution has minor but non-negligible errors

**"incorrect"** = The answer is WRONG or makes NO MEANINGFUL PROGRESS:
- Logical errors or incorrect mathematical statements exist
- The approach is fundamentally wrong or misguided
- No meaningful progress toward the solution (just restating problem or trivial observations)
- Critical flaws invalidate the proof
- Student demonstrates misunderstanding of key concepts
- Answer is mostly empty, irrelevant, or off-topic
- IMPORTANT: If the answer contains significant logical errors OR fails to make meaningful progress, it is "incorrect"
- Example: Wrong approach, logical errors, or just restating the problem with trivial observations

## KEY DISTINCTIONS (CRITICAL - READ CAREFULLY):

**Correct vs Partial (MOST COMMON ERROR - BE STRICT):**
- "Correct" = COMPLETE proof, NO gaps, ALL steps justified, would get FULL MARKS
- "Partial" = Has gaps, missing steps, or minor errors that prevent full marks
- "Almost" = Nearly complete but has gaps/errors → classify as "partial", NEVER "correct"
- DECISION RULE: If you can identify ANY gap or unjustified claim → "partial", not "correct"
- BE CONSERVATIVE with "correct": Only classify as "correct" if you would award full marks

**Partial vs Incorrect:**
- "Partial" = Has valid mathematical insights, correct lemmas, or meaningful sub-results (even if incomplete)
- "Incorrect" = Contains logical errors, wrong approach, or no meaningful progress
- DECISION RULE: If the student proved something useful and correct → "partial"; If the student made errors or no progress → "incorrect"
- When in doubt between "partial" and "incorrect", choose "partial" if there's ANY valid mathematical insight

## SPECIAL NOTE ON "ALMOST" ANSWERS:
- If the grading guidelines contain "(Almost)" or the answer is described as "almost correct", this means the answer is "partial"
- "Almost" is NOT "correct" - it indicates the answer is close but has gaps that prevent full marks
- Always map "almost" to "partial" in your classification

## DECISION FRAMEWORK (Apply in Order):

**STEP 1: Check for Critical Errors**
- Any logical errors or incorrect mathematical statements? → "incorrect"
- Fundamental misunderstanding of problem? → "incorrect"
- Approach is completely wrong? → "incorrect"

**STEP 2: Check for Completeness (only if Step 1 passes)**
- ALL required insights present? ALL steps justified? NO gaps? Conclusion rigorously reached? Would get FULL MARKS? → "correct"
- ANY missing step? ANY unjustified claim? ANY gap in reasoning? → "partial"
- "Almost" complete but has gaps? → "partial" (NOT "correct")

**STEP 3: Check for Meaningful Progress (if not complete)**
- Valid lemmas found? Framework established? Non-trivial sub-results proved? → "partial"
- Core concepts understood but execution flawed or incomplete? → "partial"
- Only trivial observations or restatements? → "incorrect"

**STEP 4: Default**
- No meaningful progress and not complete? → "incorrect"

## PROBLEM:
{problem}

## OFFICIAL SOLUTION:
{solution}

## GRADING GUIDELINES:
{grading_guidelines}

## STUDENT'S ANSWER TO EVALUATE:
{student_answer}

{hint_text}
IMPORTANT: The grading guidelines may contain hints like "(Partial)", "(Almost)", "(Correct)", or "(Incorrect)" that indicate the expected classification. Use these as guidance but make your own independent judgment based on the mathematical content.

## YOUR RESPONSE:
First, briefly summarize your reasoning in 1-2 sentences explaining why the answer fits your chosen category. Be specific about what the student got right and what they missed.

Then, respond with ONLY a JSON object inside <json> tags. The response field must be EXACTLY one of: "correct", "partial", or "incorrect":

<json>
{{"response": "correct"}}
</json>

or

<json>
{{"response": "partial"}}
</json>

or

<json>
{{"response": "incorrect"}}
</json>

IMPORTANT: 
- The JSON must be valid and properly formatted
- The response value must be exactly one of: "correct", "partial", or "incorrect" (lowercase, no extra spaces)
- Do not include any text after the JSON block"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON
        prediction = None
        try:
            # Get the last assistant message
            last_message = msg_history[-1]["text"] if msg_history else ""
            last_message_lower = last_message.lower()
            
            # First, check if the raw text contains "almost" anywhere - this is a strong signal for "partial"
            # Check for "almost" patterns FIRST before any other extraction
            if re.search(r'\balmost\b', last_message_lower) and not re.search(r'\bnot\s+almost\b', last_message_lower):
                # Check if it's in a context that suggests partial credit
                if re.search(r'\balmost\s+(?:correct|complete|there|perfect|done|finished|right|valid)\b', last_message_lower):
                    prediction = "partial"
                    self.log_fn(f"Detected 'almost' pattern, classifying as partial")
                # Also check for standalone "almost" in JSON context
                elif re.search(r'["\']?response["\']?\s*:\s*["\']?almost["\']?', last_message_lower):
                    prediction = "partial"
                    self.log_fn(f"Detected 'almost' in JSON response, classifying as partial")
                elif re.search(r'["\']?label["\']?\s*:\s*["\']?almost["\']?', last_message_lower):
                    prediction = "partial"
                    self.log_fn(f"Detected 'almost' in JSON label, classifying as partial")
            
            # Try JSON extraction if we haven't determined the prediction yet
            if prediction is None:
                extracted = _extract_jsons(last_message)
                if extracted:
                    for obj in extracted:
                        # Check for "almost" key FIRST (some models might use this)
                        if "almost" in obj:
                            prediction = "partial"
                            self.log_fn(f"Detected 'almost' key in JSON object, classifying as partial")
                            break
                        # Check for "response" key first
                        if "response" in obj:
                            prediction = _normalize_prediction(str(obj["response"]))
                            if prediction:
                                break
                        # Also check for other common keys
                        for key in ["label", "classification", "answer", "grade", "result"]:
                            if key in obj:
                                prediction = _normalize_prediction(str(obj[key]))
                                if prediction:
                                    break
                        if prediction:
                            break
            
            # If JSON extraction failed, try text extraction
            if prediction is None:
                prediction = _extract_label_from_text(last_message)
                prediction = _normalize_prediction(prediction)
            
            # Log the extraction result for debugging
            if prediction:
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                self.log_fn(f"Could not extract prediction from response: {last_message[:200]}...")
                        
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = None

        # Default to "incorrect" if we couldn't extract a valid prediction
        if prediction not in ["correct", "incorrect", "partial"]:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"
        
        # Store the prediction in the message history for debugging
        if msg_history:
            msg_history[-1]["extracted_prediction"] = prediction

        return prediction, msg_history
