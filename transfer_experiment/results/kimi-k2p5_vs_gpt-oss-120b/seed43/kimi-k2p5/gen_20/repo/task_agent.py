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
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field with regex
                try:
                    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                    if response_match:
                        results.append({"response": response_match.group(1)})
                except Exception:
                    continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks like ```json...```.
    
    Also handles plain JSON objects that might not be wrapped in tags.
    """
    results = []
    
    # Try to find JSON in markdown code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.strip())
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try to extract just the response field with regex
                try:
                    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', match)
                    if response_match:
                        results.append({"response": response_match.group(1)})
                except Exception:
                    continue
    
    # Try to find plain JSON objects (looking for {"response": ...} pattern)
    # This handles cases where the LLM outputs JSON without proper tags
    plain_json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    plain_matches = re.findall(plain_json_pattern, text, re.DOTALL)
    for match in plain_matches:
        try:
            results.append({"response": match})
        except Exception:
            continue
    
    return results or None


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects with flexible parsing.
    
    Tries multiple strategies:
    1. Look for <json>...</json> tags
    2. Look for ```json...``` code blocks
    3. Look for plain JSON objects with "response" field
    4. Look for raw JSON objects in the text
    """
    results = []
    
    # Strategy 1: Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Strategy 2: Try markdown code blocks and plain JSON
    results = _extract_json_from_markdown(text)
    if results:
        return results
    
    # Strategy 3: Look for raw JSON objects (brace matching)
    # Find all potential JSON starting points
    for match in re.finditer(r'\{', text):
        start = match.start()
        # Try to find matching closing brace
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_obj = json.loads(text[start:i+1])
                        results.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    break
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer, points

        Returns:
            (prediction, msg_history)
        """
        # Extract key fields for better context
        domain = inputs.get('domain', 'Unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert mathematical grader evaluating IMO-level solutions. Your task is to classify the student's solution into exactly ONE of four categories.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

=== CLASSIFICATION CRITERIA ===

Choose exactly ONE label:

**Correct**: Complete, rigorous, correct solution. ALL claims proven, ALL cases handled, NO gaps. The solution would receive full marks.

**Almost**: The solution has a COMPLETE STRUCTURE with only MINOR issues. The main proof framework is fully present and sound. Issues are limited to: small calculation errors, missing one trivial case, minor technical flaws, or slight gaps that are easy to fix. The hard mathematical work is DONE - it's 90-95% complete. Think: "This is essentially correct, just needs polishing."

**Partial**: Shows SUBSTANTIAL progress with KEY mathematical insights, but the solution structure is INCOMPLETE. The student found important invariants, key lemmas, or viable approaches central to the problem, but significant work remains to complete the proof. The main framework is NOT fully built. Think: "They found the key idea, but there's still substantial work to do."

**Incorrect**: NO meaningful mathematical progress. Just restating the problem, trivial observations, wrong approach, no valuable insights. When in doubt, choose Incorrect.

=== CRITICAL DISTINCTIONS (READ CAREFULLY) ===

**Correct vs Almost vs Partial vs Incorrect**:

CORRECT = Complete proof from start to finish, all cases handled, all claims rigorously proven. No gaps. No missing steps. Full marks.

ALMOST = Complete proof structure present from start to finish. All main arguments are there. Only minor issues: small calculation errors, missing one trivial case, minor technical flaw. The hard work is done. An expert would say "this is essentially correct, just needs polishing."

PARTIAL = INCOMPLETE proof structure. Student found KEY insights (invariant, lemma, viable approach) but significant sections are missing. The main framework is NOT fully built. An expert would say "they found the key idea but need to finish the proof."

INCORRECT = NO meaningful mathematical progress. Just restating the problem, trivial observations, wrong approach, no valuable insights. Be CONSERVATIVE.

**KEY TEST for Almost vs Partial**:
- Can you trace a complete logical path from given conditions to the conclusion with only minor gaps? → ALMOST
- Are there major sections completely missing or only partially developed? → PARTIAL

**KEY TEST for Partial vs Incorrect**:
- Did the student find a KEY insight that genuinely helps solve the problem (invariant, lemma, correct approach)? → PARTIAL
- Is it just restating the problem or making trivial observations with no real insight? → INCORRECT

=== DECISION PROCESS ===

Follow this step-by-step STRICTLY:

STEP 1: Check for "Correct"
- Does the solution have a complete, rigorous proof?
- Are ALL claims proven?
- Are ALL cases handled?
- Are there NO gaps or missing steps?
- If YES to all → "Correct"
- If NO → Continue to STEP 2

STEP 2: Check for "Almost" 
- Is the proof structure COMPLETE from start to finish?
- Are all main arguments present?
- Are issues limited to minor ones (calculation errors, small gaps, missing trivial case)?
- Is the hard mathematical work essentially done (90-95% complete)?
- If YES to all → "Almost"
- If NO → Continue to STEP 3

STEP 3: Check for "Partial"
- Does the solution show SUBSTANTIAL progress?
- Did the student find KEY mathematical insights (invariants, lemmas, viable approach)?
- Is the proof structure INCOMPLETE with significant work remaining?
- CRITICAL: Is there more than just restating the problem or trivial observations?
- If YES to all → "Partial"
- If NO → "Incorrect"

STEP 4: "Incorrect"
- No meaningful mathematical progress
- Just restating the problem or trivial observations
- Wrong approach with no valuable insights

=== DETAILED EXAMPLES ===

**Correct**:
- Complete rigorous proof with all cases handled, all claims proven
- Every step is justified, no logical gaps
- Would receive full marks

**Almost**:
- Complete proof with one small calculation error
- Complete proof missing one trivial boundary case  
- Complete proof with a minor technical flaw that's easy to fix
- All main arguments present, just needs polishing
- An expert would say "essentially correct"

**Partial**:
- Found the key invariant but only proved it for special cases
- Identified the right lemma but didn't complete the inductive step
- Has the main insight but the proof structure is incomplete
- Made meaningful progress beyond restating but significant gaps remain
- An expert would say "good start but incomplete"

**Incorrect**:
- Only restating the problem in different words
- Trivial observations that don't help solve the problem
- Wrong approach with no valuable insights
- Random attempts that don't lead anywhere
- No genuine mathematical progress

=== CRITICAL WARNINGS - READ CAREFULLY ===

1. **BE CONSERVATIVE WITH PARTIAL**: "Partial" requires GENUINE KEY INSIGHTS that significantly advance toward the solution. Just mentioning some observations, formulas, or definitions is NOT enough. If the student hasn't found a real insight that helps solve the problem, it's INCORRECT.

2. **ALMOST REQUIRES COMPLETE STRUCTURE**: "Almost" means the proof is essentially complete (90-95%). If there are significant gaps, missing sections, or major parts of the proof are not developed, it's PARTIAL, not Almost.

3. **WHEN IN DOUBT, CHOOSE LOWER**: If you're unsure between two categories, always choose the LOWER one (more conservative). It's better to under-grade than over-grade.

4. **DON'T BE FOOLED BY LENGTH**: A long response with many words but no real mathematical insight is still INCORRECT. Length ≠ Quality.

5. **CHECK THE GRADING GUIDELINES**: The grading guidelines provided show what constitutes Partial vs Almost vs Incorrect for this specific problem. Use them as your primary reference.

=== COMMON ERRORS TO AVOID ===

1. **Don't over-grade Partial**: If the solution just restates the problem or makes trivial observations, it's INCORRECT, not Partial.

2. **Don't under-grade Almost**: If the proof structure is complete with only minor issues, it's ALMOST, not Partial.

3. **Don't over-grade Almost**: If there are significant gaps in the proof structure, it's PARTIAL, not Almost.

4. **Be conservative with Partial**: The student must have found a GENUINE key insight that helps solve the problem. Just writing down some formulas or making observations is NOT enough.

5. **Don't over-grade Correct**: If there are ANY gaps or unproven claims, it's not Correct. Correct requires perfection.

6. **Don't under-grade Partial**: If the student genuinely found a key insight that advances toward the solution, it's PARTIAL, not Incorrect.

=== GRADING STRATEGY ===

When evaluating, ask yourself:
1. What is the KEY insight needed to solve this problem?
2. Did the student find this insight or something equivalent?
3. How much of the proof structure is complete?

Be HONEST and CONSERVATIVE. It's better to slightly under-grade than over-grade.

**FINAL CHECK**: Before giving your answer, verify:
- If you chose "Partial": Does the student have a GENUINE key insight? If not, change to "Incorrect".
- If you chose "Almost": Is the proof structure 90-95% complete? If not, change to "Partial".
- If you chose "Correct": Is there ANY gap or unproven claim? If yes, change to "Almost" or "Partial".

Provide your evaluation in this exact JSON format:

<json>
{{
    "response": "LABEL: Your brief reasoning here"
}}
</json>

Where LABEL is one of: Correct, Almost, Partial, Incorrect

CRITICAL RULES:
1. Start with the exact label followed by a colon (e.g., "Correct: ..." or "Partial: ...")
2. Put your reasoning after the colon
3. Wrap everything in <json>...</json> tags
4. No text outside the JSON tags - ONLY output the JSON block
5. Follow the 4-step decision process above carefully
6. When in doubt between two categories, choose the LOWER one (more conservative)
7. "Partial" requires GENUINE key insights, not just restating or trivial observations
8. "Correct" requires a PERFECT solution with NO gaps or errors

EXAMPLE OUTPUT:
<json>
{{
    "response": "Partial: The student found the key invariant but didn't complete the inductive step."
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            
            # PRIORITY 0: Quick check for direct label at the very start of the message
            # This handles cases where LLM outputs just "Correct" or "Partial: ..." directly
            quick_label = self._quick_extract_label(last_message)
            if quick_label != "None":
                prediction = quick_label
                self.log_fn(f"Quick extracted prediction: {prediction}")
                return str(prediction), msg_history
            
            # PRIORITY 1: Try JSON extraction first (most reliable)
            extracted = _extract_json_flexible(last_message)
            if extracted:
                # Try to get response from the last JSON object
                last_json = extracted[-1]
                if isinstance(last_json, dict):
                    if "response" in last_json:
                        response_text = last_json["response"]
                        # Extract just the label from the response value
                        prediction = self._extract_label_from_response(response_text)
                        self.log_fn(f"Extracted label from JSON response field: {prediction}")
                    else:
                        # If no "response" key, try to find any string value that looks like a label
                        found_label = False
                        for key, value in last_json.items():
                            if isinstance(value, str):
                                label = self._extract_label_from_response(value)
                                if label != "None":
                                    prediction = label
                                    found_label = True
                                    self.log_fn(f"Extracted label from JSON key '{key}': {prediction}")
                                    break
                        if not found_label:
                            # If no label found in values, use the first value as string
                            if last_json:
                                first_value = list(last_json.values())[0]
                                prediction = str(first_value)
                                self.log_fn(f"Using first JSON value as prediction: {prediction}")
                            else:
                                prediction = "None"
                else:
                    # If extracted is not a dict, try to use it directly
                    prediction = str(last_json)
                    self.log_fn(f"Using non-dict JSON value: {prediction}")
            else:
                # PRIORITY 2: Try to extract label directly from the text (fallback)
                direct_label = self._extract_label(last_message)
                if direct_label != "None":
                    prediction = direct_label
                    self.log_fn(f"Directly extracted prediction from text: {prediction}")
                else:
                    self.log_fn(f"Could not extract prediction from text, using None")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any of the expected labels from raw text
            try:
                last_message = msg_history[-1]["text"] if msg_history else ""
                prediction = self._extract_label(last_message)
                self.log_fn(f"Fallback extraction result: {prediction}")
            except Exception as e2:
                self.log_fn(f"Fallback extraction also failed: {e2}")
                prediction = "None"

        return str(prediction), msg_history

    def _extract_label_from_response(self, text: str) -> str:
        """Extract label from the response value in JSON (strict parsing).
        
        This is used when we have a JSON response value like "Partial: The solution..."
        We want to extract just the label at the start, not from anywhere in the text.
        
        Returns one of: "Correct", "Almost", "Partial", "Incorrect", or "None"
        """
        if not text:
            return "None"
        
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # Strategy 1: Look for explicit label at the very start followed by colon or space
        # This handles "Partial: ..." or "Partial ..." format
        label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)\b[:\s]', text_lower)
        if label_prefix_match:
            return label_prefix_match.group(1).capitalize()
        
        # Strategy 2: Look for label at start of any line (for multi-line responses)
        for line in text_lower.split('\n'):
            line = line.strip()
            if line:
                label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)\b[:\s]', line)
                if label_prefix_match:
                    return label_prefix_match.group(1).capitalize()
        
        # Strategy 3: Look for label as a standalone word at the very start
        standalone_match = re.match(r'^(correct|almost|partial|incorrect)\b', text_lower)
        if standalone_match:
            return standalone_match.group(1).capitalize()
        
        # Strategy 4: Look for label as standalone word at start of any line
        for line in text_lower.split('\n'):
            line = line.strip()
            if line:
                standalone_match = re.match(r'^(correct|almost|partial|incorrect)\b', line)
                if standalone_match:
                    return standalone_match.group(1).capitalize()
        
        # Strategy 5: Look for label in quotes anywhere in text
        quoted_match = re.search(r'["\'](correct|almost|partial|incorrect)["\']', text_lower)
        if quoted_match:
            return quoted_match.group(1).capitalize()
        
        # Strategy 6: Look for label in parentheses or brackets
        paren_match = re.search(r'[\(\[](correct|almost|partial|incorrect)[\)\]]', text_lower)
        if paren_match:
            return paren_match.group(1).capitalize()
        
        # Strategy 7: Look for label as a complete word (not substring) anywhere in text
        # This is a fallback - only use if no other patterns matched
        # Check in order of specificity
        for label in ['correct', 'almost', 'partial', 'incorrect']:
            # Use word boundary to ensure we're matching complete words
            pattern = r'\b' + label + r'\b'
            if re.search(pattern, text_lower):
                return label.capitalize()
        
        return "None"

    def _quick_extract_label(self, text: str) -> str:
        """Quickly extract label from the very start of the text.
        
        This is the fastest extraction method that checks for labels at the
        very beginning of the response, before any JSON parsing.
        
        Returns one of: "Correct", "Almost", "Partial", "Incorrect", or "None"
        """
        if not text:
            return "None"
        
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # Check for label at the very start followed by colon, space, or word boundary
        # This handles "Correct: ...", "Partial ...", or just "Correct" at the start
        match = re.match(r'^(correct|almost|partial|incorrect)\b[:\s]*', text_lower)
        if match:
            return match.group(1).capitalize()
        
        return "None"

    def _extract_label(self, text: str) -> str:
        """Extract just the label (Correct/Almost/Partial/Incorrect) from text.
        
        This is a more lenient version used for fallback extraction from raw text.
        
        Returns one of: "Correct", "Almost", "Partial", "Incorrect", or "None"
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Strategy 1: Look for explicit label at the start of the response
        label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)[\s:]', text_lower)
        if label_prefix_match:
            return label_prefix_match.group(1).capitalize()
        
        # Strategy 2: Look for explicit label at the start of any line
        for line in text_lower.split('\n'):
            line = line.strip()
            if line:
                label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)[\s:]', line)
                if label_prefix_match:
                    return label_prefix_match.group(1).capitalize()
        
        # Strategy 3: Look for labels as standalone words at the very start
        standalone_match = re.match(r'^(correct|almost|partial|incorrect)\b', text_lower)
        if standalone_match:
            return standalone_match.group(1).capitalize()
        
        # Strategy 4: Look for labels as standalone words at the start of any line
        for line in text_lower.split('\n'):
            line = line.strip()
            if line:
                standalone_match = re.match(r'^(correct|almost|partial|incorrect)\b', line)
                if standalone_match:
                    return standalone_match.group(1).capitalize()
        
        # Strategy 5: Look for labels in JSON-like format with explicit boundaries
        json_label_match = re.search(r'["\']?(response|label|classification|grade|evaluation|result)["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)["\']?', text_lower)
        if json_label_match:
            return json_label_match.group(2).capitalize()
        
        # Strategy 6: Look for labels in parentheses or brackets
        paren_match = re.search(r'[\(\[](correct|almost|partial|incorrect)[\)\]]', text_lower)
        if paren_match:
            return paren_match.group(1).capitalize()
        
        # Strategy 7: Look for labels followed by reasoning indicators
        reasoning_match = re.search(r'\b(correct|almost|partial|incorrect)\s*[-:]\s*', text_lower)
        if reasoning_match:
            return reasoning_match.group(1).capitalize()
        
        # Strategy 8: Look for labels in quotes
        quoted_match = re.search(r'["\'](correct|almost|partial|incorrect)["\']', text_lower)
        if quoted_match:
            return quoted_match.group(1).capitalize()
        
        # Strategy 9: Look for labels as standalone words with word boundaries
        # Check in order of specificity (most specific first)
        # Check for "Almost" first (higher priority to avoid confusion with "correct")
        almost_match = re.search(r'\balmost\b', text_lower)
        if almost_match:
            return "Almost"
        
        # Check for "Incorrect" before "Correct" to avoid "inCorrect" issues
        incorrect_match = re.search(r'\bincorrect\b', text_lower)
        if incorrect_match:
            return "Incorrect"
        
        # Check for "Partial"
        partial_match = re.search(r'\bpartial\b', text_lower)
        if partial_match:
            return "Partial"
        
        # Check for "Correct" last (but not when preceded by "in")
        correct_match = re.search(r'(?<!in)\bcorrect\b', text_lower)
        if correct_match:
            return "Correct"
        
        return "None"
