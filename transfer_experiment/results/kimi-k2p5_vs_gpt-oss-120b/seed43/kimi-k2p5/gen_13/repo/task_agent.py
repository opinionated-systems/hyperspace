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
            continue
    
    # Try to find plain JSON objects (looking for {"response": ...} pattern)
    # This handles cases where the LLM outputs JSON without proper tags
    plain_json_pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    plain_matches = re.findall(plain_json_pattern, text, re.DOTALL)
    for match in plain_matches:
        try:
            # Reconstruct the JSON object
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
        
        instruction = f"""You are an expert mathematical grader evaluating IMO-level solutions. Your task is to classify a student's solution into exactly one of four categories.

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

You must classify the solution into exactly ONE of these four categories:

**Correct (Score 7)**: The solution is complete, rigorous, and correct.
- ALL required claims are proven
- ALL cases are handled
- NO logical gaps or errors
- Matches the official solution's conclusion

**Almost (Score 1)**: The solution is nearly complete with only minor issues.
- Main proof structure is sound and complete
- Contains only small verification mistakes, trivial gaps, or minor errors
- Would require only a small fix to be fully correct
- Has all key insights but may have minor technical flaws
- The solution looks "almost done" - an expert can see it's 90-95% complete

**Partial (Score 0-6)**: The solution shows SUBSTANTIAL progress with KEY mathematical insights.
- Student has found KEY invariants, lemmas, or insights that are CENTRAL to solving the problem
- There is a VIABLE approach that could lead to a solution with more work
- Student has made MEANINGFUL mathematical progress BEYOND restating the problem
- An expert would say "this student is on the right track" or "they found the key idea"
- The solution contains genuine mathematical content that advances toward the solution
- Examples: Found the key invariant but didn't complete the proof; Proved a key lemma; Has the right approach with significant progress

**Incorrect (Score 0)**: The solution has NO meaningful mathematical progress.
- NO valuable insight found - nothing that would help solve the problem
- Approach is fundamentally flawed or completely wrong
- Just restating the problem, given conditions, or making trivial observations anyone could make
- Critical errors in reasoning that invalidate the entire approach
- The solution shows no real understanding of the problem's mathematical structure
- Examples: Only restating the problem; Random calculations with no direction; Wrong approach with no useful insights

=== CRITICAL DISTINCTIONS ===

**Almost vs Partial** - THIS IS THE MOST IMPORTANT DISTINCTION:
- "Almost" = Solution is STRUCTURALLY COMPLETE, just needs minor fixes
  * The proof framework is complete and correct
  * Only small calculation errors, one trivial case missing, or minor technical issues
  * An expert would say "this is essentially correct, just needs polishing"
  * Example: Complete proof with one small calculation error → "Almost"
  * Example: Complete proof missing one trivial case → "Almost"
  
- "Partial" = Significant work remains, major portions missing
  * The student found key insights but the proof is far from complete
  * Major gaps in the argument that require substantial additional work
  * An expert would say "good start, but lots of work remains"
  * Example: Finding a key invariant but not completing the proof → "Partial"
  * Example: Having the right approach but stopping halfway → "Partial"

**Partial vs Incorrect** - BE CONSERVATIVE WITH "Partial":
- "Partial" = Student found something SUBSTANTIAL and mathematically valuable (key lemma, invariant, viable approach). The solution must contain genuine mathematical progress that an expert would recognize as advancing toward the solution.
- "Incorrect" = No valuable mathematical insight, or approach is completely wrong. When in doubt, choose "Incorrect" - do NOT give "Partial" for trivial observations or restatements.

=== DETAILED EXAMPLES ===

**Correct examples:**
- Complete, rigorous proof with all cases handled → "Correct"
- Proof matches official solution's approach and conclusion → "Correct"

**Almost examples:**
- Complete proof with one small calculation error → "Almost"
- Complete proof missing one trivial case → "Almost"
- Proof is structurally sound but has a minor technical flaw → "Almost"
- Solution is 90-95% complete, just needs small fixes → "Almost"

**Partial examples (must have SUBSTANTIAL key insights):**
- Found the KEY invariant/lemma that is central to the solution but didn't complete the full proof → "Partial"
- Has the RIGHT approach with MEANINGFUL progress (e.g., proved a key sub-lemma, set up the right framework) → "Partial"
- Solution contains genuine mathematical content that advances significantly toward the solution → "Partial"
- An expert would say "they found the key idea, just need to finish" → "Partial"

**Incorrect examples (NO meaningful progress):**
- Just restating the problem statement or given conditions → "Incorrect"
- Making trivial observations that anyone could make without solving → "Incorrect"
- Wrong approach with no useful insights → "Incorrect"
- Random calculations or attempts with no coherent direction → "Incorrect"
- Fundamental logical errors that show no understanding → "Incorrect"

=== DECISION PROCESS ===

Follow this step-by-step process:

1. **First check**: Is the solution complete and correct? 
   - ALL claims proven? ALL cases handled? NO gaps? → "Correct"
   - Otherwise continue...

2. **Second check**: Is the solution STRUCTURALLY COMPLETE with only minor issues?
   - Is the main proof framework sound and complete?
   - Are there only small errors (calculation, one case, minor technical)?
   - Would it require only a small fix to be fully correct?
   - Does it look 90-95% complete? → "Almost"
   - Otherwise continue...

3. **Third check**: Does it show SUBSTANTIAL progress with KEY insights?
   - Has the student found KEY invariants, lemmas, or insights that are CENTRAL to the problem?
   - Is there a VIABLE approach with MEANINGFUL progress (not just the start)?
   - Does the solution contain genuine mathematical content BEYOND restating the problem?
   - Would an expert say "they found the key idea" or "this is real progress"?
   - ONLY if ALL of the above are true → "Partial"
   - Otherwise continue...

4. **Default**: → "Incorrect" (be conservative - when in doubt, it's Incorrect)

=== IMPORTANT GUIDANCE ===

- Be GENEROUS with "Almost" when the solution is structurally complete
- Be CONSERVATIVE with "Partial" - only use it for solutions with SUBSTANTIAL key insights
- "Almost" means the hard work is done, just minor fixes needed
- "Partial" means the student found the KEY idea but significant work remains
- "Incorrect" means NO valuable insight - when in doubt, choose "Incorrect"
- When in doubt between "Almost" and "Partial", ask: "Is the proof framework complete?"
- When in doubt between "Partial" and "Incorrect", ask: "Did they find something an expert would value?"

=== FEW-SHOT EXAMPLES ===

Example 1 - Student writes only: "We need to find all pairs (a,b) such that..." and restates the problem.
→ This is "Incorrect" - no meaningful progress, just restating.

Example 2 - Student finds: "The key invariant is that a+b stays constant mod 3, and here's why..." but doesn't complete.
→ This is "Partial" - found a KEY insight but didn't finish.

Example 3 - Student gives a complete proof but has one small calculation error in the final step.
→ This is "Almost" - structurally complete, just a minor fix needed.

Example 4 - Student gives a complete, rigorous proof with all cases handled.
→ This is "Correct" - fully complete solution.

Example 5 - Student tries random operations without finding the invariant, gets nowhere.
→ This is "Incorrect" - no valuable insight, wrong approach.

Provide your evaluation in the following JSON format:

<json>
{{
    "response": "Correct: Your brief reasoning here"
}}
</json>

OR

<json>
{{
    "response": "Almost: Your brief reasoning here"
}}
</json>

OR

<json>
{{
    "response": "Partial: Your brief reasoning here"
}}
</json>

OR

<json>
{{
    "response": "Incorrect: Your brief reasoning here"
}}
</json>

CRITICAL: Your response MUST start with the exact label (Correct, Almost, Partial, or Incorrect) followed by a colon and then your reasoning. The entire response must be wrapped in <json>...</json> tags. Do not include any text outside the JSON tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"] if msg_history else ""
            
            # First try to extract label directly from the text (handles cases where LLM doesn't use proper JSON)
            direct_label = self._extract_label(last_message)
            if direct_label != "None":
                prediction = direct_label
                self.log_fn(f"Directly extracted prediction: {prediction}")
            else:
                # Try JSON extraction
                extracted = _extract_json_flexible(last_message)
                if extracted:
                    # Try to get response from the last JSON object
                    last_json = extracted[-1]
                    if isinstance(last_json, dict):
                        if "response" in last_json:
                            response_text = last_json["response"]
                            # Extract just the label (Correct/Incorrect/Partial/Almost) from the response
                            prediction = self._extract_label(response_text)
                        else:
                            # If no "response" key, try to find any string value that looks like a label
                            for key, value in last_json.items():
                                if isinstance(value, str):
                                    label = self._extract_label(value)
                                    if label != "None":
                                        prediction = label
                                        break
                            else:
                                # If no label found in values, use the first value as string
                                prediction = str(list(last_json.values())[0]) if last_json else "None"
                self.log_fn(f"Extracted prediction from JSON: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Fallback: try to extract any of the expected labels from raw text
            try:
                last_message = msg_history[-1]["text"] if msg_history else ""
                prediction = self._extract_label(last_message)
            except Exception:
                prediction = "None"

        return str(prediction), msg_history

    def _extract_label(self, text: str) -> str:
        """Extract just the label (Correct/Almost/Partial/Incorrect) from text.
        
        Returns one of: "Correct", "Almost", "Partial", "Incorrect", or "None"
        
        Priority order: Correct > Almost > Partial > Incorrect
        (More specific/positive labels are checked first to avoid false negatives)
        """
        if not text:
            return "None"
        
        text_lower = text.lower()
        
        # Strategy 1: Look for explicit label at the start of the response
        # This is the most reliable format we requested from the LLM
        label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)[\s:]', text_lower)
        if label_prefix_match:
            label = label_prefix_match.group(1).capitalize()
            return label
        
        # Strategy 2: Look for explicit label at the start of any line (multiline response)
        for line in text_lower.split('\n'):
            line = line.strip()
            label_prefix_match = re.match(r'^(correct|almost|partial|incorrect)[\s:]', line)
            if label_prefix_match:
                label = label_prefix_match.group(1).capitalize()
                return label
        
        # Strategy 3: Look for labels in JSON-like format with quotes
        # This handles cases like: "response": "Almost: ..." or "label": "Correct"
        json_label_match = re.search(r'["\']?(response|label|classification|grade|evaluation|result)["\']?\s*[:=]\s*["\']?(correct|almost|partial|incorrect)', text_lower)
        if json_label_match:
            label = json_label_match.group(2).capitalize()
            return label
        
        # Strategy 4: Look for labels in parentheses like "(Correct)" or "[Partial]"
        paren_match = re.search(r'[\(\[](correct|almost|partial|incorrect)[\)\]]', text_lower)
        if paren_match:
            label = paren_match.group(1).capitalize()
            return label
        
        # Strategy 5: Look for labels followed by reasoning indicators
        reasoning_match = re.search(r'\b(correct|almost|partial|incorrect)\s*[-:]\s*', text_lower)
        if reasoning_match:
            label = reasoning_match.group(1).capitalize()
            return label
        
        # Strategy 6: Look for labels in quotes
        quoted_match = re.search(r'["\'](correct|almost|partial|incorrect)["\']', text_lower)
        if quoted_match:
            label = quoted_match.group(1).capitalize()
            return label
        
        # Strategy 7: Look for labels as standalone words with word boundaries
        # Priority: Correct > Almost > Partial > Incorrect
        # We check in order of specificity to avoid false matches
        
        # Check for "Correct" first (but be careful about "Incorrect")
        # Use word boundary and ensure it's not preceded by "in"
        correct_match = re.search(r'(?<!in)\bcorrect\b', text_lower)
        if correct_match:
            return "Correct"
        
        # Check for "Almost" - be more aggressive in finding it
        # Look for "almost" as a standalone word
        almost_match = re.search(r'\balmost\b', text_lower)
        if almost_match:
            return "Almost"
        
        # Check for "Partial"
        partial_match = re.search(r'\bpartial\b', text_lower)
        if partial_match:
            return "Partial"
        
        # Check for "Incorrect" last
        incorrect_match = re.search(r'\bincorrect\b', text_lower)
        if incorrect_match:
            return "Incorrect"
        
        return "None"
