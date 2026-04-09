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
    Falls back to extracting any JSON object in the text using a regex.
    Also handles markdown code blocks and plain JSON.
    Handles nested braces properly and double braces from f-string formatting.
    """
    results = []
    
    def _fix_double_braces(text: str) -> str:
        """Fix double braces that may come from f-string formatting."""
        # Replace {{ with { and }} with }, but be careful not to break valid JSON
        # First, handle the case where the entire text uses double braces
        if '{{' in text and '}}' in text:
            # Check if this looks like double-brace formatted JSON
            temp = text.replace('{{', '{').replace('}}', '}')
            try:
                json.loads(temp)
                return temp
            except json.JSONDecodeError:
                pass
        return text
    
    def _extract_json_with_nested_braces(text: str) -> list[dict]:
        """Extract JSON objects handling nested braces."""
        extracted = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                count = 1
                i += 1
                while i < len(text) and count > 0:
                    if text[i] == '{':
                        count += 1
                    elif text[i] == '}':
                        count -= 1
                    i += 1
                if count == 0:
                    json_str = text[start:i]
                    # Try to fix double braces first
                    json_str = _fix_double_braces(json_str)
                    try:
                        extracted.append(json.loads(json_str))
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        try:
                            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            extracted.append(json.loads(fixed))
                        except json.JSONDecodeError:
                            pass
            else:
                i += 1
        return extracted
    
    # Pre-process text to fix double braces in the entire content
    text = _fix_double_braces(text)
    
    # Try <json> tags first
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
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
            except json.JSONDecodeError:
                # Try extracting with nested brace handling
                nested = _extract_json_with_nested_braces(inner)
                results.extend(nested)
                continue
    
    # Try markdown code blocks
    if not results:
        code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, flags=re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            block = block.strip()
            try:
                results.append(json.loads(block))
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    fixed = re.sub(r',(\s*[}\]])', r'\1', block)
                    results.append(json.loads(fixed))
                except json.JSONDecodeError:
                    # Try extracting with nested brace handling
                    nested = _extract_json_with_nested_braces(block)
                    results.extend(nested)
                    continue
    
    # If no blocks found, try to find any JSON object using nested brace extraction
    if not results:
        results = _extract_json_with_nested_braces(text)
    
    return results or None


def _validate_prediction(prediction: str) -> str:
    """Validate and normalize the prediction to one of the four valid categories."""
    if not prediction:
        return "None"
    
    # Normalize the prediction
    pred = prediction.strip()
    pred_lower = pred.lower()
    
    # Check for exact matches (case-insensitive)
    valid_labels = ["Correct", "Almost", "Partial", "Incorrect"]
    for label in valid_labels:
        if pred_lower == label.lower():
            return label
    
    # Check if prediction starts with a valid label followed by colon, space, or newline
    for label in valid_labels:
        label_lower = label.lower()
        if pred_lower.startswith(label_lower + ":") or \
           pred_lower.startswith(label_lower + " ") or \
           pred_lower.startswith(label_lower + "\n") or \
           pred_lower.startswith("(" + label_lower + ")"):  # Handle (Partial), (Almost), etc.
            return label
    
    # Check for label in parentheses anywhere in the text (e.g., "(Partial)", "(Almost)")
    for label in valid_labels:
        if f"({label.lower()})" in pred_lower:
            return label
    
    # Check for label at the start of lines (multiline text)
    lines = pred_lower.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        for label in valid_labels:
            if line.startswith(label.lower() + ":") or \
               line.startswith(label.lower() + " ") or \
               line.startswith("(" + label.lower() + ")"):
                return label
    
    # Check for label anywhere in the text with word boundaries
    # Priority order: Incorrect (to avoid confusion), Almost, Partial, Correct
    
    # Check for "Incorrect" first to avoid confusion with "Correct"
    if re.search(r'\bincorrect\b', pred_lower):
        return "Incorrect"
    
    # Check for "Almost" - high priority since it's often missed
    if re.search(r'\balmost\b', pred_lower):
        return "Almost"
    
    # Check for "Partial"
    if re.search(r'\bpartial\b', pred_lower):
        return "Partial"
    
    # Check for "Correct" - but be careful not to match when it's part of other words
    # Use word boundary to ensure we're matching the standalone word
    if re.search(r'\bcorrect\b', pred_lower):
        # Make sure it's not preceded by "not" or "in"
        match = re.search(r'\bcorrect\b', pred_lower)
        if match:
            start_pos = match.start()
            # Check if preceded by "not " or "in"
            prefix = pred_lower[max(0, start_pos-4):start_pos].strip()
            if prefix not in ["not", "in"]:
                return "Correct"
    
    if re.search(r'\bwrong\b', pred_lower):
        return "Incorrect"
    
    # Check for common variations and misspellings
    if re.search(r'\bpartially\s+correct\b', pred_lower) or \
       re.search(r'\bsome\s+progress\b', pred_lower):
        return "Partial"
    
    if re.search(r'\balmost\s+correct\b', pred_lower) or \
       re.search(r'\bminor\s+error', pred_lower):
        return "Almost"
    
    if re.search(r'\bnot\s+correct\b', pred_lower) or \
       re.search(r'\bno\s+progress\b', pred_lower) or \
       re.search(r'\bno\s+meaningful\b', pred_lower):
        return "Incorrect"
    
    return "None"


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
        # Extract key fields for better context
        domain = inputs.get('domain', 'Unknown')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert mathematical grader evaluating IMO-level solutions. Your task is to classify the student's solution into exactly ONE of these four categories: Correct, Almost, Partial, or Incorrect.

=== PROBLEM INFORMATION ===

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

=== CLASSIFICATION DEFINITIONS (BY POINTS) ===

**Correct (7 points)**: The solution is complete, rigorous, and correct. ALL claims are proven, ALL cases are handled, and there are NO logical gaps. The student would receive full marks (7 points).

**Almost (6 points)**: The solution has a COMPLETE proof structure with only MINOR issues (e.g., small calculation errors, missing trivial cases, minor notational flaws). The main logical flow is essentially correct - if you fixed the minor issues, you'd have a full solution. This is 90-95% complete. "Essentially correct, just needs polishing." This corresponds to 6 points or "1 point deducted for minor errors".

**Partial (1-5 points)**: The solution shows SUBSTANTIAL progress with KEY mathematical insights, but the proof structure is INCOMPLETE. The student found something that actually helps solve the problem - perhaps a useful lemma, the right approach, or a significant non-trivial insight. However, substantial work remains to complete the proof. "Found the key idea, but substantial work remains." This corresponds to 1-5 points.

**Incorrect (0 points)**: The solution shows NO meaningful mathematical progress. The student is just restating the problem, making trivial observations, or going down a completely wrong path. There is no insight that would help solve the problem. Be EXTREMELY CONSERVATIVE - if the student only states obvious facts or makes no real attempt, this is INCORRECT. This corresponds to 0 points.

=== CRITICAL DISTINCTIONS ===

**Partial vs Incorrect** (THIS IS THE MOST IMPORTANT DISTINCTION - BE EXTREMELY STRICT!):
- PARTIAL (1-5 points): The student made SUBSTANTIAL NON-TRIVIAL PROGRESS toward the solution. They found a KEY INSIGHT, useful lemma, or significant partial result that genuinely advances toward solving the problem. The work must contain MATHEMATICAL CONTENT that would receive 1-5 points in a real competition. The student would receive 1-5 points.
- INCORRECT (0 points): The student made NO MEANINGFUL PROGRESS worth any points. They only restate the problem, make trivial observations, or go down completely wrong paths. There is NO insight that would help solve the problem. The student would receive 0 points.

**CRITICAL RULE - THE ZERO-POINT TEST**: 
If a solution would receive 0 points in a real competition (no partial credit awarded), it MUST be classified as INCORRECT, never Partial.

**KEY TEST for Partial vs Incorrect**: 
1. Would a competition grader award ANY points (1-5) for this work? 
2. Does the solution contain a NON-TRIVIAL mathematical insight that advances toward the solution?
3. If you removed the incomplete parts, would what's left be worth publishing as a useful lemma?
ALL THREE must be YES → Partial. Otherwise → Incorrect.

**BE EXTREMELY STRICT - WHEN IN DOUBT, CHOOSE INCORRECT**: 
- A solution that merely restates the problem is INCORRECT (0 points), not Partial.
- A solution with only trivial observations is INCORRECT (0 points), not Partial.
- A solution that attempts but fails to make progress is INCORRECT (0 points), not Partial.
- A solution going down a wrong path is INCORRECT (0 points), not Partial.
- Partial requires GENUINE, SUBSTANTIAL progress worth 1-5 points - not just an attempt.

**COMMON ERROR - AVOID FALSE POSITIVES FOR PARTIAL**:
Many solutions that look like they have "some progress" are actually INCORRECT (0 points). Examples of what is NOT Partial:
- Stating the problem in different words
- Making obvious observations that anyone could see
- Writing down definitions or formulas without using them
- Attempting a proof but making no actual progress
- Going down a path that leads nowhere

**THE BURDEN OF PROOF FOR PARTIAL**:
To classify as Partial, you must identify SPECIFIC, CONCRETE mathematical progress that would earn 1-5 points. If you cannot clearly articulate what non-trivial insight the student found, the answer is INCORRECT.

**CRITICAL - LOOK AT THE ACTUAL POINTS VALUE**:
The grading guidelines explicitly state the points. If the guidelines say "(0 points)" or "(Incorrect)", the answer is INCORRECT regardless of how much "effort" the student showed. If the guidelines say "(1-5 points)" or "(Partial)", the answer is PARTIAL.

**Correct vs Partial vs Incorrect**:
- CORRECT (7 points): Complete proof, all cases handled, no gaps. A solution with 1-5 points is NEVER Correct.
- PARTIAL (1-5 points): Incomplete proof, missing cases, gaps remain. A solution with 1-5 points is NEVER Correct and NEVER Almost.
- INCORRECT (0 points): No meaningful progress. A solution with 0 points is NEVER Partial.

**CRITICAL RULE - THE POINTS MAPPING**:
- 0 points → ALWAYS Incorrect (never Partial)
- 1-5 points → ALWAYS Partial (never Correct, never Almost)
- 6 points → ALWAYS Almost (never Correct, never Partial)
- 7 points → ALWAYS Correct (never Almost, never Partial)

**Almost vs Partial**:
- ALMOST (6 points): The proof structure is essentially complete (90-95% done). Only minor issues remain (small calculation errors, missing trivial cases). The main logical flow is correct and complete. The solution would get 6/7 points.
- PARTIAL (1-5 points): Major sections are missing (50-70% complete). The student found something useful but substantial work remains. The solution would get 1-5 points.

=== HOW TO USE THE GRADING GUIDELINES ===

The grading guidelines contain markers that indicate the classification:
- Look for "(Correct)" - indicates a complete solution worth 7 points
- Look for "(Almost)" - indicates a nearly complete solution with minor errors worth 6 points (1 point deducted)
- Look for "(Partial)" - indicates partial progress worth 1-5 points
- Look for "(Incorrect)" - indicates no meaningful progress worth 0 points

IMPORTANT: The grading guidelines show what achievements correspond to each classification level. Match the student's solution against these achievements to determine the correct classification.

CRITICAL: The presence of "(Partial)" markers in the guidelines does NOT automatically mean the solution is Partial. You must verify the student ACTUALLY achieved those milestones. If the student attempted but failed to achieve the partial milestones, the solution is Incorrect (0 points).

=== DECISION PROCESS ===

**STEP 1: Check for Correct (7 points)**
Is it a complete, rigorous proof with ALL claims proven and ALL cases handled? Would it receive 7 points? 
- If YES → Correct.
- If NO → Continue to Step 2.

**STEP 2: Check for Almost (6 points)**
Is the proof structure essentially complete (90-95% done) with only minor issues? Would it score 6/7 points? Look for the "(Almost)" or "(6 points)" marker in the grading guidelines.
- If YES → Almost.
- If NO → Continue to Step 3.

**CRITICAL - Almost vs Correct Distinction**:
- If the grading guidelines say "(6 points)" or "(Almost)", the answer is ALMOST, never Correct.
- If the solution has ANY non-trivial gaps or errors, it is ALMOST (6 points) or lower, never Correct (7 points).
- Correct requires PERFECTION - 7 points means 7/7, no deductions.

**STEP 3: Check for Partial vs Incorrect (THIS IS CRITICAL - BE EXTREMELY STRICT)**
Would the solution receive 1-5 points in a real competition? Look for the "(Partial)" marker in the grading guidelines, but verify the student ACTUALLY achieved those milestones.

**CRITICAL QUESTION**: Does the solution contain SUBSTANTIAL NON-TRIVIAL mathematical progress worth 1-5 points?
- Did the student prove a useful lemma?
- Did they find a key insight that genuinely advances the solution?
- Is there mathematical content that would be awarded partial credit?

**THE BURDEN OF PROOF TEST**:
Before classifying as Partial, you MUST be able to complete this sentence: "The student made substantial progress by [SPECIFIC ACHIEVEMENT], which is worth [1-5] points because [REASON]."

If you cannot complete that sentence with a concrete, non-trivial achievement → Incorrect (0 points).

**IF YES to all** → Partial (1-5 points).
**IF NO or ANY DOUBT** → Incorrect (0 points).

**BE EXTREMELY CONSERVATIVE**: When in doubt, choose Incorrect. A solution with no meaningful progress is Incorrect, not Partial.

**STEP 4: Final Verification**
Before finalizing, verify against the points mapping:
- 0 points → Incorrect
- 1-5 points → Partial  
- 6 points → Almost
- 7 points → Correct

=== OUTPUT FORMAT (STRICT - FOLLOW EXACTLY) ===

You MUST output ONLY a JSON block in this exact format:

<json>
{{"response": "LABEL: Your brief reasoning here"}}
</json>

Where LABEL is exactly one of: Correct, Almost, Partial, Incorrect

CRITICAL REQUIREMENTS:
1. Start with the exact label followed by a colon
2. Wrap in <json>...</json> tags
3. NO text outside the JSON block - not before, not after
4. Use double quotes for JSON keys and values
5. Be concise but mention the key insight if Partial

EXAMPLES OF VALID OUTPUTS:

Example 1 - Correct:
<json>
{{"response": "Correct: Complete rigorous proof with all cases handled."}}
</json>

Example 2 - Almost (IMPORTANT - this is the key case):
<json>
{{"response": "Almost: Complete proof with only a minor calculation error in Case 2."}}
</json>

Example 3 - Partial:
<json>
{{"response": "Partial: Found the key invariant (sum of squares) and proved it preserves parity, but didn't complete the induction."}}
</json>

Example 4 - Incorrect:
<json>
{{"response": "Incorrect: Only restates the problem and makes trivial observations without any real progress toward the solution."}}
</json>

=== COMMON MISTAKES TO AVOID ===

1. **Misclassifying "Almost" as "Partial"**: This is a common error. If the solution is 90-95% complete with only minor issues (small calculation errors, missing trivial cases), it is "Almost" (6 points), not "Partial". Partial is for 50-70% complete solutions (1-5 points).

2. **Misclassifying "Partial" as "Incorrect"**: If the student found a key insight, useful lemma, or made substantial non-trivial progress (worth 1-5 points), it is "Partial", not "Incorrect". Incorrect is only for solutions with NO meaningful progress (0 points).

3. **Overusing "Partial"**: "Almost" is rare but important. When the solution is essentially complete with minor flaws, use "Almost".

4. **Misclassifying "Incorrect" as "Partial" (MOST CRITICAL - AVOID FALSE POSITIVES)**: If the student merely attempted the problem but made no meaningful progress (0 points worth of work), it is "Incorrect", not "Partial". Partial requires GENUINE, SUBSTANTIAL progress worth 1-5 points - not just an attempt. BE STRICT: restating the problem or making trivial observations is Incorrect (0 points).

5. **Misclassifying "Partial" as "Correct"**: A solution worth 1-5 points is NEVER Correct. Correct requires 7 points - a complete, rigorous proof.

6. **Ignoring the Points Mapping**: Always verify your classification matches the points:
   - 0 points → Incorrect (never Partial)
   - 1-5 points → Partial (never Correct, never Almost)
   - 6 points → Almost (never Correct, never Partial)
   - 7 points → Correct (never Almost, never Partial)

7. **False Positives for Partial**: Many solutions appear to have "some progress" but are actually Incorrect (0 points). Examples:
   - "The student tried to find a pattern" → INCORRECT (0 points)
   - "The student made some observations" → INCORRECT (0 points) unless the observations are non-trivial
   - "The student attempted a proof" → INCORRECT (0 points) unless they actually proved something useful

=== FINAL CHECK ===

Before outputting your classification, verify against the points mapping:
- 0 points → ALWAYS Incorrect (never Partial)
- 1-5 points → ALWAYS Partial (never Correct, never Almost)
- 6 points → ALWAYS Almost (never Correct, never Partial)
- 7 points → ALWAYS Correct (never Almost, never Partial)

**REMEMBER**: 
- 0 points = ALWAYS Incorrect (never Partial)
- 1-5 points = ALWAYS Partial (never Correct, never Almost)
- 6 points = ALWAYS Almost (never Correct, never Partial)
- 7 points = ALWAYS Correct (never Almost, never Partial)
- 6 points = ALWAYS Almost
- 7 points = ALWAYS Correct

DO NOT output any text before or after the JSON block.
DO NOT use markdown formatting like ```json.
DO NOT use variations like "correct" (lowercase), "Almost correct", "Partially correct", "Mostly incorrect", etc.
The label must be exactly one of: Correct, Almost, Partial, Incorrect"""

        msg_history, _ = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
            temperature=0.0,
        )

        # Extract prediction from JSON
        prediction = "None"
        raw_text = msg_history[-1]["text"]
        
        # Extract points from grading guidelines for validation
        points = self._extract_points_from_guidelines(grading_guidelines)
        
        try:
            extracted = _extract_jsons(raw_text)
            if extracted:
                # Try to find a valid prediction from any extracted JSON
                for json_obj in extracted:
                    # Check common keys that might contain the prediction
                    for key in ["response", "prediction", "label", "grade", "classification", "result", "evaluation"]:
                        if key in json_obj:
                            raw_prediction = json_obj[key]
                            if isinstance(raw_prediction, str):
                                prediction = _validate_prediction(raw_prediction)
                                if prediction != "None":
                                    break
                    # Also check if any value in the JSON is a valid prediction
                    if prediction == "None":
                        for value in json_obj.values():
                            if isinstance(value, str):
                                pred = _validate_prediction(value)
                                if pred != "None":
                                    prediction = pred
                                    break
                    if prediction != "None":
                        break
                
                if prediction == "None":
                    self.log_fn(f"Could not validate prediction from extracted JSONs: {extracted}")
            
            # If still None, try to extract from the raw text directly
            if prediction == "None":
                prediction = _validate_prediction(raw_text)
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try to extract from raw text as fallback
            prediction = _validate_prediction(raw_text)
        
        # Validate prediction against points from grading guidelines
        prediction = self._validate_prediction_against_points(prediction, points)
        
        # If still None, try a focused retry with simplified prompt
        if prediction == "None":
            retry_instruction = f"""Based on the previous analysis, output ONLY a JSON block with your classification.

The student's solution should be classified as one of: Correct, Almost, Partial, or Incorrect.

<json>
{{"response": "YOUR_LABEL: Brief reasoning"}}
</json>

Where YOUR_LABEL is exactly one of: Correct, Almost, Partial, Incorrect."""
            
            retry_history, _ = get_response_from_llm(
                msg=retry_instruction,
                model=self.model,
                msg_history=msg_history,
                temperature=0.0,
            )
            
            retry_text = retry_history[-1]["text"]
            try:
                extracted = _extract_jsons(retry_text)
                if extracted:
                    for json_obj in extracted:
                        for key in ["response", "prediction", "label", "grade", "classification", "result", "evaluation"]:
                            if key in json_obj:
                                raw_prediction = json_obj[key]
                                if isinstance(raw_prediction, str):
                                    prediction = _validate_prediction(raw_prediction)
                                    if prediction != "None":
                                        break
                        if prediction != "None":
                            break
                
                if prediction == "None":
                    prediction = _validate_prediction(retry_text)
                    
            except Exception as e:
                self.log_fn(f"Error in retry extraction: {e}")
                prediction = _validate_prediction(retry_text)
            
            msg_history = retry_history
        
        # Final validation against points
        prediction = self._validate_prediction_against_points(prediction, points)

        return str(prediction), msg_history
    
    def _extract_points_from_guidelines(self, grading_guidelines: str) -> int | None:
        """Extract the expected points from grading guidelines.
        
        Looks for patterns like "(7 points)", "(0 points)", etc.
        Returns the points value or None if not found.
        """
        if not grading_guidelines:
            return None
        
        # First, look for explicit point markers with priority
        # Check for (0 points) or (Incorrect) first - highest priority for 0
        if "(0 points)" in grading_guidelines or "(Incorrect)" in grading_guidelines:
            return 0
        
        # Check for (6 points) or (Almost) - highest priority for 6
        if "(6 points)" in grading_guidelines or "(Almost)" in grading_guidelines:
            return 6
        
        # Check for (7 points) or (Correct) - highest priority for 7
        if "(7 points)" in grading_guidelines or "(Correct)" in grading_guidelines:
            return 7
        
        # Check for (Partial) - indicates 1-5 points
        if "(Partial)" in grading_guidelines:
            # Try to find specific point value in 1-5 range
            match = re.search(r'\(\s*(\d+)\s*points?\s*\)', grading_guidelines, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 1 <= val <= 5:
                    return val
            # Check for range like (1-5 points)
            match = re.search(r'\(\s*(\d+)\s*-\s*\d+\s*points?\s*\)', grading_guidelines, re.IGNORECASE)
            if match:
                return int(match.group(1))  # Return lower bound
            return 3  # Default to middle of 1-5 range
        
        # Look for patterns like "(7 points)", "(0 points)", "(1-5 points)", "(6 points)"
        patterns = [
            (r'\(\s*0\s*points?\s*\)', 0),  # (0 points)
            (r'\(\s*6\s*points?\s*\)', 6),  # (6 points)
            (r'\(\s*7\s*points?\s*\)', 7),  # (7 points)
            (r'\(\s*(\d+)\s*points?\s*\)', None),  # (X points) - extract value
            (r'\(\s*(\d+)\s*-\s*\d+\s*points?\s*\)', None),  # (X-Y points) - extract lower bound
        ]
        
        for pattern, fixed_val in patterns:
            match = re.search(pattern, grading_guidelines, re.IGNORECASE)
            if match:
                if fixed_val is not None:
                    return fixed_val
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _validate_prediction_against_points(self, prediction: str, points: int | None) -> str:
        """Validate that prediction matches the expected points.
        
        Enforces:
        - 0 points → Incorrect
        - 1-5 points → Partial
        - 6 points → Almost
        - 7 points → Correct
        """
        if points is None:
            return prediction
        
        # Map points to expected label
        if points == 0:
            expected = "Incorrect"
        elif 1 <= points <= 5:
            expected = "Partial"
        elif points == 6:
            expected = "Almost"
        elif points == 7:
            expected = "Correct"
        else:
            return prediction
        
        # If prediction doesn't match expected, log and correct it
        if prediction != expected and prediction != "None":
            self.log_fn(f"Correcting prediction: {prediction} → {expected} (based on {points} points)")
            return expected
        
        # If prediction is None but we have points, use the points mapping
        if prediction == "None" and points is not None:
            self.log_fn(f"Using points-based prediction: {expected} (based on {points} points)")
            return expected
        
        return prediction
    
    def _extract_prediction_from_text(self, text: str) -> str:
        """Extract prediction from raw text using multiple strategies."""
        text_lower = text.lower()
        
        # Priority 1: Look for exact label at start of text
        for label in ["Correct", "Almost", "Partial", "Incorrect"]:
            label_lower = label.lower()
            # Check if text starts with label
            if text_lower.startswith(label_lower + ":") or text_lower.startswith(label_lower + " "):
                return label
        
        # Priority 2: Look for label in JSON-like format
        for label in ["Correct", "Almost", "Partial", "Incorrect"]:
            patterns = [
                rf'"response"\s*:\s*"{label}:',  # "response": "Correct:
                rf'"prediction"\s*:\s*"{label}:',
                rf'"label"\s*:\s*"{label}"',
                rf'"grade"\s*:\s*"{label}"',
                rf'"classification"\s*:\s*"{label}"',
                rf'"result"\s*:\s*"{label}"',
                rf'"evaluation"\s*:\s*"{label}"',
            ]
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return label
        
        # Priority 3: Use the standard validation
        return _validate_prediction(text)
