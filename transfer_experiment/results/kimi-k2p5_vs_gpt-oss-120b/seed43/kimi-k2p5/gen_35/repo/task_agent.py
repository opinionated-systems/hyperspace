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

**Incorrect (0 points)**: The solution shows NO meaningful mathematical progress. The student is just restating the problem, making trivial observations, or going down a completely wrong path. There is no insight that would help solve the problem. Be CONSERVATIVE - if the student only states obvious facts or makes no real attempt, this is INCORRECT. This corresponds to 0 points.

=== CRITICAL DISTINCTIONS ===

**Partial vs Incorrect** (THIS IS THE MOST IMPORTANT DISTINCTION):
- PARTIAL (1-5 points): The student made SUBSTANTIAL NON-TRIVIAL PROGRESS toward the solution. They found a KEY INSIGHT, useful lemma, or significant partial result that genuinely advances toward solving the problem. Even if incomplete, the work shows real mathematical understanding. The student would receive 1-5 points.
- INCORRECT (0 points): The student made NO MEANINGFUL PROGRESS. They only restate the problem, make trivial observations, or go down completely wrong paths. There is no insight that would help solve the problem. The student would receive 0 points.

KEY TEST for Partial vs Incorrect: 
1. Did the student find something that would be worth 1-5 points? 
2. If you removed the incomplete parts, would what's left be publishable as a useful lemma or partial result? 
3. Does the solution contain ANY non-trivial mathematical insight that advances toward the solution?
If YES to all → Partial. If NO → Incorrect.

**BE CONSERVATIVE**: When in doubt between Partial and Incorrect, choose Incorrect. A solution that merely restates the problem or makes trivial observations is Incorrect, not Partial.

**Almost vs Partial**:
- ALMOST (6 points): The proof structure is essentially complete (90-95% done). Only minor issues remain (small calculation errors, missing trivial cases). The main logical flow is correct and complete. The solution would get 6/7 points.
- PARTIAL (1-5 points): Major sections are missing (50-70% complete). The student found something useful but substantial work remains. The solution would get 1-5 points.

**Correct vs Partial**:
- CORRECT (7 points): Complete proof, all cases handled, no gaps.
- PARTIAL (1-5 points): Incomplete proof, missing cases, gaps remain. A solution with 1-5 points is NEVER Correct.

=== HOW TO USE THE GRADING GUIDELINES ===

The grading guidelines contain markers that indicate the classification:
- Look for "(Correct)" - indicates a complete solution worth 7 points
- Look for "(Almost)" - indicates a nearly complete solution with minor errors worth 6 points (1 point deducted)
- Look for "(Partial)" - indicates partial progress worth 1-5 points
- Look for "(Incorrect)" - indicates no meaningful progress worth 0 points

IMPORTANT: The grading guidelines show what achievements correspond to each classification level. Match the student's solution against these achievements to determine the correct classification.

CRITICAL: The presence of "(Partial)" markers in the guidelines does NOT automatically mean the solution is Partial. You must verify the student ACTUALLY achieved those milestones. If the student attempted but failed to achieve the partial milestones, the solution is Incorrect (0 points).

=== DECISION PROCESS ===

1. First, check if the solution is Correct: Is it a complete, rigorous proof with ALL claims proven and ALL cases handled? Would it receive 7 points? If YES → Correct.

2. If not Correct, check if it's Almost: Is the proof structure essentially complete with only minor issues? Would it score 6/7 points? Look for the "(Almost)" marker in the grading guidelines. If YES → Almost.

3. If not Almost, check if it's Partial: Did the student find a KEY INSIGHT that advances the solution? Is there SUBSTANTIAL non-trivial progress? Would it receive 1-5 points? Look for the "(Partial)" marker in the grading guidelines, but verify the student actually achieved those milestones. If YES → Partial.

4. Otherwise → Incorrect (0 points).

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

1. **Misclassifying "Almost" as "Partial"**: This is the most common error. If the solution is 90-95% complete with only minor issues (small calculation errors, missing trivial cases), it is "Almost", not "Partial". Partial is for 50-70% complete solutions (1-5 points).

2. **Misclassifying "Partial" as "Incorrect"**: If the student found a key insight, useful lemma, or made substantial non-trivial progress (worth 1-5 points), it is "Partial", not "Incorrect". Incorrect is only for solutions with NO meaningful progress (0 points).

3. **Overusing "Partial"**: "Almost" is rare but important. When the solution is essentially complete with minor flaws, use "Almost".

4. **Misclassifying "Incorrect" as "Partial"**: If the student merely attempted the problem but made no meaningful progress (0 points worth of work), it is "Incorrect", not "Partial". Partial requires SUBSTANTIAL non-trivial progress.

5. **Misclassifying "Partial" as "Correct"**: A solution worth 1-5 points is NEVER Correct. Correct requires 7 points - a complete, rigorous proof.

=== FINAL CHECK ===

Before outputting your classification, verify:
- Did you check for the "(Almost)" marker in the grading guidelines?
- Is the solution 90-95% complete (6 points)? → Use "Almost"
- Is the solution 50-70% complete with key insights (1-5 points)? → Use "Partial"
- Is there no meaningful progress (0 points)? → Use "Incorrect"
- Is the solution complete with all cases handled (7 points)? → Use "Correct"

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

        return str(prediction), msg_history
