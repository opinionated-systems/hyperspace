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
from collections import Counter

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
            # Try to find valid JSON within the content
            try:
                # Look for JSON object pattern with balanced braces
                json_start = inner.find('{')
                if json_start != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(inner[json_start:], start=json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i
                                break
                    if json_end != -1:
                        results.append(json.loads(inner[json_start:json_end+1]))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```)."""
    # Try both ```json and ``` patterns with more flexible matching
    patterns = [
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(\{[\s\S]*?\})\n?```',
        r'```\s*\n?(.*?)\n?```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            match_clean = match.strip()
            try:
                return json.loads(match_clean)
            except json.JSONDecodeError:
                # Try to find valid JSON within the content using balanced braces
                try:
                    json_start = match_clean.find('{')
                    if json_start != -1:
                        brace_count = 0
                        json_end = -1
                        for i, char in enumerate(match_clean[json_start:], start=json_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i
                                    break
                        if json_end != -1:
                            return json.loads(match_clean[json_start:json_end+1].strip())
                except json.JSONDecodeError:
                    continue
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Returns:
        (validated_grade, is_valid)
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Remove common prefixes/suffixes that might be attached
    pred_clean = re.sub(r'^(?:grade|score|mark|points?)\s*[:=]?\s*', '', pred_lower, flags=re.IGNORECASE)
    pred_clean = re.sub(r'\s*(?:points?|marks?|score|grade)?$', '', pred_clean, flags=re.IGNORECASE)
    pred_clean = pred_clean.strip()
    
    # Remove any surrounding quotes
    pred_clean = pred_clean.strip('"\'')
    
    # Strict validation: only accept single digit 0-7
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for fractional grades like "3 out of 7" or "5 out of 7"
    out_of_match = re.search(r'\b([0-7])\s+out\s+of\s+7\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bdeserves\s+full\b',
        r'\baward\s+full\b',
        r'\bgive\s+full\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
        r'\bblank\b',
        r'\bempty\b',
        r'\bno\s+meaningful\b',
        r'\bno\s+progress\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
    # Check for spelled-out numbers
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7"
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', pred_lower):
            return digit, True
    
    # If no clear grade found, mark as invalid
    return pred_clean, False


def _extract_grade_from_text(text: str) -> tuple[str, str]:
    """Extract a grade from arbitrary text using multiple strategies.
    
    Returns:
        (grade, method) where method indicates how the grade was found
    """
    if not text:
        return "None", "no_text"
    
    text_lower = text.lower()
    
    # Strategy 1: Look for explicit grade patterns with digits 0-7 (highest priority)
    grade_patterns = [
        r'["\']response["\']\s*:\s*["\']?([0-7])["\']?(?!\d)',  # JSON response field
        r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*["\']?([0-7])["\']?\b',
        r'(?:grade|score|mark|final grade|final score)\s+is\s+["\']?([0-7])["\']?\b',
        r'\bgrade\s+([0-7])\s*(?:points?)?\b',
        r'\b(?:award|assign|give)\s+([0-7])\s*(?:points?)?\b',
        r'\b([0-7])\s*points?\b',
        r'\bscore\s+of\s+([0-7])\b',
        r'\b([0-7])\s*/\s*7\b',  # Fractional notation
        r'\b([0-7])\s+out\s+of\s+7\b',
    ]
    
    for pattern in grade_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1), "pattern_match"
    
    # Strategy 2: Look for JSON-like structures with response field
    json_response_patterns = [
        r'\{\s*["\']?response["\']?\s*:\s*["\']?([0-7])["\']?\s*\}',
        r'["\']response["\']\s*:\s*([0-7])(?:\s*[,}\]])',
    ]
    for pattern in json_response_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1), "json_response_field"
    
    # Strategy 3: Look for standalone digits 0-7 at word boundaries (lower priority)
    # Avoid matching digits that are part of larger numbers or dates
    standalone_patterns = [
        r'(?:^|\s)["\']?([0-7])["\']?(?:\s|$|[,\.])',  # Standalone with punctuation
        r'\bgrade\s+is\s+["\']?([0-7])["\']?\b',
        r'\bscore\s+is\s+["\']?([0-7])["\']?\b',
    ]
    for pattern in standalone_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1), "standalone_digit"
    
    # Strategy 4: Look for full credit indicators -> 7
    full_indicators = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bdeserves\s+full\b',
        r'\baward\s+full\b',
        r'\bgive\s+full\b',
        r'\bmaximum\s*(?:score|points?|credit)?\b',
    ]
    for indicator in full_indicators:
        if re.search(indicator, text_lower):
            return "7", "full_credit_indicator"
    
    # Strategy 5: Look for zero credit indicators -> 0
    zero_indicators = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
        r'\bblank\b',
        r'\bempty\b',
        r'\bno\s+meaningful\b',
        r'\bno\s+progress\b',
        r'\bno\s+credit\b',
    ]
    for indicator in zero_indicators:
        if re.search(indicator, text_lower):
            return "0", "zero_credit_indicator"
    
    # Strategy 6: Look for spelled-out numbers
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7"
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', text_lower):
            return digit, "spelled_out_number"
    
    # Strategy 7: Last resort - find any digit 0-7 that's not part of a larger number
    # Look for digits preceded by non-digit and followed by non-digit
    last_resort_match = re.search(r'(?:^|\D)([0-7])(?:\D|$)', text_lower)
    if last_resort_match:
        return last_resort_match.group(1), "last_resort_digit"
    
    return "None", "no_match_found"


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning.
        
        This method constructs a detailed prompt that guides the LLM to evaluate
        student solutions according to IMO grading standards (0-7 point scale).
        
        Args:
            inputs: Dictionary containing problem data with keys:
                - domain: Problem domain (e.g., "Mathematics")
                - problem: The problem statement
                - solution: The official solution
                - grading_guidelines: Specific grading criteria
                - student_answer: The student's submitted answer
                
        Returns:
            A formatted prompt string ready for LLM consumption
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with years of experience evaluating mathematical proofs. Your task is to evaluate the student's solution and assign a precise grade from 0-7.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## IMO Grading Scale (0-7) - USE THESE CRITERIA EXACTLY

### Grade 7 (Complete Solution)
- Complete, correct solution with rigorous proof and proper justification
- All steps are logically sound and clearly presented
- May have minor typos that don't affect mathematical correctness

### Grade 6 (Near-Complete with Minor Flaw)
- Essentially complete and correct solution
- One small gap in reasoning, or a typo in calculation that doesn't affect the main argument
- The flaw is easily fixable and doesn't invalidate the overall approach

### Grade 5 (Significant Progress)
- Most key ideas from the official solution are present
- Substantial solution elements with correct major steps
- Some gaps remain, but the core approach is sound

### Grade 4 (Good Partial Progress)
- Multiple correct key steps demonstrated
- Solution is incomplete or has notable errors in later parts
- Shows genuine understanding of the problem structure

### Grade 3 (Some Genuine Progress)
- At least one key idea or meaningful step toward solution
- Demonstrates understanding of what needs to be proved/found
- Work is relevant and shows mathematical insight

### Grade 2 (Minimal Progress)
- Some relevant ideas or observations
- Little substantive work toward the solution
- Shows awareness of problem but limited progress

### Grade 1 (Very Minimal Progress)
- Some awareness of problem structure
- Essentially no useful work toward solution
- May restate problem or make trivial observations

### Grade 0 (No Credit)
- No meaningful progress
- Completely incorrect approach
- Blank submission or irrelevant work

## Detailed Grading Instructions

### Step 1: Understand the Problem
- Read the problem statement carefully
- Identify what needs to be proved, found, or constructed
- Note any special conditions or constraints

### Step 2: Analyze the Official Solution
- Identify the key ideas and main proof steps
- Note critical lemmas, constructions, or techniques
- Understand the logical flow from start to finish

### Step 3: Evaluate the Student's Solution
- Identify the student's overall approach/strategy
- Check each claim: Is it correct? Is it justified?
- Look for correct key ideas, even if presentation is messy
- Note any original or alternative valid approaches

### Step 4: Assign Partial Credit
- Award 1 point for each significant correct key idea
- Consider: Does this step represent genuine mathematical progress?
- Don't penalize for poor notation if logic is sound
- Don't require the student to use the same method as the official solution

### Step 5: Determine Final Grade
- Sum up the points from partial credit assessment
- Verify the grade matches the quality of work demonstrated
- Double-check: Would another expert grader agree?

## Critical Grading Principles

### DO:
- Give full credit (7) only for complete, rigorous proofs
- Award partial credit for correct key insights, even if final answer is wrong
- Recognize valid alternative approaches not in the official solution
- Consider the student's demonstrated understanding, not just the final result
- Be consistent: similar quality work should get similar grades

### DON'T:
- Give full credit just for correct final answer without proper proof
- Penalize for minor notational issues if logic is sound
- Ignore partial progress - award points for correct intermediate steps
- Be swayed by length; short correct solutions deserve full credit
- Assume unstated steps are "obvious" - check if they're actually justified
- Penalize for different but valid proof techniques

## Common Mistakes to Avoid
- Grade inflation: Don't give 6-7 for incomplete solutions
- Grade deflation: Don't give 0-1 when genuine progress was made
- Overvaluing final answers: A correct guess without proof gets 0-1
- Undervaluing partial work: Multiple correct steps deserve 3-4, not 1-2

## Output Format (CRITICAL - FOLLOW EXACTLY)
Respond with ONLY this JSON format. No other text before or after:

<json>
{{
    "reasoning": "Your detailed analysis here. Structure as:\n1) Problem summary and key requirements\n2) Student's approach and strategy\n3) Key correct elements found (list each with point value)\n4) Errors or gaps identified\n5) Comparison to official solution\n6) Point-by-point justification summing to final grade\n7) Final grade confirmation",
    "response": "X"
}}
</json>

STRICT RULES - READ CAREFULLY:
1. The "response" field MUST contain ONLY a single digit from 0-7.
2. Do NOT include any text, words, or explanations in the response field.
3. Do NOT use quotes around the number in the response field (the JSON format already provides quotes).
4. Example CORRECT output: "response": "5"
5. Example INCORRECT outputs: "response": "5 points", "response": "Grade: 5", "response": "five"
6. The response field should contain ONLY one of these exact strings: "0", "1", "2", "3", "4", "5", "6", or "7"
7. Double-check your JSON is valid before outputting.
8. Ensure proper escaping of quotes within the reasoning field.

FINAL VERIFICATION CHECKLIST:
- [ ] I have analyzed the student's approach thoroughly
- [ ] I have identified all correct key ideas and steps
- [ ] I have noted all errors and gaps
- [ ] My grade reflects the actual quality of work (not just the final answer)
- [ ] The response field contains ONLY a single digit 0-7
- [ ] The JSON format is valid and properly escaped
- [ ] Another expert grader would likely agree with my assessment"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced robustness.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            # Handle different message formats
            last_msg = ""
            if msg_history:
                last_entry = msg_history[-1]
                if isinstance(last_entry, dict):
                    # Try common keys for message content
                    last_msg = last_entry.get("text") or last_entry.get("content", "")
                    if not last_msg and "message" in last_entry:
                        msg_obj = last_entry["message"]
                        if isinstance(msg_obj, dict):
                            last_msg = msg_obj.get("content", "")
            
            if not last_msg:
                return prediction, reasoning
            
            # Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                    # Clean up the prediction - remove any extra quotes
                    prediction = prediction.strip('"\'')
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                    prediction = prediction.strip('"\'')
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Fallback: try to find any JSON-like structure with response field
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg, re.DOTALL)
            for json_match in reversed(json_matches):  # Try last match first
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "None")).strip()
                    pred = pred.strip('"\'')
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        break
                except json.JSONDecodeError:
                    continue
            
            # If still no prediction, try broader JSON extraction with balanced braces
            if prediction == "None":
                # Look for JSON objects with balanced braces
                def find_json_objects(text):
                    """Find all JSON objects in text using balanced brace counting."""
                    objects = []
                    i = 0
                    while i < len(text):
                        if text[i] == '{':
                            start = i
                            brace_count = 1
                            i += 1
                            while i < len(text) and brace_count > 0:
                                if text[i] == '{':
                                    brace_count += 1
                                elif text[i] == '}':
                                    brace_count -= 1
                                i += 1
                            if brace_count == 0:
                                objects.append(text[start:i])
                        else:
                            i += 1
                    return objects
                
                json_objects = find_json_objects(last_msg)
                for json_str in reversed(json_objects):
                    try:
                        data = json.loads(json_str)
                        if "response" in data:
                            prediction = str(data["response"]).strip()
                            prediction = prediction.strip('"\'')
                            if "reasoning" in data:
                                reasoning = str(data["reasoning"])
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            # Last resort: use the comprehensive text extraction
            if prediction == "None":
                extracted_grade, method = _extract_grade_from_text(last_msg)
                if extracted_grade != "None":
                    prediction = extracted_grade
                    self.log_fn(f"Extracted grade via text analysis (method: {method})")
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", []

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
        
        # Log the reasoning and validation result
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
        
        # If grade is invalid, use the comprehensive text extraction on the raw response
        if not is_valid and response:
            extracted_grade, method = _extract_grade_from_text(response)
            if extracted_grade != "None":
                validated_grade = extracted_grade
                is_valid = True
                self.log_fn(f"Text extraction found grade: {validated_grade} (method: {method})")
        
        # If still invalid, retry with a clearer, more forceful prompt
        if not is_valid:
            self.log_fn(f"Grade validation failed, retrying with clearer prompt...")
            retry_instruction = instruction + """

⚠️⚠️⚠️ CRITICAL ERROR IN PREVIOUS ATTEMPT ⚠️⚠️⚠️

Your previous response did not follow the required format. The "response" field MUST contain ONLY a single digit from 0-7.

CORRECT FORMAT:
<json>
{
    "reasoning": "Your detailed analysis here...",
    "response": "5"
}
</json>

INCORRECT FORMATS (DO NOT USE):
- "response": "5 points" 
- "response": "Grade: 5"
- "response": "five"
- "response": "The grade is 5"

STRICT REQUIREMENT: The response field must contain ONLY one of: "0", "1", "2", "3", "4", "5", "6", or "7"

Please respond with valid JSON in the exact format specified."""
            try:
                retry_response, retry_msg_history, retry_info = get_response_from_llm(
                    msg=retry_instruction,
                    model=self.model,
                    msg_history=[],
                )
                retry_prediction, retry_reasoning = self._extract_prediction(retry_msg_history)
                validated_grade, is_valid = _validate_grade(retry_prediction, grading_guidelines)
                self.log_fn(f"Retry result: grade={validated_grade}, valid={is_valid}")
                if is_valid:
                    msg_history = retry_msg_history
                    if retry_reasoning:
                        self.log_fn(f"Retry reasoning: {retry_reasoning[:200]}...")
                else:
                    # Try text extraction on retry response
                    extracted_grade, method = _extract_grade_from_text(retry_response)
                    if extracted_grade != "None":
                        validated_grade = extracted_grade
                        is_valid = True
                        self.log_fn(f"Retry text extraction found grade: {validated_grade} (method: {method})")
            except Exception as e:
                self.log_fn(f"Retry LLM call failed: {e}")

        # Second retry with even simpler prompt if still invalid
        if not is_valid:
            self.log_fn(f"Second retry with simplified prompt...")
            simple_prompt = f"""Grade this IMO solution on a scale of 0-7.

Problem: {inputs.get('problem', '')[:500]}

Student Answer: {inputs.get('student_answer', '')[:1000]}

Respond ONLY with a JSON object in this exact format:
<json>
{{
    "reasoning": "Brief analysis of the solution quality",
    "response": "X"
}}
</json>

Where X is exactly one digit: 0, 1, 2, 3, 4, 5, 6, or 7.
The response field must contain ONLY the digit, nothing else."""
            try:
                simple_response, simple_msg_history, simple_info = get_response_from_llm(
                    msg=simple_prompt,
                    model=self.model,
                    msg_history=[],
                )
                simple_prediction, simple_reasoning = self._extract_prediction(simple_msg_history)
                validated_grade, is_valid = _validate_grade(simple_prediction, grading_guidelines)
                self.log_fn(f"Simple retry result: grade={validated_grade}, valid={is_valid}")
                if is_valid:
                    msg_history = simple_msg_history
                else:
                    # Try text extraction on simple response
                    extracted_grade, method = _extract_grade_from_text(simple_response)
                    if extracted_grade != "None":
                        validated_grade = extracted_grade
                        is_valid = True
                        self.log_fn(f"Simple retry text extraction found grade: {validated_grade} (method: {method})")
            except Exception as e:
                self.log_fn(f"Simple retry LLM call failed: {e}")

        # Final fallback: if still invalid, use the best guess from the original prediction
        if not is_valid and prediction != "None":
            # Try to extract any digit from the invalid prediction
            digit_match = re.search(r'[0-7]', prediction)
            if digit_match:
                validated_grade = digit_match.group(0)
                is_valid = True
                self.log_fn(f"Final fallback: extracted digit {validated_grade} from prediction")

        # Ultimate fallback: if we have grading guidelines, try to infer from them
        if not is_valid and grading_guidelines:
            # Look for grade indicators in the guidelines
            grade_indicators = re.findall(r'\b([0-7])\s*(?:points?)?', grading_guidelines.lower())
            if grade_indicators:
                # Use the most common grade mentioned in guidelines as a hint
                most_common = Counter(grade_indicators).most_common(1)[0][0]
                validated_grade = most_common
                is_valid = True
                self.log_fn(f"Ultimate fallback: using grade {validated_grade} from guidelines")

        return str(validated_grade), msg_history
