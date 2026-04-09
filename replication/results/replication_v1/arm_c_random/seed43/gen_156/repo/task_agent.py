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
    
    # Strict validation: only accept single digit 0-7
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
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
    return pred_clean, Falsesearch(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for grade patterns like "grade of X" or "score: X"
    grade_of_match = re.search(r'(?:grade|score|mark)s?\s+(?:of|is|:|=)\s*([0-7])\b', pred_lower)
    if grade_of_match:
        return grade_of_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
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
- **7 points**: Complete, correct solution with rigorous proof and proper justification. All steps are logically sound.
- **6 points**: Minor flaw (e.g., small gap in reasoning, typo in calculation) in an otherwise complete and correct solution.
- **5 points**: Significant progress with substantial solution elements. Most key ideas present but some gaps remain.
- **4 points**: Good partial progress. Multiple correct key steps but solution incomplete or has notable errors.
- **3 points**: Some genuine progress. At least one key idea or meaningful step toward solution.
- **2 points**: Minimal progress. Some relevant ideas or observations but little substantive work.
- **1 point**: Very minimal progress. Some awareness of problem structure but essentially no useful work.
- **0 points**: No meaningful progress, completely incorrect approach, or blank submission.

## Grading Instructions
1. **Read carefully**: First understand what the problem asks and what the official solution demonstrates.
2. **Analyze the student's approach**: Identify their strategy and key mathematical ideas.
3. **Check correctness**: Verify each claim and calculation in the student's work.
4. **Compare to official solution**: Note which parts they completed correctly vs. where they went wrong.
5. **Assess partial credit**: Award points for each correct key insight or meaningful step, even if final answer is wrong.
6. **Be precise**: The grade must reflect the actual quality of work, not your guess at their intent.

## Common Grading Pitfalls to Avoid
- Don't give full credit just for correct final answer without proper proof
- Don't penalize for minor notational issues if logic is sound
- Don't ignore partial progress - award points for correct intermediate steps
- Don't be swayed by length; short correct solutions deserve full credit
- Don't assume unstated steps are obvious; check if they're actually justified
- Don't over-penalize for missing edge cases if the main argument is correct
- Don't under-grade solutions that use alternative valid approaches

## Few-Shot Examples

### Example 1: Complete Solution (Grade 7)
Problem: Prove that for all positive real numbers a, b, c: (a+b+c)(1/a+1/b+1/c) ≥ 9
Student's Answer: By AM-GM, a+b+c ≥ 3∛(abc) and 1/a+1/b+1/c ≥ 3/∛(abc). Multiplying gives (a+b+c)(1/a+1/b+1/c) ≥ 9. Equality holds when a=b=c.
Grade: 7 - Complete proof with correct application of AM-GM inequality and verification of equality condition.

### Example 2: Near-Complete with Minor Flaw (Grade 6)
Problem: Prove that the sum of the first n odd numbers is n².
Student's Answer: We use induction. Base case: n=1, sum is 1 = 1². Assume true for n=k, so 1+3+...+(2k-1) = k². For n=k+1, we add (2k+1) to both sides: 1+3+...+(2k-1)+(2k+1) = k² + 2k + 1 = (k+1)². Thus by induction, the formula holds for all n.
Grade: 6 - The proof is essentially complete and correct. The induction structure is sound, base case verified, inductive step correctly executed. Minor note: could explicitly state "for all positive integers n" but this is a minor formality.

### Example 3: Significant Progress (Grade 5)
Problem: Find all primes p such that p² + 2 is also prime.
Student's Answer: Testing small primes: p=2 gives 4+2=6 (not prime), p=3 gives 9+2=11 (prime), p=5 gives 25+2=27 (not prime), p=7 gives 49+2=51 (not prime). For p>3, p² ≡ 1 (mod 3), so p²+2 ≡ 0 (mod 3), making it divisible by 3 and greater than 3, hence not prime. Therefore only p=3 works.
Grade: 5 - Student correctly identified the pattern, tested cases, and found the key modular arithmetic argument. However, they should explicitly justify why p² ≡ 1 (mod 3) for p>3 (since p not divisible by 3 means p ≡ ±1 mod 3). The core insight is present but the modular arithmetic justification needs more detail.

### Example 4: Good Partial Progress (Grade 4)
Problem: Prove that in any triangle, the sum of the medians is greater than 3/4 of the perimeter.
Student's Answer: Let the sides be a, b, c and medians be m_a, m_b, m_c. By the triangle inequality applied to the medians, we know m_a + m_b > c/2, m_b + m_c > a/2, m_c + m_a > b/2. Adding these: 2(m_a + m_b + m_c) > (a+b+c)/2, so m_a + m_b + m_c > (a+b+c)/4.
Grade: 4 - Student correctly applied triangle inequality to medians and derived a lower bound. However, they only proved the sum exceeds 1/4 of the perimeter, not 3/4. Multiple correct steps but didn't reach the target bound.

### Example 5: Some Genuine Progress (Grade 3)
Problem: Find all functions f: R→R such that f(x+y) = f(x) + f(y) for all x,y.
Student's Answer: Let x=y=0, then f(0) = 2f(0), so f(0) = 0. Let y=0, then f(x) = f(x) + f(0), which is consistent. The function seems linear.
Grade: 3 - Student found f(0)=0 correctly and made a relevant observation about linearity. However, they didn't prove linearity or find all solutions (f(x)=cx for rational c, with additional constraints for continuous solutions). Some genuine progress with correct initial steps.

### Example 6: Minimal Progress (Grade 2)
Problem: Prove there are infinitely many primes of the form 4k+1.
Student's Answer: Primes of form 4k+1 are 5, 13, 17, 29, 37... They seem to keep appearing. Maybe we can use a similar argument to Euclid's proof.
Grade: 2 - Student listed examples and recognized a connection to Euclid's proof method, but provided no actual proof structure or meaningful progress toward the solution.

### Example 7: Very Minimal Progress (Grade 1)
Problem: Prove that the harmonic series diverges.
Student's Answer: The series is 1 + 1/2 + 1/3 + 1/4 + ... It keeps adding smaller and smaller terms. I'm not sure if it converges or diverges.
Grade: 1 - Student wrote down the series correctly and showed awareness of the problem, but made no meaningful mathematical progress toward proving divergence.

### Example 8: Incorrect Approach (Grade 0)
Problem: Prove there are infinitely many primes.
Student's Answer: The numbers 2, 3, 5, 7, 11 are all prime. I think there are infinitely many because numbers go on forever.
Grade: 0 - No valid mathematical reasoning. Listing examples and stating intuition without proof.

## Output Format (CRITICAL - FOLLOW EXACTLY)
Respond with ONLY this JSON format. No other text before or after:

<json>
{{
    "reasoning": "Your detailed analysis here. Structure as: 1) Student's approach summary, 2) Key correct elements found, 3) Errors or gaps identified, 4) Comparison to official solution, 5) Justification for the specific grade assigned",
    "response": "X"
}}
</json>

STRICT RULES:
- "response" field: ONLY a single digit 0-7. No quotes, no spaces, no explanation, no punctuation.
- "reasoning" field: Complete analysis as described above.
- Ensure valid JSON with proper quotes and commas.
- Example correct response field: "response": "5"
- Example incorrect response field: "response": "5 points" or "response": "grade 5"

## Final Check Before Responding
- Verify your grade matches the IMO 0-7 scale criteria exactly
- Ensure your reasoning justifies why that specific grade was chosen
- Double-check that the response field contains ONLY a single digit 0-7
- Confirm your JSON is valid and properly formatted"""

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
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
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
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        break
                except json.JSONDecodeError:
                    continue
            
            # If still no prediction, try broader JSON extraction with balanced braces
            if prediction == "None":
                # Look for any JSON object in the text using balanced brace matching
                def find_json_objects(text):
                    """Find all valid JSON objects in text using balanced brace matching."""
                    objects = []
                    i = 0
                    while i < len(text):
                        if text[i] == '{':
                            brace_count = 1
                            j = i + 1
                            while j < len(text) and brace_count > 0:
                                if text[j] == '{':
                                    brace_count += 1
                                elif text[j] == '}':
                                    brace_count -= 1
                                j += 1
                            if brace_count == 0:
                                candidate = text[i:j]
                                try:
                                    data = json.loads(candidate)
                                    objects.append(data)
                                except json.JSONDecodeError:
                                    pass
                            i = j
                        else:
                            i += 1
                    return objects
                
                json_objects = find_json_objects(last_msg)
                for data in reversed(json_objects):
                    if "response" in data:
                        prediction = str(data["response"]).strip()
                        if "reasoning" in data:
                            reasoning = str(data["reasoning"])
                        break
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b',
                    r'(?:grade|score|mark|final grade|final score)\s+is\s+([0-7])\b',
                    r'["\']response["\']\s*:\s*["\']?([0-7])["\']?',
                    r'\bgrade\s+([0-7])\s*(?:points?)?\b',
                    r'\b(?:the\s+)?(?:grade|score)\s+(?:should\s+)?(?:be\s+)?([0-7])\b',
                    r'\b(?:award|assign|give)\s+(?:a\s+)?([0-7])\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
                        break
            
            # Extract reasoning if not found yet
            if not reasoning and last_msg:
                # Try to extract reasoning from various patterns
                reasoning_patterns = [
                    r'["\']reasoning["\']\s*:\s*["\']([^"\']+)["\']',
                    r'["\']reasoning["\']\s*:\s*"([^"]*)"',
                    r'reasoning[:\s]+(.+?)(?:\n\n|\Z)',
                ]
                for pattern in reasoning_patterns:
                    match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                    if match:
                        reasoning = match.group(1).strip()
                        break
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def _build_verification_prompt(self, inputs: dict, initial_grade: str, reasoning: str) -> str:
        """Build a verification prompt to double-check the grading decision.
        
        This implements a self-consistency check where the model reviews its own
        grading decision to catch potential errors or biases.
        """
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert IMO grader performing an independent verification check. You must review the initial grading decision as if you are a second expert grader who has not seen the first grader's assessment.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Initial Grading Decision (for reference only)
Initial Grade: {initial_grade}
Initial Reasoning: {reasoning}

## Your Independent Verification Task
IMPORTANT: Grade this solution independently first, BEFORE looking at the initial grade. Then compare your independent assessment with the initial grade.

Step 1 - Independent Assessment:
- Read the student's solution carefully
- Apply IMO grading standards (0-7 scale) based on the criteria
- Determine what grade YOU would assign

Step 2 - Comparison:
- Compare your independent grade with the initial grade
- If they match exactly or differ by only 1 point, the initial grade is likely correct
- If they differ by 2 or more points, one of the assessments may be incorrect

Step 3 - Critical Review Questions:
1. Did the initial grader potentially over-grade (give too many points for incomplete work)?
2. Did the initial grader potentially under-grade (miss correct elements or valid approaches)?
3. Are there alternative valid approaches the initial grader might have missed?
4. Is the initial grade consistent with the specific IMO 0-7 scale definitions?

## IMO Grading Scale (0-7) - USE EXACTLY
- **7**: Complete, correct solution with rigorous proof
- **6**: Minor flaw in otherwise complete solution
- **5**: Significant progress, most key ideas present
- **4**: Good partial progress, multiple correct steps
- **3**: Some genuine progress, at least one key idea
- **2**: Minimal progress, relevant observations
- **1**: Very minimal progress, awareness of problem
- **0**: No meaningful progress

## Output Format
Respond with ONLY this JSON format:

<json>
{{
    "independent_grade": "X",
    "verification_reasoning": "Your analysis: 1) Your independent assessment, 2) Comparison with initial grade, 3) Whether initial grade was accurate, 4) Any discrepancies found",
    "verified_grade": "X",
    "confidence": "high/medium/low",
    "discrepancy_found": true/false
}}
</json>

Field definitions:
- "independent_grade": The grade YOU would assign based on your independent assessment (0-7)
- "verification_reasoning": Your detailed analysis
- "verified_grade": The final verified grade - use your independent grade if you found a significant discrepancy, otherwise use the initial grade (0-7)
- "confidence": Your confidence in the final grade (high/medium/low)
- "discrepancy_found": true if your independent grade differs from initial by 2+ points, false otherwise

CRITICAL: All grade fields must contain ONLY a single digit 0-7."""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning, validation, and verification.

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
        
        # If grade is invalid, try to extract from the full response text
        if not is_valid and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")
            else:
                # Try to find grade patterns in the full response
                grade_patterns = [
                    (r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b', 'numeric'),
                    (r'["\']response["\']\s*:\s*["\']?([0-7])["\']?', 'json_field'),
                    (r'\bfull\s*(?:credit|points?|score|marks?)?\b', 'full'),
                    (r'\bcorrect\s*(?:solution|answer)?\b', 'correct'),
                    (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                    (r'\bwrong\s*(?:solution|answer)?\b', 'zero'),
                ]
                for pattern, pattern_type in grade_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        if pattern_type in ['full', 'correct']:
                            validated_grade = "7"
                        elif pattern_type == 'zero':
                            validated_grade = "0"
                        else:
                            validated_grade = match.group(1)
                        is_valid = True
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break
        
        # If still invalid, retry with a clearer prompt
        if not is_valid:
            self.log_fn(f"Grade validation failed, retrying with clearer prompt...")
            retry_instruction = instruction + """

CRITICAL REMINDER: Your response field MUST contain ONLY a single digit from 0-7. 
- Correct: "response": "5"
- Incorrect: "response": "5 points" or "response": "Grade: 5"
- The response field should be ONLY the digit, nothing else.

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
                    reasoning = retry_reasoning
                    if retry_reasoning:
                        self.log_fn(f"Retry reasoning: {retry_reasoning[:200]}...")
            except Exception as e:
                self.log_fn(f"Retry LLM call failed: {e}")

        # Final fallback: if still invalid, use the best guess from the original prediction
        if not is_valid and prediction != "None":
            # Try to extract any digit from the invalid prediction
            digit_match = re.search(r'[0-7]', prediction)
            if digit_match:
                validated_grade = digit_match.group(0)
                is_valid = True
                self.log_fn(f"Final fallback: extracted digit {validated_grade} from prediction")

        # Self-consistency verification: double-check high-stakes grades
        # Verify grades that are extreme (0 or 7) or if confidence is uncertain
        final_grade = str(validated_grade)
        if is_valid and final_grade in ["0", "7"]:
            try:
                self.log_fn(f"Performing verification check for grade {final_grade}...")
                verification_prompt = self._build_verification_prompt(inputs, final_grade, reasoning)
                verify_response, verify_msg_history, verify_info = get_response_from_llm(
                    msg=verification_prompt,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract verification grade
                verify_json = _extract_jsons(verify_response)
                if verify_json:
                    verify_data = verify_json[-1]
                    verified_grade = str(verify_data.get("verified_grade", final_grade)).strip()
                    independent_grade = str(verify_data.get("independent_grade", verified_grade)).strip()
                    confidence = verify_data.get("confidence", "medium")
                    discrepancy_found = verify_data.get("discrepancy_found", False)
                    
                    # Validate the verification grade
                    verified_grade, verify_valid = _validate_grade(verified_grade, grading_guidelines)
                    independent_grade, _ = _validate_grade(independent_grade, grading_guidelines)
                    
                    if verify_valid:
                        # If verification disagrees significantly, use the more conservative grade
                        original_int = int(final_grade) if final_grade.isdigit() else -1
                        verified_int = int(verified_grade) if verified_grade.isdigit() else -1
                        independent_int = int(independent_grade) if independent_grade.isdigit() else -1
                        
                        # Check for significant discrepancy (2+ points difference)
                        if abs(original_int - verified_int) >= 2 or abs(original_int - independent_int) >= 2 or discrepancy_found:
                            self.log_fn(f"Verification found significant discrepancy: original={final_grade}, verified={verified_grade}, independent={independent_grade}")
                            # Use the more conservative (lower) grade when there's disagreement
                            # This prevents over-grading while still catching under-grading
                            conservative_grade = min(original_int, verified_int, independent_int)
                            final_grade = str(conservative_grade)
                            self.log_fn(f"Using conservative grade: {final_grade}")
                        elif confidence == "low":
                            self.log_fn(f"Low confidence in verification, keeping original grade {final_grade}")
                        else:
                            self.log_fn(f"Verification confirmed grade: {verified_grade}")
                            final_grade = str(verified_grade)
            except Exception as e:
                self.log_fn(f"Verification check failed: {e}, keeping original grade")

        return final_grade, msg_history
