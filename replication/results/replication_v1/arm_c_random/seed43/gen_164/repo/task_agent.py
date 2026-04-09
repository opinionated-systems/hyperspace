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
    
    # Remove any surrounding quotes
    pred_clean = pred_clean.strip('"\'')
    
    # Strict validation: only accept single digit 0-7
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for grade patterns like "grade of X" or "score: X"
    grade_of_match = re.search(r'(?:grade|score|mark)s?\s+(?:of|is|:|=)\s*([0-7])\b', pred_lower)
    if grade_of_match:
        return grade_of_match.group(1), True
    
    # Check for JSON-like patterns: "response": "5" or 'response': '5'
    json_match = re.search(r'["\']?response["\']?\s*:\s*["\']?([0-7])["\']?', pred_lower)
    if json_match:
        return json_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bmax(?:imum)?\s*(?:score|points?|grade)?\b',
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
        r'\bno\s*progress\b',
        r'\bno\s*meaningful\b',
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
    
    # Check for digit followed by /7 pattern (e.g., "5/7" meaning 5 out of 7)
    out_of_match = re.search(r'\b([0-7])\s*/\s*(?:7|total|max)\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
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

### Grade 7 (Complete Solution)
- Complete, correct solution with rigorous proof
- All logical steps are sound and properly justified
- May have minor notational issues that don't affect correctness
- All key insights from official solution are present

### Grade 6 (Near-Complete with Minor Flaw)
- Essentially complete and correct solution
- One small gap in reasoning, or minor calculation error, or typo
- The flaw doesn't significantly impact the overall correctness
- Most of the proof structure is sound

### Grade 5 (Significant Progress)
- Substantial solution elements present
- Most key ideas from official solution are included
- Some gaps remain but the core approach is correct
- Significant progress toward complete solution

### Grade 4 (Good Partial Progress)
- Multiple correct key steps demonstrated
- Solution is incomplete or has notable errors
- Shows understanding of problem structure
- Several meaningful insights present

### Grade 3 (Some Genuine Progress)
- At least one key idea or meaningful step toward solution
- Demonstrates some understanding of the problem
- Work is relevant and partially correct
- Not just random attempts

### Grade 2 (Minimal Progress)
- Some relevant ideas or observations
- Little substantive work toward solution
- Shows awareness of problem but limited progress
- May have some correct but trivial observations

### Grade 1 (Very Minimal Progress)
- Some awareness of problem structure
- Essentially no useful work toward solution
- May restate problem or make trivial observations
- No meaningful mathematical progress

### Grade 0 (No Credit)
- No meaningful progress
- Completely incorrect approach
- Blank submission or irrelevant content
- No understanding of problem demonstrated

## Step-by-Step Grading Process

### Step 1: Understand the Problem
- Read the problem statement carefully
- Identify what needs to be proved or found
- Note any special conditions or constraints

### Step 2: Analyze the Official Solution
- Identify the key insights and techniques used
- Note the logical structure and critical steps
- Understand what constitutes a complete solution

### Step 3: Evaluate Student's Work
- Identify the student's overall approach
- Check each claim and calculation for correctness
- Note which key insights they discovered
- Identify gaps, errors, or missing steps

### Step 4: Compare to Rubric
- Match student's progress to the grade descriptions above
- Consider: What would another expert grader assign?
- Be consistent with IMO grading standards

### Step 5: Assign Grade
- Choose the grade that best matches the student's work
- Ensure the grade reflects actual demonstrated work
- Document your reasoning clearly

## Common Grading Pitfalls to AVOID

1. **Don't reward correct final answers without proof** - A correct answer with invalid reasoning gets 0, not 7
2. **Don't penalize for minor notation** - f(x) vs f(y) notation errors don't reduce grade if logic is sound
3. **Don't ignore partial credit** - Award points for each correct key insight, even if final answer is wrong
4. **Don't be swayed by length** - Short correct solutions deserve full credit; long incorrect ones get 0
5. **Don't assume unstated steps** - If a critical step is missing, the solution is incomplete
6. **Don't give benefit of the doubt** - Grade what is actually written, not what student might have meant
7. **Don't round up** - When in doubt between two grades, use the lower one (be conservative)

## Few-Shot Examples

### Example 1: Grade 7 (Complete Solution)
Problem: Prove that for all positive real numbers a, b, c: (a+b+c)(1/a+1/b+1/c) ≥ 9
Student's Answer: By AM-GM, a+b+c ≥ 3∛(abc) and 1/a+1/b+1/c ≥ 3/∛(abc). Multiplying gives (a+b+c)(1/a+1/b+1/c) ≥ 9. Equality holds when a=b=c.
Grade: 7 - Complete proof with correct application of AM-GM inequality and verification of equality condition.

### Example 2: Grade 6 (Minor Flaw)
Problem: Prove sum of first n odd numbers is n².
Student's Answer: The sum is 1+3+5+...+(2n-1). Using arithmetic series formula: n/2 × (first + last) = n/2 × (1 + 2n-1) = n/2 × 2n = n². [Forgot to mention this works for all n]
Grade: 6 - Correct proof but minor omission in explicitly stating the formula applies to all n.

### Example 3: Grade 5 (Significant Progress)
Problem: Find all functions f: R→R such that f(x+y) = f(x) + f(y).
Student's Answer: Let x=y=0, then f(0) = 2f(0), so f(0) = 0. Let y=-x, then f(0) = f(x) + f(-x), so f(-x) = -f(x). For rational q, f(qx) = qf(x). The function is linear on rationals.
Grade: 5 - Found f(0)=0, proved oddness, and showed linearity on rationals. Missing: extension to reals and proof that f(x)=cx for some constant c.

### Example 4: Grade 4 (Good Partial Progress)
Problem: Prove there are infinitely many primes.
Student's Answer: Assume there are finitely many primes p₁, p₂, ..., pₙ. Consider N = p₁×p₂×...×pₙ + 1. This number N is not divisible by any of the primes p₁ through pₙ.
Grade: 4 - Correctly set up proof by contradiction and constructed the key number N. Missing: explicit conclusion that N is either prime or has a prime factor not in the list, leading to contradiction.

### Example 5: Grade 3 (Some Genuine Progress)
Problem: Prove triangle inequality for complex numbers |z₁ + z₂| ≤ |z₁| + |z₂|.
Student's Answer: Let z₁ = a+bi and z₂ = c+di. Then |z₁ + z₂|² = (a+c)² + (b+d)² = a² + 2ac + c² + b² + 2bd + d².
Grade: 3 - Correctly expanded |z₁ + z₂|². Missing: comparison to |z₁|² + |z₂|² + 2|z₁||z₂| and application of Cauchy-Schwarz or direct algebraic completion.

### Example 6: Grade 2 (Minimal Progress)
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Student's Answer: Base case: n=1, LHS=1, RHS=1(2)/2=1. ✓ For induction, assume true for n=k, so 1+2+...+k = k(k+1)/2.
Grade: 2 - Correct base case and induction hypothesis setup. Missing: inductive step showing it holds for k+1.

### Example 7: Grade 1 (Very Minimal Progress)
Problem: Find the maximum of f(x) = x(1-x) on [0,1].
Student's Answer: This is a quadratic function. The maximum occurs at the vertex.
Grade: 1 - Correctly identified function type and that maximum is at vertex. Missing: any calculation of vertex location or maximum value.

### Example 8: Grade 0 (No Credit)
Problem: Prove there are infinitely many primes.
Student's Answer: The numbers 2, 3, 5, 7, 11 are all prime. I think there are infinitely many because numbers go on forever.
Grade: 0 - No valid mathematical reasoning. Listing examples and stating intuition without proof.

## Output Format (CRITICAL - FOLLOW EXACTLY)
Respond with ONLY this JSON format. No other text before or after:

<json>
{{
    "reasoning": "Your detailed analysis here. Structure as: 1) Student's approach summary, 2) Key correct elements found, 3) Errors or gaps identified, 4) Comparison to official solution, 5) Explicit justification for the specific grade assigned using the rubric above",
    "response": "X"
}}
</json>

STRICT RULES:
- "response" field: ONLY a single digit 0-7. No quotes, no spaces, no explanation, no punctuation.
- "reasoning" field: Complete analysis as described above, referencing specific rubric criteria.
- Ensure valid JSON with proper quotes and commas.
- Example correct response field: "response": "5"
- Example incorrect response field: "response": "5 points" or "response": "grade 5"
- When uncertain between two grades, choose the LOWER grade (be conservative)."""

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
                # Look for JSON objects with balanced braces
                def find_json_objects(text):
                    """Find all JSON-like objects with balanced braces."""
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
                            if "reasoning" in data:
                                reasoning = str(data["reasoning"])
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*[:=]\s*([0-7])\b',
                    r'(?:grade|score|mark|final grade|final score)\s+is\s+([0-7])\b',
                    r'["\']response["\']\s*:\s*["\']?([0-7])["\']?',
                    r'\bgrade\s+([0-7])\s*(?:points?)?\b',
                    r'\b(?:the\s+)?grade\s+(?:should\s+be\s+|is\s+)?([0-7])\b',
                    r'\b(?:i\s+)?(?:assign|give|award)\s+(?:a\s+)?(?:grade\s+)?(?:of\s+)?([0-7])\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
                        break
            
            # Extract reasoning if not found yet - look for reasoning field patterns
            if not reasoning:
                reasoning_patterns = [
                    r'["\']reasoning["\']\s*:\s*["\']([^"\']+)["\']',
                    r'["\']reasoning["\']\s*:\s*"([^"]*)"',
                    r'reasoning[:\s]+(.+?)(?=\n\n|\Z|grade:|score:)',
                ]
                for pattern in reasoning_patterns:
                    reasoning_match = re.search(pattern, last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
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
        
        return f"""You are an expert IMO grader performing an independent verification check. You must review the initial grading decision as if you are a second expert grader who has not seen the first grader's work.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Initial Grading Decision (for reference only)
Grade: {initial_grade}
Reasoning: {reasoning}

## Your Verification Task
Perform a COMPLETE independent re-evaluation of the student's work. Do NOT simply agree with the initial grade. Apply the full grading rubric yourself:

### Step 1: Independent Analysis
- Analyze the student's solution without bias from the initial grade
- Identify all correct elements and all errors/gaps
- Compare against the official solution's key insights

### Step 2: Apply Rubric
Match the work to the appropriate grade level:
- **7**: Complete, rigorous proof with all key insights
- **6**: Near-complete with only minor flaw
- **5**: Significant progress, most key ideas present
- **4**: Good partial progress, multiple correct steps
- **3**: Some genuine progress, at least one key idea
- **2**: Minimal progress, relevant observations
- **1**: Very minimal progress, awareness only
- **0**: No meaningful progress

### Step 3: Determine Confidence
- **high**: Clear match to rubric, unambiguous grade
- **medium**: Some ambiguity but reasonable confidence
- **low**: Borderline case, significant uncertainty

## Critical Verification Questions
1. Did the initial grader potentially OVER-grade (give too much credit)?
2. Did the initial grader potentially UNDER-grade (miss correct elements)?
3. Is there a pattern in the initial grader's reasoning that suggests bias?
4. Would a panel of expert graders likely agree with the initial assessment?

## Output Format
Respond with ONLY this JSON format:

<json>
{{
    "independent_analysis": "Your own analysis of the student's work, independent of the initial grade",
    "verification_reasoning": "Your assessment of whether the initial grade was correct, including any discrepancies found",
    "verified_grade": "X",
    "confidence": "high/medium/low",
    "discrepancy_type": "none/over_grade/under_grade/uncertain"
}}
</json>

STRICT RULES:
- "verified_grade" field: ONLY a single digit 0-7
- Be objective and independent - don't defer to the initial grade
- If you disagree with the initial grade, state this clearly in verification_reasoning
- The discrepancy_type indicates: none (agree), over_grade (initial too high), under_grade (initial too low), uncertain (can't determine)"""

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

        # Self-consistency verification: double-check all grades for accuracy
        # This helps catch both over-grading and under-grading errors
        final_grade = str(validated_grade)
        if is_valid:
            try:
                self.log_fn(f"Performing verification check for grade {final_grade}...")
                verification_prompt = self._build_verification_prompt(inputs, final_grade, reasoning)
                verify_response, verify_msg_history, verify_info = get_response_from_llm(
                    msg=verification_prompt,
                    model=self.model,
                    msg_history=[],
                )
                
                # Extract verification data
                verify_json = _extract_jsons(verify_response)
                if verify_json:
                    verify_data = verify_json[-1]
                    verified_grade = str(verify_data.get("verified_grade", final_grade)).strip()
                    confidence = verify_data.get("confidence", "medium")
                    discrepancy_type = verify_data.get("discrepancy_type", "none")
                    verification_reasoning = verify_data.get("verification_reasoning", "")
                    
                    # Validate the verification grade
                    verified_grade, verify_valid = _validate_grade(verified_grade, grading_guidelines)
                    
                    if verify_valid:
                        original_int = int(final_grade) if final_grade.isdigit() else -1
                        verified_int = int(verified_grade) if verified_grade.isdigit() else -1
                        grade_diff = abs(original_int - verified_int)
                        
                        # Log verification details
                        self.log_fn(f"Verification: original={final_grade}, verified={verified_grade}, "
                                  f"confidence={confidence}, discrepancy={discrepancy_type}")
                        
                        # Decision logic based on verification results
                        if grade_diff >= 2:
                            # Significant discrepancy - use conservative approach
                            self.log_fn(f"Significant grade discrepancy detected ({grade_diff} points)")
                            # For borderline cases, prefer the lower grade to avoid over-grading
                            if discrepancy_type == "over_grade":
                                final_grade = str(verified_grade)
                                self.log_fn(f"Using verified grade {final_grade} (corrected over-grade)")
                            elif discrepancy_type == "under_grade":
                                # Be conservative - only upgrade if confidence is high
                                if confidence == "high":
                                    final_grade = str(verified_grade)
                                    self.log_fn(f"Upgrading to {final_grade} (high confidence correction)")
                                else:
                                    self.log_fn(f"Keeping original grade {final_grade} (low confidence in upgrade)")
                            else:
                                # Uncertain discrepancy - use lower grade
                                final_grade = str(min(original_int, verified_int))
                                self.log_fn(f"Using conservative grade {final_grade}")
                        elif grade_diff == 1:
                            # Small discrepancy - use confidence to decide
                            if confidence == "high" and discrepancy_type in ["over_grade", "under_grade"]:
                                final_grade = str(verified_grade)
                                self.log_fn(f"Adjusting grade to {final_grade} (high confidence, 1-point correction)")
                            elif confidence == "low":
                                # For low confidence with 1-point diff, prefer lower grade
                                final_grade = str(min(original_int, verified_int))
                                self.log_fn(f"Using conservative grade {final_grade} (low confidence)")
                            else:
                                self.log_fn(f"Keeping original grade {final_grade} (medium confidence, minor diff)")
                        else:
                            # Grades agree
                            if confidence == "low":
                                self.log_fn(f"Grades agree but with low confidence - keeping {final_grade}")
                            else:
                                self.log_fn(f"Verification confirmed grade: {final_grade}")
                    else:
                        self.log_fn(f"Verification grade invalid, keeping original: {final_grade}")
                else:
                    self.log_fn(f"Could not extract verification JSON, keeping original grade: {final_grade}")
            except Exception as e:
                self.log_fn(f"Verification check failed: {e}, keeping original grade {final_grade}")

        return final_grade, msg_history
