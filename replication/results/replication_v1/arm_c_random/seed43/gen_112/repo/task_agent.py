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
    Enhanced to handle nested JSON structures and edge cases.
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
        
        # Try direct parsing first
        try:
            results.append(json.loads(inner))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to find valid JSON within the content using brace counting
        try:
            json_start = inner.find('{')
            if json_start == -1:
                continue
                
            # Use brace counting to find the matching closing brace
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
                potential_json = inner[json_start:json_end+1]
                results.append(json.loads(potential_json))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: try simple find/rfind approach
        try:
            json_start = inner.find('{')
            json_end = inner.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                results.append(json.loads(inner[json_start:json_end+1]))
        except json.JSONDecodeError:
            continue
            
    return results or None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Enhanced to handle nested JSON structures and edge cases.
    """
    # Try json-specific blocks first
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also try generic code blocks
    generic_pattern = r'```\s*(.*?)\s*```'
    generic_matches = re.findall(generic_pattern, text, re.DOTALL)
    all_matches = matches + [m for m in generic_matches if m not in matches]
    
    for match in all_matches:
        match = match.strip()
        
        # Try direct parsing first
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            pass
        
        # Try brace counting for nested structures
        try:
            json_start = match.find('{')
            if json_start == -1:
                continue
                
            brace_count = 0
            json_end = -1
            for i, char in enumerate(match[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            
            if json_end != -1:
                potential_json = match[json_start:json_end+1]
                return json.loads(potential_json)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Last resort: simple find/rfind
        try:
            json_start = match.find('{')
            json_end = match.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                return json.loads(match[json_start:json_end+1].strip())
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
    
    # Strict validation: only accept single digit 0-7
    # This ensures consistent output format
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    numeric_match = re.search(r'\b([0-7])\b', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bvalid\s*(?:solution|proof|answer)?\b',
        r'\bsound\s*(?:solution|proof|reasoning)?\b',
        r'\bexcellent\s*(?:solution|work|answer)?\b',
        r'\bwell\s*done\b',
        r'\bfully\s*correct\b',
        r'\bcompletely\s*correct\b',
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
        r'\bempty\s*(?:answer|solution)?\b',
        r'\bblank\b',
        r'\binvalid\s*(?:solution|proof|approach)?\b',
        r'\bno\s*meaningful\s*(?:progress|work)?\b',
        r'\bcompletely\s*wrong\b',
        r'\btotally\s*incorrect\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True
    
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

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with extensive experience in mathematical problem evaluation.

Your task is to evaluate a student's solution to a mathematical problem with precision and consistency.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions for Evaluation

1. **Initial Assessment**: Read the student's answer completely before forming any conclusions.

2. **Step-by-Step Analysis**: 
   - Compare each step of the student's solution to the official solution
   - Identify any logical errors, computational mistakes, or gaps in reasoning
   - Note any creative alternative approaches that may be valid

3. **Partial Credit Evaluation**:
   - IMO problems use a 0-7 point scale
   - Award points for meaningful progress toward the solution
   - Consider partial results, correct methodology with errors, or incomplete proofs

4. **Final Grade Determination**:
   - 7 points: Complete, correct solution with proper justification
   - 6 points: Minor flaw in an otherwise complete solution
   - 5 points: Significant progress with substantial solution elements
   - 3-4 points: Partial progress with some correct elements
   - 1-2 points: Minimal progress or some relevant ideas
   - 0 points: No meaningful progress or completely incorrect

## Common Grading Scenarios

**Full Credit (7 points)**: The solution is complete, correct, and well-justified. All steps are logically sound and the final answer matches the official solution.

**Near-Complete (6 points)**: The solution is essentially correct but has a minor flaw (e.g., a small computational error, missing edge case, or slight gap in reasoning that doesn't affect the main result).

**Substantial Progress (5 points)**: The student made significant progress, perhaps solving a major part of the problem or providing a mostly correct approach with some significant gaps.

**Partial Progress (3-4 points)**: The student has some correct elements, perhaps a good initial approach, some correct calculations, or partial results that show understanding.

**Minimal Progress (1-2 points)**: The student shows some relevant ideas or understanding of the problem, but little concrete progress toward the solution.

**No Credit (0 points)**: The answer is blank, completely irrelevant, or shows no understanding of the problem.

## Partial Credit Guidelines

When assigning partial credit, consider:
- **Correct approach with errors (4-5 points)**: Student understands the method but makes execution errors
- **Correct intermediate results (3-4 points)**: Student gets some key lemmas or partial results correct
- **Good setup but no progress (2-3 points)**: Student correctly interprets the problem but cannot proceed
- **Relevant ideas only (1-2 points)**: Student mentions concepts related to the solution but cannot apply them
- **Wrong approach but some correct math (1-2 points)**: Method is wrong but calculations are correct

## IMO Grading Scale Reference (USE THIS EXACTLY)

- **7 points**: Complete, correct solution with proper justification. All steps logically sound.
- **6 points**: Minor flaw in otherwise complete solution (small error, missing edge case)
- **5 points**: Significant progress with substantial solution elements (major part solved)
- **4 points**: Good partial progress (correct approach with significant gaps)
- **3 points**: Some partial progress (correct intermediate results, good setup)
- **2 points**: Minimal progress (relevant ideas, correct interpretation but no solution)
- **1 point**: Very minimal progress (some relevant concepts mentioned)
- **0 points**: No meaningful progress, blank, or completely incorrect

**Important**: Always provide the specific numeric grade (0-7) that best matches the student's work based on the official solution and grading guidelines.

## Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags. The JSON must have exactly these two fields:

<json>
{{
    "reasoning": "Your detailed analysis here. Explain the student's approach, identify errors or gaps, compare to official solution, and justify your grade...",
    "response": "X"
}}
</json>

CRITICAL RULES:
- The "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
- Do NOT include quotes around the number, explanations, or any other text in the response field
- The "reasoning" field should contain your complete analysis (can be multiple sentences)
- Ensure the JSON is valid and properly formatted
- Double-check that your grade matches your reasoning (e.g., don't give 7 if you mention major errors)

EXAMPLES OF VALID RESPONSES:

Example 1 - Full credit:
<json>
{{
    "reasoning": "The student provided a complete and correct solution with proper justification. All steps are logically sound and match the official solution. The proof is rigorous and covers all cases.",
    "response": "7"
}}
</json>

Example 2 - Near complete:
<json>
{{
    "reasoning": "The solution is essentially correct with a minor computational error in the final step. The approach is valid and most of the proof is correct, but there's a small arithmetic mistake that doesn't affect the main result.",
    "response": "6"
}}
</json>

Example 3 - Significant progress:
<json>
{{
    "reasoning": "The student made significant progress, correctly identifying the key lemma and proving it. However, the final application of the lemma to solve the main problem is incomplete.",
    "response": "5"
}}
</json>

Example 4 - Partial progress:
<json>
{{
    "reasoning": "The student has a good initial approach and correctly set up the problem. They made some progress with correct intermediate calculations but couldn't complete the solution due to a logical gap.",
    "response": "4"
}}
</json>

Example 5 - Some progress:
<json>
{{
    "reasoning": "The student correctly interpreted the problem and mentioned some relevant concepts, but made limited concrete progress toward the solution.",
    "response": "3"
}}
</json>

Example 6 - Minimal progress:
<json>
{{
    "reasoning": "The student shows some understanding of the problem domain and mentions a relevant concept, but provides little concrete progress.",
    "response": "2"
}}
</json>

Example 7 - Very minimal progress:
<json>
{{
    "reasoning": "The student mentions a concept related to the problem but cannot apply it or make any meaningful progress.",
    "response": "1"
}}
</json>

Example 8 - No credit:
<json>
{{
    "reasoning": "The answer is blank or shows no understanding of the problem. No meaningful mathematical content is provided.",
    "response": "0"
}}
</json>"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced extraction.
        
        Uses multiple extraction strategies to handle various response formats.
        
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
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                last_json = extracted[-1]
                if "response" in last_json:
                    prediction = str(last_json["response"]).strip()
                if "reasoning" in last_json:
                    reasoning = str(last_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                return prediction, reasoning
            
            # Strategy 3: Try to find JSON with brace counting for nested structures
            start = last_msg.find('{')
            while start != -1:
                # Use brace counting to find matching closing brace
                brace_count = 0
                json_end = -1
                for i, char in enumerate(last_msg[start:], start=start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i
                            break
                
                if json_end != -1:
                    try:
                        potential_json = last_msg[start:json_end+1]
                        fallback = json.loads(potential_json)
                        if "response" in fallback:
                            prediction = str(fallback["response"]).strip()
                            if "reasoning" in fallback:
                                reasoning = str(fallback["reasoning"])
                            return prediction, reasoning
                    except json.JSONDecodeError:
                        pass
                
                # Look for next opening brace
                start = last_msg.find('{', start + 1)
            
            # Strategy 4: Look for grade patterns in text
            # Look for "Grade: X" or "Final grade: X" patterns
            grade_patterns = [
                r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b',
                r'(?:grade|score|mark)\s+is\s+([0-7])\b',
                r'(?:grade|score|mark)\s+of\s+([0-7])\b',
                r'\bgives?\s+(?:a\s+)?([0-7])\s*(?:points?)?\b',
                r'\baward\s+(?:a\s+)?([0-7])\b',
                r'\bassign\s+(?:a\s+)?([0-7])\b',
                r'\b(?:the\s+)?(?:grade|score)\s+(?:should\s+be|is)\s+([0-7])\b',
                r'\b(?:deserves?|earns?)\s+(?:a\s+)?([0-7])\b',
                r'\b(?:student|this)\s+(?:gets?|receives?)\s+(?:a\s+)?([0-7])\b',
                r'\b(?:i\s+)?(?:would\s+)?(?:give|assign|award)\s+(?:a\s+)?([0-7])\b',
            ]
            for pattern in grade_patterns:
                grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    # Try to extract reasoning from surrounding text
                    reasoning_match = re.search(r'(?:reasoning|analysis|explanation|assessment|evaluation)[:\s]+(.+?)(?:\n\n|\Z)', last_msg, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                    break
            
            # Strategy 5: Look for explicit grade mentions at end of text
            if prediction == "None":
                end_grade_match = re.search(r'\bgrade\s*:?\s*([0-7])\s*$', last_msg, re.IGNORECASE)
                if end_grade_match:
                    prediction = end_grade_match.group(1)
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def _build_retry_prompt(self, inputs: dict, previous_response: str, error_reason: str) -> str:
        """Build a retry prompt when the initial response was invalid.
        
        Args:
            inputs: Original problem inputs
            previous_response: The previous LLM response that failed validation
            error_reason: Why the previous response was invalid
            
        Returns:
            A formatted retry prompt
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Your previous response was invalid and needs to be corrected.

## Previous Response (INVALID)
{previous_response}

## Error
{error_reason}

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}

## Instructions for Evaluation (RETRY - READ CAREFULLY)

Your previous response could not be parsed correctly. This is a RETRY attempt. Please provide a valid JSON response following these CRITICAL rules:

### CRITICAL RULES:
1. The "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
2. Do NOT include quotes around the number, explanations, or any other text in the response field
3. The "reasoning" field should contain your complete analysis
4. Ensure the JSON is valid and properly formatted
5. The grade in "response" MUST match your reasoning - if you say the solution has errors, don't give it 7 points

### IMO Grading Scale Reference (USE THIS EXACTLY)

- **7 points**: Complete, correct solution with proper justification
- **6 points**: Minor flaw in otherwise complete solution
- **5 points**: Significant progress with substantial solution elements
- **4 points**: Good partial progress (correct approach with significant gaps)
- **3 points**: Some partial progress (correct intermediate results, good setup)
- **2 points**: Minimal progress (relevant ideas, correct interpretation but no solution)
- **1 point**: Very minimal progress (some relevant concepts mentioned)
- **0 points**: No meaningful progress, blank, or completely incorrect

### Output Format (STRICT REQUIREMENT)

You MUST respond with a valid JSON object wrapped in <json> tags:

<json>
{{
    "reasoning": "Your detailed analysis here. Be specific about what the student did right and wrong, then justify your grade.",
    "response": "X"
}}
</json>

Where X is exactly one digit: 0, 1, 2, 3, 4, 5, 6, or 7

### EXAMPLE OF CORRECT RESPONSE:
<json>
{{
    "reasoning": "The student correctly identified the approach but made a computational error in step 3. The final answer is incorrect, but the method is sound.",
    "response": "4"
}}
</json>

### IMPORTANT:
- Double-check your JSON is valid before responding
- Ensure the grade matches your reasoning
- Use ONLY the digits 0-7, no other characters in the response field"""

    def _verify_grade_consistency(self, grade: str, reasoning: str) -> tuple[str, bool]:
        """Verify that the grade is consistent with the reasoning.
        
        Uses a comprehensive approach to detect contradictions between
        the assigned grade and the reasoning provided.
        
        Args:
            grade: The extracted grade
            reasoning: The reasoning text
            
        Returns:
            (verified_grade, is_consistent)
        """
        if not reasoning or grade == "None":
            return grade, True
            
        reasoning_lower = reasoning.lower()
        
        # Define positive and negative indicators
        positive_indicators = [
            "correct", "valid", "sound", "complete", "proper", "rigorous",
            "well-justified", "logically sound", "matches the official",
            "all steps are correct", "correctly proves", "valid proof",
            "correct solution", "perfect solution", "fully correct",
            "completely correct", "excellent", "well done"
        ]
        
        negative_indicators = [
            "incorrect", "wrong", "error", "mistake", "flawed", "invalid",
            "does not solve", "fails to", "missing", "incomplete",
            "gap", "cannot", "unable", "does not prove", "no progress",
            "blank", "empty", "irrelevant", "no meaningful"
        ]
        
        # Count positive and negative indicators
        positive_count = sum(1 for ind in positive_indicators if ind in reasoning_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in ind in reasoning_lower)
        
        # High grade (6-7) with many negative indicators is suspicious
        if grade in ["6", "7"] and negative_count > positive_count + 1:
            self.log_fn(f"Grade {grade} but reasoning has more negative ({negative_count}) than positive ({positive_count}) indicators")
            return grade, False
            
        # Low grade (0-1) with many positive indicators is suspicious
        if grade in ["0", "1"] and positive_count > negative_count + 1:
            self.log_fn(f"Grade {grade} but reasoning has more positive ({positive_count}) than negative ({negative_count}) indicators")
            return grade, False
        
        # Severe contradictions
        if grade == "7":
            severe_negative = [
                "completely wrong", "totally incorrect", "fundamentally flawed",
                "does not solve", "fails to prove", "incorrect solution",
                "no credit", "zero points", "blank answer"
            ]
            for indicator in severe_negative:
                if indicator in reasoning_lower:
                    self.log_fn(f"Grade 7 but reasoning contains '{indicator}' - severe inconsistency")
                    return grade, False
                    
        if grade == "0":
            severe_positive = [
                "completely correct", "fully correct", "perfect solution",
                "correct solution", "valid proof", "correctly proves",
                "all steps correct", "complete solution", "full credit"
            ]
            for indicator in severe_positive:
                if indicator in reasoning_lower:
                    self.log_fn(f"Grade 0 but reasoning contains '{indicator}' - severe inconsistency")
                    return grade, False
                    
        return grade, True

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning, validation, and retry.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)
        grading_guidelines = inputs.get("grading_guidelines", "")
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=[],
                )
            except Exception as e:
                self.log_fn(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return "0", []  # Return 0 instead of "None" on complete failure
                continue

            # Extract prediction with enhanced extraction
            prediction, reasoning = self._extract_prediction(msg_history)
            
            # Validate the grade
            validated_grade, is_valid = _validate_grade(prediction, grading_guidelines)
            
            # Log the reasoning and validation result
            if reasoning:
                self.log_fn(f"Reasoning: {reasoning[:200]}...")
            self.log_fn(f"Attempt {attempt + 1}: Extracted grade: {prediction}, Validated: {validated_grade}, Is valid: {is_valid}")
            
            # If grade is invalid, try to extract from the full response text
            if not is_valid and response:
                # Try to find any numeric grade in the response with word boundaries
                numeric_match = re.search(r'\b([0-7])\b', response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    self.log_fn(f"Fallback extraction found grade: {validated_grade}")
                else:
                    # Try to find grade patterns in the full response
                    grade_patterns = [
                        (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b', 'numeric'),
                        (r'(?:grade|score|mark)\s+is\s+([0-7])\b', 'numeric'),
                        (r'(?:grade|score|mark)\s+of\s+([0-7])\b', 'numeric'),
                        (r'\bfull\s*(?:credit|points?|score)?\b', 'full'),
                        (r'\bcomplete\s*(?:solution|answer|credit)?\b', 'full'),
                        (r'\ball\s*(?:points?|credit|marks?)?\b', 'full'),
                        (r'\bperfect\s*(?:score|solution)?\b', 'full'),
                        (r'\bcorrect\s*(?:solution|answer)?\b', 'full'),
                        (r'\bvalid\s*(?:solution|proof|answer)?\b', 'full'),
                        (r'\bsound\s*(?:solution|proof|reasoning)?\b', 'full'),
                        (r'\bexcellent\s*(?:solution|work|answer)?\b', 'full'),
                        (r'\bwell\s*done\b', 'full'),
                        (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                        (r'\bwrong\s*(?:solution|answer)?\b', 'zero'),
                        (r'\bnone\b', 'zero'),
                        (r'\bempty\s*(?:answer|solution)?\b', 'zero'),
                        (r'\bblank\b', 'zero'),
                        (r'\binvalid\s*(?:solution|proof|approach)?\b', 'zero'),
                        (r'\bno\s*meaningful\s*(?:progress|work)?\b', 'zero'),
                    ]
                    for pattern, pattern_type in grade_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            if pattern_type == 'full':
                                validated_grade = "7"
                            elif pattern_type == 'zero':
                                validated_grade = "0"
                            else:
                                validated_grade = match.group(1)
                            is_valid = True
                            self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                            break
            
            # Verify grade-reasoning consistency (only for severe contradictions)
            if is_valid and reasoning:
                validated_grade, is_consistent = self._verify_grade_consistency(validated_grade, reasoning)
                if not is_consistent and attempt < max_retries:
                    self.log_fn(f"Grade-reasoning inconsistency detected, retrying...")
                    error_reason = f"The grade '{validated_grade}' contradicts your reasoning. Your reasoning states the solution is {'correct' if validated_grade == '0' else 'incorrect'}, but you assigned {'0' if validated_grade == '0' else '7'} points. Please provide a consistent grade."
                    instruction = self._build_retry_prompt(inputs, response, error_reason)
                    continue
            
            # If we have a valid grade, return it
            if is_valid:
                return str(validated_grade), msg_history
            
            # If invalid and we have retries left, build retry prompt
            if attempt < max_retries:
                error_reason = f"Could not extract a valid grade from the response. The response field contained: '{prediction}'. Please provide a valid single digit (0-7) in the response field."
                instruction = self._build_retry_prompt(inputs, response, error_reason)
                self.log_fn(f"Invalid grade, retrying with corrected prompt...")
        
        # All retries exhausted, return best effort (default to 0 for safety)
        self.log_fn(f"All retries exhausted, returning best effort: {validated_grade}")
        return str(validated_grade) if validated_grade not in ["None", ""] else "0", msg_history
