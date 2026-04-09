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
    # Try both ```json and ``` patterns
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                # Try to find valid JSON within the content using balanced braces
                try:
                    json_start = match.find('{')
                    if json_start != -1:
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
                            return json.loads(match[json_start:json_end+1].strip())
                except json.JSONDecodeError:
                    continue
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool, float]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with strict support for IMO 0-7 point scale.
    Only accepts single digit grades 0-7 for consistency.
    
    Returns:
        (validated_grade, is_valid, confidence_score)
    """
    if not prediction or prediction == "None":
        return "None", False, 0.0
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Strict validation: only accept single digit 0-7 (highest confidence)
    if pred_clean in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        return pred_clean, True, 1.0
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True, 0.95
    
    # Check for fractional grades like "3 out of 7" or "5 out of 7"
    out_of_match = re.search(r'\b([0-7])\s+out\s+of\s+7\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True, 0.95
    
    # Check for numeric grades embedded in text (0-7 for IMO problems)
    # Look for standalone digits or digits at word boundaries
    numeric_match = re.search(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True, 0.85
    
    # Check for full credit patterns -> 7
    full_patterns = [
        r'\bfull\s*(?:credit|points?|score|marks?)?\b',
        r'\bcomplete\s*(?:solution|answer|credit)?\b',
        r'\ball\s*(?:points?|credit|marks?)?\b',
        r'\bperfect\s*(?:score|solution)?\b',
        r'\bcorrect\s*(?:solution|answer)?\b',
        r'\bseven\s*(?:points?)?\b',
        r'\bmaximum\s*(?:score|points?|credit)?\b',
    ]
    for pattern in full_patterns:
        if re.search(pattern, pred_lower):
            return "7", True, 0.75
    
    # Check for zero/no credit patterns -> 0
    zero_patterns = [
        r'\bno\s*(?:credit|points?|score|marks?)?\b',
        r'\bzero\s*(?:credit|points?|score|marks?)?\b',
        r'\b0\s*(?:points?|credit|score|marks?)?\b',
        r'\bincorrect\s*(?:solution|answer)?\b',
        r'\bwrong\s*(?:solution|answer)?\b',
        r'\bnone\b',
        r'\bempty\s*(?:answer|solution|response)?\b',
        r'\bblank\b',
    ]
    for pattern in zero_patterns:
        if re.search(pattern, pred_lower):
            return "0", True, 0.75
    
    # Check for spelled-out numbers
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7"
    }
    for word, digit in number_words.items():
        if re.search(rf'\b{word}\b', pred_lower):
            return digit, True, 0.7
    
    # Check for grade ranges that indicate uncertainty - extract lower bound (conservative)
    range_patterns = [
        r'\b([0-7])\s*[-~]\s*([0-7])\b',
        r'\bbetween\s+([0-7])\s+and\s+([0-7])\b',
        r'\b([0-7])\s+or\s+([0-7])\b',
    ]
    for pattern in range_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            # Return the lower bound for conservative grading
            return match.group(1), True, 0.5
    
    # Check for partial credit indicators with numeric values
    partial_patterns = [
        (r'\bpartial\s+(?:credit|points?|score)?\s*(?:of\s*)?([0-7])\b', 0.6),
        (r'\b([0-7])\s+points?\s+(?:partial|partial\s+credit)\b', 0.6),
    ]
    for pattern, conf in partial_patterns:
        match = re.search(pattern, pred_lower)
        if match:
            return match.group(1), True, conf
    
    # If no clear grade found, mark as invalid
    return pred_clean, False, 0.0


def _aggregate_grades(grades: list[str], confidences: list[float]) -> tuple[str, float]:
    """Aggregate multiple grade predictions using weighted voting.
    
    Args:
        grades: List of grade strings (0-7)
        confidences: List of confidence scores for each grade
        
    Returns:
        (aggregated_grade, aggregate_confidence)
    """
    if not grades:
        return "None", 0.0
    
    # Count weighted votes for each grade
    vote_counts = {}
    for grade, conf in zip(grades, confidences):
        if grade in vote_counts:
            vote_counts[grade] += conf
        else:
            vote_counts[grade] = conf
    
    # Find the grade with highest weighted votes
    best_grade = max(vote_counts.keys(), key=lambda g: (vote_counts[g], int(g)))
    best_score = vote_counts[best_grade]
    
    # Calculate aggregate confidence
    total_votes = sum(vote_counts.values())
    if total_votes > 0:
        # Confidence is the proportion of weighted votes for the winning grade
        # plus a bonus for consensus
        agreement_ratio = best_score / total_votes
        # Count how many samples agree with the best grade
        agreement_count = sum(1 for g in grades if g == best_grade)
        consensus_bonus = min(0.1 * (agreement_count - 1), 0.2)  # Up to 0.2 bonus for consensus
        aggregate_confidence = min(agreement_ratio + consensus_bonus, 1.0)
    else:
        aggregate_confidence = 0.0
    
    return best_grade, aggregate_confidence


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.num_samples = 3  # Number of samples for uncertain cases
        self.confidence_threshold = 0.8  # Threshold for multi-sample grading

    def _build_prompt(self, inputs: dict, include_examples: bool = True) -> str:
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
            include_examples: Whether to include detailed grading examples
                
        Returns:
            A formatted prompt string ready for LLM consumption
        """
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        examples_section = ""
        if include_examples:
            examples_section = """
## Detailed Grading Examples with Rubric Alignment

Example 1 - Grade 7 (Full Credit):
Student wrote a complete proof with all necessary steps, clear logical flow, and proper justification for each claim. Even though the notation differed from the official solution, the mathematical reasoning was sound and complete. All key insights from the official solution were present.
→ Grade: 7
→ Reasoning: Complete solution with valid proof structure, all necessary lemmas proved, and correct conclusion.

Example 2 - Grade 6 (Minor Flaw):
Student had the correct approach and nearly complete solution, but made a small arithmetic error in the final calculation that didn't affect the overall method. The core insight and most of the proof were correct. One minor gap in reasoning that could be easily fixed.
→ Grade: 6
→ Reasoning: Nearly complete with correct approach, minor error in final step doesn't invalidate the main proof structure.

Example 3 - Grade 5 (Significant Progress):
Student proved the main lemma correctly and outlined the approach to the main problem, but missed connecting the lemma to the final conclusion. Had most key ideas but one significant gap in the logical chain.
→ Grade: 5
→ Reasoning: Significant progress with substantial solution elements, most key ideas present but notable gap in final connection.

Example 4 - Grade 4 (Good Partial Progress):
Student correctly identified the key lemma and proved it, but didn't connect it to the main problem. Had about half of the necessary components for a complete solution. Multiple correct steps but incomplete overall.
→ Grade: 4
→ Reasoning: Good partial progress with several correct key steps, has main idea but couldn't fully execute the complete proof.

Example 5 - Grade 3 (Some Meaningful Progress):
Student demonstrated at least one key insight or correct step toward the solution. Showed understanding of the problem structure but limited execution. One solid correct idea present.
→ Grade: 3
→ Reasoning: Some meaningful progress with at least one key idea or correct step demonstrated, shows understanding but limited execution.

Example 6 - Grade 2 (Minimal Progress):
Student understood what the problem was asking and made some relevant observations, but didn't make significant progress toward a solution. Had some correct ideas but couldn't develop them into concrete steps.
→ Grade: 2
→ Reasoning: Minimal progress with some relevant ideas but little concrete progress toward solution.

Example 7 - Grade 1 (Very Minimal Progress):
Student restated the problem or made trivial observations that show some engagement but no real mathematical progress. Barely relevant to the actual solution.
→ Grade: 1
→ Reasoning: Very minimal progress, barely relevant to problem, may have restated problem or made trivial observations.

Example 8 - Grade 0 (No Credit):
Student's answer was completely unrelated to the problem, or blank, or contained only incorrect statements with no valid mathematical reasoning. No meaningful engagement with the problem.
→ Grade: 0
→ Reasoning: No meaningful progress, completely incorrect, blank, or nonsense. No valid mathematical reasoning present.
"""

        return f"""You are an expert IMO (International Mathematical Olympiad) grader. Evaluate the student's solution and assign a grade from 0-7.

## Problem
{problem}

## Official Solution
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer
{student_answer}
{examples_section}

## IMO Grading Scale (0-7) - BE PRECISE AND CONSISTENT

### Grade 7: Complete Solution (Full Credit)
- Complete, correct solution with proper justification for ALL steps
- All key insights from the official solution are present (may use different approach)
- Valid proof structure with no gaps in reasoning
- Mathematical reasoning is sound throughout

### Grade 6: Minor Flaw
- Nearly complete solution with correct approach
- One minor gap in reasoning OR small calculation error
- Core insight and most of the proof are correct
- Error does not invalidate the main proof structure

### Grade 5: Significant Progress
- Substantial solution elements present
- Most key ideas demonstrated but with notable gaps
- Main lemma or key insight proved but not fully connected to conclusion
- Clear understanding of problem structure shown

### Grade 4: Good Partial Progress
- Several correct key steps demonstrated
- Has the main idea but couldn't fully execute complete proof
- About half of necessary components for complete solution
- Multiple correct mathematical statements present

### Grade 3: Some Meaningful Progress
- At least one key insight or correct step toward solution
- Shows understanding of problem but limited execution
- One solid correct idea that advances toward solution
- Not just trivial observations

### Grade 2: Minimal Progress
- Some relevant ideas but little concrete progress
- Shows engagement with problem but limited development
- Some correct observations but not key insights
- Understanding of what problem is asking

### Grade 1: Very Minimal Progress
- Barely relevant to the problem
- May have restated the problem or made trivial observations
- Little to no mathematical progress
- Minimal engagement with actual problem content

### Grade 0: No Credit
- No meaningful progress
- Completely incorrect or unrelated to problem
- Blank or nonsense answer
- No valid mathematical reasoning present

## Critical Grading Principles
1. Award partial credit generously for correct mathematical ideas, even if incomplete
2. A correct key insight or lemma deserves at least 2-3 points (Grade 3+)
3. Multiple correct steps with good reasoning deserve 4-5 points
4. Only deduct for actual mathematical errors, not for missing "elegant" steps
5. Check if the student's approach, while different from official solution, is mathematically valid
6. Consider the difficulty of the problem - harder problems may warrant more generous partial credit
7. Be consistent: similar levels of progress should receive similar grades
8. When uncertain between two grades, use the HIGHER grade (be generous with partial credit)

## Your Task
1. Read the problem carefully and understand what is being asked
2. Study the official solution to identify key steps and insights
3. Analyze the student's answer line by line
4. Identify ALL correct mathematical statements, valid proof techniques, and key insights
5. Note any errors, gaps, or invalid reasoning (distinguish between minor gaps and major errors)
6. Compare the student's progress to the official solution's structure
7. Consider if alternative approaches are mathematically valid
8. Assign the final grade (0-7) based on the rubric above

## Output Format (CRITICAL - FOLLOW EXACTLY)
Respond with ONLY this JSON format:

<json>
{{
    "reasoning": "Step-by-step analysis: (1) What the student did correctly with specific citations, (2) What errors or gaps exist and their severity, (3) How this compares to official solution's key steps, (4) Clear justification for the specific grade assigned referencing the rubric",
    "response": "X"
}}
</json>

CRITICAL RULES - READ CAREFULLY:
1. "response" field MUST contain ONLY a single digit: 0, 1, 2, 3, 4, 5, 6, or 7
2. NO quotes around the digit in the response field
3. NO text like "Grade: 5" or "5/7" or "five" - ONLY the digit itself
4. NO explanation in the response field
5. "reasoning" field: Your complete analysis (be thorough, specific, and reference the rubric)
6. Ensure valid JSON with proper formatting
7. The response field must be parseable as a single integer 0-7
8. Example CORRECT: "response": "5"
9. Example INCORRECT: "response": "Grade: 5", "response": "5/7", "response": "five"
10. Double-check your JSON is valid before responding
"""

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history with enhanced robustness.
        
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
                for json_obj in extracted:
                    if "response" in json_obj:
                        pred = str(json_obj["response"]).strip()
                        # Validate it's a single digit 0-7
                        if pred in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                            prediction = pred
                            if "reasoning" in json_obj:
                                reasoning = str(json_obj["reasoning"])
                            return prediction, reasoning
                        # If not a clean digit, store it but continue looking
                        elif pred and pred != "None":
                            prediction = pred
                            if "reasoning" in json_obj:
                                reasoning = str(json_obj["reasoning"])
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    pred = str(md_json["response"]).strip()
                    if pred in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                        prediction = pred
                        if "reasoning" in md_json:
                            reasoning = str(md_json["reasoning"])
                        return prediction, reasoning
                    elif pred and pred != "None":
                        prediction = pred
                        if "reasoning" in md_json:
                            reasoning = str(md_json["reasoning"])
            
            # Strategy 3: Try to find any JSON-like structure with response field
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg)
            for json_match in json_matches:
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "")).strip()
                    if pred in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                    elif pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Try to find JSON with balanced braces (more complex structures)
            brace_start = last_msg.find('{')
            if brace_start != -1:
                brace_count = 0
                for i, char in enumerate(last_msg[brace_start:], start=brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                json_str = last_msg[brace_start:i+1]
                                parsed = json.loads(json_str)
                                if "response" in parsed:
                                    pred = str(parsed["response"]).strip()
                                    if pred in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                                        prediction = pred
                                        if "reasoning" in parsed:
                                            reasoning = str(parsed["reasoning"])
                                        return prediction, reasoning
                                    elif pred and pred != "None":
                                        prediction = pred
                                        if "reasoning" in parsed:
                                            reasoning = str(parsed["reasoning"])
                            except json.JSONDecodeError:
                                pass
                            break
            
            # Strategy 5: Look for explicit grade patterns in text
            if prediction == "None" or prediction not in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_patterns = [
                    r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b',
                    r'\bgrade\s+is\s+([0-7])\b',
                    r'\bscore\s+is\s+([0-7])\b',
                    r'\bassigned\s+(?:grade|score)\s+([0-7])\b',
                    r'\b(?:the\s+)?grade\s+(?:of\s+)?([0-7])\b',
                ]
                for pattern in grade_patterns:
                    grade_match = re.search(pattern, last_msg, re.IGNORECASE)
                    if grade_match:
                        prediction = grade_match.group(1)
                        return prediction, reasoning
            
            # Strategy 6: Look for standalone digits 0-7 in the response
            if prediction == "None" or prediction not in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                # Find standalone digits that are likely grades
                standalone_match = re.search(r'(?:^|\s|[^0-9])([0-7])(?:\s|[^0-9]|$)', last_msg)
                if standalone_match:
                    prediction = standalone_match.group(1)
                    return prediction, reasoning
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning

    def _grade_single(self, inputs: dict, include_examples: bool = True) -> tuple[str, str, list[dict], float]:
        """Get a single grade prediction with enhanced extraction and validation.
        
        Returns:
            (grade, reasoning, msg_history, confidence)
        """
        instruction = self._build_prompt(inputs, include_examples=include_examples)
        grading_guidelines = inputs.get("grading_guidelines", "")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            return "None", "", [], 0.0

        # Extract prediction with enhanced extraction
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade with confidence score
        validated_grade, is_valid, confidence = _validate_grade(prediction, grading_guidelines)
        
        # If grade is invalid or low confidence, try to extract from the full response text
        if (not is_valid or confidence < 0.8) and response:
            # Try to find any numeric grade in the response
            numeric_match = re.search(r'\b([0-7])\b', response)
            if numeric_match:
                validated_grade = numeric_match.group(1)
                is_valid = True
                confidence = max(confidence, 0.85)
                self.log_fn(f"Fallback extraction found grade: {validated_grade}")
            else:
                # Try to find grade patterns in the full response
                grade_patterns = [
                    (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])\b', 'numeric'),
                    (r'\bfull\s*(?:credit|points?|score|marks?)?\b', 'full'),
                    (r'\bcorrect\s*(?:solution|answer)?\b', 'full'),
                    (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                    (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
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
                        confidence = max(confidence, 0.75)
                        self.log_fn(f"Pattern-based fallback found grade: {validated_grade}")
                        break
        
        # If still invalid or low confidence, retry with a clearer prompt
        if not is_valid or confidence < 0.5:
            self.log_fn(f"Grade validation failed or low confidence ({confidence}), retrying with clearer prompt...")
            retry_instruction = instruction + """

⚠️ CRITICAL REMINDER ⚠️
Your previous response did not follow the required format. 
The "response" field MUST contain ONLY a single digit from 0-7.
Examples of CORRECT responses: "response": "5"
Examples of INCORRECT responses: "response": "Grade: 5", "response": "5/7", "response": "five points"

Please respond with ONLY the digit in the response field."""
            try:
                retry_response, retry_msg_history, retry_info = get_response_from_llm(
                    msg=retry_instruction,
                    model=self.model,
                    msg_history=[],
                )
                retry_prediction, retry_reasoning = self._extract_prediction(retry_msg_history)
                validated_grade, is_valid, retry_confidence = _validate_grade(retry_prediction, grading_guidelines)
                self.log_fn(f"Retry result: grade={validated_grade}, valid={is_valid}, confidence={retry_confidence}")
                if is_valid:
                    msg_history = retry_msg_history
                    reasoning = retry_reasoning
                    confidence = max(confidence, retry_confidence)
            except Exception as e:
                self.log_fn(f"Retry LLM call failed: {e}")

        return str(validated_grade), reasoning, msg_history, confidence

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with multi-sample grading for uncertain cases.
        
        Uses consensus-based grading with confidence weighting to improve accuracy.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # First attempt with examples
        grade, reasoning, msg_history, confidence = self._grade_single(inputs, include_examples=True)
        
        # Log the initial result
        if reasoning:
            self.log_fn(f"Initial reasoning: {reasoning[:200]}...")
        self.log_fn(f"Initial grade: {grade}, Confidence: {confidence}")
        
        # If confidence is high enough, return the result
        if confidence >= self.confidence_threshold:
            return str(grade), msg_history
        
        # For uncertain cases, use multi-sample grading
        self.log_fn(f"Low confidence ({confidence}), using multi-sample grading with {self.num_samples} samples...")
        
        grades = [grade]
        confidences = [confidence]
        all_reasonings = [reasoning]
        
        # Collect additional samples with temperature variation for diversity
        for i in range(self.num_samples - 1):
            try:
                # Use a simpler prompt without examples for subsequent samples
                # Vary the approach to get diverse perspectives
                sample_grade, sample_reasoning, sample_history, sample_conf = self._grade_single(
                    inputs, include_examples=(i == 0)  # Only include examples on first retry
                )
                self.log_fn(f"Sample {i+2}: grade={sample_grade}, confidence={sample_conf}")
                
                # Validate the sample - only accept valid grades
                if sample_grade in ["0", "1", "2", "3", "4", "5", "6", "7"] and sample_conf > 0:
                    grades.append(sample_grade)
                    confidences.append(sample_conf)
                    all_reasonings.append(sample_reasoning)
                elif sample_grade != "None":
                    # Try to validate the grade
                    validated, is_valid, val_conf = _validate_grade(sample_grade, "")
                    if is_valid and validated in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                        grades.append(validated)
                        confidences.append(max(sample_conf, val_conf))
                        all_reasonings.append(sample_reasoning)
            except Exception as e:
                self.log_fn(f"Sample {i+2} failed: {e}")
                continue
        
        # Aggregate the grades if we have multiple valid samples
        if len(grades) > 1:
            final_grade, aggregate_confidence = _aggregate_grades(grades, confidences)
            self.log_fn(f"Multi-sample results: grades={grades}, aggregated={final_grade}, confidence={aggregate_confidence}")
            
            # Use the reasoning from the sample with highest confidence that matches the final grade
            best_reasoning = ""
            best_conf_for_grade = 0.0
            for g, r, c in zip(grades, all_reasonings, confidences):
                if g == final_grade and c > best_conf_for_grade:
                    best_reasoning = r
                    best_conf_for_grade = c
            
            if best_reasoning:
                self.log_fn(f"Selected reasoning: {best_reasoning[:200]}...")
            
            # If aggregate confidence is still low, consider the median grade
            if aggregate_confidence < 0.5 and len(grades) >= 3:
                # Sort grades and take median for robustness
                sorted_grades = sorted(grades, key=lambda x: (int(x), confidences[grades.index(x)]))
                median_grade = sorted_grades[len(sorted_grades) // 2]
                self.log_fn(f"Low aggregate confidence, using median grade: {median_grade}")
                
                # Find reasoning for median grade
                for g, r, c in zip(grades, all_reasonings, confidences):
                    if g == median_grade and c > best_conf_for_grade:
                        best_reasoning = r
                        best_conf_for_grade = c
                
                return str(median_grade), msg_history
            
            return str(final_grade), msg_history
        
        # If multi-sample failed, return the initial result
        return str(grade), msg_history
