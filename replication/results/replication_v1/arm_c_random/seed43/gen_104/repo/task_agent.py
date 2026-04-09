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
    
    # Check for quoted digits like "5" or '5'
    quoted_match = re.search(r'["\']([0-7])["\']', pred_clean)
    if quoted_match:
        return quoted_match.group(1), True, 0.98
    
    # Check for fractional grades like "3/7" or "5 / 7" - extract just the numerator
    fractional_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_clean)
    if fractional_match:
        return fractional_match.group(1), True, 0.95
    
    # Check for fractional grades like "3 out of 7" or "5 out of 7"
    out_of_match = re.search(r'\b([0-7])\s+out\s+of\s+7\b', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True, 0.95
    
    # Check for "X points" or "X marks" patterns
    points_match = re.search(r'\b([0-7])\s*(?:points?|marks?|score)\b', pred_lower)
    if points_match:
        return points_match.group(1), True, 0.9
    
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
    
    # If no clear grade found, mark as invalid
    return pred_clean, False, 0.0


def _aggregate_grades(grades: list[str], confidences: list[float]) -> tuple[str, float]:
    """Aggregate multiple grade predictions using weighted voting with outlier detection.
    
    Args:
        grades: List of grade strings (0-7)
        confidences: List of confidence scores for each grade
        
    Returns:
        (aggregated_grade, aggregate_confidence)
    """
    if not grades:
        return "None", 0.0
    
    # Filter out "None" grades
    valid_pairs = [(g, c) for g, c in zip(grades, confidences) if g != "None"]
    if not valid_pairs:
        return "None", 0.0
    
    valid_grades, valid_confs = zip(*valid_pairs)
    
    # Convert to integers for statistical analysis
    grade_ints = [int(g) for g in valid_grades]
    
    # Outlier detection: if we have 3+ samples, check for outliers
    if len(grade_ints) >= 3:
        mean_grade = sum(grade_ints) / len(grade_ints)
        std_grade = (sum((g - mean_grade) ** 2 for g in grade_ints) / len(grade_ints)) ** 0.5
        
        # Identify outliers (more than 1.5 std devs from mean)
        outlier_threshold = 1.5 * std_grade if std_grade > 0 else 0
        filtered_pairs = []
        for g, c in zip(valid_grades, valid_confs):
            g_int = int(g)
            if abs(g_int - mean_grade) <= outlier_threshold:
                filtered_pairs.append((g, c))
            else:
                # Reduce confidence for outliers instead of removing entirely
                filtered_pairs.append((g, c * 0.5))
        
        if filtered_pairs:
            valid_grades, valid_confs = zip(*filtered_pairs)
    
    # Count weighted votes for each grade
    vote_counts = {}
    for grade, conf in zip(valid_grades, valid_confs):
        if grade in vote_counts:
            vote_counts[grade] += conf
        else:
            vote_counts[grade] = conf
    
    # Find the grade with highest weighted votes
    # Tie-breaker: prefer grades closer to the median (more conservative)
    median_grade = sorted([int(g) for g in valid_grades])[len(valid_grades) // 2]
    
    def grade_sort_key(g):
        # Primary: vote count, Secondary: distance from median (prefer closer), Tertiary: grade value
        g_int = int(g)
        return (vote_counts[g], -abs(g_int - median_grade), g_int)
    
    best_grade = max(vote_counts.keys(), key=grade_sort_key)
    best_score = vote_counts[best_grade]
    
    # Calculate aggregate confidence
    total_votes = sum(vote_counts.values())
    if total_votes > 0:
        # Confidence is the proportion of weighted votes for the winning grade
        agreement_ratio = best_score / total_votes
        # Count how many samples agree with the best grade
        agreement_count = sum(1 for g in valid_grades if g == best_grade)
        # Consensus bonus: up to 0.2 for full agreement
        consensus_bonus = min(0.1 * (agreement_count - 1), 0.2)
        # Agreement quality: bonus for high-confidence agreement
        high_conf_agreement = sum(1 for g, c in zip(valid_grades, valid_confs) if g == best_grade and c >= 0.8)
        quality_bonus = min(0.05 * high_conf_agreement, 0.1)
        aggregate_confidence = min(agreement_ratio + consensus_bonus + quality_bonus, 1.0)
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
## Detailed Grading Examples

Example 1 - Grade 7 (Full Credit):
Student wrote a complete proof with all necessary steps, clear logical flow, and proper justification for each claim. Even though the notation differed from the official solution, the mathematical reasoning was sound and complete.
→ Grade: 7

Example 2 - Grade 6 (Minor Flaw):
Student had the correct approach and nearly complete solution, but made a small arithmetic error in the final calculation that didn't affect the overall method. The core insight and most of the proof were correct.
→ Grade: 6

Example 3 - Grade 5 (Substantial Progress):
Student had most key ideas present but with notable gaps. The main approach was correct and well-developed, but missing some connections or having significant gaps in justification.
→ Grade: 5

Example 4 - Grade 4 (Good Partial Progress):
Student correctly identified the key lemma and proved it, but didn't connect it to the main problem. Had about half of the necessary components for a complete solution.
→ Grade: 4

Example 5 - Grade 3 (Some Progress):
Student demonstrated at least one key idea or correct step. Shows understanding of the problem but limited execution.
→ Grade: 3

Example 6 - Grade 2 (Minimal Progress):
Student understood what the problem was asking and made some relevant observations, but didn't make significant progress toward a solution. Had some correct ideas but couldn't develop them.
→ Grade: 2

Example 7 - Grade 1 (Very Minimal Progress):
Student barely engaged with the problem. May have restated the problem or made trivial observations without meaningful mathematical progress.
→ Grade: 1

Example 8 - Grade 0 (No Credit):
Student's answer was completely unrelated to the problem, or blank, or contained only incorrect statements with no valid mathematical reasoning.
→ Grade: 0
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

### Grade Definitions with Specific Criteria:
- **7 (Complete)**: Full solution with all key steps, correct reasoning, and proper justification. Alternative valid approaches are acceptable. No significant gaps or errors.
- **6 (Near-Complete)**: Correct approach with minor flaw only: small calculation error, slight gap in reasoning, or incomplete justification for ONE non-critical step. Core insight must be correct.
- **5 (Substantial)**: Most key ideas present (60-80% of solution). Main approach correct but with notable gaps in execution or missing connections between steps. Core insight demonstrated.
- **4 (Good Partial)**: About half the solution correct. Key lemma proved OR main structure outlined with some correct steps, but significant portions missing or incorrect.
- **3 (Some Progress)**: At least one key idea or correct non-trivial step demonstrated. Shows genuine engagement with problem structure, not just restating.
- **2 (Minimal)**: Relevant observations about the problem, some correct ideas, but little concrete progress. Shows understanding of what problem asks.
- **1 (Very Minimal)**: Barely relevant engagement. May restate problem or make trivial observations without mathematical development.
- **0 (No Credit)**: Blank, nonsense, completely incorrect, or unrelated to problem.

### IMO-Specific Grading Principles (FOLLOW STRICTLY):
1. **Partial Credit is Standard**: IMO awards points for correct progress, not just complete solutions
2. **Key Insight Rule**: A correct key lemma, insight, or non-trivial step = minimum 2-3 points
3. **Progressive Scoring**: Each correct substantial step adds to the score proportionally
4. **Alternative Approaches**: Different but mathematically valid methods deserve full credit
5. **Error vs Gap**: Distinguish between mathematical errors (deduct) and gaps in writing (minor deduction)
6. **Difficulty Adjustment**: Harder problems warrant more generous partial credit for equivalent progress
7. **Consistency**: Similar levels of mathematical progress must receive similar grades
8. **Generosity Bias**: When uncertain between two grades, prefer the higher (IMO convention)

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

RULES:
- "response" field: ONLY a single digit 0-7 (no quotes, no text, no explanation, no fractions)
- "reasoning" field: Your complete analysis (be thorough, specific, and reference the rubric)
- Ensure valid JSON with proper formatting
- The response field must be parseable as a single integer 0-7
- Do NOT write "Grade: 5" or "5/7" in the response field - only "5"
"""

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
                # If we got a valid prediction, return it
                if prediction != "None" and prediction:
                    return prediction, reasoning
            
            # Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                if "response" in md_json:
                    prediction = str(md_json["response"]).strip()
                if "reasoning" in md_json:
                    reasoning = str(md_json["reasoning"])
                if prediction != "None" and prediction:
                    return prediction, reasoning
            
            # Fallback: try to find any JSON-like structure with response field
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[^{}]*"response"[^{}]*\}'
            json_matches = re.findall(json_pattern, last_msg)
            for json_match in json_matches:
                try:
                    fallback = json.loads(json_match)
                    pred = str(fallback.get("response", "")).strip()
                    if pred and pred != "None":
                        prediction = pred
                        if "reasoning" in fallback:
                            reasoning = str(fallback["reasoning"])
                        return prediction, reasoning
                except json.JSONDecodeError:
                    continue
            
            # Try to find JSON with balanced braces (more complex structures)
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
                                    if pred and pred != "None":
                                        prediction = pred
                                        if "reasoning" in parsed:
                                            reasoning = str(parsed["reasoning"])
                                        return prediction, reasoning
                            except json.JSONDecodeError:
                                pass
                            break
            
            # Last resort: look for grade patterns in text
            if prediction == "None":
                # Look for "Grade: X" or "Final grade: X" patterns
                grade_match = re.search(r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', last_msg, re.IGNORECASE)
                if grade_match:
                    prediction = grade_match.group(1)
                    
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
        if not is_valid or confidence < 0.8:
            # Get the full response text from the last message
            full_response = ""
            if msg_history:
                last_entry = msg_history[-1]
                if isinstance(last_entry, dict):
                    full_response = last_entry.get("text") or last_entry.get("content", "")
            
            if full_response:
                # Try to find any numeric grade in the response
                numeric_match = re.search(r'\b([0-7])\b', full_response)
                if numeric_match:
                    validated_grade = numeric_match.group(1)
                    is_valid = True
                    confidence = max(confidence, 0.85)
                    self.log_fn(f"Fallback extraction found grade: {validated_grade}")
                else:
                    # Try to find grade patterns in the full response
                    grade_patterns = [
                        (r'(?:grade|score|mark|final grade|final score)\s*:?\s*([0-7])', 'numeric'),
                        (r'\bfull\s*(?:credit|points?|score|marks?)?\b', 'full'),
                        (r'\bcorrect\s*(?:solution|answer)?\b', 'full'),
                        (r'\bno\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bzero\s*(?:credit|points?|score|marks?)?\b', 'zero'),
                        (r'\bincorrect\s*(?:solution|answer)?\b', 'zero'),
                    ]
                    for pattern, pattern_type in grade_patterns:
                        match = re.search(pattern, full_response, re.IGNORECASE)
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
            retry_instruction = instruction + "\n\nCRITICAL: Your response field MUST contain ONLY a single digit from 0-7. No other text, no quotes, no explanation. Just the digit."
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

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Validate inputs
        if not inputs or not isinstance(inputs, dict):
            self.log_fn("Invalid inputs provided")
            return "None", []
        
        # Check for required fields
        required_fields = ["problem", "student_answer"]
        missing_fields = [f for f in required_fields if not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Missing required fields: {missing_fields}")
            # Still try to grade, but log the issue
        
        # First attempt with examples
        try:
            grade, reasoning, msg_history, confidence = self._grade_single(inputs, include_examples=True)
        except Exception as e:
            self.log_fn(f"Initial grading failed: {e}")
            return "None", []
        
        # Log the initial result
        if reasoning:
            self.log_fn(f"Initial reasoning: {reasoning[:200]}...")
        self.log_fn(f"Initial grade: {grade}, Confidence: {confidence}")
        
        # If we got a valid grade with high confidence, return it
        if grade != "None" and confidence >= self.confidence_threshold:
            return str(grade), msg_history
        
        # For uncertain cases or invalid grades, use multi-sample grading
        self.log_fn(f"Low confidence ({confidence}) or invalid grade, using multi-sample grading with {self.num_samples} samples...")
        
        grades = []
        confidences = []
        all_reasonings = []
        
        # Only add initial result if it's valid
        if grade != "None":
            grades.append(grade)
            confidences.append(confidence)
            all_reasonings.append(reasoning)
        
        # Collect additional samples
        for i in range(self.num_samples):
            try:
                # Use examples only on first sample for variety
                sample_grade, sample_reasoning, sample_history, sample_conf = self._grade_single(
                    inputs, include_examples=(i == 0)
                )
                self.log_fn(f"Sample {i+1}: grade={sample_grade}, confidence={sample_conf}")
                
                # Validate the sample
                if sample_grade != "None" and sample_conf > 0:
                    grades.append(sample_grade)
                    confidences.append(sample_conf)
                    all_reasonings.append(sample_reasoning)
            except Exception as e:
                self.log_fn(f"Sample {i+1} failed: {e}")
                continue
        
        # Aggregate the grades if we have any valid ones
        if len(grades) > 0:
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
            
            return str(final_grade), msg_history
        
        # If all attempts failed, return the initial result or "None"
        self.log_fn("All grading attempts failed, returning initial result")
        return str(grade) if grade != "None" else "0", msg_history  # Default to 0 if completely failed
