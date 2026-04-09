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
import time
from typing import Callable, Any

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Retry a function with exponential backoff."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
    raise last_exception


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
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
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in <json> block: {e}")
            continue
    return results or None


def _extract_json_flexible(text: str) -> dict | None:
    """Extract JSON using multiple strategies with enhanced robustness."""
    # Strategy 1: <json> tags
    json_blocks = _extract_jsons(text)
    if json_blocks:
        return json_blocks[-1]
    
    # Strategy 2: ```json code blocks
    json_code_pattern = r'```json\s*(.*?)```'
    matches = re.findall(json_code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            fixed = _fix_json_string(match.strip())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: ``` code blocks (any language)
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            fixed = _fix_json_string(match.strip())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Find JSON objects directly (smart brace matching)
    best_json = None
    best_length = 0
    
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidate = text[start_idx:i+1]
                try:
                    parsed = json.loads(candidate)
                    if len(candidate) > best_length:
                        best_json = parsed
                        best_length = len(candidate)
                except json.JSONDecodeError:
                    fixed = _fix_json_string(candidate)
                    try:
                        parsed = json.loads(fixed)
                        if len(fixed) > best_length:
                            best_json = parsed
                            best_length = len(fixed)
                    except json.JSONDecodeError:
                        pass
                start_idx = -1
    
    if best_json is not None:
        return best_json
    
    # Strategy 5: Look for key-value patterns
    kv_pattern = r'["\']?(\w+)["\']?\s*:\s*["\']?([^"\'\n,}]+)["\']?'
    matches = re.findall(kv_pattern, text)
    if matches:
        result = {}
        for key, value in matches:
            value = value.strip().strip('"\'').lower()
            if value in ["correct", "incorrect", "partial", "almost"]:
                result[key] = value
        if result:
            return result
    
    return None


def _fix_json_string(text: str) -> str:
    """Fix common JSON formatting issues in LLM responses."""
    text = text.strip()
    
    # Replace single quotes with double quotes (carefully)
    result = []
    in_string = False
    string_char = None
    i = 0
    while i < len(text):
        char = text[i]
        if not in_string:
            if char in '"\'':
                in_string = True
                string_char = char
                result.append('"')
            elif char == '{':
                result.append(char)
            elif char == '}':
                # Remove trailing comma before closing brace
                while result and result[-1] in ' \t,':
                    result.pop()
                result.append(char)
            else:
                result.append(char)
        else:
            if char == string_char:
                # Check if it's escaped
                backslash_count = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    backslash_count += 1
                    j -= 1
                if backslash_count % 2 == 0:
                    # Not escaped, end of string
                    in_string = False
                    string_char = None
                    result.append('"')
                else:
                    result.append(char)
            elif char == '"':
                # Escape double quotes inside string
                result.append('\\"')
            elif char == '\n':
                # Replace newlines with escaped newlines
                result.append('\\n')
            else:
                result.append(char)
        i += 1
    
    fixed = ''.join(result)
    
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    return fixed


def _normalize_prediction(raw_value) -> str:
    """Normalize a raw prediction value to one of the valid labels."""
    if raw_value is None:
        return "unknown"
    
    raw_str = str(raw_value).lower().strip()
    
    # Direct matches
    if raw_str in ["correct", "incorrect", "partial", "almost"]:
        return raw_str
    
    # Check for exact matches with quotes removed
    clean_str = raw_str.strip('"\'')
    if clean_str in ["correct", "incorrect", "partial", "almost"]:
        return clean_str
    
    # Handle common variations
    if raw_str in ["true", "yes", "right", "valid", "1", "full", "complete", "perfect", "flawless"]:
        return "correct"
    if raw_str in ["false", "no", "wrong", "invalid", "0", "none", "fail", "error", "bad"]:
        return "incorrect"
    if raw_str in ["part", "partially", "incomplete", "half", "some", "mostly wrong", "mostly incorrect"]:
        return "partial"
    if raw_str in ["almost correct", "close", "minor errors", "nearly", "mostly correct", "trivial errors"]:
        return "almost"
    
    # Check for substring matches (more specific first)
    if "almost" in raw_str or "nearly" in raw_str or "minor" in raw_str:
        return "almost"
    if "partial" in raw_str or "incomplete" in raw_str or "half" in raw_str:
        return "partial"
    # Be careful with "correct" - "incorrect" contains "correct"
    if "incorrect" in raw_str or "wrong" in raw_str or "invalid" in raw_str:
        return "incorrect"
    if "correct" in raw_str or "right" in raw_str:
        return "correct"
    
    return "unknown"


def _is_valid_prediction(prediction: str) -> bool:
    """Check if a prediction is one of the valid labels."""
    return prediction in ["correct", "incorrect", "partial", "almost"]


def _parse_grading_guidelines(guidelines: str) -> dict:
    """Parse grading guidelines to extract rubric indicators with enhanced accuracy."""
    result = {
        "has_partial": False,
        "has_almost": False,
        "has_correct": False,
        "has_incorrect": False,
        "partial_context": [],
        "almost_context": [],
        "correct_context": [],
        "incorrect_context": [],
        "score_hints": {},
        "primary_category": None,
        "confidence": 0.0,
    }
    
    if not guidelines:
        return result
    
    guidelines_lower = guidelines.lower()
    
    # Look for explicit markers with more comprehensive patterns
    # Priority order matters - more specific patterns first
    markers = [
        # Exact rubric markers (highest priority)
        (r'\(\s*Partial\s*\)', "partial", 1.0),
        (r'\(\s*Almost\s*\)', "almost", 1.0),
        (r'\(\s*Correct\s*\)', "correct", 1.0),
        (r'\(\s*Incorrect\s*\)', "incorrect", 1.0),
        # Credit-based markers
        (r'\(\s*Full\s*Credit\s*\)', "correct", 0.95),
        (r'\(\s*No\s*Credit\s*\)', "incorrect", 0.95),
        (r'\(\s*Half\s*Credit\s*\)', "partial", 0.95),
        (r'\(\s*Most\s*Credit\s*\)', "almost", 0.95),
        (r'\(\s*Some\s*Credit\s*\)', "partial", 0.90),
        # Score-based patterns
        (r'\(\s*\d+\s*/\s*\d+\s*\)', "score_based", 0.80),
        # Alternative markers
        (r'\[\s*Partial\s*\]', "partial", 0.90),
        (r'\[\s*Almost\s*\]', "almost", 0.90),
        (r'\[\s*Correct\s*\]', "correct", 0.90),
        (r'\[\s*Incorrect\s*\]', "incorrect", 0.90),
    ]
    
    category_scores = {"partial": 0, "almost": 0, "correct": 0, "incorrect": 0}
    
    for pattern, label, confidence in markers:
        matches = list(re.finditer(pattern, guidelines, re.IGNORECASE))
        if matches:
            if label != "score_based":
                result[f"has_{label}"] = True
                category_scores[label] = max(category_scores[label], confidence)
            
            for match in matches:
                # Extract context around the marker (expanded range for more context)
                start = max(0, match.start() - 150)
                end = min(len(guidelines), match.end() + 200)
                context = guidelines[start:end].replace('\n', ' ').strip()
                if label != "score_based":
                    result[f"{label}_context"].append(context)
    
    # Extract score/point information with more detail
    score_patterns = [
        (r'(\d+)\s*points?', "points"),
        (r'score:\s*(\d+)', "score"),
        (r'\((\d+)\s*/\s*(\d+)\)', "fraction"),
        (r'(\d+)\s*/\s*(\d+)\s*points?', "fraction_points"),
        (r'award\s+(\d+)', "award"),
        (r'give\s+(\d+)', "give"),
    ]
    
    extracted_scores = []
    max_points = None
    
    for pattern, score_type in score_patterns:
        if score_type == "fraction":
            matches = re.findall(pattern, guidelines, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    try:
                        score, total = int(match[0]), int(match[1])
                        extracted_scores.append(score)
                        if max_points is None or total > max_points:
                            max_points = total
                        # Infer category from score ratio
                        if total > 0:
                            ratio = score / total
                            if ratio >= 0.9:
                                category_scores["correct"] = max(category_scores["correct"], 0.85)
                            elif ratio >= 0.7:
                                category_scores["almost"] = max(category_scores["almost"], 0.85)
                            elif ratio >= 0.4:
                                category_scores["partial"] = max(category_scores["partial"], 0.85)
                            else:
                                category_scores["incorrect"] = max(category_scores["incorrect"], 0.85)
                    except ValueError:
                        pass
        else:
            matches = re.findall(pattern, guidelines, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        match = match[0]
                    score = int(match)
                    extracted_scores.append(score)
                except (ValueError, IndexError):
                    pass
    
    if extracted_scores:
        result["score_hints"]["extracted_scores"] = extracted_scores
        result["score_hints"]["max_points"] = max_points
    
    # Look for keywords that indicate quality with weighted scoring
    quality_keywords = {
        "correct": [
            ("complete", 0.7), ("correct", 0.8), ("valid", 0.6), 
            ("proper", 0.6), ("full", 0.7), ("perfect", 0.9),
            ("flawless", 0.9), ("excellent", 0.8), ("right answer", 0.8),
        ],
        "almost": [
            ("minor", 0.7), ("small", 0.6), ("slight", 0.6), 
            ("typo", 0.8), ("nearly", 0.8), ("close", 0.7),
            ("trivial", 0.7), ("insignificant", 0.6), ("almost correct", 0.9),
            ("mostly correct", 0.8), ("small error", 0.75),
        ],
        "partial": [
            ("partial", 0.8), ("incomplete", 0.7), ("missing", 0.6), 
            ("some", 0.5), ("half", 0.7), ("progress", 0.6),
            ("attempt", 0.5), ("partial credit", 0.9), ("partially correct", 0.8),
            ("on the right track", 0.7), ("meaningful progress", 0.75),
        ],
        "incorrect": [
            ("wrong", 0.8), ("invalid", 0.7), ("error", 0.6), 
            ("incorrect", 0.8), ("fail", 0.7), ("no credit", 0.9),
            ("fundamentally wrong", 0.9), ("does not work", 0.7),
            ("completely wrong", 0.9), ("no progress", 0.7),
        ],
    }
    
    for category, keywords in quality_keywords.items():
        for keyword, weight in keywords:
            if keyword in guidelines_lower:
                result[f"has_{category}"] = True
                category_scores[category] = max(category_scores[category], weight)
    
    # Determine primary category based on scores
    if any(v > 0 for v in category_scores.values()):
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        if best_score > 0:
            result["primary_category"] = best_category
            result["confidence"] = best_score
    
    return result


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract key information from inputs for better prompting
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        points = inputs.get('points', '')
        reward = inputs.get('reward', '')
        
        # Parse grading guidelines to extract rubric indicators
        rubric = _parse_grading_guidelines(grading_guidelines)
        
        # Build rubric context for the prompt with enhanced signals
        rubric_context = ""
        
        # Primary category signal (highest priority)
        if rubric.get("primary_category") and rubric.get("confidence", 0) > 0.7:
            primary = rubric["primary_category"]
            confidence = rubric["confidence"]
            rubric_context += f"\n\n[STRONG RUBRIC SIGNAL: The grading guidelines strongly indicate this solution should be classified as '{primary}' (confidence: {confidence:.0%}). This is the PRIMARY classification hint.]"
        
        # Individual marker signals
        if rubric["has_partial"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Partial)' or related markers, indicating partial credit. Consider 'partial' if the solution shows some progress but is incomplete.]"
            if rubric["partial_context"]:
                rubric_context += f"\nContext: {rubric['partial_context'][0][:250]}"
        if rubric["has_almost"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Almost)' or related markers, indicating near-correctness with minor issues. Consider 'almost' if the solution is nearly correct with only trivial errors.]"
            if rubric["almost_context"]:
                rubric_context += f"\nContext: {rubric['almost_context'][0][:250]}"
        if rubric["has_correct"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Correct)' or related markers, indicating full correctness. Consider 'correct' if the solution is essentially flawless.]"
            if rubric["correct_context"]:
                rubric_context += f"\nContext: {rubric['correct_context'][0][:250]}"
        if rubric["has_incorrect"]:
            rubric_context += "\n\n[RUBRIC MARKER: The grading guidelines contain '(Incorrect)' or related markers, indicating no credit. Consider 'incorrect' if the solution is fundamentally wrong.]"
            if rubric["incorrect_context"]:
                rubric_context += f"\nContext: {rubric['incorrect_context'][0][:250]}"
        
        # Score-based guidance with interpretation
        if rubric.get("score_hints", {}).get("extracted_scores"):
            scores = rubric["score_hints"]["extracted_scores"]
            max_pts = rubric["score_hints"].get("max_points")
            if max_pts and max_pts > 0:
                # Calculate ratio for interpretation
                avg_score = sum(scores) / len(scores)
                ratio = avg_score / max_pts
                ratio_guidance = ""
                if ratio >= 0.9:
                    ratio_guidance = "This high score ratio suggests 'correct' classification."
                elif ratio >= 0.7:
                    ratio_guidance = "This score ratio suggests 'almost' classification."
                elif ratio >= 0.4:
                    ratio_guidance = "This score ratio suggests 'partial' classification."
                else:
                    ratio_guidance = "This low score ratio suggests 'incorrect' classification."
                rubric_context += f"\n\n[SCORE INFORMATION: Extracted scores: {scores} out of {max_pts}. {ratio_guidance}]"
        
        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems (like IMO, Putnam, etc.).

Your task is to evaluate the student's answer and classify it into EXACTLY ONE of these four categories:

1. "correct" - The student's answer is fully correct, complete, and rigorous. It matches or exceeds the official solution in completeness and correctness. Only use this if the solution is essentially flawless with no significant errors or omissions.

2. "almost" - The student's answer is nearly correct with only minor errors, typos, or small omissions that don't significantly affect the overall correctness. The core logic is sound but there are small imperfections. The student demonstrates strong understanding with only trivial mistakes.

3. "partial" - The student's answer has some correct elements, shows meaningful progress, or demonstrates understanding of key concepts, but is incomplete, has significant gaps, or contains major errors that prevent it from being fully correct. The student is on the right track but hasn't reached a complete solution.

4. "incorrect" - The student's answer is fundamentally wrong, contains major errors, shows no meaningful progress toward the solution, or is irrelevant/off-track. The approach or answer is completely wrong.

=== CLASSIFICATION DECISION TREE (FOLLOW THIS ORDER) ===

Step 1: RUBRIC MARKERS (HIGHEST PRIORITY)
The Grading Guidelines below contain explicit markers like (Partial), (Almost), (Correct), (Incorrect), (Full Credit), (No Credit), (Half Credit), (Most Credit). These are STRONG signals of the intended classification. 
- If you see (Correct) or (Full Credit) → strongly consider "correct"
- If you see (Almost) or (Most Credit) → strongly consider "almost"  
- If you see (Partial) or (Half Credit) or (Some Credit) → strongly consider "partial"
- If you see (Incorrect) or (No Credit) → strongly consider "incorrect"
Follow these markers unless the student's answer clearly contradicts them.

Step 2: SCORE INTERPRETATION
If the rubric shows scores like "(X/Y) points":
- Score ratio ≥ 90% → likely "correct"
- Score ratio 70-89% → likely "almost"
- Score ratio 40-69% → likely "partial"
- Score ratio < 40% → likely "incorrect"

Step 3: SOLUTION ANALYSIS (if no clear rubric markers)
Compare the student's solution against the official solution:
- Same final answer with valid reasoning? → "correct"
- Approach correct but minor computational/notational errors? → "almost"
- Correct initial steps but incomplete or with significant errors? → "partial"
- Approach fundamentally wrong or completely missing? → "incorrect"

Step 4: PROBLEM DIFFICULTY CONTEXT
- For difficult problems, partial credit is given for meaningful progress
- For easier problems, the standard is higher for "correct" and "almost"

=== PROBLEM STATEMENT ===
{problem}

=== OFFICIAL SOLUTION ===
{solution}

=== GRADING GUIDELINES (RUBRIC) ===
{grading_guidelines}
{rubric_context}

=== STUDENT'S ANSWER ===
{student_answer}

=== ADDITIONAL SIGNALS ===
Points: {points}
Reward: {reward}

=== GRADING INSTRUCTIONS ===
Carefully analyze:
1. What the problem is asking for
2. What the official solution provides as the correct answer
3. The grading guidelines above - these contain specific rubric markers indicating what constitutes each grade level
4. The student's answer - compare it against the official solution and rubric

Pay special attention to:
- Does the student identify the correct answer/approach?
- Are the key steps of the proof/solution present?
- Are there errors in reasoning or calculations?
- How complete is the solution?
- What do the grading guidelines explicitly indicate about this solution?
- Are there partial credit markers that suggest the intended classification?

=== REASONING PROCESS ===
Before giving your final answer, think through:
1. What is the core question/problem being asked?
2. What is the official solution's approach and answer?
3. What does the student's answer provide?
4. How does the rubric/grading guidelines classify this type of answer?
5. Which category best matches the student's work?

=== RESPONSE FORMAT (STRICT) ===
You MUST respond ONLY in the following JSON format. Do not include any other text, explanations, or markdown outside the JSON block:

<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

The response field must contain EXACTLY one of these four lowercase words: correct, almost, partial, incorrect.
- Use "correct" only for flawless solutions
- Use "almost" for solutions with only minor/trivial errors
- Use "partial" for incomplete solutions with some correct progress
- Use "incorrect" for fundamentally wrong solutions"""

        # Use retry with backoff for LLM call to handle transient failures
        def _call_llm():
            return get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        
        try:
            response, msg_history, info = retry_with_backoff(
                _call_llm,
                max_retries=3,
                base_delay=1.0,
                exceptions=(Exception,),
            )
        except Exception as e:
            logger.error(f"LLM call failed after retries: {e}")
            return "unknown", []

        # Extract prediction from response with enhanced robustness
        prediction = "unknown"
        try:
            last_message = msg_history[-1]["text"]
            self.log_fn(f"Raw response: {last_message[:500]}...")
            
            # Try flexible JSON extraction first
            extracted = _extract_json_flexible(last_message)
            if extracted:
                # Try to get the response value
                if isinstance(extracted, dict):
                    if "response" in extracted:
                        prediction = _normalize_prediction(extracted["response"])
                    elif len(extracted) == 1:
                        # If only one key, use its value
                        prediction = _normalize_prediction(list(extracted.values())[0])
                    else:
                        # Try to find a value that looks like a grade
                        for key, value in extracted.items():
                            normalized = _normalize_prediction(value)
                            if _is_valid_prediction(normalized):
                                prediction = normalized
                                break
            
            # If still unknown, try direct text extraction with priority ordering
            if not _is_valid_prediction(prediction):
                text_lower = last_message.lower()
                
                # Priority 1: Look for exact quoted labels in JSON-like context
                json_patterns = [
                    (r'"response"\s*:\s*"(correct|almost|partial|incorrect)"', 1),
                    (r'"(correct|almost|partial|incorrect)"', 1),
                    (r"'(correct|almost|partial|incorrect)'", 1),
                ]
                for pattern, group in json_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        candidate = match.group(group).lower()
                        if _is_valid_prediction(candidate):
                            prediction = candidate
                            break
                
                # Priority 2: Look for labels in code blocks or after colons
                if not _is_valid_prediction(prediction):
                    colon_patterns = [
                        (r'response\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
                        (r'classification\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
                        (r'grade\s*[=:]\s*(correct|almost|partial|incorrect)\b', 1),
                    ]
                    for pattern, group in colon_patterns:
                        match = re.search(pattern, text_lower)
                        if match:
                            candidate = match.group(group).lower()
                            if _is_valid_prediction(candidate):
                                prediction = candidate
                                break
                
                # Priority 3: Check for standalone words at word boundaries
                if not _is_valid_prediction(prediction):
                    # Check in order of specificity (almost before correct to avoid substring issues)
                    if re.search(r'\balmost\b', text_lower):
                        prediction = "almost"
                    elif re.search(r'\bpartial\b', text_lower):
                        prediction = "partial"
                    elif re.search(r'\bincorrect\b', text_lower) or re.search(r'\bwrong\b', text_lower):
                        prediction = "incorrect"
                    elif re.search(r'\bcorrect\b', text_lower):
                        # Make sure "correct" isn't part of "incorrect" by checking context
                        for match in re.finditer(r'\bcorrect\b', text_lower):
                            start = max(0, match.start() - 3)
                            context = text_lower[start:match.start()]
                            if not context.endswith('in') and not context.endswith('not '):
                                prediction = "correct"
                                break
            
            # Final fallback: look for the first occurrence of any valid label
            # with smart disambiguation
            if not _is_valid_prediction(prediction):
                text_lower = last_message.lower()
                # Find all occurrences of each label
                positions = []
                for label in ["correct", "almost", "partial", "incorrect"]:
                    for match in re.finditer(r'\b' + label + r'\b', text_lower):
                        positions.append((match.start(), label))
                
                if positions:
                    positions.sort()
                    # Check for "not correct" or "incorrect" patterns
                    for i, (pos, label) in enumerate(positions):
                        if label == "correct":
                            # Check if preceded by "not" or "in"
                            context_before = text_lower[max(0, pos-20):pos]
                            if "not " in context_before[-4:] or context_before.endswith("in"):
                                continue  # Skip this "correct" as it's likely "incorrect"
                            prediction = label
                            break
                        elif label == "incorrect":
                            prediction = label
                            break
                        elif label == "almost":
                            prediction = label
                            break
                        elif label == "partial":
                            prediction = label
                            break
                    else:
                        # If no valid prediction found after filtering, use first
                        prediction = positions[0][1]
            
            # Use rubric primary category as fallback if LLM response is unclear
            if prediction == "unknown" and rubric.get("primary_category") and rubric.get("confidence", 0) > 0.8:
                prediction = rubric["primary_category"]
                self.log_fn(f"Using rubric primary category as fallback: {prediction}")
            
            self.log_fn(f"Extracted prediction: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Emergency fallback to rubric if available
            if rubric.get("primary_category"):
                prediction = rubric["primary_category"]

        return str(prediction), msg_history
