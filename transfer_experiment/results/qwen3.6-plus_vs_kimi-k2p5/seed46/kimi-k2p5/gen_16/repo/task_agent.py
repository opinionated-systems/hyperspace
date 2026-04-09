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

__version__ = "4.2.0"


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
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ```, ``` JSON ... ```, or plain ``` ... ``` blocks
    pattern = r'```\s*(?:\w+)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies, from most to least specific."""
    # Try <json> tags first (most specific)
    results = _extract_jsons(text)
    if results:
        return results

    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results

    # Try to find JSON objects directly using brace-matching (handles nested JSON)
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find matching closing brace
            depth = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    if results:
        return results

    # Fallback: try simple regex for flat JSON objects
    results = []
    pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            continue

    return results or None


def _extract_grade_from_text(text: str) -> str | None:
    """Extract grade directly from text using pattern matching."""
    text_lower = text.lower()
    
    # Look for explicit grade assignments with various patterns
    patterns = [
        # JSON-style patterns (most specific first)
        (r'"response"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"grade"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"label"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"result"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"evaluation"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"answer"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        (r'"prediction"\s*:\s*"(correct|incorrect|partial|almost)"', 1),
        # Unquoted patterns
        (r'response\s*:\s*(correct|incorrect|partial|almost)', 1),
        (r'grade\s*:\s*(correct|incorrect|partial|almost)', 1),
        (r'label\s*:\s*(correct|incorrect|partial|almost)', 1),
        # Standalone with word boundaries (least specific)
        (r'\b(correct|incorrect|partial|almost)\b', 0),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(group)
    
    return None


def _clean_json_response(text: str) -> str:
    """Clean common JSON formatting issues from LLM responses."""
    # Remove common prefixes that LLMs add
    prefixes_to_remove = [
        r'^\s*Here is the JSON:\s*',
        r'^\s*Here is my response:\s*',
        r'^\s*Response:\s*',
        r'^\s*Grade:\s*',
        r'^\s*The grade is:\s*',
        r'^\s*JSON:\s*',
        r'^\s*Output:\s*',
        r'^\s*Result:\s*',
        r'^\s*Evaluation:\s*',
        r'^\s*Answer:\s*',
        r'^\s*My answer is:\s*',
        r'^\s*The answer is:\s*',
        r'^\s*Final answer:\s*',
        r'^\s*Decision:\s*',
        r'^\s*Based on my analysis.*?:\s*',
    ]
    
    cleaned = text
    for pattern in prefixes_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove trailing text after the JSON block
    # Find the last </json> or ``` and truncate after it
    last_json_end = cleaned.rfind('</json>')
    last_code_end = cleaned.rfind('```')
    
    if last_json_end != -1:
        end_pos = last_json_end + 7
        if end_pos < len(cleaned):
            # Check if there's only whitespace/trailing text
            trailing = cleaned[end_pos:].strip()
            if trailing and not trailing.startswith('{'):
                cleaned = cleaned[:end_pos]
    elif last_code_end != -1:
        # Make sure this is after a JSON block
        code_start = cleaned.rfind('```', 0, last_code_end)
        if code_start != -1:
            between = cleaned[code_start:last_code_end].lower()
            if 'json' in between or '{' in between:
                end_pos = last_code_end + 3
                if end_pos < len(cleaned):
                    trailing = cleaned[end_pos:].strip()
                    if trailing and not trailing.startswith('{'):
                        cleaned = cleaned[:end_pos]
    
    return cleaned.strip()


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
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')

        instruction = f"""You are an expert mathematical grader evaluating IMO-style competition problems. Your task is to carefully analyze a student's answer and assign exactly one of four grades.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

---

GRADE DEFINITIONS - READ CAREFULLY:

**Correct**: The answer is 100% complete and correct with ZERO flaws.
- All key steps from the official solution are present (or equivalent alternative approach)
- The proof is complete with no gaps
- All reasoning is mathematically valid
- Use ONLY when the solution is perfect

**Incorrect**: The answer is fundamentally wrong with no meaningful progress.
- The approach is completely wrong or misguided
- No key insights from the official solution are present
- The reasoning contains critical errors
- Use when there is no substantial correct content

**Partial**: The answer shows meaningful progress but is incomplete or has significant gaps.
- Some key insights from the official solution are present
- There is a valid approach or partial proof structure
- Missing critical steps or contains non-trivial gaps
- May have some correct lemmas or observations
- Use when there is real mathematical progress but not close to complete

**Almost**: The answer is nearly complete with only minor issues.
- Most key steps are present and correct
- The main proof structure is sound
- Only minor gaps, typos, or easily fixable errors
- The core mathematical argument is valid
- Use when the solution is very close to correct but has small flaws

---

DETAILED GRADING CRITERIA:

**Key Distinctions:**

1. Partial vs Almost (CRITICAL - Most Common Error):
   
   PARTIAL means:
   - Missing MAJOR proof components OR has significant logical gaps
   - The solution is NOT close to complete (typically 25-70% complete)
   - Missing entire sections of the proof
   - Has the key insight but didn't develop it into a full solution
   - Examples of PARTIAL:
     * Found the invariant but didn't prove sufficiency
     * Started the right approach but stopped halfway
     * Has the key insight but proof has critical gaps
     * Only solved a special case, not the general problem
     * Has correct setup but major calculation errors that break the proof
     * Missing the hardest/most critical step of the proof
   
   ALMOST means:
   - Has ALL major components, only minor issues remain
   - The solution IS close to complete (typically 80-95% there)
   - Examples of ALMOST:
     * Complete proof with one small calculation error
     * All key steps present, just missing a trivial verification
     * Correct approach, complete proof, minor notation issues
     * Small logical gap that is easily fixable
     * Has all main ideas, just needs to connect them more clearly
     * One lemma is stated without proof but is intuitively obvious

2. Almost vs Correct:
   - Almost: Has minor flaws that need fixing (even small ones)
   - Correct: Absolutely perfect, no flaws whatsoever

3. Incorrect vs Partial:
   - Incorrect: No meaningful mathematical progress toward solution
     * Random guessing or completely wrong approach
     * No valid mathematical reasoning
   - Partial: Has at least one key insight or valid approach element
     * Shows understanding of the problem structure
     * Has valid mathematical reasoning even if incomplete

---

STEP-BY-STEP GRADING PROCESS:

STEP 1: ANALYZE THE OFFICIAL SOLUTION
   - Identify the KEY INSIGHTS (the "aha" moments)
   - Note the MAIN PROOF STRUCTURE (what are the major sections)
   - List CRITICAL LEMMAS/TECHNIQUES (essential components)
   - Count the total number of major steps/components

STEP 2: ANALYZE THE STUDENT'S ANSWER
   - Did they identify the MAIN APPROACH? (same as official or equivalent?)
   - Which KEY INSIGHTS are present? (list them)
   - What major steps/components are MISSING?
   - What is the COMPLETENESS LEVEL?
     * Count how many major steps are present vs missing
     * 0-25% complete → Incorrect
     * 25-70% complete → Partial
     * 75-90% complete → Almost
     * 95-99% complete → Almost
     * 100% complete → Correct

STEP 3: CHECK GRADING GUIDELINES
   - Look for specific achievements mentioned
   - Note what earns "Partial" vs "Almost" in the guidelines
   - Follow any specific criteria in the guidelines

STEP 4: MAKE YOUR DECISION
   - Be OBJECTIVE and CONSISTENT
   - Use the COMPLETENESS PERCENTAGE as your primary guide
   - When in doubt between Partial/Almost:
     * Ask: "Is this 75%+ complete with only minor issues?"
     * If YES → Almost
     * If NO or has major gaps → Partial

---

DECISION FRAMEWORK:

When in doubt, use these rules:
   - Doubt between Incorrect/Partial → choose Partial (if any valid insight exists)
   - Doubt between Partial/Almost → choose Partial (unless clearly 75%+ complete)
   - Doubt between Almost/Correct → choose Almost

---

Respond in this exact JSON format:
<json>
{{
    "response": "Correct" or "Incorrect" or "Partial" or "Almost"
}}
</json>

IMPORTANT: Your response must be valid JSON. Do not include any text outside the JSON tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = self._extract_prediction(response)
        return str(prediction), msg_history

    def _extract_prediction(self, response: str) -> str:
        """Extract the grade prediction from the LLM response.
        
        Uses multiple strategies to find a valid grade label.
        """
        prediction = "None"
        
        try:
            # Clean the response first
            cleaned_response = _clean_json_response(response)
            
            # Strategy 1: Extract JSON and look for response field
            extracted = _extract_any_json(cleaned_response)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict):
                        # Check for response field first (most common)
                        if "response" in item:
                            val = item["response"]
                            if isinstance(val, str):
                                prediction = val.strip()
                                break
                            elif isinstance(val, (int, float)):
                                prediction = str(val)
                                break
                        # Check alternative field names
                        for key in ["grade", "label", "result", "evaluation", "answer", "prediction"]:
                            if key in item:
                                val = item[key]
                                if isinstance(val, str):
                                    prediction = val.strip()
                                    break
                                elif isinstance(val, (int, float)):
                                    prediction = str(val)
                                    break
                        if prediction != "None":
                            break

            # Normalize to lowercase for comparison
            if prediction != "None":
                prediction = str(prediction).strip().lower()

            # Strategy 2: Use the dedicated grade extraction function
            if prediction == "None":
                grade = _extract_grade_from_text(response)
                if grade:
                    prediction = grade

            # Strategy 3: Pattern matching in text if still no match
            if prediction == "None":
                prediction = self._extract_from_text_patterns(response)

        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Final validation
        return self._validate_prediction(prediction, response)

    def _extract_from_text_patterns(self, text: str) -> str:
        """Extract grade label using text pattern matching."""
        text_lower = text.lower()
        
        # Define patterns to check in order of specificity (most specific first)
        patterns = [
            # Exact JSON-like patterns with quotes (most specific)
            ('"response": "correct"', "correct"),
            ('"response":"correct"', "correct"),
            ('"response": "incorrect"', "incorrect"),
            ('"response":"incorrect"', "incorrect"),
            ('"response": "partial"', "partial"),
            ('"response":"partial"', "partial"),
            ('"response": "almost"', "almost"),
            ('"response":"almost"', "almost"),
            # Grade field patterns
            ('"grade": "correct"', "correct"),
            ('"grade":"correct"', "correct"),
            ('"grade": "incorrect"', "incorrect"),
            ('"grade":"incorrect"', "incorrect"),
            ('"grade": "partial"', "partial"),
            ('"grade":"partial"', "partial"),
            ('"grade": "almost"', "almost"),
            ('"grade":"almost"', "almost"),
            # Label field patterns
            ('"label": "correct"', "correct"),
            ('"label":"correct"', "correct"),
            ('"label": "incorrect"', "incorrect"),
            ('"label":"incorrect"', "incorrect"),
            ('"label": "partial"', "partial"),
            ('"label":"partial"', "partial"),
            ('"label": "almost"', "almost"),
            ('"label":"almost"', "almost"),
            # Result field patterns
            ('"result": "correct"', "correct"),
            ('"result":"correct"', "correct"),
            ('"result": "incorrect"', "incorrect"),
            ('"result":"incorrect"', "incorrect"),
            ('"result": "partial"', "partial"),
            ('"result":"partial"', "partial"),
            ('"result": "almost"', "almost"),
            ('"result":"almost"', "almost"),
            # Standalone quoted values
            ('"correct"', "correct"),
            ('"incorrect"', "incorrect"),
            ('"partial"', "partial"),
            ('"almost"', "almost"),
            # Unquoted field patterns
            ('response: correct', "correct"),
            ('response:correct', "correct"),
            ('response: incorrect', "incorrect"),
            ('response:incorrect', "incorrect"),
            ('response: partial', "partial"),
            ('response:partial', "partial"),
            ('response: almost', "almost"),
            ('response:almost', "almost"),
            ('grade: correct', "correct"),
            ('grade:correct', "correct"),
            ('grade: incorrect', "incorrect"),
            ('grade:incorrect', "incorrect"),
            ('grade: partial', "partial"),
            ('grade:partial', "partial"),
            ('grade: almost', "almost"),
            ('grade:almost', "almost"),
        ]
        
        for pattern, label in patterns:
            if pattern in text_lower:
                return label
        
        # Check for standalone words with word boundaries
        # Order matters: check longer/more specific words first
        # Use word boundaries to avoid partial matches
        if re.search(r'\bpartial\b', text_lower):
            return "partial"
        if re.search(r'\balmost\b', text_lower):
            return "almost"
        if re.search(r'\bincorrect\b', text_lower):
            return "incorrect"
        if re.search(r'\bcorrect\b', text_lower):
            return "correct"
        
        return "None"

    def _validate_prediction(self, prediction: str, original_response: str) -> str:
        """Validate and normalize the prediction to a valid label."""
        valid_labels = ["correct", "incorrect", "partial", "almost"]
        
        pred_lower = prediction.lower().strip()
        
        # Direct match
        if pred_lower in valid_labels:
            return pred_lower
        
        # Handle common variations and typos with more comprehensive matching
        variations = {
            "correct": ["correct", "right", "true", "valid", "perfect", "complete", "full", "accurate", 
                       "100%", "flawless", "exact", "precise", "proper"],
            "incorrect": ["incorrect", "wrong", "false", "invalid", "error", "none", "fail", 
                         "fails", "failed", "failure", "unsuccessful", "rejected"],
            "partial": ["partial", "part", "some", "incomplete", "half", "partially", 
                       "partial credit", "partially correct", "partial solution"],
            "almost": ["almost", "nearly", "close", "minor", "small issues", "nearly correct", 
                      "mostly correct", "minor error", "tiny mistake", "almost there", 
                      "very close", "just missed", "slight error"],
        }
        
        # Check for variations with word boundaries for better accuracy
        for label, variants in variations.items():
            for variant in variants:
                # Use word boundary check for single words
                if ' ' not in variant:
                    if re.search(r'\b' + re.escape(variant) + r'\b', pred_lower):
                        return label
                else:
                    # For multi-word phrases, check if contained
                    if variant in pred_lower:
                        return label
        
        # Try to find any valid label as substring (last resort)
        for label in valid_labels:
            if label in pred_lower:
                return label
        
        # Try to extract from the original response one more time
        grade = _extract_grade_from_text(original_response)
        if grade and grade in valid_labels:
            return grade
        
        # Try pattern matching on original response
        text_grade = self._extract_from_text_patterns(original_response)
        if text_grade != "None" and text_grade in valid_labels:
            return text_grade
        
        # Log the issue and default to incorrect
        self.log_fn(f"Could not determine valid label from: '{prediction}'")
        self.log_fn(f"Original response: {original_response[:200]}...")
        return "incorrect"
