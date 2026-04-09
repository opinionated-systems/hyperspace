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

# Valid grade labels
_VALID_GRADES = {"correct", "partial", "incorrect"}


def _extract_guideline_label(guidelines: str) -> str | None:
    """Extract explicit grade label from grading guidelines.
    
    The guidelines may contain labels like:
    - "(Partial)" → "partial"
    - "(Almost)" → "partial" (minor but non-negligible issues)
    - "(Incorrect)" → "incorrect" or "partial" depending on content
    - "(Correct)" → verify independently
    
    Returns the label type if found, None otherwise.
    
    Priority order: (Incorrect) > (Almost) > (Partial) > (Correct)
    This ensures we catch the most restrictive label first.
    """
    if not guidelines:
        return None
    
    guidelines_lower = guidelines.lower()
    
    # Priority 1: (Incorrect) - most restrictive
    if re.search(r'[\(\[<\{]incorrect[\)\]>\}]', guidelines_lower):
        return "incorrect"
    
    # Priority 2: (Almost) - means minor but NON-NEGLIGIBLE mistakes
    if re.search(r'[\(\[<\{]almost[\)\]>\}]', guidelines_lower):
        return "partial"
    
    # Priority 3: (Partial) - explicit partial credit
    if re.search(r'[\(\[<\{]partial[\)\]>\}]', guidelines_lower):
        return "partial"
    
    # Priority 4: (Correct) - explicit full credit (less common)
    if re.search(r'[\(\[<\{]correct[\)\]>\}]', guidelines_lower):
        return "correct"
    
    return None


def _validate_grade_with_guidelines(predicted_grade: str, guideline_hint: str | None, grading_guidelines: str = "") -> str:
    """Validate and potentially correct the predicted grade based on guideline hints.
    
    This function applies hard rules for specific guideline labels to ensure
    consistent grading. The LLM's judgment is respected unless it clearly
    contradicts the explicit guideline label meaning.
    
    Label meanings:
    - (Almost) = "partial" (minor but non-negligible issues, never correct)
    - (Partial) = "partial" (incomplete but has progress, never correct)
    - (Incorrect) = "incorrect" or "partial" (no/wrong progress)
    - (Correct) = verify independently (label is just a claim)
    
    Returns the validated (and possibly corrected) grade.
    """
    if not guideline_hint:
        return predicted_grade
    
    guidelines_lower = (grading_guidelines or "").lower()
    
    # Check for specific labels using regex for robustness
    has_almost_label = re.search(r'[\(\[<\{]almost[\)\]>\}]', guidelines_lower) is not None
    has_partial_label = re.search(r'[\(\[<\{]partial[\)\]>\}]', guidelines_lower) is not None
    has_incorrect_label = re.search(r'[\(\[<\{]incorrect[\)\]>\}]', guidelines_lower) is not None
    has_correct_label = re.search(r'[\(\[<\{]correct[\)\]>\}]', guidelines_lower) is not None
    
    # HARD RULES: These override LLM judgment when there's a clear contradiction
    
    # Rule 1: (Almost) labeled answers MUST be "partial" (never "correct" or "incorrect")
    # "Almost" explicitly means minor but NON-NEGLIGIBLE mistakes exist
    if has_almost_label:
        return "partial"
    
    # Rule 2: (Partial) labeled answers MUST be "partial" (never "correct")
    # "Partial" explicitly means incomplete answer
    if has_partial_label:
        return "partial"
    
    # Rule 3: (Incorrect) labeled answers should not be "correct"
    # But can be "partial" if there's any valid progress
    if has_incorrect_label and predicted_grade == "correct":
        # Check if there's any indication of partial progress in guidelines
        progress_indicators = [
            'some', 'partial', 'progress', 'part', 'step', 'idea', 'approach',
            'observation', 'invariant', 'proved', 'shown', 'valid', 'correct'
        ]
        has_progress = any(indicator in guidelines_lower for indicator in progress_indicators)
        if has_progress:
            return "partial"
        return "incorrect"
    
    return predicted_grade


def _normalize_grade(grade: str) -> str | None:
    """Normalize a grade string to one of the valid grades.
    
    Handles common variations and misspellings.
    """
    if not grade:
        return None
    
    # Clean up the grade string
    grade_clean = grade.lower().strip().strip('"\'`)').strip('.,;:!?')
    grade_lower = grade_clean.strip()
    
    # Direct match
    if grade_lower in _VALID_GRADES:
        return grade_lower
    
    # Check for negation patterns first
    negation_patterns = [
        r'\bnot\s+correct\b',
        r'\bnot\s+right\b',
        r'\bnot\s+valid\b',
        r'\bisn\'t\s+correct\b',
        r'\bnot\s+partial\b',
    ]
    for pattern in negation_patterns:
        if re.search(pattern, grade_lower):
            return "incorrect"
    
    # Common variations
    variations = {
        "correct": [
            "correct", "right", "true", "valid", "full", "full credit", "complete",
            "fully correct", "all correct", "perfect", "excellent", "good",
            "accurate", "proper", "sound", "flawless", "error-free", "100%",
            "full marks", "full score", "correct answer", "correct solution",
            "entirely correct", "completely correct", "totally correct",
            "yes", "pass", "accepted", "approved", "satisfactory", "success",
            "successful", "complete solution", "valid solution", "right answer",
            "true answer", "accurate answer", "proper answer", "sound answer",
            "flawless answer", "perfect answer", "excellent answer", "good answer",
            "correct work", "right work", "valid work", "sound work",
            "full points", "all points", "maximum points", "max points",
            "100 points", "100% correct", "completely right", "totally right",
            "entirely right", "fully right", "all right", "alright",
            "fully accurate", "completely accurate", "totally accurate",
            "entirely accurate", "perfectly correct", "absolutely correct",
            "definitely correct", "certainly correct", "undoubtedly correct",
        ],
        "partial": [
            "partial", "part", "partially correct", "almost", "some credit", "half",
            "partial credit", "incomplete", "mostly correct", "nearly correct",
            "some progress", "partial solution", "partial answer", "half credit",
            "partially right", "partially valid", "minor errors", "small mistakes",
            "almost correct", "almost right", "nearly right", "mostly right",
            "partial marks", "partial score", "some marks", "in progress",
            "half correct", "half right", "50%", "partially done", "partially",
            "mostly", "nearly", "close", "incomplete solution", "incomplete answer",
            "almost there", "nearly there", "close to correct", "close to right",
            "mostly accurate", "mostly valid", "mostly sound", "mostly proper",
            "some correct", "some right", "some valid", "some proper",
            "partial work", "incomplete work", "unfinished", "unfinished solution",
            "unfinished answer", "unfinished work", "half done", "half finished",
            "partial success", "limited success", "some success", "moderate success",
            "50 percent", "fifty percent", "half marks", "half points",
            "partial points", "some points", "limited points", "moderate points",
            "almost there", "getting there", "on the right track", "good start",
            "some understanding", "partial understanding", "incomplete but correct",
            "mostly there", "nearly complete", "almost complete", "close to complete",
            "almost perfect", "nearly perfect", "mostly perfect",
            "almost flawless", "nearly flawless", "mostly flawless",
            "close to perfect", "close to flawless", "close to complete",
            "nearly complete", "almost complete", "mostly complete",
            "partially complete", "partially correct", "partially accurate",
            "partially valid", "partially sound", "partially proper",
            "partially right", "partially true", "partially successful",
            "somewhat correct", "somewhat right", "somewhat accurate",
            "fairly correct", "fairly right", "fairly accurate",
            "reasonably correct", "reasonably right", "reasonably accurate",
            "relatively correct", "relatively right", "relatively accurate",
        ],
        "incorrect": [
            "incorrect", "wrong", "false", "invalid", "no credit", "none", "zero",
            "fully wrong", "completely wrong", "totally wrong", "bad", "poor",
            "erroneous", "mistaken", "flawed", "unsound", "no progress", "failed",
            "0%", "zero credit", "no marks", "zero marks", "wrong answer",
            "incorrect answer", "wrong solution", "incorrect solution",
            "not correct", "not right", "not valid", "entirely wrong",
            "fail", "rejected", "unacceptable", "unsatisfactory",
            "wrong work", "bad work", "poor work", "flawed work",
            "erroneous work", "mistaken work", "unsound work", "invalid work",
            "false answer", "false solution", "false work",
            "invalid answer", "invalid solution", "invalid work",
            "zero points", "no points", "0 points", "zero percent",
            "0 percent", "none correct", "none right", "none valid",
            "not accurate", "not proper", "not sound", "not flawless",
            "completely false", "totally false", "entirely false",
            "fully false", "all wrong", "all false", "all incorrect",
            "fail answer", "fail solution", "fail work",
            "rejected answer", "rejected solution", "rejected work",
            "unacceptable answer", "unacceptable solution", "unacceptable work",
            "completely incorrect", "totally incorrect", "entirely incorrect",
            "fully incorrect", "absolutely incorrect", "definitely incorrect",
            "certainly incorrect", "undoubtedly incorrect", "clearly incorrect",
            "obviously incorrect", "plainly incorrect", "simply incorrect",
            "just incorrect", "only incorrect", "merely incorrect",
            "utterly wrong", "completely wrong", "totally wrong",
            "entirely wrong", "fully wrong", "absolutely wrong",
            "definitely wrong", "certainly wrong", "undoubtedly wrong",
            "clearly wrong", "obviously wrong", "plainly wrong",
            "simply wrong", "just wrong", "only wrong", "merely wrong",
        ],
    }
    
    for canonical, variants in variations.items():
        if grade_lower in variants:
            return canonical
    
    # Check for partial matches with word boundaries
    if re.search(r'\bpartial\w*\b', grade_lower) or re.search(r'\balmost\b', grade_lower):
        return "partial"
    
    # Check for "correct" but exclude cases with negation or "incorrect"
    if re.search(r'\bcorrect\b', grade_lower):
        if "incorrect" not in grade_lower and "not " not in grade_lower:
            return "correct"
    
    # Check for incorrect/wrong patterns
    if re.search(r'\b(incorrect|wrong|false)\b', grade_lower):
        return "incorrect"
    
    # Check for numeric patterns
    if re.search(r'\b0\b', grade_lower) or re.search(r'\b0%\b', grade_lower) or re.search(r'\bzero\b', grade_lower):
        return "incorrect"
    if re.search(r'\b100\b', grade_lower) or re.search(r'\b100%\b', grade_lower) or re.search(r'\bfull\b', grade_lower):
        return "correct"
    if re.search(r'\b50\b', grade_lower) or re.search(r'\b50%\b', grade_lower) or re.search(r'\bhalf\b', grade_lower):
        return "partial"
    
    # Check for common phrases that indicate grades
    if re.search(r'\b(full|complete|perfect)\s+(?:credit|marks|score|solution)\b', grade_lower):
        return "correct"
    if re.search(r'\b(partial|some|half)\s+(?:credit|marks|score|progress)\b', grade_lower):
        return "partial"
    if re.search(r'\b(no|zero)\s+(?:credit|marks|score|progress)\b', grade_lower):
        return "incorrect"
    
    return None


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks and markdown code blocks.
    
    Uses robust parsing with multiple fallback strategies for malformed JSON.
    """
    results = []
    
    def _try_parse_json(inner: str) -> dict | None:
        """Try to parse JSON with various fixes for common issues."""
        inner = inner.strip()
        if not inner:
            return None
            
        # Try direct parse first
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        
        # Fix 1: Handle trailing commas
        try:
            fixed = re.sub(r',\s*}', '}', inner)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 2: Handle single quotes (convert to double)
        try:
            fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', inner)
            fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 3: Handle unquoted keys
        try:
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 4: Handle Python-style True/False/None
        try:
            fixed = inner.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 5: Handle newlines in strings
        try:
            fixed = re.sub(r'("[^"]*?)\n([^"]*?")', r'\1\\n\2', inner)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 6: Handle escaped quotes that might be malformed
        try:
            fixed = inner.replace('\\"', '"').replace("\\'", "'")
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Fix 7: Extract response field directly with regex patterns
        response_patterns = [
            r'"response"\s*:\s*"(correct|partial|incorrect)"',
            r"'response'\s*:\s*'(correct|partial|incorrect)'",
            r'"response"\s*:\s*\\"(correct|partial|incorrect)\\"',
            r'"response"\s*:\s*(correct|partial|incorrect)\b',
            r'"(?:grade|verdict|result|evaluation)"\s*:\s*"(correct|partial|incorrect)"',
            r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?',
            r'response\s*[=:]\s*["\']?(correct|partial|incorrect)["\']?',
            r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]',
        ]
        for pattern in response_patterns:
            try:
                response_match = re.search(pattern, inner, re.IGNORECASE)
                if response_match:
                    return {"response": response_match.group(1).lower()}
            except Exception:
                pass
        
        # Fix 8: Extract any valid grade from the text
        try:
            for grade in _VALID_GRADES:
                if f'"{grade}"' in inner.lower() or f"'{grade}'" in inner.lower():
                    return {"response": grade}
        except Exception:
            pass
        
        return None
    
    # Find <json>...</json> blocks (case insensitive)
    text_lower = text.lower()
    search_from = 0
    while True:
        start = text_lower.find("<json>", search_from)
        if start == -1:
            break
        end = text_lower.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Find markdown code blocks: ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL | re.IGNORECASE):
        inner = match.group(1).strip()
        parsed = _try_parse_json(inner)
        if parsed:
            results.append(parsed)
    
    # Find bare JSON objects with response field
    bare_json_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(correct|partial|incorrect)["\']?\s*\}'
    for match in re.finditer(bare_json_pattern, text, re.IGNORECASE):
        try:
            parsed = _try_parse_json(match.group(0))
            if parsed:
                results.append(parsed)
        except Exception:
            pass
    
    # Find any response field pattern in the text
    response_patterns = [
        r'"response"\s*:\s*"?(correct|partial|incorrect)"?',
        r"['\"]response['\"]\s*:\s*['\"]?(correct|partial|incorrect)['\"]?",
        r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]',
    ]
    for pattern in response_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                grade = match.group(1).lower()
                if grade in _VALID_GRADES:
                    results.append({"response": grade})
            except Exception:
                pass
    
    return results or None


def _extract_grade_fallback(text: str) -> str | None:
    """Fallback: scan the text for a grade keyword with context awareness.
    
    Focuses on the last section where the LLM typically states its conclusion.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    last_section = text_lower[-2000:] if len(text_lower) > 2000 else text_lower
    
    # Strategy 1: JSON-like response field (highest priority)
    json_patterns = [
        r'"response"\s*:\s*"(correct|partial|incorrect)"',
        r'["\']response["\']\s*:\s*["\']?(correct|partial|incorrect)["\']?',
        r'"response"\s*:\s*\[\s*"?(correct|partial|incorrect)"?\s*\]',
    ]
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 2: Explicit final decision patterns
    final_patterns = [
        r'(?:grade|verdict|assessment|result|evaluation)\s*[:\-=]\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:grade|verdict|assessment|result|evaluation)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:conclusion|therefore|thus)\s*,?\s*(?:the\s+)?(?:grade|result|verdict)\s+is\s*["\']?(correct|partial|incorrect)["\']?',
        r'(?:this\s+(?:answer|solution|work)\s+is)\s*["\']?(correct|partial|incorrect)["\']?',
    ]
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_section, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Strategy 3: "is X" patterns with negation check
    is_matches = list(re.finditer(r'\bis\s+(correct|partial|incorrect)\b', last_section, re.IGNORECASE))
    for match in reversed(is_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-50):start_pos]
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 4: Standalone grade words with negation check
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', last_section))
    for match in reversed(standalone_matches):
        grade = match.group(1).lower()
        start_pos = match.start()
        context = last_section[max(0, start_pos-50):start_pos]
        negation_indicators = ['not ', 'isn\'t', 'is not', 'never', 'neither', 'nor', 'no ']
        if any(neg in context for neg in negation_indicators):
            continue
        return grade
    
    # Strategy 5: Try full text
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower()
    
    # Absolute last resort
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    if standalone_matches:
        return standalone_matches[-1].group(1).lower()

    return None


def _extract_grade_from_response(text: str) -> str | None:
    """Extract grade from LLM response using multiple strategies.
    
    Priority order:
    1. JSON blocks with response field (highest priority)
    2. Fallback extraction with patterns
    3. Last resort: standalone words
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Strategy 1: Try JSON extraction first (most reliable)
    try:
        extracted = _extract_jsons(text)
        if extracted:
            # Use the last JSON object (most likely to be the final answer)
            for obj in reversed(extracted):
                grade = str(obj.get("response", "")).strip().lower()
                normalized = _normalize_grade(grade)
                if normalized:
                    return normalized
    except Exception:
        pass
    
    # Strategy 2: Try fallback extraction
    try:
        fallback = _extract_grade_fallback(text)
        if fallback:
            normalized = _normalize_grade(fallback)
            if normalized:
                return normalized
    except Exception:
        pass
    
    # Strategy 3: Last resort - look for any valid grade in the text
    standalone_matches = list(re.finditer(r'\b(correct|partial|incorrect)\b', text_lower))
    if standalone_matches:
        return standalone_matches[-1].group(1).lower()
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_file = log_file
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines,
                    student_answer

        Returns:
            (prediction, msg_history)
        """
        problem = inputs.get("problem", "")
        official_solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        # Extract guideline label for validation
        guideline_hint = _extract_guideline_label(grading_guidelines)
        
        if guideline_hint:
            self.log_fn(f"Guideline label detected: '{guideline_hint}'")

        # Build instruction with clear guidance based on label
        instruction = self._build_instruction(
            problem, official_solution, grading_guidelines, student_answer, guideline_hint
        )

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response
        prediction = self._extract_prediction(msg_history, response)
        
        # Apply hard validation rules based on guideline labels
        prediction = self._apply_hard_rules(prediction, guideline_hint, grading_guidelines)

        return str(prediction), msg_history

    def _build_instruction(
        self, 
        problem: str, 
        official_solution: str, 
        grading_guidelines: str, 
        student_answer: str,
        guideline_hint: str | None
    ) -> str:
        """Build the grading instruction prompt."""
        
        # Build label-specific guidance
        label_guidance = ""
        guidelines_lower = (grading_guidelines or "").lower()
        
        if guideline_hint == "partial" and re.search(r'[\(\[<\{]almost[\)\]>\}]', guidelines_lower):
            label_guidance = """
⚠️  CRITICAL: The grading guidelines are labeled (ALMOST).
This means the answer has MINOR but NON-NEGLIGIBLE issues.
You MUST grade this as "partial" - never "correct" or "incorrect".
"""
        elif guideline_hint == "partial" and re.search(r'[\(\[<\{]partial[\)\]>\}]', guidelines_lower):
            label_guidance = """
⚠️  CRITICAL: The grading guidelines are labeled (PARTIAL).
This means the answer is INCOMPLETE but has SOME valid progress.
You MUST grade this as "partial" - never "correct".
"""
        elif guideline_hint == "incorrect":
            label_guidance = """
ℹ️  NOTE: The grading guidelines are labeled (INCORRECT).
This typically means "incorrect", but check if ANY valid progress exists.
If there's ANY valid reasoning → "partial", otherwise "incorrect".
"""
        elif guideline_hint == "correct":
            label_guidance = """
ℹ️  NOTE: The grading guidelines are labeled (CORRECT).
You must INDEPENDENTLY VERIFY - don't blindly trust the label.
Only grade "correct" if TRULY FLAWLESS.
"""

        return f"""You are an expert mathematics competition grader. Evaluate the student's answer and assign exactly ONE grade: "correct", "partial", or "incorrect".

════════════════════════════════════════════════════════════════
GRADE DEFINITIONS
════════════════════════════════════════════════════════════════

【correct】= PERFECT (100%) - USE ONLY WHEN FLAWLESS
  • Complete solution with all logical steps valid
  • No errors, gaps, or calculation mistakes
  • Final answer matches official solution exactly
  • RARE - be very strict about this

【partial】= PARTIAL CREDIT (1-99%) - USE FOR IMPERFECT WORK
  • Has valid progress but incomplete or flawed
  • Contains errors, missing steps, or logical gaps
  • Shows understanding but execution is imperfect
  • "Almost correct" or "mostly correct" → this is "partial"
  • MOST COMMON grade for competition answers

【incorrect】= NO CREDIT (0%) - USE WHEN NO VALID PROGRESS
  • No understanding of the problem
  • Completely wrong approach
  • No meaningful mathematical reasoning
  • UNCOMMON - only when truly no valid work exists
{label_guidance}

════════════════════════════════════════════════════════════════
DECISION FRAMEWORK
════════════════════════════════════════════════════════════════

Step 1: Is the answer TRULY PERFECT? (zero flaws, complete)
  → If YES → "correct"
  → If NO → Step 2

Step 2: Does the answer show ANY meaningful progress?
  → If YES → "partial"  
  → If NO → "incorrect"

════════════════════════════════════════════════════════════════
PROBLEM
════════════════════════════════════════════════════════════════
{problem}

════════════════════════════════════════════════════════════════
OFFICIAL SOLUTION
════════════════════════════════════════════════════════════════
{official_solution}

════════════════════════════════════════════════════════════════
GRADING GUIDELINES
════════════════════════════════════════════════════════════════
{grading_guidelines}

════════════════════════════════════════════════════════════════
STUDENT'S ANSWER
════════════════════════════════════════════════════════════════
{student_answer}

════════════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════════════

Provide your decision in <json> tags:

<json>
{{
    "reasoning": "Your detailed explanation here.",
    "response": "correct" | "partial" | "incorrect"
}}
</json>

CRITICAL: The "response" field MUST be exactly: "correct", "partial", or "incorrect" (lowercase, no quotes around the value in the JSON string value).

Now provide your grading decision."""

    def _extract_prediction(self, msg_history: list[dict], response: str) -> str:
        """Extract and validate the grade from the LLM response."""
        # Get assistant text
        assistant_text = ""
        if msg_history:
            last_msg = msg_history[-1]
            assistant_text = last_msg.get("content", "") or last_msg.get("text", "") or ""
            if isinstance(assistant_text, list):
                parts = []
                for block in assistant_text:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                assistant_text = "\n".join(parts) if parts else ""
        if not assistant_text and response:
            assistant_text = response
        
        if not assistant_text:
            return "partial"
        
        # Try to extract grade
        extracted = _extract_grade_from_response(assistant_text)
        if extracted:
            self.log_fn(f"Extracted grade: {extracted}")
            return extracted
        
        self.log_fn("No grade extracted, defaulting to partial")
        return "partial"

    def _apply_hard_rules(
        self, 
        prediction: str, 
        guideline_hint: str | None, 
        grading_guidelines: str
    ) -> str:
        """Apply hard validation rules based on guideline labels."""
        guidelines_lower = (grading_guidelines or "").lower()
        
        # Rule 1: (Almost) and (Partial) labels MUST be "partial"
        if re.search(r'[\(\[<\{](almost|partial)[\)\]>\}]', guidelines_lower):
            if prediction != "partial":
                self.log_fn(f"Grade ENFORCED from '{prediction}' to 'partial' due to (Almost)/(Partial) label")
                return "partial"
        
        # Rule 2: (Incorrect) label should not be "correct"
        if re.search(r'[\(\[<\{]incorrect[\)\]>\}]', guidelines_lower) and prediction == "correct":
            # Check for progress indicators
            progress_indicators = ['some', 'partial', 'progress', 'part', 'step', 'idea', 'approach',
                                   'observation', 'invariant', 'proved', 'shown', 'valid']
            has_progress = any(indicator in guidelines_lower for indicator in progress_indicators)
            if has_progress:
                self.log_fn("Grade ENFORCED from 'correct' to 'partial' due to (Incorrect) label with progress")
                return "partial"
            else:
                self.log_fn("Grade ENFORCED from 'correct' to 'incorrect' due to (Incorrect) label")
                return "incorrect"
        
        # Validate the prediction is a valid grade
        if prediction not in _VALID_GRADES:
            self.log_fn(f"Invalid grade '{prediction}', defaulting to partial")
            return "partial"
        
        return prediction
