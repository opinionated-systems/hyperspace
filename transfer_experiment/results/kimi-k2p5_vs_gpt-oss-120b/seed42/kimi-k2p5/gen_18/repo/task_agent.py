"""
Task agent: solves a given task with a single LLM call.
Reimplemented from facebookresearch/HyperAgents task_agent.py.
"""

from __future__ import annotations

import json
import logging
import re

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


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
        except json.JSONDecodeError:
            continue
    return results or None


def _normalize_prediction(prediction: str) -> str | None:
    """Normalize a prediction string to one of the allowed categories."""
    if not prediction:
        return None
    
    pred_lower = prediction.lower().strip()
    
    # Remove common prefixes/suffixes
    pred_clean = re.sub(r'^(the\s+|a\s+|an\s+|is\s+|it\s+|this\s+|that\s+)', '', pred_lower)
    pred_clean = re.sub(r'\s+(answer|classification|category|result|grade)$', '', pred_clean)
    
    # Check for exact matches first
    allowed_categories = ["correct", "incorrect", "partial", "almost"]
    if pred_clean in allowed_categories:
        return pred_clean.capitalize()
    
    # Check for compound patterns first (before single word patterns)
    # These are phrases that indicate "Almost" category
    almost_compound_patterns = [
        r'\balmost\s+correct\b', r'\bnearly\s+correct\b', r'\balmost\s+perfect\b',
        r'\bclose\s+to\s+correct\b', r'\bessentially\s+correct\b',
        r'\bmostly\s+correct\b', r'\bcorrect\s+except\s+for\b',
        r'\bwould\s+be\s+correct\b', r'\balmost\s+complete\b',
        r'\bnearly\s+complete\b', r'\bessentially\s+complete\b',
        r'\bmostly\s+complete\b', r'\bpractically\s+correct\b',
        r'\bpractically\s+complete\b', r'\bvirtually\s+correct\b',
        r'\bvirtually\s+complete\b', r'\bwould\s+be\s+perfect\b',
        r'\bwould\s+be\s+complete\b', r'\bcould\s+be\s+correct\b',
        r'\bcorrect\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)\b',
        r'\bcomplete\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)\b',
    ]
    for pattern in almost_compound_patterns:
        if re.search(pattern, pred_lower):
            return "Almost"
    
    # Now check single word patterns in priority order
    # Check Almost first (most specific to avoid being caught by other patterns)
    if re.search(r'\balmost\b', pred_lower):
        return "Almost"
    if re.search(r'\bnearly\b', pred_lower):
        return "Almost"
    if re.search(r'\bessentially\b', pred_lower):
        return "Almost"
    if re.search(r'\bpractically\b', pred_lower):
        return "Almost"
    if re.search(r'\bvirtually\b', pred_lower):
        return "Almost"
    if re.search(r'\bminor\s+(?:error|mistake|issue|flaw)', pred_lower):
        return "Almost"
    if re.search(r'\bsmall\s+(?:error|mistake|issue|flaw)', pred_lower):
        return "Almost"
    if re.search(r'\btiny\s+(?:error|mistake|issue|flaw)', pred_lower):
        return "Almost"
    if re.search(r'\bsign\s+error', pred_lower):
        return "Almost"
    if re.search(r'\barithmetic\s+error', pred_lower):
        return "Almost"
    if re.search(r'\bcalculation\s+error', pred_lower):
        return "Almost"
    if re.search(r'\btypo\b', pred_lower):
        return "Almost"
    
    # Check Partial patterns
    if re.search(r'\bpartial\b', pred_lower):
        return "Partial"
    if re.search(r'\bpartly\b', pred_lower):
        return "Partial"
    if re.search(r'\bincomplete\b', pred_lower):
        return "Partial"
    if re.search(r'\bon\s+the\s+right\s+track\b', pred_lower):
        return "Partial"
    if re.search(r'\bgood\s+start\b', pred_lower):
        return "Partial"
    if re.search(r'\bin\s+progress\b', pred_lower):
        return "Partial"
    if re.search(r'\bunfinished\b', pred_lower):
        return "Partial"
    
    # Check Incorrect patterns
    if re.search(r'\bincorrect\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bwrong\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bfalse\b', pred_lower):
        return "Incorrect"
    if re.search(r'\binvalid\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bnot\s+correct\b', pred_lower):
        return "Incorrect"
    if re.search(r'\berroneous\b', pred_lower):
        return "Incorrect"
    
    # Check Correct patterns last (least specific)
    if re.search(r'\bcorrect\b', pred_lower):
        return "Correct"
    if re.search(r'\bright\b', pred_lower):
        return "Correct"
    if re.search(r'\bvalid\b', pred_lower):
        return "Correct"
    if re.search(r'\btrue\b', pred_lower):
        return "Correct"
    if re.search(r'\bperfect\b', pred_lower):
        return "Correct"
    if re.search(r'\bcomplete\b', pred_lower):
        return "Correct"
    if re.search(r'\baccurate\b', pred_lower):
        return "Correct"
    
    return None


def _extract_response_flexible(text: str) -> str | None:
    """Extract the classification from model response using multiple strategies."""
    if not text:
        return None
    
    text_lower = text.lower()
    text_upper = text.upper()
    
    # Strategy 1: Look for explicit classification statements with high confidence
    # Check "Almost" first as it's the most commonly missed
    explicit_patterns = [
        # Direct classification statements - Almost
        (r'\bclassification\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        (r'\bgrade\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        (r'\bcategory\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        (r'\bclassify\s+(?:this|it)\s+as\s+["\']?almost["\']?\b', "Almost"),
        (r'\bis\s+["\']?almost["\']?\b', "Almost"),
        (r'\bthis\s+is\s+["\']?almost["\']?\b', "Almost"),
        (r'\bthe\s+answer\s+is\s+["\']?almost["\']?\b', "Almost"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*almost["\']?\b', "Almost"),
        (r'\bshould\s+be\s+["\']?almost["\']?\b', "Almost"),
        (r'\bwould\s+be\s+["\']?almost["\']?\b', "Almost"),
        (r'\bevaluation\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        (r'\bverdict\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        
        # Direct classification statements - Partial
        (r'\bclassification\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bgrade\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bcategory\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bclassify\s+(?:this|it)\s+as\s+["\']?partial["\']?\b', "Partial"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*partial["\']?\b', "Partial"),
        (r'\bevaluation\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bverdict\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        
        # Direct classification statements - Incorrect
        (r'\bclassification\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bgrade\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bcategory\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*incorrect["\']?\b', "Incorrect"),
        (r'\bevaluation\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bverdict\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        
        # Direct classification statements - Correct
        (r'\bclassification\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bgrade\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bcategory\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*correct["\']?\b', "Correct"),
        (r'\bevaluation\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bverdict\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
    ]
    
    for pattern, category in explicit_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return category
    
    # Strategy 2: Try JSON extraction from <json> tags
    json_results = _extract_jsons(text)
    if json_results:
        for result in json_results:
            if isinstance(result, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            return "Correct" if val else "Incorrect"
    
    # Strategy 3: Try to find JSON in markdown code blocks
    markdown_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\n?```'
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        try:
            json_obj = json.loads(match.group(1))
            if isinstance(json_obj, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in json_obj:
                        val = json_obj[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Look for standalone category mentions at end of text or emphasized
    # Check Almost first (most commonly missed)
    almost_standalone = [
        r'(?:^|\n)\s*["\']?almost["\']?\s*[.!?]?\s*(?:$|\n)',
        r'\*\*almost\*\*',
        r'\balmost\b[.!?]?\s*$',
        r'^\s*almost\s*$',
        r'"almost"',
        r"'almost'",
        r'#\s*almost\b',
        r'\(almost\)',
        r'\[almost\]',
    ]
    for pattern in almost_standalone:
        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
            return "Almost"
    
    # Then other categories
    standalone_patterns = [
        (r'(?:^|\n)\s*["\']?partial["\']?\s*[.!?]?\s*(?:$|\n)', "Partial"),
        (r'(?:^|\n)\s*["\']?incorrect["\']?\s*[.!?]?\s*(?:$|\n)', "Incorrect"),
        (r'(?:^|\n)\s*["\']?correct["\']?\s*[.!?]?\s*(?:$|\n)', "Correct"),
        (r'\*\*partial\*\*', "Partial"),
        (r'\*\*incorrect\*\*', "Incorrect"),
        (r'\*\*correct\*\*', "Correct"),
        (r'^\s*partial\s*$', "Partial"),
        (r'^\s*incorrect\s*$', "Incorrect"),
        (r'^\s*correct\s*$', "Correct"),
        (r'"partial"', "Partial"),
        (r'"incorrect"', "Incorrect"),
        (r'"correct"', "Correct"),
        (r"'partial'", "Partial"),
        (r"'incorrect'", "Incorrect"),
        (r"'correct'", "Correct"),
    ]
    for pattern, category in standalone_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
            return category
    
    # Strategy 5: Look for direct category mentions with word boundaries
    # Check Almost first (most commonly missed)
    if re.search(r'\bALMOST\b', text_upper):
        return "Almost"
    
    # Then other categories in order
    for category in ["Partial", "Incorrect", "Correct"]:
        if re.search(rf'\b{category.upper()}\b', text_upper):
            return category
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem."""
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate the student's answer and classify it into exactly one of four categories.

## The Four Categories (CHOOSE EXACTLY ONE):

1. **Correct** (7 points): The solution is 100% perfect. No errors, no gaps, no issues whatsoever. The proof is complete and rigorous.

2. **Almost** (6 points): The solution is 90-99% complete with only 1-2 TINY issues. Examples: arithmetic error, typo, minor notation issue, one small edge case, sign error. The fix would take less than 30 seconds to explain. The core logic is sound and complete.

3. **Partial** (1-3 points): The solution shows good understanding and correct approach, but has MAJOR gaps or missing components. The student is on the right track but significant work remains. Multiple issues or missing critical steps.

4. **Incorrect** (0 points): The approach is fundamentally wrong, or the student fails to demonstrate understanding of the correct method. Wrong method, circular reasoning, or verification by example.

## CRITICAL DECISION RULES:

**Rule 1 - Correct vs Almost**: 
- If there's ANY flaw (even a tiny typo, sign error, or arithmetic mistake) → **Almost**
- Only if 100% perfect with zero issues → **Correct**
- When in doubt, choose Almost over Correct

**Rule 2 - Almost vs Partial (MOST IMPORTANT - COMMON ERROR)**:
- **Almost**: 1-2 tiny issues that are easily fixable (typo, arithmetic error, one missing edge case)
- **Partial**: 3+ issues OR any major conceptual gap OR incomplete proof structure
- Key test: "Would this solution be competition-ready with a 30-second fix?" 
  - Yes (just fix a typo/sign) → **Almost**
  - No (need to add proof steps, fix logic) → **Partial**
- **COMMON MISTAKE**: Don't classify "almost complete with minor errors" as Partial - that's Almost!

**Rule 3 - Partial vs Incorrect**:
- Does student demonstrate understanding of the correct approach/method? 
  - Yes (right idea, wrong execution or incomplete) → **Partial**
  - No (completely wrong method, no understanding) → **Incorrect**
- Key test: "Are they on the right track?" 
  - Yes → Partial
  - No → Incorrect

## Problem Statement:
```
{problem}
```

## Official Solution:
```
{solution}
```

## Grading Guidelines:
```
{grading_guidelines}
```

## Student Answer:
```
{student_answer}
```

## DETAILED EXAMPLES WITH EXPLANATIONS:

**Correct Example**:
- Student provides complete, rigorous proof with all steps correct
- No errors, no gaps, no typos, perfect notation
- Every claim is justified, all cases covered

**Almost Examples** (1-2 tiny issues, easily fixable):
- "2+2=5" in an otherwise perfect proof (one arithmetic error)
- "Correct formula but calculated 100×101/2=5051 instead of 5050" (calculation error)
- "Complete proof but forgot to check n=0 case" (one missing edge case)
- "Correct approach, one sign error: wrote -b instead of +b" (typo/sign error)
- "Proof is complete but has a minor notation inconsistency" (tiny issue)
- "Used correct method, made one arithmetic mistake in final answer" (one error)

**Partial Examples** (major gaps, incomplete but on right track):
- "Started induction correctly, set up base case, but didn't complete the inductive step" (incomplete)
- "Identified the key invariant but didn't prove it works for all cases" (major gap)
- "Correct method but missing proof of lemma 1 and lemma 2" (multiple gaps)
- "Good approach, proved 2 out of 3 required conditions" (incomplete)
- "Understood the problem, set up equations correctly, but couldn't solve them" (incomplete execution)

**Incorrect Examples** (wrong approach, no understanding):
- "Proved the statement by checking examples n=1,2,3,4,5" (verification ≠ proof)
- "Used completely wrong method that doesn't apply here" (wrong approach)
- "Assumed what needed to be proved" (circular reasoning)
- "Made up a formula that has no basis" (no understanding)
- "Proof has fundamental logical flaw" (wrong reasoning)

## STEP-BY-STEP EVALUATION PROCESS:

1. **Read the student answer completely**
2. **Compare to official solution**: What matches? What's different?
3. **Identify ALL issues** (errors, gaps, typos, missing steps)
4. **Count and categorize issues**:
   - Tiny issues: typos, arithmetic errors, sign errors, one missing edge case
   - Major issues: missing proof steps, logical gaps, incomplete reasoning
5. **Apply decision rules**:
   - Any flaw at all? → Not Correct
   - 1-2 tiny issues only? → Almost
   - 3+ issues OR major gaps? → Partial
   - Wrong approach? → Incorrect
6. **Double-check your classification**:
   - Did you miss any tiny errors that would make this "Almost" instead of "Correct"?
   - Are you sure this is "Partial" and not "Almost"? (common error!)
   - Are you sure this is "Incorrect" and not "Partial"?

## Response Format (REQUIRED):
You MUST respond with a JSON object in this exact format:

<json>
{{
    "response": "Correct" | "Almost" | "Partial" | "Incorrect"
}}
</json>

**CRITICAL REMINDERS**:
- "Almost" means ALMOST PERFECT - 1-2 tiny fixable issues only
- "Partial" means significant work remains - don't confuse with Almost!
- When the grading guidelines mention "minor mistake" or "small gap" → use **Almost**
- When the grading guidelines mention "incomplete" or "didn't complete" → use **Partial**

Output your classification now:"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using flexible extraction
        prediction = "None"
        response_text = ""
        try:
            response_text = msg_history[-1]["text"] if msg_history else ""
            extracted = _extract_response_flexible(response_text)
            if extracted:
                prediction = extracted
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                # Try one more time with normalization on the raw text
                normalized = _normalize_prediction(response_text)
                if normalized:
                    prediction = normalized
                    self.log_fn(f"Normalized prediction: {prediction}")
                else:
                    self.log_fn(f"Could not extract prediction from response: {response_text[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction against allowed categories
        allowed_categories = ["Correct", "Incorrect", "Partial", "Almost"]
        if prediction not in allowed_categories:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to None")
            prediction = "None"
        
        # Post-processing: Check if the response text contains strong indicators for "Almost"
        if prediction in ["Correct", "Partial", "None"] and response_text:
            response_lower = response_text.lower()
            strong_almost_indicators = [
                # Single tiny issues
                r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'(?:one|single|1)\s+(?:arithmetic|calculation|computation|sign)\s+(?:error|mistake)',
                r'sign\s+(?:error|mistake)',
                r'typo\b',
                r'(?:small|minor)\s+typo',
                # Almost correct patterns
                r'essentially\s+correct',
                r'essentially\s+complete',
                r'nearly\s+correct',
                r'nearly\s+complete',
                r'nearly\s+perfect',
                r'almost\s+correct',
                r'almost\s+perfect',
                r'almost\s+complete',
                r'95%\s+(?:correct|complete)',
                r'99%\s+(?:correct|complete)',
                r'mostly\s+correct',
                r'correct\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'would\s+be\s+correct\s+(?:if|with|after)',
                r'would\s+be\s+perfect',
                r'would\s+be\s+complete',
                r'could\s+be\s+correct',
                # Fix patterns
                r'just\s+needs?\s+(?:a\s+)?(?:minor|small|tiny)\s+fix',
                r'only\s+needs?\s+(?:a\s+)?(?:minor|small|tiny)\s+fix',
                r'needs\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:fix|correction)',
                r'(?:simple|trivial|minor|small)\s+(?:fix|correction)',
                # Off by patterns
                r'off\s+by\s+(?:one|1|a\s+factor)',
                r'forgot\s+(?:to|the)\s+(?:check|include|mention)',
                # Missing tiny things
                r'missing\s+(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)',
                r'missing\s+(?:one|a\s+single)\s+(?:minor|small|tiny)',
                r'missing\s+(?:one|a\s+single)\s+(?:case|step|detail)',
                r'(?:just|only)\s+missing\s+(?:a\s+)?(?:minor|small|tiny)',
                # Arithmetic/calculation errors
                r'(?:arithmetic|calculation|computation)\s+(?:error|mistake)\s+(?:only|just)',
                r'(?:small|minor|tiny)\s+(?:arithmetic|calculation|computation)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:one|a\s+single)\s+(?:error|mistake|issue)',
                r'(?:forgot|missed)\s+(?:to\s+)?(?:check|include|verify|prove)',
                r'(?:would|could)\s+be\s+(?:correct|right|valid)\s+with',
                r'(?:minor|small|tiny)\s+(?:correction|adjustment|change)',
                r'(?:simple|trivial)\s+(?:error|mistake|issue|fix)',
                r'(?:easily|quickly)\s+(?:fixed|corrected|remedied)',
                r'(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|thing|detail|part)',
                r'(?:nearly|almost|practically)\s+(?:complete|finished|done|perfect)',
                r'(?:just|only)\s+(?:a\s+)?(?:bit|little|slightly)\s+(?:off|wrong|incorrect)',
                r'(?:small|minor)\s+(?:oversight|omission|lapse)',
                r'(?:one|single)\s+(?:exception|edge\s+case|special\s+case)',
                r'(?:forgot|missed)\s+(?:the|to)\s+(?:case|condition|constraint)',
                r'(?:notation|variable|symbol)\s+(?:confusion|error|issue|inconsistency)',
                r'(?:sign|plus|minus)\s+(?:error|mistake|confusion)',
                r'off\s+by\s+(?:a\s+)?(?:factor|sign)',
                r'(?:should|needs\s+to)\s+be\s+(?:negative|positive|the\s+opposite)',
                # Minor mistake patterns from grading guidelines
                r'minor\s+mistake',
                r'minor\s+error',
                r'small\s+mistake',
                r'small\s+error',
                r'small\s+gap',
                r'minor\s+gap',
                r'tiny\s+mistake',
                r'tiny\s+error',
            ]
            for pattern in strong_almost_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Almost' indicator, changing from '{prediction}' to 'Almost'")
                    prediction = "Almost"
                    break
        
        # Check grading guidelines for Almost indicators (when prediction is Correct or Partial)
        if prediction in ["Correct", "Partial"] and grading_guidelines:
            guidelines_lower = grading_guidelines.lower()
            # Look for "(Almost)" section in grading guidelines
            if '(almost)' in guidelines_lower or 'almost)' in guidelines_lower:
                # Check if the response mentions things in the Almost section
                almost_section_match = re.search(r'\(almost\)[^\(]*', guidelines_lower)
                if almost_section_match:
                    almost_section = almost_section_match.group(0)
                    # Check for key phrases that indicate Almost
                    almost_phrases = [
                        'minor mistake', 'small gap', 'minor error', 'small mistake',
                        'almost complete', 'not completed', 'minor issue', 'small error',
                        'tiny mistake', 'tiny error', 'sign error', 'typo'
                    ]
                    for phrase in almost_phrases:
                        if phrase in almost_section and phrase in response_text.lower():
                            self.log_fn(f"Post-processing: Grading guidelines mention Almost with '{phrase}', changing to 'Almost'")
                            prediction = "Almost"
                            break
        
        # Post-processing: Check if "Incorrect" should be "Partial"
        if prediction == "Incorrect" and response_text:
            response_lower = response_text.lower()
            strong_partial_indicators = [
                # Understanding and approach indicators
                r'(?:good|correct|right)\s+(?:start|approach|direction|idea|method)',
                r'on\s+the\s+right\s+track',
                r'right\s+idea',
                r'correct\s+method',
                r'correct\s+approach',
                r'good\s+understanding',
                r'understands\s+the\s+problem',
                r'understands\s+the\s+concept',
                r'(?:significant|substantial|good|decent)\s+progress',
                # Partial work indicators
                r'partial\s+(?:solution|proof|answer|result)',
                r'incomplete\s+but\s+(?:correct|valid|good|promising)',
                r'incomplete\s+proof',
                r'partially\s+correct',
                r'started\s+(?:correctly|well)',
                r'began\s+(?:correctly|well)',
                r'set\s+up\s+(?:correctly|properly)',
                r'identified\s+(?:the\s+)?(?:key|correct|right)',
                r'found\s+(?:the\s+)?(?:correct|right|key)',
                r'correctly\s+identified',
                r'correctly\s+determined',
                r'correctly\s+proved',
                r'correctly\s+showed',
                r'correctly\s+derived',
                r'correctly\s+stated',
                r'correct\s+up\s+to',
                r'correct\s+until',
                r'valid\s+up\s+to',
                r'valid\s+until',
                # Progress indicators
                r'made\s+progress',
                r'shows\s+progress',
                r'headed\s+in\s+the\s+right\s+direction',
                r'going\s+in\s+the\s+right\s+direction',
                # Incomplete but correct approach
                r'did\s+not\s+complete',
                r'did\s+not\s+finish',
                r'incomplete\s+attempt',
                r'unfinished\s+but',
                r'missing\s+(?:some|several|many)\s+(?:steps|parts)',
                r'needs\s+(?:more|additional)\s+(?:work|steps|proof)',
                # Grading guideline patterns
                r'proved\s+that',
                r'showed\s+that',
                r'demonstrated\s+that',
                r'established\s+that',
            ]
            for pattern in strong_partial_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Partial' indicator, changing from 'Incorrect' to 'Partial'")
                    prediction = "Partial"
                    break
        
        # Check grading guidelines for Partial indicators (when prediction is Incorrect)
        if prediction == "Incorrect" and grading_guidelines:
            guidelines_lower = grading_guidelines.lower()
            # Look for "(Partial)" section in grading guidelines
            if '(partial)' in guidelines_lower:
                # Check if the response mentions achievements from the Partial section
                partial_section_match = re.search(r'\(partial\)[^\(]*', guidelines_lower)
                if partial_section_match:
                    partial_section = partial_section_match.group(0)
                    # Count how many partial criteria are mentioned in response
                    partial_criteria = re.findall(r'\d+\.\s+([^\n]+)', partial_section)
                    matches = 0
                    for criterion in partial_criteria:
                        # Extract key words from criterion
                        key_words = re.findall(r'\b(proved|showed|demonstrated|found|used|stated|identified|correctly)\b', criterion.lower())
                        for word in key_words:
                            if word in response_text.lower():
                                matches += 1
                                break
                    # If multiple partial criteria are met, change to Partial
                    if matches >= 1:
                        self.log_fn(f"Post-processing: Grading guidelines show Partial criteria met, changing to 'Partial'")
                        prediction = "Partial"
        
        # Post-processing: Check if "Partial" should be "Almost"
        if prediction == "Partial" and response_text:
            response_lower = response_text.lower()
            strong_almost_from_partial = [
                # Single tiny issues
                r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue)',
                r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'(?:one|single|1)\s+(?:arithmetic|calculation|computation|sign)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:a\s+)?(?:typo|sign\s+error)',
                r'(?:small|minor|tiny)\s+(?:arithmetic|calculation)\s+(?:error|mistake)',
                # Almost complete/correct patterns
                r'essentially\s+(?:correct|complete)',
                r'nearly\s+(?:correct|complete|perfect)',
                r'almost\s+(?:correct|perfect|complete)',
                r'95%\s+(?:correct|complete)',
                r'99%\s+(?:correct|complete)',
                r'mostly\s+(?:correct|complete)',
                r'practically\s+(?:correct|complete)',
                r'virtually\s+(?:correct|complete)',
                # Exception patterns
                r'correct\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'complete\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'would\s+be\s+(?:correct|perfect|complete)\s+(?:if|with|after)',
                r'could\s+be\s+(?:correct|perfect|complete)\s+(?:if|with|after)',
                # Fix patterns
                r'(?:just|only)\s+needs?\s+(?:a\s+)?(?:minor|small|tiny)\s+fix',
                r'needs\s+(?:just|only|a)\s+(?:minor|small|tiny)\s+fix',
                r'(?:simple|trivial|minor|small)\s+fix',
                # Character-level errors
                r'(?:one|single)\s+(?:character|letter|digit|sign|symbol)\s+(?:error|mistake)',
                r'(?:simple|trivial)\s+(?:error|mistake|fix|issue)',
                # Off by patterns
                r'off\s+by\s+(?:one|1|a\s+sign)',
                r'(?:sign|plus|minus)\s+(?:error|confusion|mistake)',
                # Minor issues
                r'(?:minor|small|tiny)\s+(?:issue|problem|flaw)',
                r'(?:just|only)\s+a\s+(?:bit|little|slightly)\s+(?:off|wrong)',
                # Almost there patterns
                r'almost\s+there',
                r'very\s+close\s+to\s+(?:correct|complete)',
                r'just\s+about\s+(?:correct|complete)',
            ]
            for pattern in strong_almost_from_partial:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Almost' indicator (was Partial), changing to 'Almost'")
                    prediction = "Almost"
                    break
        
        # Check grading guidelines for Almost indicators (when prediction is Partial)
        if prediction == "Partial" and grading_guidelines:
            guidelines_lower = grading_guidelines.lower()
            # Look for "(Almost)" section in grading guidelines
            if '(almost)' in guidelines_lower:
                almost_section_match = re.search(r'\(almost\)[^\(]*', guidelines_lower)
                if almost_section_match:
                    almost_section = almost_section_match.group(0)
                    # Check if response mentions "minor" or "small" issues
                    if 'minor' in almost_section or 'small' in almost_section:
                        if 'minor' in response_text.lower() or 'small' in response_text.lower() or 'tiny' in response_text.lower():
                            self.log_fn(f"Post-processing: Grading guidelines show Almost with minor/small issues, changing from Partial to 'Almost'")
                            prediction = "Almost"

        return str(prediction), msg_history
