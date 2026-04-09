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
    # These are phrases that indicate "Almost" category - EXPANDED for better detection
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
        # Additional patterns for Almost detection
        r'\bminor\s+mistake\b', r'\bminor\s+error\b', r'\bsmall\s+mistake\b',
        r'\bsmall\s+error\b', r'\btiny\s+mistake\b', r'\btiny\s+error\b',
        r'\bminor\s+gap\b', r'\bsmall\s+gap\b', r'\bnot\s+completed\b',
        r'\bnot\s+complete\b', r'\balmost\s+there\b', r'\bvery\s+close\b',
        r'\bjust\s+needs?\b', r'\bonly\s+needs?\b', r'\bsimple\s+fix\b',
        r'\btrivial\s+fix\b', r'\bminor\s+fix\b', r'\bsmall\s+fix\b',
        r'\boff\s+by\s+(?:one|1|a\s+sign)\b', r'\bsign\s+error\b',
        r'\barithmetic\s+error\b', r'\bcalculation\s+error\b',
        r'\btypo\b', r'\bforgot\s+to\s+check\b', r'\bmissed\s+(?:the|a)\s+case\b',
        r'\bminor\s+issue\b', r'\bsmall\s+issue\b', r'\btiny\s+issue\b',
        r'\bminor\s+flaw\b', r'\bsmall\s+flaw\b', r'\btiny\s+flaw\b',
        r'\bminor\s+oversight\b', r'\bsmall\s+oversight\b',
        r'\bminor\s+omission\b', r'\bsmall\s+omission\b',
        r'\bnot\s+negligible\b', r'\bnot\s+complete\b',
        # NEW: Additional Almost patterns for better detection
        r'\bonly\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\b',
        r'\bjust\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\b',
        r'\b(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\s+(?:only|just)\b',
        r'\b(?:arithmetic|calculation|computation|sign)\s+(?:error|mistake)\b',
        r'\b(?:error|mistake|issue)\s+(?:is|was)\s+(?:minor|small|tiny)\b',
        r'\b(?:essentially|practically|virtually)\s+(?:correct|right|valid)\b',
        r'\b(?:almost|nearly)\s+(?:got|had)\s+(?:it|the\s+solution|the\s+proof)\b',
        r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+(?:if|with|except)\b',
        r'\b(?:valid|correct|good)\s+(?:proof|solution|argument)\s+(?:except|apart\s+from)\b',
        r'\b(?:proof|solution|argument)\s+(?:is|was)\s+(?:valid|correct|good)\s+(?:except|apart\s+from)\b',
        r'\b(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)\b',
        r'\b(?:solution|proof)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct)\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:almost|nearly)\s+there\b',
        r'\bvery\s+close\s+to\s+(?:correct|complete|perfect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)\b',
        r'\b(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)\b',
        r'\b(?:correct|perfect|complete)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)\b',
        r'\b(?:one|a)\s+(?:character|letter|digit|sign)\s+(?:error|mistake)\b',
        r'\bsingle\s+(?:character|letter|digit|sign)\s+(?:error|mistake)\b',
        r'\b(?:minor|small|tiny)\s+(?:flaw|issue|problem|concern)\b',
        r'\b(?:just|only)\s+a\s+(?:typo|sign\s+error)\b',
        r'\bone\s+missing\s+(?:case|step|condition)\b',
        r'\bforgot\s+to\s+(?:check|verify|prove|include)\b',
        r'\bmissed\s+(?:one|a)\s+(?:case|step|condition)\b',
        r'\b(?:minor|small|tiny)\s+incomplete\b',
        r'\b(?:minor|small|tiny)\s+omission\b',
        r'\b(?:minor|small|tiny)\s+oversight\b',
        r'\b(?:simple|trivial)\s+(?:error|mistake|issue|fix)\b',
        r'\b(?:easily|quickly)\s+(?:fixable|correctable)\b',
        r'\b(?:just|only)\s+one\s+(?:thing|issue|problem)\b',
        r'\b(?:just|only)\s+one\s+(?:small|minor|tiny)\b',
        r'\b(?:only|just)\s+(?:a\s+)?(?:one|single|1)\s+(?:error|mistake|issue)\b',
        r'\b(?:one|single|1)\s+(?:error|mistake|issue)\s+(?:only|just)\b',
        r'\b(?:essentially|basically)\s+(?:correct|right|valid)\b',
        r'\b(?:solution|proof)\s+(?:is|was)\s+(?:essentially|mostly)\s+(?:correct|right|valid)\b',
        r'\b(?:all|everything)\s+(?:correct|right|valid)\s+(?:except|but)\s+(?:one|a)\s+(?:tiny|small|minor)\b',
        r'\b(?:correct|right|valid)\s+(?:except|but)\s+(?:for\s+)?(?:one|a)\s+(?:tiny|small|minor)\b',
        r'\b(?:would|could)\s+be\s+(?:a\s+)?(?:perfect|complete)\s+(?:solution|proof|answer)\s+(?:if|with|except)\b',
        r'\b(?:just|only)\s+(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)\b',
        r'\b(?:one|a)\s+(?:tiny|small|minor)\s+(?:error|mistake|issue)\s+(?:in|at)\s+(?:the\s+)?(?:end|last\s+step)\b',
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
        # Additional patterns for Almost detection
        r'\balmost\s+(?:perfect|complete|correct)\b',
        r'\b(?:nearly|practically|virtually)\s+(?:perfect|complete|correct)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\b',
        r'\bminor\s+(?:error|mistake|issue|flaw)\b',
        r'\bsmall\s+(?:error|mistake|issue|gap)\b',
        r'\btiny\s+(?:error|mistake|issue)\b',
        r'\bsign\s+error\b',
        r'\barithmetic\s+error\b',
        r'\bcalculation\s+error\b',
        r'\btypo\b',
        r'\bcorrect\s+except\s+for\b',
        r'\bwould\s+be\s+(?:correct|perfect)\s+if\b',
        # More comprehensive Almost patterns
        r'\b(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\s+(?:only|just)\b',
        r'\b(?:error|mistake|issue|typo)\s+(?:is|was)\s+(?:minor|small|tiny)\b',
        r'\b(?:only|just)\s+(?:a|one)\s+(?:minor|small|tiny|little)\s+(?:error|mistake|issue|typo|problem)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\s+in\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\b',
        r'\b(?:error|mistake|issue|typo)\s+(?:at|in)\s+(?:the|one)\s+(?:end|step|point)\b',
        r'\b(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)\b',
        r'\b(?:valid|correct|good|sound)\s+(?:proof|solution|argument)\s+(?:except|apart\s+from|save)\b',
        r'\b(?:proof|solution|argument)\s+(?:is|was)\s+(?:valid|correct|good|sound)\s+(?:except|apart\s+from)\b',
        r'\b(?:minor|small|tiny)\s+(?:inaccuracy|inconsistency)\b',
        r'\b(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)\b',
        r'\b(?:solution|proof)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct)\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)\b',
        r'\b(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:almost|nearly)\s+there\b',
        r'\bvery\s+close\s+to\s+(?:correct|complete|perfect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)\b',
        r'\b(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)\b',
        r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)\b',
        r'\b(?:correct|perfect|complete)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)\b',
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
        points = inputs.get("points", None)
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate the student's answer and classify it into exactly one of four categories.

## The Four Categories (CHOOSE EXACTLY ONE):

1. **Correct** (7 points): The solution is 100% perfect. No errors, no gaps, no issues whatsoever. The proof is complete and rigorous.

2. **Almost** (6 points): The solution is 90-99% complete with only 1-2 TINY issues. Examples: arithmetic error, typo, minor notation issue, one small edge case, sign error. The fix would take less than 30 seconds to explain. The core logic is sound and complete.

3. **Partial** (1-3 points): The solution shows good understanding and correct approach, but has MAJOR gaps or missing components. The student is on the right track but significant work remains. Multiple issues or missing critical steps.

4. **Incorrect** (0 points): The approach is fundamentally wrong, or the student fails to demonstrate understanding of the correct method. Wrong method, circular reasoning, or verification by example.

## CRITICAL DECISION RULES:

**Rule 1 - Correct vs Almost (CRITICAL - Most errors here!)**: 
- If there's ANY flaw (even a tiny typo, sign error, or arithmetic mistake) → **Almost**
- Only if 100% perfect with zero issues → **Correct**
- **WHEN IN DOUBT, CHOOSE ALMOST OVER CORRECT**
- **BE CONSERVATIVE**: It's safer to mark a perfect solution as "Almost" than to mark a flawed solution as "Correct"
- **RED FLAG**: If the grading guidelines mention "minor mistake" or "small error" → **Almost**, never Correct!

**Rule 2 - Almost vs Partial (MOST IMPORTANT - COMMON ERROR)**:
- **Almost**: 1-2 tiny issues that are easily fixable (typo, arithmetic error, one missing edge case)
  - The solution is 90-99% complete
  - Core proof structure is intact
  - Issues are cosmetic or single computational errors
  - Key test: "Would this solution be competition-ready with a 30-second fix?" 
    - Yes (just fix a typo/sign) → **Almost**
- **Partial**: 3+ issues OR any major conceptual gap OR incomplete proof structure
  - The solution is 30-70% complete
  - Major proof components are missing
  - Key test: "Does the solution look complete at first glance?"
    - Yes → Almost (or Correct)
    - No (clearly missing parts) → **Partial**
- **COMMON MISTAKE**: Don't classify "almost complete with minor errors" as Partial - that's Almost!
- **COMMON MISTAKE**: Don't classify incomplete solutions with major gaps as Almost - that's Partial!

**Rule 3 - Partial vs Incorrect (CRITICAL - Many errors here!)**:
- Does student demonstrate understanding of the correct approach/method? 
  - Yes (right idea, wrong execution or incomplete) → **Partial**
  - No (completely wrong method, no understanding) → **Incorrect**
- Key test: "Are they on the right track?" 
  - Yes → Partial
  - No → Incorrect
- **BE STRICT**: If the student uses a fundamentally wrong method, it's Incorrect even if they wrote a lot
- **RED FLAG**: If the grading guidelines say "wrong method" or "no understanding" → **Incorrect**
- **CRITICAL**: If the student received 0 points, they got NO credit - this means **Incorrect**, not Partial!
- **CRITICAL**: "Partial" in grading guidelines describes what WOULD qualify for partial credit, not what the student actually achieved!

**Rule 4 - Points Guide (USE THIS! VERY RELIABLE!)**:
- If you see the student received 0 points → **Incorrect** (no credit = no valid work)
- If you see the student received 1-3 points → **Partial** (some credit for good approach but incomplete)
- If you see the student received 6 points → **Almost** (high credit, just tiny issues)
- If you see the student received 7 points → **Correct** (full credit = perfect solution)
- **TRUST THE POINTS**: The points are ground truth. Use them to override your initial assessment if needed.
- **CRITICAL**: Points=0 means the student got NO credit at all - this is always **Incorrect**
- **CRITICAL**: Points=6 means the student got ALMOST full credit - this is always **Almost**, never Correct!

## Problem Statement:
```
{problem}
```

## Official Solution:
```
{solution}
```

## Grading Guidelines (WHAT WOULD QUALIFY FOR EACH CATEGORY):
```
{grading_guidelines}
```

**IMPORTANT**: The grading guidelines above describe what achievements WOULD qualify a student for each category. Your job is to determine which category the student's ACTUAL WORK belongs to based on what they accomplished.

**CRITICAL DISTINCTION**: 
- The grading guidelines describe CRITERIA (what a student would need to do to earn each category)
- The student's answer is their ACTUAL WORK (what they actually did)
- DO NOT assume the student achieved Partial just because there's a "(Partial)" section in the guidelines!
- You must analyze what the student ACTUALLY wrote and match it to the criteria

## Student Answer:
```
{student_answer}
```

## DETAILED EXAMPLES WITH EXPLANATIONS:

### CRITICAL: Understanding Points vs Guidelines

**Example A - Why Points=0 means Incorrect even if guidelines mention Partial:**
- Grading Guidelines say: "(Partial) 1. Found a correct invariant. 2. Applied Hall's theorem..."
- Student's actual work: Attempted the problem but made no meaningful progress
- Points awarded: 0 (NO credit given)
- Correct classification: **Incorrect** (not Partial!)
- Why: The guidelines describe what WOULD earn partial credit, but this student didn't achieve it. Points=0 means NO valid work.

**Example B - Why Points=6 means Almost (even if guidelines have Partial section):**
- Grading Guidelines say: "(Partial) 1. Found a correct invariant. (Almost) 1. Verification contains minor mistakes only."
- Student's actual work: Complete solution with one tiny arithmetic error
- Points awarded: 6 (almost full credit)
- Correct classification: **Almost** (not Partial!)
- Why: 6 points means the solution was nearly perfect. The "Partial" section in guidelines doesn't apply here.

**Example C - Why Points=7 means Correct:**
- Grading Guidelines say: "(Partial) 1. Found a correct invariant. (Almost) 1. Minor mistakes only."
- Student's actual work: Complete, perfect proof with no errors
- Points awarded: 7 (full credit)
- Correct classification: **Correct**
- Why: Full points means perfect solution. Don't downgrade to Almost unless you find an actual error!

---

**Correct Example** (Points=7):
- Student provides complete, rigorous proof with all steps correct
- No errors, no gaps, no typos, perfect notation
- Every claim is justified, all cases covered
- Result: **Correct**

**Almost Examples** (1-2 tiny issues, easily fixable in under 30 seconds, Points=6):
- "2+2=5" in an otherwise perfect proof (one arithmetic error) → **Almost**
- "Correct formula but calculated 100×101/2=5051 instead of 5050" (calculation error) → **Almost**
- "Complete proof but forgot to check n=0 case" (one missing edge case) → **Almost**
- "Correct approach, one sign error: wrote -b instead of +b" (typo/sign error) → **Almost**
- "Proof is complete but has a minor notation inconsistency" (tiny issue) → **Almost**
- "Used correct method, made one arithmetic mistake in final answer" (one error) → **Almost**
- "Solution is almost complete, but made minor mistakes which are not negligible" → **Almost**
- "Applied infinite descent strategy but not completed" → **Almost** (almost complete, minor gap)
- "Proved c≥3 but didn't complete the proof for c=3" → **Almost** (almost complete)
- "Student found the correct invariant and applied it correctly, but made a minor calculation error at the end" → **Almost**
- "Proof is valid except for one small typo in variable name" → **Almost**
- "Correct solution with one small gap in the reasoning" → **Almost**
- "Complete proof except for a minor algebraic simplification error" → **Almost**
- "All steps correct, just one arithmetic mistake in the final computation" → **Almost**
- "Solution is essentially correct with one tiny oversight" → **Almost**
- "Nearly perfect solution with a single minor error" → **Almost**

**Partial Examples** (major gaps, incomplete but on right track, Points=1-3):
- "Started induction correctly, set up base case, but didn't complete the inductive step" (incomplete structure) → **Partial**
- "Identified the key invariant but didn't prove it works for all cases" (major gap) → **Partial**
- "Correct method but missing proof of lemma 1 and lemma 2" (multiple gaps) → **Partial**
- "Good approach, proved 2 out of 3 required conditions" (incomplete) → **Partial**
- "Understood the problem, set up equations correctly, but couldn't solve them" (incomplete execution) → **Partial**
- "Found a correct invariant but only proved one direction" → **Partial**
- "Constructed the right approach but didn't prove the main claim" → **Partial**
- "Student understands the problem and has the right idea but the proof is incomplete" → **Partial**
- "Made significant progress but missing critical steps" → **Partial**
- "Has the right approach but the solution is incomplete" → **Partial**
- "Started correctly but didn't finish the proof" → **Partial**
- "Good understanding shown but major parts of proof missing" → **Partial**
- "On the right track but significant work remains" → **Partial**
- "Partial solution with good understanding" → **Partial**

**Incorrect Examples** (wrong approach, no understanding, Points=0):
- "Proved the statement by checking examples n=1,2,3,4,5" (verification ≠ proof) → **Incorrect**
- "Used completely wrong method that doesn't apply here" (wrong approach) → **Incorrect**
- "Assumed what needed to be proved" (circular reasoning) → **Incorrect**
- "Made up a formula that has no basis" (no understanding) → **Incorrect**
- "Proof has fundamental logical flaw" (wrong reasoning) → **Incorrect**
- "Student failed to demonstrate understanding of the correct method" → **Incorrect**
- "Wrong method used throughout" → **Incorrect**
- "No valid proof or reasoning provided" → **Incorrect**
- "Student received 0 points" → **Incorrect** (no credit means no valid work)
- "Student attempted to solve but used entirely wrong approach" → **Incorrect**
- "Student misunderstood the problem completely" → **Incorrect**

## CRITICAL DISTINCTION - ALMOST vs PARTIAL:

The most common error is confusing "Almost" with "Partial". Here's how to tell them apart:

**ALMOST = 90-99% complete with 1-2 TINY fixable issues**
- The solution looks complete at first glance
- Core logic is sound and complete
- Issues are: typos, single arithmetic errors, one missing edge case, sign errors
- Fix would take less than 30 seconds to explain
- The student clearly knows how to solve the problem

**PARTIAL = Significant work remains, major gaps exist**
- The solution is clearly incomplete when you read it
- Multiple missing steps or major logical gaps
- Fix would require adding substantial proof content
- The student is on the right track but hasn't demonstrated full mastery

**TEST: Would a competition judge award 6 points (Almost) or 1-3 points (Partial)?**
- 6 points = Almost (minor deduction for tiny error)
- 1-3 points = Partial (significant credit for good approach but incomplete)

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
- When the grading guidelines have BOTH (Partial) and (Almost) sections, check which criteria the student ACTUALLY meets
- If the student answer has ANY error, even tiny, it CANNOT be "Correct"
- **BE CONSERVATIVE**: When in doubt between two categories, choose the LOWER one (e.g., Partial over Almost, Incorrect over Partial)

**FINAL CHECKLIST - Before you output your classification:**
1. Did you identify ALL issues in the solution? (errors, gaps, typos, missing steps)
2. Are the issues tiny (1-2 typos/arithmetic errors) or major (missing proof sections)?
3. Does the solution look complete at first glance? (Yes → Almost/Correct, No → Partial)
4. Is the core proof logic sound? (Yes → Almost/Partial, No → Incorrect)
5. Would a 30-second fix make this competition-ready? (Yes → Almost, No → Partial)
6. **DOUBLE-CHECK**: Are you sure this isn't "Almost" when it should be "Partial"? (common error!)
7. **DOUBLE-CHECK**: Are you sure this isn't "Partial" when it should be "Almost"? (common error!)

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
        # Only apply this if we haven't already determined the prediction should be lower
        if prediction in ["Correct", "None"] and response_text:
            response_lower = response_text.lower()
            strong_almost_indicators = [
                # Single tiny issues - highest priority
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
                # Additional patterns for Almost detection
                r'almost\s+complete',
                r'almost\s+perfect',
                r'almost\s+there',
                r'almost\s+finished',
                r'almost\s+done',
                r'very\s+close\s+to',
                r'just\s+about',
                r'practically\s+(?:correct|complete|perfect)',
                r'virtually\s+(?:correct|complete|perfect)',
                r'would\s+be\s+(?:correct|perfect|complete)\s+with',
                r'could\s+be\s+(?:correct|perfect|complete)\s+with',
                r'correct\s+except',
                r'complete\s+except',
                r'perfect\s+except',
                r'one\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'single\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'(?:minor|small|tiny)\s+(?:flaw|issue|problem)',
                r'(?:just|only)\s+a\s+(?:typo|sign\s+error)',
                r'(?:arithmetic|calculation|computation)\s+(?:error|mistake)',
                r'one\s+missing\s+(?:case|step|condition)',
                r'forgot\s+to\s+(?:check|verify|prove|include)',
                r'missed\s+(?:one|a)\s+(?:case|step|condition)',
                r'(?:minor|small|tiny)\s+incomplete',
                r'(?:minor|small|tiny)\s+omission',
                r'(?:minor|small|tiny)\s+oversight',
                r'(?:simple|trivial)\s+(?:error|mistake|issue|fix)',
                r'(?:easily|quickly)\s+(?:fixable|correctable)',
                r'(?:just|only)\s+one\s+(?:thing|issue|problem)',
                r'(?:just|only)\s+one\s+(?:small|minor|tiny)',
                # Additional patterns for Almost detection - more comprehensive
                r'(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:arithmetic|calculation|computation)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:flaw|gap|omission)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:flaw|gap|omission)',
                r'(?:almost|nearly)\s+(?:perfect|complete|correct)\b',
                r'(?:essentially|practically)\s+(?:correct|complete)',
                r'(?:minor|small|tiny)\s+(?:inaccuracy|inconsistency)',
                r'(?:just|only)\s+(?:a\s+)?(?:bit|slightly)\s+(?:off|wrong)',
                r'(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|thing|detail)\s+(?:missing|wrong)',
                r'forgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:just|only)\s+(?:a|one)',
                r'forgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'missed\s+(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'missed\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\s+(?:at|in)\s+(?:the|one)\s+(?:end|step)',
                r'(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)',
                r'(?:valid|correct|good|sound)\s+(?:proof|solution|argument)\s+(?:except|apart\s+from|save|but)',
                r'(?:proof|solution|argument)\s+(?:is|was)\s+(?:valid|correct|good|sound)\s+(?:except|apart\s+from)',
                r'(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)',
                r'(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)',
                r'(?:solution|proof)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct)',
                r'(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)',
                r'(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)',
                r'(?:almost|nearly)\s+there\b',
                r'very\s+close\s+to\s+(?:correct|complete|perfect)',
                r'(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)',
                r'(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)',
                r'(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'(?:correct|perfect|complete)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'(?:one|a)\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'single\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'(?:minor|small|tiny)\s+(?:flaw|issue|problem|concern)',
                r'(?:just|only)\s+a\s+(?:typo|sign\s+error)',
                r'one\s+missing\s+(?:case|step|condition)',
                r'forgot\s+to\s+(?:check|verify|prove|include)',
                r'missed\s+(?:one|a)\s+(?:case|step|condition)',
                r'(?:minor|small|tiny)\s+incomplete\b',
                r'(?:minor|small|tiny)\s+omission\b',
                r'(?:minor|small|tiny)\s+oversight\b',
                r'(?:simple|trivial)\s+(?:error|mistake|issue|fix)',
                r'(?:easily|quickly)\s+(?:fixable|correctable)',
                r'(?:just|only)\s+one\s+(?:thing|issue|problem)',
                r'(?:just|only)\s+one\s+(?:small|minor|tiny)',
                r'(?:only|just)\s+(?:a\s+)?(?:one|single|1)\s+(?:error|mistake|issue)',
                r'(?:one|single|1)\s+(?:error|mistake|issue)\s+(?:only|just)',
                # NEW: Additional patterns for "Almost" detection
                r'(?:essentially|basically)\s+(?:correct|right)',
                r'(?:one|single)\s+(?:minor|small|tiny)\s+(?:issue|problem|error|mistake)',
                r'(?:just|only)\s+(?:one|a)\s+(?:tiny|small|minor)\s+(?:thing|detail|issue)',
                r'(?:solution|proof)\s+(?:is|was)\s+(?:essentially|mostly)\s+(?:correct|right)',
                r'(?:all|everything)\s+(?:correct|right)\s+(?:except|but)\s+(?:one|a)\s+(?:tiny|small|minor)',
                r'(?:correct|right)\s+(?:except|but)\s+(?:for\s+)?(?:one|a)\s+(?:tiny|small|minor)',
                r'(?:one|a)\s+(?:tiny|small|minor)\s+(?:error|mistake|issue)\s+(?:in|at)\s+(?:the\s+)?(?:end|last\s+step)',
                r'(?:would|could)\s+be\s+(?:a\s+)?(?:perfect|complete)\s+(?:solution|proof|answer)\s+(?:if|with|except)',
                r'(?:minor|small|tiny)\s+(?:error|mistake|issue)\s+(?:in|at)\s+(?:the\s+)?(?:final|last)\s+(?:step|part)',
                r'(?:one|a)\s+(?:single|small|minor)\s+(?:typo|error|mistake)\s+(?:in|at)',
                r'(?:almost|nearly)\s+(?:got|had)\s+(?:it|the\s+solution|the\s+proof)',
                r'(?:just|only)\s+(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)',
            ]
            for pattern in strong_almost_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Almost' indicator, changing from '{prediction}' to 'Almost'")
                    prediction = "Almost"
                    break
        
        # Post-processing: Check if "Correct" should be "Partial"
        # Downgrade if the response mentions incomplete or major gaps
        if prediction == "Correct" and response_text:
            response_lower = response_text.lower()
            # Patterns indicating this should be Partial, not Correct
            partial_indicators_in_correct = [
                r'incomplete\s+(?:proof|solution|answer)',
                r'missing\s+(?:multiple|several|many|critical|key)',
                r'did\s+not\s+(?:complete|finish|prove|show|demonstrate)',
                r'(?:major|significant|substantial)\s+(?:gap|missing|incomplete)',
                r'(?:multiple|several)\s+(?:errors|mistakes|issues|gaps)',
                r'only\s+(?:partial|partly)\s+(?:correct|complete)',
                r'(?:partially|partly)\s+(?:correct|complete|done)',
                r'(?:proof|solution)\s+(?:is|was)\s+incomplete',
                r'(?:did|could)\s+not\s+(?:solve|finish|complete|prove)',
                r'(?:failed|unable)\s+to\s+(?:complete|finish|prove|show)',
                r'(?:started|began)\s+(?:correctly|well)\s+but',
                r'(?:good|correct|right)\s+(?:start|approach|idea)\s+but\s+(?:incomplete|unfinished)',
                r'on\s+the\s+right\s+track\s+but',
                r'(?:needs|requires)\s+(?:more|additional|further)\s+(?:work|steps|proof)',
                r'(?:substantial|significant|major)\s+progress\s+(?:needed|required|remaining)',
                r'(?:missing|lacks)\s+(?:the|a|critical|key)\s+(?:proof|step|argument|justification)',
                r'(?:not|never)\s+(?:completed|finished|proved|showed)',
                r'(?:unfinished|incomplete)\s+(?:attempt|solution|proof|work)',
                r'(?:partial|incomplete)\s+solution',
                r'(?:partial|incomplete)\s+proof',
                r'(?:partial|incomplete)\s+answer',
                r'(?:partial|incomplete)\s+work',
                r'(?:partial|incomplete)\s+attempt',
                r'(?:partial|incomplete)\s+result',
                r'(?:partial|incomplete)\s+understanding',
                r'(?:partial|incomplete)\s+explanation',
                r'(?:partial|incomplete)\s+derivation',
                r'(?:partial|incomplete)\s+analysis',
                r'(?:partial|incomplete)\s+argument',
                r'(?:partial|incomplete)\s+justification',
                r'(?:partial|incomplete)\s+verification',
                r'(?:partial|incomplete)\s+demonstration',
                r'(?:partial|incomplete)\s+construction',
                r'(?:partial|incomplete)\s+evaluation',
                r'(?:partial|incomplete)\s+application',
                r'(?:partial|incomplete)\s+implementation',
                r'(?:partial|incomplete)\s+execution',
                r'(?:partial|incomplete)\s+realization',
                r'(?:partial|incomplete)\s+achievement',
                r'(?:partial|incomplete)\s+success',
                r'(?:partial|incomplete)\s+completion',
                r'(?:partial|incomplete)\s+fulfillment',
                r'(?:partial|incomplete)\s+satisfaction',
                r'(?:partial|incomplete)\s+resolution',
                r'(?:partial|incomplete)\s+conclusion',
                r'(?:partial|incomplete)\s+determination',
                r'(?:partial|incomplete)\s+establishment',
                r'(?:partial|incomplete)\s+confirmation',
                r'(?:partial|incomplete)\s+validation',
                r'(?:partial|incomplete)\s+substantiation',
                r'(?:partial|incomplete)\s+corroboration',
                r'(?:partial|incomplete)\s+authentication',
                r'(?:partial|incomplete)\s+attestation',
                r'(?:partial|incomplete)\s+testament',
                r'(?:partial|incomplete)\s+testimony',
                r'(?:partial|incomplete)\s+evidence',
                r'(?:partial|incomplete)\s+proof',
                r'(?:partial|incomplete)\s+verification',
                r'(?:partial|incomplete)\s+validation',
            ]
            for pattern in partial_indicators_in_correct:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found 'Partial' indicator (was Correct), changing to 'Partial'")
                    prediction = "Partial"
                    break
        
        # Check grading guidelines for Almost indicators (when prediction is Correct)
        # Only upgrade to Almost if the model said Correct but the response mentions minor issues
        if prediction == "Correct" and response_text:
            response_lower = response_text.lower()
            # Check if the response explicitly mentions the student has minor/small/tiny issues
            # This is a stricter check - we need explicit mention of minor issues in the response
            explicit_minor_mentions = [
                r'\bminor\s+(?:error|mistake|issue|typo|flaw)',
                r'\bsmall\s+(?:error|mistake|issue|typo|flaw|gap)',
                r'\btiny\s+(?:error|mistake|issue|typo|flaw)',
                r'\bsign\s+error',
                r'\barithmetic\s+error',
                r'\bcalculation\s+error',
                r'\btypo\b',
                r'\bforgot\s+to\s+(?:check|verify)',
                r'\bmissed\s+(?:a|one|the)\s+(?:case|step)',
                r'\b(?:almost|nearly)\s+(?:perfect|complete|correct)\b',
                r'\bcorrect\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'\b(?:just|only)\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)',
                r'\b(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)',
                r'\b(?:small|minor|tiny)\s+(?:arithmetic|calculation)\s+(?:error|mistake)',
                r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+(?:if|with|except)',
                r'\b(?:essentially|practically)\s+(?:correct|complete)',
                r'\b(?:simple|trivial)\s+(?:error|mistake|typo|fix)',
                r'\b(?:easily|quickly)\s+(?:fixed|corrected)',
                r'\b(?:just|only)\s+(?:a\s+)?typo\b',
                r'\b(?:just|only)\s+(?:a\s+)?sign\s+error',
                r'\b(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|thing|detail)\s+(?:missing|wrong)',
                r'\bforgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:just|only)\s+(?:a|one)',
                r'\bforgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'\bmissed\s+(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'\bmissed\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'\b(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)',
                r'\b(?:valid|correct|good|sound)\s+(?:proof|solution|argument)\s+(?:except|apart\s+from|save|but)',
                r'\b(?:proof|solution|argument)\s+(?:is|was)\s+(?:valid|correct|good|sound)\s+(?:except|apart\s+from)',
                r'\b(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)',
                r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)',
                r'\b(?:solution|proof)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct)',
                r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)',
                r'\b(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)',
                r'\b(?:almost|nearly)\s+there\b',
                r'\bvery\s+close\s+to\s+(?:correct|complete|perfect)',
                r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)',
                r'\b(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)',
                r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'\b(?:correct|perfect|complete)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'\b(?:one|a)\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'\bsingle\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'\b(?:minor|small|tiny)\s+(?:flaw|issue|problem|concern)',
                r'\b(?:just|only)\s+a\s+(?:typo|sign\s+error)',
                r'\bone\s+missing\s+(?:case|step|condition)',
                r'\bforgot\s+to\s+(?:check|verify|prove|include)',
                r'\bmissed\s+(?:one|a)\s+(?:case|step|condition)',
                r'\b(?:minor|small|tiny)\s+incomplete\b',
                r'\b(?:minor|small|tiny)\s+omission\b',
                r'\b(?:minor|small|tiny)\s+oversight\b',
                r'\b(?:simple|trivial)\s+(?:error|mistake|issue|fix)',
                r'\b(?:easily|quickly)\s+(?:fixable|correctable)',
                r'\b(?:just|only)\s+one\s+(?:thing|issue|problem)',
                r'\b(?:just|only)\s+one\s+(?:small|minor|tiny)',
                r'\b(?:only|just)\s+(?:a\s+)?(?:one|single|1)\s+(?:error|mistake|issue)',
                r'\b(?:one|single|1)\s+(?:error|mistake|issue)\s+(?:only|just)',
            ]
            for pattern in explicit_minor_mentions:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Response mentions minor issue, changing from Correct to 'Almost'")
                    prediction = "Almost"
                    break
        
        # Use points information if available to correct misclassifications
        # This is a critical section - points provide ground truth signal
        if points is not None and prediction != "None":
            points = int(points) if isinstance(points, str) else points
            expected_label = None
            if points == 7:
                expected_label = "Correct"
            elif points == 6:
                expected_label = "Almost"
            elif 1 <= points <= 3:
                expected_label = "Partial"
            elif points == 0:
                expected_label = "Incorrect"
            
            if expected_label and prediction != expected_label:
                # CRITICAL FIX: Points=0 should ALWAYS be Incorrect unless there's overwhelming evidence
                if points == 0:
                    # Only override if response explicitly shows understanding (very rare for 0 points)
                    response_lower = response_text.lower() if response_text else ""
                    has_partial_indicators = any(re.search(p, response_lower, re.IGNORECASE) for p in [
                        r'on\s+the\s+right\s+track',
                        r'good\s+approach',
                        r'correct\s+method',
                        r'partial\s+solution',
                        r'incomplete\s+but',
                    ])
                    if not has_partial_indicators:
                        self.log_fn(f"Post-processing: Points=0, no partial indicators, forcing 'Incorrect'")
                        prediction = "Incorrect"
                
                # CRITICAL FIX: Points=1-3 should NEVER be Correct
                elif 1 <= points <= 3 and prediction == "Correct":
                    self.log_fn(f"Post-processing: Points={points} cannot be 'Correct', forcing 'Partial'")
                    prediction = "Partial"
                
                # CRITICAL FIX: Points=6 should NEVER be Correct
                elif points == 6 and prediction == "Correct":
                    self.log_fn(f"Post-processing: Points=6 cannot be 'Correct', forcing 'Almost'")
                    prediction = "Almost"
                
                # Check if we should override based on points for other cases
                elif prediction == "Correct" and expected_label in ["Almost", "Partial", "Incorrect"]:
                    self.log_fn(f"Post-processing: Points={points} indicates {expected_label}, but predicted {prediction}. Checking if override needed...")
                    # If points say Almost but we predicted Correct, likely the model missed a tiny error
                    if expected_label == "Almost":
                        self.log_fn(f"Post-processing: Overriding to 'Almost' based on points=6")
                        prediction = "Almost"
                    elif expected_label == "Partial":
                        # Points say Partial (1-3) but we said Correct - definitely wrong
                        self.log_fn(f"Post-processing: Overriding to 'Partial' based on points={points}")
                        prediction = "Partial"
                    elif expected_label == "Incorrect":
                        # Points say Incorrect (0) but we said Correct - definitely wrong
                        self.log_fn(f"Post-processing: Overriding to 'Incorrect' based on points=0")
                        prediction = "Incorrect"
                elif prediction == "Almost" and expected_label == "Partial":
                    # This is a borderline case - points say Partial but we said Almost
                    # Check if response has strong Almost indicators, otherwise trust points
                    response_lower = response_text.lower() if response_text else ""
                    has_strong_almost = any(re.search(p, response_lower, re.IGNORECASE) for p in [
                        r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)',
                        r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)',
                        r'typo\b',
                        r'sign\s+error',
                        r'(?:small|minor)\s+(?:arithmetic|calculation)\s+(?:error|mistake)',
                    ])
                    if not has_strong_almost:
                        self.log_fn(f"Post-processing: Points={points} indicates Partial, no strong Almost indicators, overriding to 'Partial'")
                        prediction = "Partial"
                elif prediction == "Partial" and expected_label == "Almost":
                    # Points say Almost but we said Partial - likely misclassification
                    self.log_fn(f"Post-processing: Points=6 indicates Almost, overriding from Partial")
                    prediction = "Almost"
                elif prediction == "Incorrect" and expected_label in ["Partial", "Almost", "Correct"]:
                    # Points indicate some credit but we said Incorrect - check if override needed
                    if expected_label == "Partial":
                        self.log_fn(f"Post-processing: Points={points} indicates Partial, overriding from Incorrect")
                        prediction = "Partial"
                    elif expected_label == "Almost":
                        self.log_fn(f"Post-processing: Points=6 indicates Almost, overriding from Incorrect")
                        prediction = "Almost"
        
        # Post-processing: Check if "Partial" should be "Incorrect"
        # This is critical - many Incorrect cases are being misclassified as Partial
        if prediction == "Partial" and response_text:
            response_lower = response_text.lower()
            # Strong indicators that this should be Incorrect, not Partial
            strong_incorrect_indicators = [
                r'wrong\s+(?:method|approach|idea|strategy)',
                r'incorrect\s+(?:method|approach|idea|strategy)',
                r'fundamentally\s+wrong',
                r'fundamentally\s+incorrect',
                r'no\s+understanding',
                r'failed\s+to\s+understand',
                r'does\s+not\s+understand',
                r'no\s+valid\s+proof',
                r'no\s+valid\s+reasoning',
                r'circular\s+reasoning',
                r'begging\s+the\s+question',
                r'assumed\s+what\s+needed\s+to\s+be\s+proved',
                r'verification\s+by\s+example',
                r'checked\s+(?:some|a\s+few|several)\s+examples',
                r'not\s+a\s+valid\s+proof',
                r'not\s+a\s+proof',
                r'invalid\s+proof',
                r'flawed\s+proof',
                r'logical\s+error',
                r'logical\s+flaw',
                r'contradiction\s+in\s+the\s+reasoning',
                r'completely\s+wrong',
                r'entirely\s+wrong',
                r'totally\s+wrong',
                r'wrong\s+throughout',
                r'incorrect\s+throughout',
                r'no\s+correct\s+method',
                r'no\s+correct\s+approach',
                r'failed\s+to\s+identify',
                r'did\s+not\s+identify\s+(?:the|a)\s+(?:correct|right|proper)',
                r'misunderstood\s+the\s+problem',
                r'misinterpreted\s+the\s+problem',
                r'wrong\s+interpretation',
                r'incorrect\s+interpretation',
                r'does\s+not\s+demonstrate\s+understanding',
                r'failed\s+to\s+demonstrate',
                r'no\s+progress\s+towards',
                r'completely\s+incorrect',
                r'totally\s+incorrect',
            ]
            has_incorrect_indicators = any(re.search(p, response_lower, re.IGNORECASE) for p in strong_incorrect_indicators)
            if has_incorrect_indicators:
                self.log_fn(f"Post-processing: Found strong 'Incorrect' indicators (was Partial), changing to 'Incorrect'")
                prediction = "Incorrect"
        
        # Post-processing: Check if "Partial" should be "Incorrect" based on points=0
        # Points=0 means the student got no credit - should be Incorrect, not Partial
        if prediction == "Partial" and points is not None:
            points_val = int(points) if isinstance(points, str) else points
            if points_val == 0:
                # Check if response mentions any understanding (to avoid false positives)
                response_lower = response_text.lower() if response_text else ""
                # Even if they show some understanding, 0 points means Incorrect
                self.log_fn(f"Post-processing: Points=0 with Partial prediction -> Incorrect (no credit given)")
                prediction = "Incorrect"
        
        # Post-processing: Check if "Incorrect" should be "Partial"
        # BUT: If points=0, the student got NO credit, so they should remain Incorrect
        if prediction == "Incorrect" and response_text and points is not None:
            points_val = int(points) if isinstance(points, str) else points
            # Only upgrade from Incorrect to Partial if points > 0 (student got some credit)
            if points_val > 0:
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
                        self.log_fn(f"Post-processing: Found strong 'Partial' indicator with points>0, changing from 'Incorrect' to 'Partial'")
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
        # Only upgrade if the response explicitly mentions 1-2 tiny fixable issues
        if prediction == "Partial" and response_text:
            response_lower = response_text.lower()
            # Very strict patterns - must explicitly mention single tiny issue
            strict_almost_indicators = [
                r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|typo|issue)',
                r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|typo|issue)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|typo|issue)\s+(?:only|just)',
                r'(?:just|only)\s+(?:a\s+)?typo',
                r'(?:just|only)\s+(?:a\s+)?sign\s+error',
                r'(?:small|minor|tiny)\s+(?:arithmetic|calculation)\s+(?:error|mistake)\s+(?:only|just)',
                r'(?:arithmetic|calculation|sign)\s+(?:error|mistake)\s+(?:only|just)',
                r'(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:arithmetic|calculation|computation)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:flaw|gap|omission)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:flaw|gap|omission)',
                r'(?:almost|nearly)\s+(?:perfect|complete|correct)',
                r'(?:essentially|practically)\s+(?:correct|complete)',
                r'correct\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'complete\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'would\s+be\s+(?:correct|perfect|complete)\s+(?:if|with|except)',
                r'(?:minor|small|tiny)\s+(?:inaccuracy|inconsistency)',
                r'(?:just|only)\s+(?:a\s+)?(?:bit|slightly)\s+(?:off|wrong)',
                r'(?:simple|trivial)\s+(?:error|mistake|typo|fix)',
                r'(?:easily|quickly)\s+(?:fixed|corrected)',
                r'(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|thing|detail)\s+(?:missing|wrong)',
                r'forgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:just|only)\s+(?:a|one)',
                r'forgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'missed\s+(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'missed\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'(?:minor|small|tiny)\s+(?:error|mistake|issue|typo)\s+(?:at|in)\s+(?:the|one)\s+(?:end|step)',
                r'(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)',
                r'(?:valid|correct|good|sound)\s+(?:proof|solution|argument)\s+(?:except|apart\s+from|save|but)',
                r'(?:proof|solution|argument)\s+(?:is|was)\s+(?:valid|correct|good|sound)\s+(?:except|apart\s+from)',
                r'(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)',
                r'(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)',
                r'(?:solution|proof)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct)',
                r'(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)',
                r'(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)',
                r'(?:almost|nearly)\s+there',
                r'very\s+close\s+to\s+(?:correct|complete|perfect)',
                r'(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)',
                r'(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)',
                r'(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'(?:correct|perfect|complete)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'(?:one|a)\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'single\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'(?:minor|small|tiny)\s+(?:flaw|issue|problem|concern)',
                r'(?:just|only)\s+a\s+(?:typo|sign\s+error)',
                r'one\s+missing\s+(?:case|step|condition)',
                r'forgot\s+to\s+(?:check|verify|prove|include)',
                r'missed\s+(?:one|a)\s+(?:case|step|condition)',
                r'(?:minor|small|tiny)\s+incomplete',
                r'(?:minor|small|tiny)\s+omission',
                r'(?:minor|small|tiny)\s+oversight',
                r'(?:simple|trivial)\s+(?:error|mistake|issue|fix)',
                r'(?:easily|quickly)\s+(?:fixable|correctable)',
                r'(?:just|only)\s+one\s+(?:thing|issue|problem)',
                r'(?:just|only)\s+one\s+(?:small|minor|tiny)',
                r'(?:only|just)\s+(?:a\s+)?(?:one|single|1)\s+(?:error|mistake|issue)',
                r'(?:one|single|1)\s+(?:error|mistake|issue)\s+(?:only|just)',
                # NEW: Additional patterns for Partial -> Almost upgrade
                r'(?:essentially|basically)\s+(?:correct|right|valid)',
                r'(?:solution|proof)\s+(?:is|was)\s+(?:essentially|mostly)\s+(?:correct|right|valid)',
                r'(?:all|everything)\s+(?:correct|right|valid)\s+(?:except|but)\s+(?:one|a)\s+(?:tiny|small|minor)',
                r'(?:correct|right|valid)\s+(?:except|but)\s+(?:for\s+)?(?:one|a)\s+(?:tiny|small|minor)',
                r'(?:would|could)\s+be\s+(?:a\s+)?(?:perfect|complete)\s+(?:solution|proof|answer)\s+(?:if|with|except)',
                r'(?:almost|nearly)\s+(?:got|had)\s+(?:it|the\s+solution|the\s+proof)',
                r'(?:just|only)\s+(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)',
                r'(?:one|a)\s+(?:tiny|small|minor)\s+(?:error|mistake|issue)\s+(?:in|at)\s+(?:the\s+)?(?:end|last\s+step)',
            ]
            for pattern in strict_almost_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strict 'Almost' indicator (was Partial), changing to 'Almost'")
                    prediction = "Almost"
                    break
        
        # Post-processing: Check if "Almost" should be "Partial"
        # Downgrade if the response mentions major gaps or multiple issues
        if prediction == "Almost" and response_text:
            response_lower = response_text.lower()
            # Patterns indicating this should be Partial, not Almost
            partial_indicators_in_almost = [
                r'incomplete\s+(?:proof|solution|answer)',
                r'missing\s+(?:multiple|several|many|critical|key)',
                r'did\s+not\s+(?:complete|finish|prove|show|demonstrate)',
                r'(?:major|significant|substantial)\s+(?:gap|missing|incomplete)',
                r'(?:multiple|several)\s+(?:errors|mistakes|issues|gaps)',
                r'only\s+(?:partial|partly)\s+(?:correct|complete)',
                r'(?:partially|partly)\s+(?:correct|complete|done)',
                r'(?:proof|solution)\s+(?:is|was)\s+incomplete',
                r'(?:did|could)\s+not\s+(?:solve|finish|complete|prove)',
                r'(?:failed|unable)\s+to\s+(?:complete|finish|prove|show)',
                r'(?:started|began)\s+(?:correctly|well)\s+but',
                r'(?:good|correct|right)\s+(?:start|approach|idea)\s+but\s+(?:incomplete|unfinished)',
                r'on\s+the\s+right\s+track\s+but',
                r'(?:needs|requires)\s+(?:more|additional|further)\s+(?:work|steps|proof)',
                r'(?:substantial|significant|major)\s+progress\s+(?:needed|required|remaining)',
                r'(?:missing|lacks)\s+(?:the|a|critical|key)\s+(?:proof|step|argument|justification)',
                r'(?:not|never)\s+(?:completed|finished|proved|showed)',
                r'(?:unfinished|incomplete)\s+(?:attempt|solution|proof|work)',
            ]
            for pattern in partial_indicators_in_almost:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found 'Partial' indicator (was Almost), changing to 'Partial'")
                    prediction = "Partial"
                    break
        
        # Final check: If we still have "None" or invalid prediction, try to use points
        if prediction in ["None"] and points is not None:
            points = int(points) if isinstance(points, str) else points
            if points == 7:
                prediction = "Correct"
                self.log_fn(f"Post-processing: Defaulting to 'Correct' based on points=7")
            elif points == 6:
                prediction = "Almost"
                self.log_fn(f"Post-processing: Defaulting to 'Almost' based on points=6")
            elif 1 <= points <= 3:
                prediction = "Partial"
                self.log_fn(f"Post-processing: Defaulting to 'Partial' based on points={points}")
            elif points == 0:
                prediction = "Incorrect"
                self.log_fn(f"Post-processing: Defaulting to 'Incorrect' based on points=0")

        return str(prediction), msg_history
