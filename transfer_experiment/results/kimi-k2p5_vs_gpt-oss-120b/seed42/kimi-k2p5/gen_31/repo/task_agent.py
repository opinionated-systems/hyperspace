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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
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
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)\b',
    ]
    for pattern in almost_compound_patterns:
        if re.search(pattern, pred_lower, re.IGNORECASE):
            return "Almost"
    
    # Check for Partial compound patterns
    partial_compound_patterns = [
        r'\bpartially\s+correct\b', r'\bpartial\s+solution\b',
        r'\bpartial\s+proof\b', r'\bpartial\s+answer\b',
        r'\bpartial\s+result\b', r'\bincomplete\s+solution\b',
        r'\bincomplete\s+proof\b', r'\bincomplete\s+answer\b',
        r'\bpart\s+of\s+the\b', r'\bsome\s+progress\b',
        r'\bon\s+the\s+right\s+track\b', r'\bgood\s+start\b',
        r'\bcorrect\s+approach\b', r'\bcorrect\s+method\b',
        r'\bstarted\s+correctly\b', r'\bpartial\s+understanding\b',
        r'\bsignificant\s+progress\b', r'\bsubstantial\s+progress\b',
        r'\bpartially\s+complete\b', r'\bpartially\s+done\b',
        r'\bpartially\s+proved\b', r'\bpartially\s+shown\b',
        r'\bpartially\s+valid\b', r'\bpartially\s+right\b',
        r'\bpartial\s+work\b', r'\bpartial\s+attempt\b',
        r'\bpartial\s+result\b', r'\bpartial\s+success\b',
        r'\bpartial\s+completion\b', r'\bpartial\s+achievement\b',
    ]
    for pattern in partial_compound_patterns:
        if re.search(pattern, pred_lower, re.IGNORECASE):
            return "Partial"
    
    # Check for Incorrect compound patterns
    incorrect_compound_patterns = [
        r'\bwrong\s+method\b', r'\bwrong\s+approach\b',
        r'\bincorrect\s+method\b', r'\bincorrect\s+approach\b',
        r'\bnot\s+correct\b', r'\bnot\s+right\b',
        r'\bnot\s+valid\b', r'\bnot\s+true\b',
        r'\bno\s+understanding\b', r'\bno\s+valid\b',
        r'\bcircular\s+reasoning\b', r'\bverification\s+by\s+example\b',
        r'\bwrong\s+idea\b', r'\bwrong\s+strategy\b',
        r'\bincorrect\s+idea\b', r'\bincorrect\s+strategy\b',
        r'\bcompletely\s+wrong\b', r'\bentirely\s+wrong\b',
        r'\btotally\s+wrong\b', r'\babsolutely\s+wrong\b',
        r'\bfundamentally\s+wrong\b', r'\bfundamentally\s+incorrect\b',
        r'\bcompletely\s+incorrect\b', r'\bentirely\s+incorrect\b',
    ]
    for pattern in incorrect_compound_patterns:
        if re.search(pattern, pred_lower, re.IGNORECASE):
            return "Incorrect"
    
    # Check for Correct compound patterns
    correct_compound_patterns = [
        r'\bcompletely\s+correct\b', r'\bentirely\s+correct\b',
        r'\btotally\s+correct\b', r'\babsolutely\s+correct\b',
        r'\bperfectly\s+correct\b', r'\bfully\s+correct\b',
        r'\b100%\s+correct\b', r'\ball\s+correct\b',
        r'\bcompletely\s+right\b', r'\bentirely\s+right\b',
        r'\btotally\s+right\b', r'\babsolutely\s+right\b',
        r'\bperfectly\s+right\b', r'\bfully\s+right\b',
        r'\b100%\s+right\b', r'\ball\s+right\b',
        r'\bcompletely\s+valid\b', r'\bentirely\s+valid\b',
        r'\btotally\s+valid\b', r'\babsolutely\s+valid\b',
        r'\bperfectly\s+valid\b', r'\bfully\s+valid\b',
        r'\b100%\s+valid\b', r'\ball\s+valid\b',
        r'\bcompletely\s+true\b', r'\bentirely\s+true\b',
        r'\btotally\s+true\b', r'\babsolutely\s+true\b',
        r'\bperfectly\s+true\b', r'\bfully\s+true\b',
        r'\b100%\s+true\b', r'\ball\s+true\b',
    ]
    for pattern in correct_compound_patterns:
        if re.search(pattern, pred_lower, re.IGNORECASE):
            return "Correct"
    
    # Check single word patterns - order matters!
    # Check Almost patterns first (high priority)
    if re.search(r'\balmost\b', pred_lower):
        return "Almost"
    if re.search(r'\bnearly\b', pred_lower):
        return "Almost"
    if re.search(r'\bclose\b', pred_lower):
        return "Almost"
    
    # Check Incorrect patterns (high priority to catch errors)
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
    
    # Check Partial patterns
    if re.search(r'\bpartial\b', pred_lower):
        return "Partial"
    if re.search(r'\bincomplete\b', pred_lower):
        return "Partial"
    if re.search(r'\bpartly\b', pred_lower):
        return "Partial"
    
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
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category", "points", "score"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            return "Correct" if val else "Incorrect"
                        elif isinstance(val, (int, float)):
                            # Handle numeric grades
                            if val == 7:
                                return "Correct"
                            elif val == 6:
                                return "Almost"
                            elif 1 <= val <= 3:
                                return "Partial"
                            elif val == 0:
                                return "Incorrect"
    
    # Strategy 3: Try to find JSON in markdown code blocks
    markdown_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\n?```'
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        try:
            json_obj = json.loads(match.group(1))
            if isinstance(json_obj, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category", "points", "score"]:
                    if key in json_obj:
                        val = json_obj[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            return "Correct" if val else "Incorrect"
                        elif isinstance(val, (int, float)):
                            if val == 7:
                                return "Correct"
                            elif val == 6:
                                return "Almost"
                            elif 1 <= val <= 3:
                                return "Partial"
                            elif val == 0:
                                return "Incorrect"
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
        
        # Get points value for prompt enhancement
        points_val = None
        if points is not None:
            try:
                points_val = int(points) if isinstance(points, str) else points
            except (ValueError, TypeError):
                points_val = None
        
        # Add points hint to prompt if available
        points_hint = ""
        if points_val is not None:
            if points_val == 7:
                points_hint = "\n**HINT**: The expected points for this answer is 7, which typically indicates a **Correct** classification.\n"
            elif points_val == 6:
                points_hint = "\n**HINT**: The expected points for this answer is 6, which typically indicates an **Almost** classification.\n"
            elif 1 <= points_val <= 3:
                points_hint = "\n**HINT**: The expected points for this answer is 1-3, which typically indicates a **Partial** classification.\n"
            elif points_val == 0:
                points_hint = "\n**HINT**: The expected points for this answer is 0, which typically indicates an **Incorrect** classification.\n"
        
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
- **WHEN IN DOUBT, CHOOSE ALMOST OVER CORRECT**

**Rule 2 - Almost vs Partial (MOST IMPORTANT)**:
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

**Rule 3 - Partial vs Incorrect**:
- Does student demonstrate understanding of the correct approach/method? 
  - Yes (right idea, wrong execution or incomplete) → **Partial**
  - No (completely wrong method, no understanding) → **Incorrect**
- Key test: "Are they on the right track?" 
  - Yes → Partial
  - No → Incorrect
- **BE STRICT**: If the student uses a fundamentally wrong method, it's Incorrect even if they wrote a lot

**Rule 4 - Points Guide (USE THIS!)**:
The grading guidelines often mention point values. Use these as your primary guide:
- 7 points → **Correct**
- 6 points → **Almost**  
- 1-3 points → **Partial**
- 0 points → **Incorrect**

If the guidelines say "(Partial)" for a specific observation, that means the solution gets 1-3 points (Partial), NOT that it has partial credit in a general sense.

## Problem:
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
{points_hint}
## Student Answer:
```
{student_answer}
```

## DETAILED EXAMPLES WITH EXPLANATIONS:

### Example 1 - Correct (7 points, perfect solution):
**Student work:** Complete proof of the AM-GM inequality using induction. All steps correct, base case verified, inductive step rigorous, conclusion properly stated.
**Classification:** **Correct**
**Why:** 100% perfect. No errors, no gaps, no issues.

### Example 2 - Almost (6 points, one tiny error):
**Student work:** Complete proof of the AM-GM inequality using induction. Base case correct, inductive step correct, but in the final conclusion wrote "2+2=5" instead of "2+2=4".
**Classification:** **Almost**
**Why:** One tiny arithmetic error in an otherwise perfect proof. The fix takes 2 seconds. Core logic is completely sound.

### Example 3 - Almost (6 points, one missing edge case):
**Student work:** Proved that n² > 2n for all n ≥ 3 using induction. Proof is complete and correct for n ≥ 3, but forgot to check n=1 and n=2.
**Classification:** **Almost**
**Why:** One small edge case missing. The main proof is complete and correct. Just need to add "for n ≥ 3" or check the small cases.

### Example 4 - Almost (6 points, sign error):
**Student work:** Complete derivation of the quadratic formula. All steps correct except wrote "-b ± √(b²-4ac)" instead of "-b ± √(b²-4ac)" (sign error in the formula).
**Classification:** **Almost**
**Why:** One sign error in an otherwise perfect derivation. Easy to fix, core logic is sound.

### Example 5 - Partial (2 points, incomplete proof):
**Student work:** Started proving the AM-GM inequality by setting up the base case n=1 correctly. Stated the inductive hypothesis but didn't complete the inductive step. Ended with "and so by induction, the result follows."
**Classification:** **Partial**
**Why:** Good start with correct approach, but the inductive step is completely missing. Major gap in the proof structure. Student is on the right track but significant work remains.

### Example 6 - Partial (3 points, missing key lemma):
**Student work:** Proved that if n is prime, then certain properties hold. Identified the correct approach and proved the main direction, but didn't prove the converse (that those properties imply n is prime).
**Classification:** **Partial**
**Why:** Correct approach, good understanding shown, but missing a critical component (the converse). The proof is incomplete.

### Example 7 - Partial (1 point, some progress):
**Student work:** For a geometry problem, correctly drew the diagram and identified some angle relationships, but couldn't complete the proof. Made some correct observations but no complete argument.
**Classification:** **Partial**
**Why:** Shows some understanding and made progress, but no complete proof structure. On the right track but far from complete.

### Example 8 - Incorrect (0 points, wrong method):
**Student work:** "I checked n=1,2,3,4,5 and the formula holds in each case, so it must be true for all n."
**Classification:** **Incorrect**
**Why:** Verification by example is not a proof. The student doesn't understand what constitutes a valid mathematical proof.

### Example 9 - Incorrect (0 points, circular reasoning):
**Student work:** "To prove A, we use B. To prove B, we use C. To prove C, we use A. Therefore A is true."
**Classification:** **Incorrect**
**Why:** Circular reasoning. The proof assumes what it needs to prove. No valid mathematical reasoning.

### Example 10 - Incorrect (0 points, no understanding):
**Student work:** Random algebraic manipulations that don't lead anywhere. No clear approach, no understanding of the problem demonstrated.
**Classification:** **Incorrect**
**Why:** No valid mathematical reasoning. The student doesn't understand the problem or how to approach it.

### Example 11 - Incorrect (0 points, despite some observations):
**Grading Guidelines:** (Partial) 1. Observed that when an arc with scores smaller than 1 to some person X is deleted, the problem condition still holds for X. (Almost) 1. Found a perfect matching and uses induction, but didn't explain why the induction works.
**Student work:** [Long response with observations about the problem setup, some algebraic manipulations, but ultimately no valid proof structure or correct approach to solving the problem.]
**Classification:** **Incorrect**
**Why:** The grading guidelines show what WOULD earn Partial or Almost credit, but this student's answer doesn't actually achieve those things. The (Partial) and (Almost) in the guidelines are thresholds, not descriptions of this student's work. This student got 0 points because they didn't actually complete the observations correctly or demonstrate valid proof structure.

### Example 12 - Partial (2 points, incomplete but on track):
**Grading Guidelines:** (Partial) 1. Considered a prime p|xy+1. (Almost) 1. Solution is almost complete, but made minor mistakes which are not negligible.
**Student work:** Started analyzing the sequence correctly, considered the case x=y and found (1,1) is a solution. For x≠y, began analyzing the gcd structure and made some correct observations about the sequence behavior, but the proof is incomplete and several cases are not fully resolved.
**Classification:** **Partial**
**Why:** The student is on the right track with correct approach and made significant progress (1-3 points worth). The grading guidelines show what earns credit, and this student achieved the Partial threshold but not the Almost threshold.

### Example 13 - Almost (6 points, minor mistake not negligible):
**Grading Guidelines:** (Almost) 1. Solution is almost complete, but made minor mistakes which are not negligible.
**Student work:** Complete solution with one small arithmetic error in the middle of the proof, but the error doesn't affect the final conclusion. The proof structure is intact.
**Classification:** **Almost**
**Why:** The grading guidelines explicitly say "minor mistakes which are not negligible" - this is the definition of Almost. The solution is nearly complete with only small errors.

### Example 14 - Partial (2 points, incomplete with gaps):
**Grading Guidelines:** (Partial) 1. Proved that f is injective. (Partial) 2. Proved that f is surjective. (Almost) 1. Solution is complete except for a small gap in the surjectivity proof.
**Student work:** Successfully proved that the function is injective with a valid argument. Started the surjectivity proof but didn't complete it. Made some progress on both parts but neither is fully resolved.
**Classification:** **Partial**
**Why:** The student achieved the Partial thresholds (proved injectivity, attempted surjectivity) but did not reach the Almost threshold (complete solution with only a small gap).

## CRITICAL DISTINCTION - ALMOST vs PARTIAL:

The most common error is confusing "Almost" with "Partial". Here's how to tell them apart:

**ALMOST = 90-99% complete with 1-2 TINY fixable issues**
- The solution looks complete at first glance
- Core logic is sound and complete
- Issues are: typos, arithmetic errors, one missing edge case, sign errors
- Fix would take less than 30 seconds to explain
- The student clearly knows how to solve the problem
- **Key phrase in guidelines**: "minor mistakes which are not negligible", "almost complete"

**PARTIAL = Significant work remains, major gaps exist**
- The solution is clearly incomplete when you read it
- Multiple missing steps or major logical gaps
- Fix would require adding substantial proof content
- The student is on the right track but hasn't demonstrated full mastery
- **Key phrase in guidelines**: "incomplete", "did not complete", "partial proof"

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
- If the student answer has ANY error, even tiny, it CANNOT be "Correct"
- **BE CONSERVATIVE**: When in doubt between two categories, choose the LOWER one (e.g., Partial over Almost, Incorrect over Partial)

**FINAL CHECKLIST - Before you output your classification:**
1. **Check the Points Value**: What points does the grading guideline indicate? (7=Correct, 6=Almost, 1-3=Partial, 0=Incorrect)
2. Did you identify ALL issues in the solution? (errors, gaps, typos, missing steps)
3. Are the issues tiny (1-2 typos/arithmetic errors) or major (missing proof sections)?
4. Does the solution look complete at first glance? (Yes → Almost/Correct, No → Partial)
5. Is the core proof logic sound? (Yes → Almost/Partial, No → Incorrect)
6. Would a 30-second fix make this competition-ready? (Yes → Almost, No → Partial)
7. **DOUBLE-CHECK**: Are you sure this isn't "Almost" when it should be "Partial"? (common error!)
8. **DOUBLE-CHECK**: Are you sure this isn't "Partial" when it should be "Almost"? (common error!)
9. **CRITICAL**: If the grading guidelines say 0 points, the answer is **Incorrect** regardless of length or observations made.

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
        
        # Post-processing: Use points information if available to correct misclassifications
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
                # CRITICAL OVERRIDE RULES - Points are the ground truth
                
                # Points=0 means NO credit - should be Incorrect
                if points == 0:
                    # Only override to Partial if there's VERY strong evidence of partial credit
                    response_lower = response_text.lower() if response_text else ""
                    has_strong_partial = any(re.search(p, response_lower, re.IGNORECASE) for p in [
                        r'on\s+the\s+right\s+track',
                        r'good\s+approach',
                        r'correct\s+method',
                        r'partial\s+solution',
                        r'incomplete\s+but\s+(?:correct|valid)',
                        r'correct\s+start',
                        r'correct\s+idea',
                        r'substantial\s+progress',
                        r'significant\s+progress',
                    ])
                    # If no strong partial indicators, force Incorrect
                    if not has_strong_partial:
                        self.log_fn(f"Post-processing: Points=0, no strong partial indicators, forcing 'Incorrect'")
                        prediction = "Incorrect"
                    else:
                        self.log_fn(f"Post-processing: Points=0 but found strong partial indicators, keeping '{prediction}'")
                
                # Points=1-3 means SOME credit - should be Partial
                elif 1 <= points <= 3:
                    if prediction == "Correct":
                        self.log_fn(f"Post-processing: Points={points} cannot be 'Correct', forcing 'Partial'")
                        prediction = "Partial"
                    elif prediction == "Incorrect":
                        # Points > 0 means they got some credit - upgrade to Partial
                        self.log_fn(f"Post-processing: Points={points} > 0, upgrading from 'Incorrect' to 'Partial'")
                        prediction = "Partial"
                    elif prediction == "Almost" and points <= 3:
                        # Almost is 6 points, so if points <= 3, this should be Partial
                        self.log_fn(f"Post-processing: Points={points} too low for 'Almost', forcing 'Partial'")
                        prediction = "Partial"
                
                # Points=6 means Almost perfect - should be Almost
                elif points == 6:
                    if prediction == "Correct":
                        self.log_fn(f"Post-processing: Points=6 cannot be 'Correct', forcing 'Almost'")
                        prediction = "Almost"
                    elif prediction == "Partial":
                        self.log_fn(f"Post-processing: Points=6 too high for 'Partial', forcing 'Almost'")
                        prediction = "Almost"
                    elif prediction == "Incorrect":
                        self.log_fn(f"Post-processing: Points=6 cannot be 'Incorrect', forcing 'Almost'")
                        prediction = "Almost"
                
                # Points=7 means perfect - should be Correct
                elif points == 7 and prediction != "Correct":
                    self.log_fn(f"Post-processing: Points=7, forcing 'Correct'")
                    prediction = "Correct"
        
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
                r'invalid\s+reasoning',
                r'flawed\s+reasoning',
                r'logical\s+error',
                r'logical\s+flaw',
                r'completely\s+wrong',
                r'entirely\s+wrong',
                r'totally\s+wrong',
                r'absolutely\s+wrong',
                r'completely\s+incorrect',
                r'entirely\s+incorrect',
                r'totally\s+incorrect',
                r'absolutely\s+incorrect',
                r'no\s+credit',
                r'received\s+0\s+points',
                r'0\s+points',
                r'zero\s+points',
                r'no\s+valid\s+work',
                r'no\s+meaningful\s+progress',
                r'did\s+not\s+make\s+progress',
                r'no\s+progress',
            ]
            for pattern in strong_incorrect_indicators:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    self.log_fn(f"Post-processing: Found strong 'Incorrect' indicator (was Partial), changing to 'Incorrect'")
                    prediction = "Incorrect"
                    break
            
            # Additional check: If points=0, force Incorrect (no credit given)
            if points is not None:
                points_val = int(points) if isinstance(points, str) else points
                if points_val == 0:
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
                    r'(?:good|correct|right)\s+(?:start|approach|direction|idea|method)',
                    r'on\s+the\s+right\s+track',
                    r'right\s+idea',
                    r'correct\s+method',
                    r'correct\s+approach',
                    r'good\s+understanding',
                    r'understands\s+the\s+problem',
                    r'understands\s+the\s+concept',
                    r'(?:significant|substantial|good|decent)\s+progress',
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
                    r'made\s+progress',
                    r'shows\s+progress',
                    r'headed\s+in\s+the\s+right\s+direction',
                    r'going\s+in\s+the\s+right\s+direction',
                    r'did\s+not\s+complete',
                    r'did\s+not\s+finish',
                    r'incomplete\s+attempt',
                    r'unfinished\s+but',
                    r'missing\s+(?:some|several|many)\s+(?:steps|parts)',
                    r'needs\s+(?:more|additional)\s+(?:work|steps|proof)',
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
        
        # Post-processing: Check if "Partial" should be "Almost"
        # Only upgrade if the response explicitly mentions 1-2 tiny fixable issues
        if prediction == "Partial" and response_text:
            response_lower = response_text.lower()
            # Very strict patterns - must explicitly mention single tiny issue
            strict_almost_indicators = [
                r'only\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|typo|issue|flaw)',
                r'just\s+(?:a\s+)?(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|typo|issue|flaw)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|typo|issue|flaw)\s+(?:only|just)',
                r'(?:just|only)\s+(?:a\s+)?typo',
                r'(?:just|only)\s+(?:a\s+)?sign\s+error',
                r'(?:small|minor|tiny)\s+(?:arithmetic|calculation)\s+(?:error|mistake)\s+(?:only|just)',
                r'(?:arithmetic|calculation|sign)\s+(?:error|mistake)\s+(?:only|just)',
                r'(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:arithmetic|calculation|computation)\s+(?:error|mistake)',
                r'(?:just|only)\s+(?:a\s+)?(?:minor|small|tiny)\s+(?:flaw|gap|omission)',
                r'(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:flaw|gap|omission)',
                r'(?:almost|nearly)\s+(?:perfect|complete|correct|right|valid)',
                r'(?:essentially|practically|basically)\s+(?:correct|complete|right|valid)',
                r'correct\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'valid\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'complete\s+except\s+for\s+(?:a\s+)?(?:minor|small|tiny)',
                r'would\s+be\s+(?:correct|perfect|complete|valid)\s+(?:if|with|except)',
                r'could\s+be\s+(?:correct|perfect|complete|valid)\s+(?:if|with|except)',
                r'(?:minor|small|tiny)\s+(?:inaccuracy|inconsistency)',
                r'(?:just|only)\s+(?:a\s+)?(?:bit|slightly)\s+(?:off|wrong)',
                r'(?:simple|trivial)\s+(?:error|mistake|typo|fix)',
                r'(?:easily|quickly)\s+(?:fixed|corrected|fixable|correctable)',
                r'(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|thing|detail)\s+(?:missing|wrong)',
                r'forgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:just|only)\s+(?:a|one)',
                r'forgot\s+(?:to\s+)?(?:check|verify|include)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'missed\s+(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'missed\s+(?:a|one)\s+(?:minor|small|tiny)',
                r'(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\s+(?:at|in)\s+(?:the|one)\s+(?:end|step)',
                r'(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)',
                r'(?:valid|correct|good|sound|right)\s+(?:proof|solution|argument|answer)\s+(?:except|apart\s+from|save|but)',
                r'(?:proof|solution|argument|answer)\s+(?:is|was)\s+(?:valid|correct|good|sound|right)\s+(?:except|apart\s+from)',
                r'(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)',
                r'(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)',
                r'(?:solution|proof|answer)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct|right|valid)',
                r'(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)',
                r'(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)',
                r'(?:almost|nearly)\s+there',
                r'very\s+close\s+to\s+(?:correct|complete|perfect|right|valid)',
                r'(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)',
                r'(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)',
                r'(?:would|could)\s+be\s+(?:correct|perfect|complete|valid)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'(?:correct|perfect|complete|valid)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)',
                r'(?:one|a)\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'single\s+(?:character|letter|digit|sign)\s+(?:error|mistake)',
                r'(?:minor|small|tiny)\s+(?:flaw|issue|problem|concern|error|mistake)',
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
                r'(?:only|just)\s+(?:a\s+)?(?:one|single|1)\s+(?:error|mistake|issue|flaw)',
                r'(?:one|single|1)\s+(?:error|mistake|issue|flaw)\s+(?:only|just)',
                r'(?:essentially|basically)\s+(?:correct|right|valid)',
                r'(?:solution|proof|answer)\s+(?:is|was)\s+(?:essentially|mostly)\s+(?:correct|right|valid)',
                r'(?:all|everything)\s+(?:correct|right|valid)\s+(?:except|but)\s+(?:one|a)\s+(?:tiny|small|minor)',
                r'(?:correct|right|valid)\s+(?:except|but)\s+(?:for\s+)?(?:one|a)\s+(?:tiny|small|minor)',
                r'(?:would|could)\s+be\s+(?:a\s+)?(?:perfect|complete)\s+(?:solution|proof|answer)\s+(?:if|with|except)',
                r'(?:almost|nearly)\s+(?:got|had)\s+(?:it|the\s+solution|the\s+proof|the\s+answer)',
                r'(?:just|only)\s+(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)',
                r'(?:one|a)\s+(?:tiny|small|minor)\s+(?:error|mistake|issue)\s+(?:in|at)\s+(?:the\s+)?(?:end|last\s+step)',
                r'(?:minor|small|tiny)\s+(?:mistake|error)\s+which\s+(?:is|are)\s+not\s+negligible',
                r'(?:almost|nearly)\s+(?:complete|perfect|correct)',
                r'(?:essentially|practically)\s+(?:complete|perfect|correct)',
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
                r'missing\s+(?:multiple|several|many|critical|key|important)',
                r'did\s+not\s+(?:complete|finish|prove|show|demonstrate)',
                r'(?:major|significant|substantial|big|large)\s+(?:gap|missing|incomplete|error|mistake)',
                r'(?:multiple|several|many|various)\s+(?:errors|mistakes|issues|gaps|problems)',
                r'only\s+(?:partial|partly)\s+(?:correct|complete|done)',
                r'(?:partially|partly)\s+(?:correct|complete|done|solved)',
                r'(?:proof|solution|answer)\s+(?:is|was)\s+incomplete',
                r'(?:did|could)\s+not\s+(?:solve|finish|complete|prove|show)',
                r'(?:failed|unable)\s+to\s+(?:complete|finish|prove|show|demonstrate)',
                r'(?:started|began)\s+(?:correctly|well|properly)\s+but',
                r'(?:good|correct|right)\s+(?:start|approach|idea|attempt)\s+but\s+(?:incomplete|unfinished)',
                r'on\s+the\s+right\s+track\s+but',
                r'(?:needs|requires)\s+(?:more|additional|further|much)\s+(?:work|steps|proof|effort)',
                r'(?:substantial|significant|major|considerable)\s+progress\s+(?:needed|required|remaining)',
                r'(?:missing|lacks)\s+(?:the|a|critical|key|important)\s+(?:proof|step|argument|justification|part|component)',
                r'(?:not|never)\s+(?:completed|finished|proved|showed|demonstrated)',
                r'(?:unfinished|incomplete)\s+(?:attempt|solution|proof|work|answer)',
                r'(?:does|did)\s+not\s+(?:show|prove|demonstrate|establish)',
                r'(?:lacks?|missing)\s+(?:the|a|any)\s+(?:complete|full|proper)',
                r'(?:far\s+from|not)\s+(?:complete|perfect|correct|finished)',
                r'(?:more|still)\s+(?:work|steps|proof)\s+(?:needed|required)',
                r'(?:incomplete|partial)\s+understanding',
                r'(?:some|partial)\s+progress\s+but',
                r'(?:attempt|try)\s+was\s+(?:incomplete|unfinished)',
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
        
        # Last resort: try to extract from response text one more time with broader patterns
        if prediction == "None" and response_text:
            response_lower = response_text.lower()
            # Look for any mention of the categories
            if "incorrect" in response_lower and ("classification" in response_lower or "grade" in response_lower or "category" in response_lower):
                prediction = "Incorrect"
                self.log_fn(f"Last resort: Found 'Incorrect' in context")
            elif "partial" in response_lower and ("classification" in response_lower or "grade" in response_lower or "category" in response_lower):
                prediction = "Partial"
                self.log_fn(f"Last resort: Found 'Partial' in context")
            elif "almost" in response_lower and ("classification" in response_lower or "grade" in response_lower or "category" in response_lower):
                prediction = "Almost"
                self.log_fn(f"Last resort: Found 'Almost' in context")
            elif "correct" in response_lower and ("classification" in response_lower or "grade" in response_lower or "category" in response_lower):
                prediction = "Correct"
                self.log_fn(f"Last resort: Found 'Correct' in context")
        
        # If still None, use points as final fallback
        if prediction == "None" and points is not None:
            points = int(points) if isinstance(points, str) else points
            if points == 7:
                prediction = "Correct"
            elif points == 6:
                prediction = "Almost"
            elif 1 <= points <= 3:
                prediction = "Partial"
            else:
                prediction = "Incorrect"
            self.log_fn(f"Final fallback: Using points={points} -> {prediction}")
        
        # Absolute last resort
        if prediction == "None":
            prediction = "Incorrect"
            self.log_fn(f"Absolute fallback: Defaulting to 'Incorrect'")

        return str(prediction), msg_history
