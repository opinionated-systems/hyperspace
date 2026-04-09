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
    """Extract JSON objects from <json>...</json> blocks with robust error handling."""
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
        
        # Try multiple parsing strategies
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Strategy 1: Fix trailing commas
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', inner)
                results.append(json.loads(fixed))
                continue
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Fix single quotes
            try:
                fixed = inner.replace("'", '"')
                results.append(json.loads(fixed))
                continue
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Extract just the value part
            try:
                # Look for "response": "value" pattern
                match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                if match:
                    results.append({"response": match.group(1)})
                    continue
            except Exception:
                pass
            
            # Strategy 4: Look for any key-value pair
            try:
                match = re.search(r'"(\w+)"\s*:\s*"([^"]+)"', inner)
                if match:
                    results.append({match.group(1): match.group(2)})
                    continue
            except Exception:
                pass
                
            continue
    return results or None


def _normalize_prediction(prediction: str) -> str | None:
    """Normalize a prediction string to one of the allowed categories.
    
    Priority order: Almost > Incorrect > Partial > Correct
    This ensures "almost correct" maps to Almost, not Correct.
    """
    if not prediction:
        return None
    
    pred_lower = prediction.lower().strip()
    
    # Remove common prefixes/suffixes and punctuation
    pred_clean = re.sub(r'^(the\s+|a\s+|an\s+|is\s+|it\s+|this\s+|that\s+)', '', pred_lower)
    pred_clean = re.sub(r'\s+(answer|classification|category|result|grade)$', '', pred_clean)
    pred_clean = re.sub(r'[.!?]+$', '', pred_clean)
    
    # Check for exact matches first
    allowed_categories = ["correct", "incorrect", "partial", "almost"]
    if pred_clean in allowed_categories:
        return pred_clean.capitalize()
    
    # ===== PRIORITY 1: ALMOST PATTERNS (highest priority) =====
    # These compound patterns MUST be checked before any single word patterns
    # to ensure "almost correct" maps to Almost, not Correct
    
    almost_compound_patterns = [
        # Core "almost" phrases
        r'\balmost\s+correct\b', r'\bnearly\s+correct\b', r'\balmost\s+perfect\b',
        r'\bclose\s+to\s+correct\b', r'\bessentially\s+correct\b',
        r'\bmostly\s+correct\b', r'\bcorrect\s+except\s+for\b',
        r'\bcomplete\s+except\s+for\b',
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
        r'\b(?:error|mistake|issue|typo|flaw)\s+(?:is|was)\s+(?:minor|small|tiny)\b',
        r'\b(?:essentially|practically|virtually)\s+(?:correct|right|valid)\b',
        r'\b(?:almost|nearly)\s+(?:got|had)\s+(?:it|the\s+solution|the\s+proof)\b',
        r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete)\s+(?:if|with|except)\b',
        r'\b(?:valid|correct|good)\s+(?:proof|solution|argument)\s+(?:except|apart\s+from)\b',
        # Additional patterns for "Almost"
        r'\bminor\s+mistakes?\s+which\s+are\s+not\s+negligible\b',
        r'\bminor\s+mistakes?\s+not\s+negligible\b',
        r'\bsmall\s+gap\s+in\s+(?:the\s+)?(?:proof|solution)\b',
        r'\bminor\s+gap\s+in\s+(?:the\s+)?(?:proof|solution)\b',
        r'\bsolution\s+is\s+essentially\s+(?:complete|correct)\b',
        r'\bproof\s+is\s+essentially\s+(?:complete|correct)\b',
        r'\bsolution\s+is\s+complete\s+except\b',
        r'\bproof\s+is\s+complete\s+except\b',
        r'\b(?:just|only)\s+a\s+(?:small|minor|tiny)\s+error\b',
        r'\b(?:just|only)\s+a\s+typo\b',
        r'\b(?:just|only)\s+a\s+sign\s+error\b',
        r'\b(?:one|single)\s+(?:small|minor|tiny)\s+error\b',
        r'\barithmetic\s+mistake\b',
        r'\bcalculation\s+mistake\b',
        r'\bsign\s+mistake\b',
        r'\boff\s+by\s+(?:one|1|a\s+sign)\b',
        r'\bsmall\s+(?:error|mistake)\s+in\s+(?:the\s+)?final\b',
        r'\bminor\s+(?:error|mistake)\s+in\s+(?:the\s+)?final\b',
        r'\btiny\s+(?:error|mistake)\s+in\s+(?:the\s+)?final\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny|little)\s+(?:error|mistake|issue|typo|problem|flaw)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\s+in\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\b',
        r'\b(?:error|mistake|issue|typo|flaw)\s+(?:at|in)\s+(?:the|one)\s+(?:end|step|point)\b',
        r'\b(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)\b',
        r'\b(?:valid|correct|good|sound|right)\s+(?:proof|solution|argument|answer)\s+(?:except|apart\s+from|save|but)\b',
        r'\b(?:proof|solution|argument|answer)\s+(?:is|was)\s+(?:valid|correct|good|sound|right)\s+(?:except|apart\s+from)\b',
        r'\b(?:minor|small|tiny)\s+(?:inaccuracy|inconsistency)\b',
        r'\b(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)\b',
        r'\b(?:solution|proof|answer)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct|right|valid)\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)\b',
        r'\b(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:almost|nearly)\s+there\b',
        r'\bvery\s+close\s+to\s+(?:correct|complete|perfect|right|valid)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)\b',
        r'\b(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)\b',
        r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete|valid)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)\b',
        r'\b(?:correct|perfect|complete|valid)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)\b',
        r'\b(?:essentially|practically|virtually)\s+(?:correct|complete|right|valid|perfect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)\b',
        r'\b(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|flaw)\s+which\s+(?:is|are)\s+not\s+negligible\b',
        r'\b(?:almost|nearly)\s+complete\b',
        r'\b(?:almost|nearly)\s+perfect\b',
        r'\b(?:almost|nearly)\s+correct\b',
    ]
    for pattern in almost_compound_patterns:
        if re.search(pattern, pred_lower):
            return "Almost"
    
    # Check Almost single word patterns - BEFORE other categories
    if re.search(r'\balmost\b', pred_lower):
        return "Almost"
    if re.search(r'\bnearly\b', pred_lower):
        return "Almost"
    if re.search(r'\bclose\b', pred_lower):
        return "Almost"
    
    # ===== PRIORITY 2: INCORRECT PATTERNS =====
    # Check for explicit "not" patterns first
    if re.search(r'\bnot\s+(?:correct|right|valid|perfect|complete)\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bincorrect\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bwrong\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bfalse\b', pred_lower):
        return "Incorrect"
    if re.search(r'\binvalid\b', pred_lower):
        return "Incorrect"
    if re.search(r'\berroneous\b', pred_lower):
        return "Incorrect"
    if re.search(r'\bflawed\b', pred_lower):
        return "Incorrect"
    
    # ===== PRIORITY 3: PARTIAL PATTERNS =====
    if re.search(r'\bpartial\b', pred_lower):
        return "Partial"
    if re.search(r'\bincomplete\b', pred_lower):
        return "Partial"
    if re.search(r'\bpartly\b', pred_lower):
        return "Partial"
    if re.search(r'\bsome\s+(?:progress|correct|valid)\b', pred_lower):
        return "Partial"
    if re.search(r'\bon\s+the\s+right\s+track\b', pred_lower):
        return "Partial"
    
    # ===== PRIORITY 4: CORRECT PATTERNS (lowest priority) =====
    # Only match if no negation or "almost" context
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
        # Direct classification statements - Almost (expanded patterns)
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
        (r'\bresult\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        (r'\bprediction\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        # JSON and structured formats for Almost
        (r'\bclassification[:\s]+almost\b', "Almost"),
        (r'\bgrade[:\s]+almost\b', "Almost"),
        (r'\bcategory[:\s]+almost\b', "Almost"),
        (r'["\']response["\']\s*:\s*["\']almost["\']', "Almost"),
        (r'\{\s*["\']?response["\']?\s*:\s*["\']?almost["\']?\s*\}', "Almost"),
        (r'\bjson\s*[:=]\s*["\']?almost["\']?\b', "Almost"),
        (r'\{[^}]*["\']?response["\']?\s*:\s*["\']?Almost["\']?[^}]*\}', "Almost"),
        (r'["\']response["\']\s*:\s*["\']Almost["\']', "Almost"),
        (r'<json>[^<]*"response"\s*:\s*"Almost"[^<]*</json>', "Almost"),
        (r'\bthe\s+classification\s+is\s+["\']?almost["\']?\b', "Almost"),
        (r'\bi\s+classify\s+this\s+as\s+["\']?almost["\']?\b', "Almost"),
        (r'\bthis\s+(?:should|would)\s+be\s+["\']?almost["\']?\b', "Almost"),
        
        # Direct classification statements - Partial
        (r'\bclassification\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bgrade\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bcategory\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bclassify\s+(?:this|it)\s+as\s+["\']?partial["\']?\b', "Partial"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*partial["\']?\b', "Partial"),
        (r'\bevaluation\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bverdict\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bresult\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        (r'\bprediction\s*[:=]\s*["\']?partial["\']?\b', "Partial"),
        
        # Direct classification statements - Incorrect
        (r'\bclassification\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bgrade\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bcategory\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*incorrect["\']?\b', "Incorrect"),
        (r'\bevaluation\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bverdict\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bresult\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        (r'\bprediction\s*[:=]\s*["\']?incorrect["\']?\b', "Incorrect"),
        
        # Direct classification statements - Correct
        (r'\bclassification\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bgrade\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bcategory\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bresponse["\']?\s*[:=]\s*["\']?\s*correct["\']?\b', "Correct"),
        (r'\bevaluation\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bverdict\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bresult\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
        (r'\bprediction\s*[:=]\s*["\']?correct["\']?\b', "Correct"),
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
            # Try to fix common JSON issues in markdown
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', match.group(1))
                json_obj = json.loads(fixed)
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
    
    # Strategy 4: Look for raw JSON objects without tags or markdown
    raw_json_pattern = r'\{\s*["\']?response["\']?\s*:\s*["\']?(\w+)["\']?\s*\}'
    for match in re.finditer(raw_json_pattern, text, re.IGNORECASE):
        val = match.group(1)
        normalized = _normalize_prediction(val.strip())
        if normalized:
            return normalized
    
    # Also try to find any JSON-like structure with classification keys
    loose_json_pattern = r'["\']?(?:response|classification|grade|category)["\']?\s*[:=]\s*["\']?(\w+)["\']?'
    for match in re.finditer(loose_json_pattern, text, re.IGNORECASE):
        val = match.group(1)
        normalized = _normalize_prediction(val.strip())
        if normalized:
            return normalized
    
    # Strategy 4b: Look for JSON with "result" or "prediction" keys
    result_json_pattern = r'["\']?(?:result|prediction|answer|output)["\']?\s*[:=]\s*["\']?(\w+)["\']?'
    for match in re.finditer(result_json_pattern, text, re.IGNORECASE):
        val = match.group(1)
        normalized = _normalize_prediction(val.strip())
        if normalized:
            return normalized
    
    # Strategy 4c: Look for any word that looks like a category in JSON-like context
    json_context_pattern = r'\{[^}]*["\']?(?:response|classification|grade|category|result|prediction)["\']?\s*[:=]\s*["\']?(Correct|Almost|Partial|Incorrect)["\']?[^}]*\}'
    for match in re.finditer(json_context_pattern, text, re.IGNORECASE):
        val = match.group(1)
        normalized = _normalize_prediction(val.strip())
        if normalized:
            return normalized
    
    # Strategy 5: Look for standalone category mentions at end of text or emphasized
    # Check Almost first (most commonly missed)
    almost_standalone = [
        r'(?:^|\n)\s*["\']?almost["\']?\s*[.!?]?\s*(?:$|\n)',
        r'\*\*almost\*\*',
        r'\*almost\*',
        r'`almost`',
        r'\balmost\b[.!?]?\s*$',
        r'^\s*almost\s*$',
        r'"almost"',
        r"'almost'",
        r'#\s*almost\b',
        r'\(almost\)',
        r'\[almost\]',
        r'\balmost\s+(?:perfect|complete|correct|right|valid)\b',
        r'\b(?:nearly|practically|virtually)\s+(?:perfect|complete|correct|right|valid)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\b',
        r'\bminor\s+(?:error|mistake|issue|flaw|gap|omission)\b',
        r'\bsmall\s+(?:error|mistake|issue|gap|flaw|omission)\b',
        r'\btiny\s+(?:error|mistake|issue|flaw|gap)\b',
        r'\bsign\s+error\b',
        r'\barithmetic\s+error\b',
        r'\barithmetic\s+mistake\b',
        r'\bcalculation\s+error\b',
        r'\bcalculation\s+mistake\b',
        r'\btypo\b',
        r'\btypographical\s+error\b',
        r'\bcorrect\s+except\s+for\b',
        r'\bvalid\s+except\s+for\b',
        r'\bcomplete\s+except\s+for\b',
        r'\bwould\s+be\s+(?:correct|perfect|complete|valid)\s+if\b',
        r'\bwould\s+be\s+(?:correct|perfect|complete|valid)\s+with\b',
        r'\bcould\s+be\s+(?:correct|perfect|complete|valid)\s+if\b',
        r'\b(?:one|single|1)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\s+(?:only|just)\b',
        r'\b(?:error|mistake|issue|typo|flaw)\s+(?:is|was)\s+(?:minor|small|tiny)\b',
        r'\b(?:only|just)\s+(?:a|one)\s+(?:minor|small|tiny|little)\s+(?:error|mistake|issue|typo|problem|flaw)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\s+in\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:error|mistake|issue|typo|flaw)\b',
        r'\b(?:error|mistake|issue|typo|flaw)\s+(?:at|in)\s+(?:the|one)\s+(?:end|step|point)\b',
        r'\b(?:otherwise|apart\s+from|except\s+for)\s+(?:a|one|this|that)\s+(?:minor|small|tiny)\b',
        r'\b(?:valid|correct|good|sound|right)\s+(?:proof|solution|argument|answer)\s+(?:except|apart\s+from|save|but)\b',
        r'\b(?:proof|solution|argument|answer)\s+(?:is|was)\s+(?:valid|correct|good|sound|right)\s+(?:except|apart\s+from)\b',
        r'\b(?:minor|small|tiny)\s+(?:inaccuracy|inconsistency)\b',
        r'\b(?:calculation|computation)\s+(?:is|was)\s+(?:slightly|a\s+bit)\s+(?:off|wrong|incorrect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:minor|small|tiny)\s+(?:detail|thing)\s+(?:missing|wrong)\b',
        r'\b(?:solution|proof|answer)\s+(?:is|was)\s+(?:almost|nearly)\s+(?:complete|perfect|correct|right|valid)\b',
        r'\b(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:minor|small|tiny)\s+(?:issue|problem|concern)\s+(?:with|in)\b',
        r'\b(?:just|only)\s+(?:one|a)\s+(?:minor|small|tiny)\s+(?:issue|problem|concern)\b',
        r'\b(?:almost|nearly)\s+there\b',
        r'\bvery\s+close\s+to\s+(?:correct|complete|perfect|right|valid)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:fix|correction)\b',
        r'\b(?:small|minor|tiny)\s+(?:fix|correction)\s+(?:needed|required)\b',
        r'\b(?:would|could)\s+be\s+(?:correct|perfect|complete|valid)\s+with\s+(?:a|one)\s+(?:small|minor|tiny)\b',
        r'\b(?:correct|perfect|complete|valid)\s+except\s+for\s+(?:a|one)\s+(?:small|minor|tiny)\b',
        r'\b(?:essentially|practically|virtually)\s+(?:correct|complete|right|valid|perfect)\b',
        r'\b(?:just|only)\s+(?:a|one)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)\b',
        r'\b(?:one|a)\s+(?:small|minor|tiny)\s+(?:step|part|piece)\s+(?:missing|wrong)\b',
        r'\b(?:minor|small|tiny)\s+(?:error|mistake|flaw)\s+which\s+(?:is|are)\s+not\s+negligible\b',
        r'\b(?:almost|nearly)\s+complete\b',
        r'\b(?:almost|nearly)\s+perfect\b',
        r'\b(?:almost|nearly)\s+correct\b',
        r'\bminor\s+mistakes?\s+which\s+are\s+not\s+negligible\b',
        r'\bminor\s+mistakes?\s+not\s+negligible\b',
        r'\bsmall\s+gap\s+in\s+(?:the\s+)?(?:proof|solution)\b',
        r'\bminor\s+gap\s+in\s+(?:the\s+)?(?:proof|solution)\b',
        r'\bsolution\s+is\s+essentially\s+(?:complete|correct)\b',
        r'\bproof\s+is\s+essentially\s+(?:complete|correct)\b',
        r'\bsolution\s+is\s+complete\s+except\b',
        r'\bproof\s+is\s+complete\s+except\b',
        r'\b(?:just|only)\s+a\s+(?:small|minor|tiny)\s+error\b',
        r'\b(?:just|only)\s+a\s+typo\b',
        r'\b(?:just|only)\s+a\s+sign\s+error\b',
        r'\b(?:one|single)\s+(?:small|minor|tiny)\s+error\b',
        r'\barithmetic\s+mistake\b',
        r'\bcalculation\s+mistake\b',
        r'\bsign\s+mistake\b',
        r'\boff\s+by\s+(?:one|1|a\s+sign)\b',
        r'\bsmall\s+(?:error|mistake)\s+in\s+(?:the\s+)?final\b',
        r'\bminor\s+(?:error|mistake)\s+in\s+(?:the\s+)?final\b',
        r'\btiny\s+(?:error|mistake)\s+in\s+(?:the\s+)?final\b',
    ]
    for pattern in almost_standalone:
        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
            return "Almost"
    
    # Then other categories
    standalone_patterns = [
        # Almost patterns (check first)
        (r'(?:^|\n)\s*["\']?almost["\']?\s*[.!?]?\s*(?:$|\n)', "Almost"),
        (r'^\s*almost\s*$', "Almost"),
        (r'"almost"', "Almost"),
        (r"'almost'", "Almost"),
        (r'`almost`', "Almost"),
        # Other categories
        (r'(?:^|\n)\s*["\']?partial["\']?\s*[.!?]?\s*(?:$|\n)', "Partial"),
        (r'(?:^|\n)\s*["\']?incorrect["\']?\s*[.!?]?\s*(?:$|\n)', "Incorrect"),
        (r'(?:^|\n)\s*["\']?correct["\']?\s*[.!?]?\s*(?:$|\n)', "Correct"),
        (r'\*\*partial\*\*', "Partial"),
        (r'\*\*incorrect\*\*', "Incorrect"),
        (r'\*\*correct\*\*', "Correct"),
        (r'\*partial\*', "Partial"),
        (r'\*incorrect\*', "Incorrect"),
        (r'\*correct\*', "Correct"),
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
    
    # Strategy 6: Use _normalize_prediction on the full text as last resort
    return _normalize_prediction(text)


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
                points_hint = "\n**POINTS = 7**: This answer received 7 points (perfect score) → classify as **Correct**\n"
            elif points_val == 6:
                points_hint = "\n**POINTS = 6**: This answer received 6 points (near-perfect, tiny issues) → classify as **Almost**\n"
            elif 1 <= points_val <= 3:
                points_hint = f"\n**POINTS = {points_val}**: This answer received {points_val} points (incomplete but on track) → classify as **Partial**\n"
            elif points_val == 0:
                points_hint = "\n**POINTS = 0**: This answer received 0 points (wrong approach) → classify as **Incorrect**\n"
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems.

## POINTS TO CLASSIFICATION MAPPING (FOLLOW THIS EXACTLY):
- **7 points** → **Correct** (100% perfect, no errors)
- **6 points** → **Almost** (90-99% complete, 1-2 tiny issues like typos, arithmetic errors, sign errors, one missing edge case)
- **1-3 points** → **Partial** (30-70% complete, significant gaps but on right track)
- **0 points** → **Incorrect** (wrong method, no understanding)

## The Four Categories with Examples:

### 1. **Correct** (7 points): 100% perfect. No errors, no gaps. The proof is complete and rigorous.
**Examples:**
- "The proof is complete and correct."
- "All steps are valid and the conclusion follows."
- "Perfect solution with no errors."

### 2. **Almost** (6 points): 90-99% complete with only 1-2 TINY issues. 
**Examples of tiny issues:**
- Arithmetic error: "2+2=5" (just a calculation mistake)
- Typo: "x^2" written as "x^3" in one place
- Sign error: Forgot a negative sign in one step
- One small edge case missing
- Minor notation issue

**Key phrases that indicate "Almost":**
- "minor mistakes which are not negligible"
- "almost complete"
- "complete except for a small gap"
- "correct except for one tiny error"
- "would be perfect if not for..."
- "just a typo"
- "only a sign error"

**If points = 6, you MUST classify as "Almost"**

### 3. **Partial** (1-3 points): Shows good understanding and correct approach, but has MAJOR gaps.
**Examples:**
- "The student proved that..." (only proved one part of a multi-part problem)
- "Found the correct approach but did not complete the proof"
- "Has the right idea but missing significant steps"
- "Started correctly but stopped halfway"

**Key phrases that indicate "Partial":**
- "incomplete proof"
- "did not complete"
- "only proved one direction"
- "missing the main argument"
- "has the right approach but..."

**If points = 1-3, you MUST classify as "Partial"**

### 4. **Incorrect** (0 points): Wrong method, circular reasoning, or verification by example.
**Examples:**
- "The approach is fundamentally wrong"
- "Circular reasoning"
- "Verified for a few cases but no general proof"
- "Completely wrong method"

**If points = 0, you MUST classify as "Incorrect"**

## FEW-SHOT EXAMPLES:

**Example 1 (Correct - 7 points):**
Student: "Proof: Let n be an integer. Then n^2 + n = n(n+1). Since n and n+1 are consecutive integers, one is even. Therefore n(n+1) is even. QED."
Classification: Correct
Reasoning: The proof is complete, rigorous, and has no errors.

**Example 2 (Almost - 6 points):**
Student: "Proof: Let n be an integer. Then n^2 + n = n(n+1). Since n and n+1 are consecutive integers, one is even. Therefore n(n+1) is even. So n^2 + n is even for all n."
Classification: Almost
Reasoning: The proof is essentially complete and correct, but the student forgot to write "QED" or explicitly state the conclusion in the final sentence. This is a tiny omission.

**Example 3 (Partial - 2 points):**
Student: "Let n be an integer. Then n^2 + n = n(n+1). Since n and n+1 are consecutive integers, one must be even."
Classification: Partial
Reasoning: The student has the right approach and made significant progress, but did not complete the proof by explicitly concluding that the product is even.

**Example 4 (Incorrect - 0 points):**
Student: "I checked n=1,2,3,4,5 and n^2+n is even in all cases. Therefore it's always even."
Classification: Incorrect
Reasoning: Verification by example is not a valid proof. No general argument was given.

## CRITICAL RULES:

**Rule 1 - POINTS ARE PRIMARY**: The points value is the MOST RELIABLE indicator:
- 7 points = **Correct**
- 6 points = **Almost** (NOT Partial!)
- 1-3 points = **Partial** (NOT Almost!)
- 0 points = **Incorrect**

**Rule 2 - Correct vs Almost**: 
- If there's ANY flaw (even tiny) → **Almost**
- Only if 100% perfect → **Correct**

**Rule 3 - Almost vs Partial (MOST COMMON ERROR)**:
- **ALMOST** = Solution looks COMPLETE at first glance, only 1-2 tiny fixable issues
- **PARTIAL** = Solution is clearly INCOMPLETE, major gaps exist
- **Test**: "Would a 30-second fix make this competition-ready?" YES → Almost, NO → Partial

**Rule 4 - Partial vs Incorrect**:
- **PARTIAL** = Right approach, on the right track, but incomplete
- **INCORRECT** = Wrong approach, no valid proof structure

## Response Format (REQUIRED):

<json>
{{
    "response": "Correct" | "Almost" | "Partial" | "Incorrect"
}}
</json>

## INPUTS:

**Problem:**
{problem}

**Official Solution:**
{solution}

**Grading Guidelines:**
{grading_guidelines}
{points_hint}
**Student Answer:**
{student_answer}

Based on the grading guidelines and student answer above, classify this solution into EXACTLY ONE category.

Output ONLY the JSON object with your classification:"""

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

        # Post-processing: Use points to correct misclassifications
        if points is not None and prediction != "None":
            try:
                points_int = int(points) if isinstance(points, str) else points
                response_lower = response_text.lower() if response_text else ""
                
                # For points=6 (Almost) - very strong signal
                if points_int == 6:
                    if prediction != "Almost":
                        # Check if the response actually supports Almost
                        if re.search(r'\b(?:almost|nearly|close|minor|small|tiny|typo|error|except|gap)\b', response_lower):
                            self.log_fn(f"Points-based correction: Points=6 + keywords -> 'Almost'")
                            prediction = "Almost"
                        elif prediction in ["Correct", "Partial"]:
                            # Force correction for 6 points
                            self.log_fn(f"Points-based correction: Points=6 overrides '{prediction}' -> 'Almost'")
                            prediction = "Almost"
                
                # For points=0 (Incorrect) - very strong signal
                elif points_int == 0:
                    if prediction != "Incorrect":
                        # Check if the response actually supports Incorrect
                        if re.search(r'\b(?:wrong|incorrect|invalid|false|error|not\s+correct|not\s+valid)\b', response_lower):
                            self.log_fn(f"Points-based correction: Points=0 + keywords -> 'Incorrect'")
                            prediction = "Incorrect"
                        elif prediction in ["Correct", "Almost", "Partial"]:
                            # Force correction for 0 points
                            self.log_fn(f"Points-based correction: Points=0 overrides '{prediction}' -> 'Incorrect'")
                            prediction = "Incorrect"
                
                # For points=1-3 (Partial) - strong signal for Partial
                elif 1 <= points_int <= 3:
                    if prediction != "Partial":
                        # Check if the response actually supports Partial
                        if re.search(r'\b(?:partial|incomplete|partly|some|gap|missing|not\s+complete|not\s+finished)\b', response_lower):
                            self.log_fn(f"Points-based correction: Points={points_int} + keywords -> 'Partial'")
                            prediction = "Partial"
                        elif prediction in ["Correct", "Almost", "Incorrect"]:
                            # Force correction for 1-3 points
                            self.log_fn(f"Points-based correction: Points={points_int} overrides '{prediction}' -> 'Partial'")
                            prediction = "Partial"
                
                # For points=7 (Correct)
                elif points_int == 7:
                    if prediction != "Correct":
                        # Check if the response actually supports Correct
                        if re.search(r'\b(?:correct|perfect|complete|valid|right|true|no\s+error|no\s+flaw)\b', response_lower):
                            self.log_fn(f"Points-based correction: Points=7 + keywords -> 'Correct'")
                            prediction = "Correct"
                        elif prediction in ["Almost", "Partial", "Incorrect"]:
                            # Force correction for 7 points
                            self.log_fn(f"Points-based correction: Points=7 overrides '{prediction}' -> 'Correct'")
                            prediction = "Correct"
                        
            except (ValueError, TypeError):
                pass
        
        # Final fallback: If still None, use points
        if prediction == "None" and points is not None:
            try:
                points_int = int(points) if isinstance(points, str) else points
                if points_int == 7:
                    prediction = "Correct"
                    self.log_fn(f"Final fallback: Points=7 -> 'Correct'")
                elif points_int == 6:
                    prediction = "Almost"
                    self.log_fn(f"Final fallback: Points=6 -> 'Almost'")
                elif 1 <= points_int <= 3:
                    prediction = "Partial"
                    self.log_fn(f"Final fallback: Points={points_int} -> 'Partial'")
                elif points_int == 0:
                    prediction = "Incorrect"
                    self.log_fn(f"Final fallback: Points=0 -> 'Incorrect'")
            except (ValueError, TypeError):
                prediction = "Incorrect"
                self.log_fn(f"Final fallback: Invalid points, defaulting to 'Incorrect'")
        
        # Absolute last resort
        if prediction == "None":
            prediction = "Incorrect"
            self.log_fn(f"Absolute fallback: Defaulting to 'Incorrect'")

        return str(prediction), msg_history
