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
    if not isinstance(text, str) or not text:
        return None
    
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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Bare JSON objects (starting with { and ending with })
    """
    if not isinstance(text, str) or not text:
        return None
    
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try ```json code blocks
    results = []
    pattern = r'```json\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    if results:
        return results
    
    # Try bare JSON objects using balanced brace matching
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            depth = 1
            j = i + 1
            in_string = False
            escape_next = False
            while j < n and depth > 0:
                char = text[j]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                i = j
                continue
        i += 1
    
    return results or None


def _analyze_reasoning_for_category(reasoning: str) -> str | None:
    """Analyze reasoning text to detect category indicators.
    
    This helps catch cases where the reasoning clearly indicates one category
    but the response field might have a different value.
    
    Args:
        reasoning: The reasoning text from the LLM response
        
    Returns:
        Suggested category or None if no strong indicators found
    """
    if not isinstance(reasoning, str) or not reasoning:
        return None
    
    reasoning_lower = reasoning.lower()
    
    # Strong indicators for "incorrect" - highest priority
    incorrect_patterns = [
        "no valid mathematical", "fundamentally wrong", "completely wrong",
        "totally wrong", "nonsense", "gibberish", "no understanding",
        "no solution", "failed", "failure", "blank", "empty",
        "fundamental misunderstanding", "wrong approach", "invalid approach",
        "does not understand", "did not understand", "nothing correct",
        "completely incorrect", "totally incorrect", "absolutely wrong",
        "fundamental error", "critical error", "serious mistake",
        "not correct", "is wrong", "are wrong", "was wrong",
        "incorrect solution", "incorrect answer", "incorrect approach",
        "no valid", "invalid reasoning", "flawed reasoning", "unsound",
        "does not demonstrate", "failed to", "unable to", "cannot solve",
        "no meaningful progress", "no valid steps", "completely failed",
        "does not solve", "did not solve", "failed to solve",
        "wrong answer", "wrong result", "wrong conclusion",
        "not a valid", "not an valid", "invalid solution",
        "no proof", "no valid proof", "missing proof", "lacks proof",
        "does not prove", "did not prove", "failed to prove",
        "erroneous", "fallacious", "bogus", "invalid"
    ]
    
    for pattern in incorrect_patterns:
        if pattern in reasoning_lower:
            return "incorrect"
    
    # Strong indicators for "partial" - check before "almost"
    # These indicate significant gaps or incomplete work
    partial_patterns = [
        "partial credit", "partially correct", "incomplete solution",
        "incomplete proof", "partial solution", "partial progress",
        "significant progress", "meaningful progress", "valid insight",
        "key insight", "correct lemma", "proved a lemma", "identified key",
        "some understanding", "partial proof", "partial result",
        "some correct", "partially valid", "partial understanding",
        "correct framework", "valid reasoning", "partial success",
        "some valid", "partially worked", "partial attempt",
        "partially successful", "some correct steps", "valid partial",
        "partially complete", "not fully correct", "not completely correct",
        "partial marks", "some credit", "partially right", "incomplete but",
        "partial understanding demonstrated", "some valid reasoning",
        "made progress", "good start", "on the right track",
        "correct direction", "valid approach", "substantial progress",
        "important insight", "major progress", "significant portion",
        "half correct", "partially worked out", "some progress",
        "incomplete execution", "started correctly", "began correctly",
        "correct beginning", "valid start", "some valid steps",
        "partially valid solution", "incomplete but valid",
        "missing steps", "incomplete reasoning", "unfinished",
        "needs more work", "requires completion", "incomplete answer",
        "partial answer", "some right", "partly correct", "partly right",
        "did not finish", "incomplete argument", "partial argument",
        "started well", "began well", "good beginning", "valid beginning",
        "significant gaps", "major gaps", "large gaps", "substantial gaps",
        "some progress", "limited progress", "modest progress",
        "reasonable start", "reasonable beginning", "reasonable attempt",
        "valid attempt", "legitimate attempt", "serious attempt",
        "demonstrates some", "shows some", "has some", "contains some",
        "identified correctly", "correctly identified", "recognized correctly",
        "correct intuition", "valid intuition", "good intuition",
        "correct strategy", "valid strategy", "good strategy",
        "correct idea", "valid idea", "good idea", "useful idea",
        "correct observation", "valid observation", "useful observation",
        "correct insight", "valid insight", "useful insight",
        "partially successful", "partially complete", "partially correct",
        "incomplete but promising", "incomplete but valid", "incomplete but sound",
        "has merit", "has value", "has substance", "has content",
        "not entirely wrong", "not completely wrong", "not totally wrong",
        "not entirely incorrect", "not completely incorrect", "not totally incorrect",
        "not entirely invalid", "not completely invalid", "not totally invalid",
        "not nonsense", "not gibberish", "not garbage", "not rubbish",
        "not trivial", "not empty", "not blank", "not missing",
        "not a failure", "not failed", "not useless", "not worthless",
        # Additional patterns to catch "partial" vs "almost" confusion
        "far from complete", "nowhere near complete", "long way from complete",
        "stopped early", "stopped prematurely", "gave up",
        "only proved", "only showed", "only demonstrated",
        "did not complete", "did not finish", "did not solve",
        "missing major", "missing significant", "lacks major", "lacks significant",
        "needs substantial", "requires substantial", "needs considerable",
        "incomplete execution", "incomplete implementation",
        "partial result", "partial answer", "incomplete solution",
        "verification contains minor mistakes", "minor mistakes only",
        "found a correct invariant", "found an invariant",
        "proved a useful", "proved one direction", "proved one part",
        "only one direction", "only one part", "partial case",
        "some cases only", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis"
    ]
    
    for pattern in partial_patterns:
        if pattern in reasoning_lower:
            return "partial"
    
    # Strong indicators for "almost" - very conservative
    # Must indicate near-completion with minor issues
    almost_patterns = [
        "almost correct", "nearly correct", "almost complete", "nearly complete",
        "minor mistake only", "minor error only", "small error only",
        "tiny mistake", "slight error only", "essentially correct",
        "fundamentally correct", "correct approach with minor",
        "correct method with minor", "nearly perfect", "almost perfect",
        "mostly correct with minor", "correct except for minor",
        "correct except for small", "correct except for tiny",
        "would be perfect if", "would be correct if", "only a minor",
        "just a minor", "just a small", "only a small", "minor typo only",
        "small typo only", "minor calculation error", "small calculation error",
        "minor arithmetic error", "slight miscalculation", "tiny miscalculation",
        "minor oversight", "small oversight", "trivial error", "insignificant error",
        "cosmetic error", "formatting error", "notation error", "sign error",
        "minor gap", "small gap", "trivial gap", "nearly solved",
        "almost solved", "essentially solved", "practically correct",
        "one minor", "a minor", "the minor", "single minor",
        "one small", "a small", "the small", "single small",
        "one tiny", "a tiny", "the tiny", "single tiny",
        "nearly there", "very close",
        "minor correction", "small correction", "tiny correction",
        "minor fix", "small fix", "tiny fix", "slight fix",
        "minor adjustment", "small adjustment", "tiny adjustment",
        "minor refinement", "small refinement", "tiny refinement",
        "minor issue only", "small issue only", "tiny issue only",
        "minor flaw only", "small flaw only", "tiny flaw only",
        "minor problem only", "small problem only", "tiny problem only",
        "minor mistake only", "small mistake only", "tiny mistake only",
        "just needs", "only needs", "simply needs", "merely needs",
        "only requires", "just requires", "simply requires", "merely requires",
        "easily fixed", "easily corrected", "easily remedied",
        "simple fix", "simple correction", "simple remedy",
        "quick fix", "quick correction", "quick remedy",
        "trivial fix", "trivial correction", "trivial remedy"
    ]
    
    # Check for "almost" indicators but also check for disqualifiers
    # REDUCED: Only the strongest disqualifiers that clearly indicate NOT "almost"
    almost_disqualifiers = [
        # Fundamental issues - clearly not "almost"
        "fundamentally wrong", "fundamental misunderstanding", "fundamental error",
        "completely wrong", "totally wrong", "absolutely wrong",
        "wrong approach", "incorrect approach", "invalid approach",
        "no understanding", "no valid mathematical", "nonsense", "gibberish",
        "blank", "empty", "no solution", "failed completely", "no attempt",
        # Major structural issues
        "major logical gap", "critical logical gap", "fatal flaw",
        "fatal error", "catastrophic error",
        # Very incomplete - clearly partial
        "far from complete", "nowhere near complete", "long way from complete",
        "less than 50%", "under 50%", "below 50%",
        "only 10%", "only 20%", "only 30%", "only 40%",
        "gave up immediately", "stopped immediately",
    ]
    
    has_disqualifier = any(d in reasoning_lower for d in almost_disqualifiers)
    
    if not has_disqualifier:
        for pattern in almost_patterns:
            if pattern in reasoning_lower:
                return "almost"
    
    # Strong indicators for "correct"
    correct_patterns = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "correct solution", "correct answer", "full marks", "full credit",
        "complete solution", "perfect solution", "excellent solution",
        "full understanding", "all correct", "everything correct",
        "correct throughout", "valid solution", "valid proof",
        "correct proof", "sound reasoning", "correct reasoning",
        "100% correct", "complete and correct", "fully solved",
        "correctly solved", "properly solved", "well done",
        "excellent work", "perfect work", "flawless", "no errors",
        "no mistakes", "accurate", "precise", "exactly right",
        "complete proof", "sound proof", "valid argument",
        "is correct", "are correct", "was correct", "were correct",
        "correctly derived", "correctly proved", "correctly shown",
        "correctly demonstrated", "correctly established", "correctly concluded",
        "valid derivation", "valid demonstration", "valid establishment",
        "sound solution", "sound answer", "sound argument",
        "rigorous proof", "rigorous solution", "rigorous argument",
        "complete and rigorous", "complete and sound", "complete and valid",
        "perfectly correct", "absolutely correct", "definitely correct",
        "undoubtedly correct", "clearly correct", "obviously correct",
        "demonstrates full", "shows full", "has full", "contains full",
        "complete understanding", "thorough understanding", "deep understanding",
        "mastered", "expertly solved", "elegantly solved", "beautifully solved"
    ]
    
    for pattern in correct_patterns:
        if pattern in reasoning_lower:
            return "correct"
    
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid categories.
    
    Args:
        prediction: Raw prediction string from LLM
        
    Returns:
        Normalized prediction: "correct", "almost", "partial", or "incorrect"
    """
    if not isinstance(prediction, str):
        return "incorrect"
    
    prediction_lower = prediction.lower().strip()
    
    # Direct matches (exact match takes highest priority)
    valid_categories = ["correct", "almost", "partial", "incorrect"]
    for cat in valid_categories:
        if prediction_lower == cat:
            return cat
    
    # Check for compound phrases FIRST before simple word matching
    # These compound phrases should take priority over simple word matches
    compound_indicators = {
        "partial": ["partially correct", "partial solution", "partial proof", "partially valid", "partially complete", 
                    "partially worked", "partially successful", "incomplete proof", "incomplete solution",
                    "partial result", "partial answer", "partial argument", "partially solved"],
        "almost": ["almost correct", "nearly correct", "almost complete", "nearly complete", 
                   "mostly correct", "mostly right", "essentially correct", "practically correct",
                   "almost solved", "nearly solved", "almost perfect", "nearly perfect"],
        "incorrect": ["not correct", "not right", "not valid", "wrong approach", "incorrect approach",
                      "not solved", "unsolved", "not a valid", "not an valid"]
    }
    
    # Check for compound indicators first
    for category, indicators in compound_indicators.items():
        for indicator in indicators:
            if indicator in prediction_lower:
                return category
    
    # Check for exact word matches with word boundaries (but be careful about context)
    words = re.findall(r'\b\w+\b', prediction_lower)
    
    # Check for standalone category words in the extracted words
    for cat in valid_categories:
        if cat in words:
            return cat
    
    # Only use word match if it's a standalone category word and not part of a compound
    for word in words:
        if word in valid_categories:
            # Check for negation context
            idx = prediction_lower.find(word)
            if idx != -1:
                before = prediction_lower[max(0, idx-10):idx]
                # Check for negation
                if "not " in before or "n't " in before:
                    continue
                # Check if preceded by "partially" (which would make it partial, not correct)
                if "partially " in before:
                    continue
                return word
    
    # Priority-based phrase matching with STRICT ordering
    # The key insight: "almost" should be very conservative - only when solution is nearly perfect
    # "partial" is for solutions with meaningful progress but significant gaps
    # "incorrect" is for solutions with no valid mathematical progress
    
    # First, check for clear "incorrect" indicators (highest priority for wrong answers)
    incorrect_indicators = [
        "no valid mathematical", "no progress", "fundamentally wrong", "completely wrong",
        "totally wrong", "nonsense", "gibberish", "invalid approach", "wrong approach",
        "no understanding", "no solution", "failed", "failure", "no attempt",
        "blank", "empty", "no answer", "missing", "irrelevant", "off topic",
        "fundamental misunderstanding", "trivial", "not valid", "does not understand",
        "did not understand", "0%", "zero", "nothing correct", "no correct",
        "completely incorrect", "totally incorrect", "absolutely wrong",
        "fundamental error", "critical error", "serious mistake", "major error",
        "not correct", "is wrong", "are wrong", "was wrong", "were wrong",
        "incorrect solution", "incorrect answer", "incorrect approach",
        "no valid", "invalid reasoning", "flawed reasoning", "unsound",
        "does not demonstrate", "failed to", "unable to", "cannot solve",
        "no meaningful progress", "no valid steps", "completely failed",
        "does not solve", "did not solve", "failed to solve", "unable to solve",
        "wrong answer", "wrong result", "wrong conclusion", "incorrect conclusion",
        "not a valid", "not an valid", "invalid solution", "invalid answer",
        "no proof", "no valid proof", "missing proof", "lacks proof",
        "does not prove", "did not prove", "failed to prove",
        "erroneous", "fallacious", "bogus", "invalid", "unsound proof",
        "not sound", "does not work", "did not work", "will not work"
    ]
    for indicator in incorrect_indicators:
        if indicator in prediction_lower:
            return "incorrect"
    
    # Check for "almost" indicators - be VERY conservative
    # "Almost" means: 85-99% complete, tiny fixable errors, would be perfect if errors fixed
    # Key distinction: "almost" requires the solution to be fundamentally correct
    # CRITICAL: Must check for disqualifying indicators first
    
    # First check if there are disqualifying indicators for "almost"
    # These indicate the solution is NOT nearly complete, so should be "partial" instead
    # REDUCED: Only the strongest disqualifiers that clearly indicate NOT "almost"
    almost_disqualifiers = [
        # Fundamental issues - clearly not "almost"
        "fundamentally wrong", "fundamental misunderstanding", "fundamental error",
        "completely wrong", "totally wrong", "absolutely wrong",
        "wrong approach", "incorrect approach", "invalid approach",
        "no understanding", "no valid mathematical", "nonsense", "gibberish",
        "blank", "empty", "no solution", "failed completely", "no attempt",
        # Major structural issues
        "major logical gap", "critical logical gap", "fatal flaw",
        "fatal error", "catastrophic error",
        # Very incomplete - clearly partial
        "far from complete", "nowhere near complete", "long way from complete",
        "less than 50%", "under 50%", "below 50%",
        "only 10%", "only 20%", "only 30%", "only 40%",
        "gave up immediately", "stopped immediately",
    ]
    has_almost_disqualifier = any(d in prediction_lower for d in almost_disqualifiers)
    
    # Only check for "almost" indicators if no disqualifiers present
    if not has_almost_disqualifier:
        almost_indicators = [
            "almost correct", "nearly correct", "almost complete", "nearly complete",
            "minor mistake only", "minor error only", "small error only",
            "tiny mistake", "slight error only", "essentially correct",
            "fundamentally correct", "correct approach with minor",
            "correct method with minor", "nearly perfect", "almost perfect",
            "mostly correct with minor", "correct except for minor",
            "correct except for small", "correct except for tiny",
            "would be perfect if", "would be correct if", "only a minor",
            "just a minor", "just a small", "only a small", "minor typo only",
            "small typo only", "minor calculation error", "small calculation error",
            "minor arithmetic error", "slight miscalculation", "tiny miscalculation",
            "minor oversight", "small oversight", "trivial error", "insignificant error",
            "cosmetic error", "formatting error", "notation error", "sign error",
            "minor gap", "small gap", "trivial gap", "nearly solved",
            "almost solved", "essentially solved", "practically correct",
            "one minor", "a minor", "the minor", "single minor",
            "one small", "a small", "the small", "single small",
            "one tiny", "a tiny", "the tiny", "single tiny",
            "85%", "90%", "95%", "99%", "nearly there", "very close",
            "minor correction", "small correction", "tiny correction",
            "minor fix", "small fix", "tiny fix", "slight fix",
            "minor adjustment", "small adjustment", "tiny adjustment",
            "minor refinement", "small refinement", "tiny refinement",
            "minor issue only", "small issue only", "tiny issue only",
            "minor flaw only", "small flaw only", "tiny flaw only",
            "minor problem only", "small problem only", "tiny problem only",
            "minor mistake only", "small mistake only", "tiny mistake only",
            "just needs", "only needs", "simply needs", "merely needs",
            "only requires", "just requires", "simply requires", "merely requires",
            "easily fixed", "easily corrected", "easily remedied",
            "simple fix", "simple correction", "simple remedy",
            "quick fix", "quick correction", "quick remedy",
            "trivial fix", "trivial correction", "trivial remedy"
        ]
        for indicator in almost_indicators:
            if indicator in prediction_lower:
                return "almost"
    
    # Check for "partial" indicators - broader than "almost" but requires valid progress
    # "Partial" means: 10-75% complete, has valid mathematical content but significant gaps
    # Key distinction: "partial" has meaningful progress but is NOT nearly complete
    # IMPORTANT: Check for partial indicators BEFORE correct indicators to avoid misclassification
    partial_indicators = [
        "partial credit", "partially correct", "incomplete solution",
        "incomplete proof", "partial solution", "partial progress",
        "significant progress", "meaningful progress", "valid insight",
        "key insight", "correct lemma", "proved a lemma", "identified key",
        "some understanding", "partial proof", "partial result",
        "some correct", "partially valid", "partial understanding",
        "correct framework", "valid reasoning", "partial success",
        "some valid", "partially worked", "partial attempt",
        "partially successful", "some correct steps", "valid partial",
        "partially complete", "not fully correct", "not completely correct",
        "partial marks", "some credit", "partially right", "incomplete but",
        "partial understanding demonstrated", "some valid reasoning",
        "made progress", "good start", "on the right track",
        "correct direction", "valid approach", "substantial progress",
        "important insight", "major progress", "significant portion",
        "half correct", "partially worked out", "some progress",
        "incomplete execution", "started correctly", "began correctly",
        "correct beginning", "valid start", "some valid steps",
        "partially valid solution", "incomplete but valid",
        "missing steps", "incomplete reasoning", "unfinished",
        "needs more work", "requires completion", "incomplete answer",
        "partial answer", "some right", "partly correct", "partly right",
        "50%", "halfway", "incomplete work", "unfinished solution",
        "10%", "20%", "25%", "30%", "40%", "60%", "70%", "75%",
        "did not finish", "incomplete argument", "partial argument",
        "started well", "began well", "good beginning", "valid beginning",
        "significant gaps", "major gaps", "large gaps", "substantial gaps",
        "some progress", "limited progress", "modest progress",
        "reasonable start", "reasonable beginning", "reasonable attempt",
        "valid attempt", "legitimate attempt", "serious attempt",
        "demonstrates some", "shows some", "has some", "contains some",
        "identified correctly", "correctly identified", "recognized correctly",
        "correct intuition", "valid intuition", "good intuition",
        "correct strategy", "valid strategy", "good strategy",
        "correct idea", "valid idea", "good idea", "useful idea",
        "correct observation", "valid observation", "useful observation",
        "correct insight", "valid insight", "useful insight",
        "partially successful", "partially complete", "partially correct",
        "incomplete but promising", "incomplete but valid", "incomplete but sound",
        "has merit", "has value", "has substance", "has content",
        "not entirely wrong", "not completely wrong", "not totally wrong",
        "not entirely incorrect", "not completely incorrect", "not totally incorrect",
        "not entirely invalid", "not completely invalid", "not totally invalid",
        "not nonsense", "not gibberish", "not garbage", "not rubbish",
        "not trivial", "not empty", "not blank", "not missing",
        "not a failure", "not failed", "not useless", "not worthless",
        # Additional patterns to better distinguish from "almost"
        "far from complete", "nowhere near complete", "long way from complete",
        "stopped early", "stopped prematurely", "gave up",
        "only proved", "only showed", "only demonstrated",
        "did not complete", "did not finish", "did not solve",
        "missing major", "missing significant", "lacks major", "lacks significant",
        "needs substantial", "requires substantial", "needs considerable",
        "incomplete execution", "incomplete implementation",
        "partial result", "partial answer", "incomplete solution",
        "verification contains minor mistakes", "minor mistakes only",
        "found a correct invariant", "found an invariant",
        "proved a useful", "proved one direction", "proved one part",
        "only one direction", "only one part", "partial case",
        "some cases only", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis"
    ]
    for indicator in partial_indicators:
        if indicator in prediction_lower:
            return "partial"
    
    # Check for "correct" indicators - must be very clear
    # These should be specific phrases that clearly indicate a complete correct solution
    correct_indicators = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "correct solution", "correct answer", "full marks", "full credit",
        "complete solution", "perfect solution", "excellent solution",
        "full understanding", "all correct", "everything correct",
        "correct throughout", "valid solution", "valid proof",
        "correct proof", "sound reasoning", "correct reasoning",
        "100% correct", "complete and correct", "fully solved",
        "correctly solved", "properly solved", "well done",
        "excellent work", "perfect work", "flawless", "no errors",
        "no mistakes", "accurate", "precise", "exactly right",
        "complete proof", "sound proof", "valid argument",
        "is correct", "are correct", "was correct", "were correct",
        "correctly derived", "correctly proved", "correctly shown",
        "correctly demonstrated", "correctly established", "correctly concluded",
        "valid derivation", "valid demonstration", "valid establishment",
        "sound solution", "sound answer", "sound argument",
        "rigorous proof", "rigorous solution", "rigorous argument",
        "complete and rigorous", "complete and sound", "complete and valid",
        "perfectly correct", "absolutely correct", "definitely correct",
        "undoubtedly correct", "clearly correct", "obviously correct",
        "demonstrates full", "shows full", "has full", "contains full",
        "complete understanding", "thorough understanding", "deep understanding",
        "mastered", "expertly solved", "elegantly solved", "beautifully solved"
    ]
    for indicator in correct_indicators:
        if indicator in prediction_lower:
            return "correct"
    
    # Check for negated correct (which means incorrect)
    if ("not correct" in prediction_lower or 
        "not fully" in prediction_lower or
        "not completely" in prediction_lower or
        "not entirely" in prediction_lower or
        "not totally" in prediction_lower or
        "not accurate" in prediction_lower or
        "not right" in prediction_lower or
        "not valid" in prediction_lower or
        "not entirely" in prediction_lower or
        "not 100%" in prediction_lower or
        "not perfect" in prediction_lower or
        "not flawless" in prediction_lower):
        return "incorrect"
    
    # Check for "wrong" or "error" without "minor" or "small" qualifier
    if ("wrong" in prediction_lower or "error" in prediction_lower or "mistake" in prediction_lower):
        # Check if it's qualified as minor/small/tiny
        if not any(q in prediction_lower for q in ["minor", "small", "tiny", "slight", "trivial", "cosmetic", "insignificant", "negligible", "minimal"]):
            return "incorrect"
    
    # Check for "mostly" - handle cases not caught by compound indicators
    if "mostly" in prediction_lower:
        # If "mostly wrong" or "mostly incorrect" -> incorrect
        if "mostly wrong" in prediction_lower or "mostly incorrect" in prediction_lower:
            return "incorrect"
        # If "mostly incomplete" or "mostly partial" -> partial
        if "mostly incomplete" in prediction_lower or "mostly partial" in prediction_lower:
            return "partial"
        # If "mostly correct" or "mostly right" -> almost (if no disqualifiers)
        if ("mostly correct" in prediction_lower or "mostly right" in prediction_lower) and not has_almost_disqualifier:
            return "almost"
    
    # Check for "nearly" - handle cases not caught by compound indicators
    if "nearly" in prediction_lower:
        if "nearly wrong" in prediction_lower or "nearly incorrect" in prediction_lower:
            return "incorrect"
        if "nearly complete" in prediction_lower or "nearly correct" in prediction_lower:
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
    
    # Check for "almost" as standalone word
    if "almost" in prediction_lower:
        # Check for strong partial qualifiers first
        if any(q in prediction_lower for q in ["incomplete", "unfinished", "missing major", "significant gap", "major gap", "large gap"]):
            return "partial"
        if not has_almost_disqualifier:
            return "almost"
        else:
            return "partial"
    
    # Check for "essentially" - often indicates "almost"
    if "essentially" in prediction_lower:
        if "essentially correct" in prediction_lower or "essentially right" in prediction_lower:
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
        if "essentially wrong" in prediction_lower or "essentially incorrect" in prediction_lower:
            return "incorrect"
    
    # Check for "practically" - often indicates "almost"
    if "practically" in prediction_lower:
        if "practically correct" in prediction_lower or "practically right" in prediction_lower:
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
    
    # Fallback: if just "correct" appears without negation or qualification
    if "correct" in prediction_lower:
        # Check for strong partial qualifiers first
        if any(q in prediction_lower for q in ["partially", "not fully", "incomplete", "unfinished", "missing", "not complete", "not entirely"]):
            return "partial"
        # Check for almost qualifiers  
        if any(q in prediction_lower for q in ["almost", "nearly", "mostly", "essentially", "practically", "virtually", "basically"]):
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
        return "correct"
    
    # Check for "right" as synonym for correct
    if "right" in prediction_lower:
        if any(q in prediction_lower for q in ["partially", "not fully", "incomplete", "not entirely", "not completely"]):
            return "partial"
        if any(q in prediction_lower for q in ["almost", "nearly", "mostly", "essentially", "practically", "virtually", "basically"]):
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
        if "not right" in prediction_lower or "wrong" in prediction_lower:
            return "incorrect"
        return "correct"
    
    # Check for "solved" indicators
    if "solved" in prediction_lower:
        if "not solved" in prediction_lower or "unsolved" in prediction_lower:
            return "incorrect"
        if "partially solved" in prediction_lower or "incompletely solved" in prediction_lower:
            return "partial"
        if "almost solved" in prediction_lower or "nearly solved" in prediction_lower:
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
        return "correct"
    
    # Check for "valid" indicators
    if "valid" in prediction_lower:
        if "not valid" in prediction_lower or "invalid" in prediction_lower:
            return "incorrect"
        if "partially valid" in prediction_lower:
            return "partial"
        if "mostly valid" in prediction_lower or "nearly valid" in prediction_lower:
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
        return "correct"
    
    # Check for "complete" indicators
    if "complete" in prediction_lower:
        if "not complete" in prediction_lower or "incomplete" in prediction_lower:
            return "partial"
        if "almost complete" in prediction_lower or "nearly complete" in prediction_lower:
            if not has_almost_disqualifier:
                return "almost"
            else:
                return "partial"
        return "correct"
    
    # Check for "sound" indicators
    if "sound" in prediction_lower:
        if "not sound" in prediction_lower or "unsound" in prediction_lower:
            return "incorrect"
        if "partially sound" in prediction_lower:
            return "partial"
        return "correct"
    
    # Check for "good" indicators - often indicates partial or better
    if "good" in prediction_lower:
        if "not good" in prediction_lower or "no good" in prediction_lower:
            return "incorrect"
        if "good start" in prediction_lower or "good beginning" in prediction_lower:
            return "partial"
        if "good progress" in prediction_lower or "good work" in prediction_lower:
            return "partial"
    
    # Check for "progress" indicators
    if "progress" in prediction_lower:
        if "no progress" in prediction_lower or "not progress" in prediction_lower:
            return "incorrect"
        if "some progress" in prediction_lower or "limited progress" in prediction_lower:
            return "partial"
        if "significant progress" in prediction_lower or "meaningful progress" in prediction_lower:
            return "partial"
    
    # Check for "understanding" indicators
    if "understanding" in prediction_lower:
        if "no understanding" in prediction_lower or "not understanding" in prediction_lower:
            return "incorrect"
        if "some understanding" in prediction_lower or "partial understanding" in prediction_lower:
            return "partial"
        if "full understanding" in prediction_lower or "complete understanding" in prediction_lower:
            return "correct"
    
    # Default fallback to incorrect for safety
    return "incorrect"


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
        # Build a more structured prompt with clear instructions
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate the student's answer and classify it into exactly ONE category.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Categories (choose exactly one):

**"correct"** - Complete, correct solution with sound mathematical reasoning. May have negligible typos. The solution demonstrates full understanding and arrives at the correct answer with valid proof. The solution is 100% complete.

**"almost"** - Nearly complete solution (80-99%) with fundamentally correct approach. The solution is VERY CLOSE to being perfect - only tiny, easily fixable issues remain:
- Small calculation errors (arithmetic mistakes, sign errors) in final steps
- Minor typos or notation issues that don't affect understanding
- Missing one trivial edge case that doesn't affect the main result
- Small gaps that are obvious to fill with minimal effort
- The core proof structure is complete and correct
- Verification contains only minor mistakes

KEY TEST for "almost": Would this solution be "correct" with just a few minutes of minor corrections? If YES → "almost".

CRITICAL: "almost" means the student clearly understands the problem deeply and has essentially solved it, just with tiny slips. The solution should feel "nearly perfect" not "incomplete".

**"partial"** - Meaningful progress (10-75%) with SOME valid mathematical content but SIGNIFICANT gaps remaining:
- Proved a useful lemma or sub-result but main proof incomplete
- Identified a key insight or correct approach but execution stopped early
- Started with valid reasoning but major portions missing
- Has correct framework but missing substantial execution steps
- Valid mathematical work exists but far from a complete solution

KEY TEST for "partial": Did the student make genuine mathematical progress but clearly stopped well short of a complete solution? If YES → "partial".

CRITICAL: "partial" means there's real substance but substantial work remains. The solution should feel "incomplete" or "unfinished" - not "nearly done".

**"incorrect"** - No valid mathematical progress. Wrong approach, fundamental misunderstanding, nonsense, or empty response. KEY TEST: Is there any valid mathematical contribution toward solving the problem? If NO → "incorrect".

## CRITICAL DECISION RULES - APPLY IN ORDER:
1. **Is it fully correct?** (Complete proof, correct answer, sound reasoning, 100% complete) → "correct"
2. **Is it nearly perfect with only tiny fixable errors?** (80-99% complete, minor calculation errors, small typos, trivial case missing - would be perfect if fixed) → "almost"
3. **Is there meaningful valid progress despite significant gaps?** (10-75% complete, proved a lemma, identified key insight, correct approach started but incomplete) → "partial"
4. **Otherwise** (wrong approach, no valid progress, nonsense) → "incorrect"

## CRITICAL DISTINCTIONS - READ CAREFULLY:

**"almost" vs "partial" - THIS IS THE HARDEST DISTINCTION:**

**"ALMOST" means NEARLY PERFECT:**
- The solution is 80-99% complete and feels "essentially done"
- Core proof structure is complete and correct
- Only tiny, easily fixable issues remain (minor arithmetic, small typos)
- Student clearly understands the problem deeply
- Would be correct with just a few minutes of corrections
- The solution feels "nearly perfect" not "incomplete"

**EXAMPLES of "almost":**
- Complete correct proof with one small arithmetic error in final step
- Correct solution with minor notation typo (wrote x instead of y)
- Valid proof missing only one trivial edge case
- Correct approach with small sign error that's easy to fix
- Solution is 80%+ complete, just needs tiny polish
- Proved main result but verification has minor mistakes
- Correct solution with one small gap that's easy to fill

**"PARTIAL" means INCOMPLETE with gaps:**
- The solution is 10-75% complete and feels "unfinished"
- There's valid mathematical content BUT major gaps remain
- Student made progress but stopped well short of completion
- Significant additional work would be needed
- The solution feels "incomplete" not "nearly done"

**EXAMPLES of "partial":**
- Proved a useful lemma but main proof not started
- Identified correct approach but only 30-50% executed
- Started valid reasoning but stopped with major work remaining
- Has correct framework but missing substantial execution
- Solution is 40-60% complete with big gaps

**DECISION HEURISTIC - USE THIS:**
1. Estimate the completeness percentage:
   - 95-100% → "correct" (or "almost" if tiny errors)
   - 80-94% → "almost" (nearly done, minor fixes needed)
   - 10-75% → "partial" (meaningful progress but far from done)
   - 0-10% → "incorrect" (no valid progress)

2. Ask: "If I fixed all errors in 5 minutes, would this be complete?"
   - YES → "almost"
   - NO, still needs substantial work → "partial"

3. Ask: "Does this feel 'nearly perfect' or 'incomplete'?"
   - Nearly perfect → "almost"
   - Incomplete → "partial"

**WHEN IN DOUBT between "almost" and "partial":**
- If the solution is 80%+ complete with minor issues → "almost"
- If the solution is 50-79% complete → "partial"
- "Almost" should be used when the solution is nearly done
- "Partial" should be used when there's significant work remaining
- If you think "this is pretty good but needs more work" → "partial"
- If you think "this is nearly perfect except for tiny issues" → "almost"

**"partial" vs "incorrect":**
- "partial": At least some valid mathematical contribution (proved something useful, identified a key insight, correct approach started)
- "incorrect": No valid progress at all (wrong approach, nonsense, empty)

## Output Format:
Respond ONLY in this JSON format:
<json>
{{
    "reasoning": "Brief explanation of why this category was chosen, focusing on completeness percentage and error severity",
    "response": "correct"
}}</json>

The response value MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase).

## FINAL INSTRUCTION:
Be precise in your classification. When deciding between "almost" and "partial", ask yourself: 
1. "What percentage complete is this solution?"
2. "Are the errors minor and easily fixable, or are there significant gaps?"
3. "Would this be a complete solution with tiny fixes, or does it need substantial additional work?"

**CRITICAL**: If the solution has ANY of the following, it is NOT "almost":
- Wrong approach or fundamental misunderstanding
- Multiple significant errors
- Less than 80% completeness
- Missing major portions of the solution
- "Partial" appears in the reasoning (indicates partial credit thinking)

**CRITICAL**: If the solution has ANY of the following, it is NOT "partial":
- 85-99% complete with only minor issues → "almost" or "correct"
- Nearly complete with tiny fixable errors → "almost"

If truly uncertain, prefer "partial" over "almost" to be conservative.

## ADDITIONAL GUIDANCE FOR EDGE CASES:

**When the solution has both correct and incorrect parts:**
- If the correct parts form a nearly complete solution with only minor errors → "almost"
- If the correct parts represent meaningful progress but significant work remains → "partial"
- If the incorrect parts dominate or the approach is fundamentally wrong → "incorrect"

**When the solution is incomplete:**
- If it's 80-99% complete with only minor gaps → "almost"
- If it's 10-75% complete with substantial gaps → "partial"
- If it's less than 10% complete or the progress is not valid → "incorrect"

**When the solution has calculation errors:**
- If the approach is correct and only minor arithmetic errors exist → "almost"
- If the approach is correct but major calculations are wrong or missing → "partial"
- If the approach itself is wrong → "incorrect"

**When the solution uses the right approach:**
- If the approach is correct and nearly fully executed → "almost" or "correct"
- If the approach is correct but execution is incomplete → "partial"
- If the approach is correct but fundamentally misapplied → "incorrect"

**When the solution mentions "partial":**
- If the reasoning uses words like "partial", "incomplete", "unfinished", "missing major" → "partial"
- "Almost" solutions should feel complete, not partial

**When the solution has verification or checking:**
- If verification contains minor mistakes only → "almost"
- If verification is incomplete or has major gaps → "partial"
- If verification is completely wrong or missing → "incorrect"

"""

        # Initialize msg_history as empty list in case of early failure
        msg_history = []
        
        # Validate inputs to prevent errors
        if not isinstance(inputs, dict):
            self.log_fn(f"Invalid inputs type: {type(inputs)}")
            return "incorrect", [{"role": "error", "text": f"Invalid inputs type: {type(inputs)}"}]
        
        # Check for required fields
        required_fields = ['problem', 'solution', 'student_answer']
        missing_fields = [f for f in required_fields if f not in inputs or not inputs.get(f)]
        if missing_fields:
            self.log_fn(f"Missing required fields: {missing_fields}")
            # Continue anyway, as grading_guidelines might be optional
        
        # Validate and sanitize input content to prevent injection issues
        try:
            problem = str(inputs.get('problem', '')).strip()
            solution = str(inputs.get('solution', '')).strip()
            student_answer = str(inputs.get('student_answer', '')).strip()
            grading_guidelines = str(inputs.get('grading_guidelines', '')).strip()
            domain = str(inputs.get('domain', 'Mathematics')).strip()
            
            # Check for empty critical fields
            if not problem or not solution or not student_answer:
                self.log_fn("Critical fields are empty after sanitization")
                return "incorrect", [{"role": "error", "text": "Critical input fields are empty"}]
        except Exception as e:
            self.log_fn(f"Error sanitizing inputs: {e}")
            return "incorrect", [{"role": "error", "text": f"Input sanitization failed: {e}"}]
        
        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "incorrect", [{"role": "error", "text": f"LLM call failed: {e}"}]

        # Extract prediction from JSON using flexible extraction
        prediction = "incorrect"  # Default to incorrect for safety
        
        # Handle edge case: msg_history is None
        if msg_history is None:
            self.log_fn("msg_history is None")
            return str(prediction), [{"role": "error", "text": "No response from LLM"}]
        
        # Handle edge case: msg_history is not a list
        if not isinstance(msg_history, list):
            self.log_fn(f"Invalid msg_history type: {type(msg_history)}")
            # Try to extract from string if possible
            if isinstance(msg_history, str):
                try:
                    prediction = _normalize_prediction(msg_history.lower())
                    self.log_fn(f"Extracted from string msg_history: {prediction}")
                except Exception:
                    pass
            return str(prediction), [{"role": "error", "text": f"Invalid response type: {type(msg_history)}"}]
        
        # Handle edge case: empty msg_history
        if len(msg_history) == 0:
            self.log_fn("msg_history is empty")
            return str(prediction), msg_history
        
        try:
            last_message = msg_history[-1]
            
            # Handle edge case: last message is not a dict
            if not isinstance(last_message, dict):
                self.log_fn(f"Last message is not a dict: {type(last_message)}")
                # Try to extract from string directly if it's a string
                if isinstance(last_message, str):
                    prediction = _normalize_prediction(last_message.lower())
                    self.log_fn(f"Extracted from string message: {prediction}")
                return str(prediction), msg_history
            
            # Handle edge case: missing 'text' key
            if "text" not in last_message:
                self.log_fn(f"Last message missing 'text' key. Keys: {list(last_message.keys())}")
                # Try to find any string value that might be the response
                for key, value in last_message.items():
                    if isinstance(value, str) and len(value) > 10:
                        prediction = _normalize_prediction(value.lower())
                        self.log_fn(f"Extracted from alternative key '{key}': {prediction}")
                        return str(prediction), msg_history
                    # Also check for nested dict with 'content' or 'message' key
                    if isinstance(value, dict):
                        for nested_key in ['content', 'message', 'text', 'response']:
                            if nested_key in value and isinstance(value[nested_key], str):
                                prediction = _normalize_prediction(value[nested_key].lower())
                                self.log_fn(f"Extracted from nested key '{key}.{nested_key}': {prediction}")
                                return str(prediction), msg_history
                return str(prediction), msg_history
            
            text_content = last_message["text"]
            
            # Handle edge case: text is not a string
            if not isinstance(text_content, str):
                self.log_fn(f"Text content is not a string: {type(text_content)}")
                # Try to convert to string if possible
                try:
                    text_content = str(text_content)
                except Exception:
                    return str(prediction), msg_history
            
            # Handle edge case: empty text
            if not text_content or not text_content.strip():
                self.log_fn("Text content is empty")
                return str(prediction), msg_history
            
            # Try to extract JSON
            extracted = _extract_json_flexible(text_content)
            
            if extracted and len(extracted) > 0:
                last_json = extracted[-1]
                
                if isinstance(last_json, dict):
                    # Try multiple possible keys for the response (prioritize 'response' as per our prompt)
                    response_keys = ["response", "answer", "prediction", "grade", "result", "category", "evaluation", "classification", "label", "output"]
                    raw_prediction = None
                    matched_key = None
                    
                    for key in response_keys:
                        if key in last_json:
                            raw_prediction = last_json[key]
                            matched_key = key
                            break
                    
                    # If no standard key found, try any key that might contain the prediction
                    if raw_prediction is None:
                        for key, value in last_json.items():
                            if isinstance(value, str):
                                val_lower = value.lower().strip()
                                if val_lower in ["correct", "almost", "partial", "incorrect"]:
                                    raw_prediction = value
                                    matched_key = key
                                    break
                    
                    # Also check reasoning field for additional context
                    reasoning_text = ""
                    if "reasoning" in last_json and isinstance(last_json["reasoning"], str):
                        reasoning_text = last_json["reasoning"].lower()
                    
                    # Analyze reasoning to detect potential misclassification
                    reasoning_suggestion = _analyze_reasoning_for_category(reasoning_text) if reasoning_text else None
                    
                    if raw_prediction is not None:
                        # Handle different types of raw_prediction
                        if isinstance(raw_prediction, str):
                            # First normalize the raw prediction
                            normalized_pred = _normalize_prediction(raw_prediction)
                            
                            # IMPROVED: Better handling of almost vs partial distinction
                            # Use combined text for more accurate classification
                            combined_text = raw_prediction
                            if reasoning_text:
                                combined_text = f"{raw_prediction} {reasoning_text}"
                            combined_prediction = _normalize_prediction(combined_text)
                            
                            # If reasoning suggests a different category, use reasoning
                            # This helps catch cases where the response field is wrong
                            if reasoning_suggestion and reasoning_suggestion != normalized_pred:
                                # Special case: if reasoning suggests "partial" but response says "almost",
                                # trust reasoning (reasoning is usually more accurate about completeness)
                                # This is the most common misclassification to catch
                                if reasoning_suggestion == "partial" and normalized_pred == "almost":
                                    self.log_fn(f"Reasoning suggests 'partial' but response says 'almost'. Using reasoning.")
                                    prediction = reasoning_suggestion
                                # Special case: if reasoning suggests "incorrect" but response says something else,
                                # trust reasoning (incorrect is a strong signal)
                                elif reasoning_suggestion == "incorrect":
                                    self.log_fn(f"Reasoning suggests 'incorrect' but response says '{normalized_pred}'. Using reasoning.")
                                    prediction = reasoning_suggestion
                                # Special case: if reasoning suggests "almost" and response says "partial",
                                # check if reasoning has strong "almost" indicators AND no disqualifiers
                                elif reasoning_suggestion == "almost" and normalized_pred == "partial":
                                    # IMPROVED: More nuanced check for "almost" vs "partial"
                                    # Check for strong "almost" indicators in combined text
                                    has_strong_almost = ("nearly correct" in combined_text or 
                                                        "almost correct" in combined_text or 
                                                        "minor mistake" in combined_text or
                                                        "minor error" in combined_text or
                                                        "small error" in combined_text or
                                                        "tiny" in combined_text or
                                                        "essentially correct" in combined_text or
                                                        "verification contains minor" in combined_text)
                                    # Check for strong "partial" indicators
                                    has_strong_partial = ("incomplete proof" in combined_text or
                                                         "unfinished" in combined_text or
                                                         "missing major" in combined_text or
                                                         "significant gap" in combined_text or
                                                         "stopped early" in combined_text or
                                                         "gave up" in combined_text or
                                                         "only proved" in combined_text or
                                                         "far from complete" in combined_text)
                                    if has_strong_almost and not has_strong_partial:
                                        self.log_fn(f"Reasoning strongly suggests 'almost' over 'partial'. Using reasoning.")
                                        prediction = reasoning_suggestion
                                    else:
                                        prediction = normalized_pred
                                # Special case: if reasoning suggests "correct" but response says something else,
                                # check if reasoning supports "correct"
                                elif reasoning_suggestion == "correct" and normalized_pred in ["almost", "partial", "incorrect"]:
                                    # Only trust "correct" if reasoning is very clear
                                    if "fully correct" in reasoning_text or "completely correct" in reasoning_text or "100%" in reasoning_text:
                                        self.log_fn(f"Reasoning strongly suggests 'correct' over '{normalized_pred}'. Using reasoning.")
                                        prediction = reasoning_suggestion
                                    else:
                                        prediction = normalized_pred
                                else:
                                    prediction = normalized_pred
                            else:
                                # Use combined prediction for better accuracy
                                prediction = combined_prediction
                        elif isinstance(raw_prediction, (int, float)):
                            # Handle numeric predictions (unlikely but possible)
                            prediction = "incorrect"
                        elif isinstance(raw_prediction, bool):
                            prediction = "correct" if raw_prediction else "incorrect"
                        else:
                            prediction = _normalize_prediction(str(raw_prediction))
                        
                        self.log_fn(f"Raw prediction from key '{matched_key}': {raw_prediction} -> Normalized: {prediction}")
                        if reasoning_suggestion:
                            self.log_fn(f"Reasoning analysis suggests: {reasoning_suggestion}")
                    else:
                        self.log_fn(f"JSON missing response key. Available keys: {list(last_json.keys())}")
                        # Try to extract from the entire JSON as string, including reasoning
                        try:
                            json_str = json.dumps(last_json).lower()
                            prediction = _normalize_prediction(json_str)
                            self.log_fn(f"Extracted from JSON string: {prediction}")
                        except Exception:
                            pass
                else:
                    self.log_fn(f"Last JSON is not a dict: {type(last_json)}")
                    # If it's a string, try to normalize it directly
                    if isinstance(last_json, str):
                        prediction = _normalize_prediction(last_json)
                        self.log_fn(f"Extracted from JSON string value: {prediction}")
            else:
                # Try to extract directly from text if no JSON found
                text_lower = text_content.lower()
                prediction = _normalize_prediction(text_lower)
                self.log_fn(f"No JSON found, extracted from text: {prediction}")
                
        except json.JSONDecodeError as e:
            self.log_fn(f"JSON decode error: {e}")
            # Try to extract from raw text as fallback
            try:
                if msg_history and isinstance(msg_history, list) and len(msg_history) > 0:
                    last_msg = msg_history[-1]
                    if isinstance(last_msg, dict):
                        text_content = last_msg.get("text", "")
                        if isinstance(text_content, str):
                            prediction = _normalize_prediction(text_content.lower())
                            self.log_fn(f"Fallback extraction after JSON error: {prediction}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction failed: {fallback_e}")
                
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        # Final validation: ensure prediction is one of the valid categories
        valid_categories = ["correct", "almost", "partial", "incorrect"]
        if prediction not in valid_categories:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to 'incorrect'")
            prediction = "incorrect"
        
        # Additional safety check: if prediction is None or empty, default to incorrect
        if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
            self.log_fn("Prediction is None or empty, defaulting to 'incorrect'")
            prediction = "incorrect"

        return str(prediction), msg_history
