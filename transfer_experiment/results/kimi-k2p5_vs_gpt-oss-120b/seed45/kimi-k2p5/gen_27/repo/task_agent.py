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
    """Analyze reasoning text to detect category indicators with improved accuracy.
    
    This helps catch cases where the reasoning clearly indicates one category
    but the response field might have a different value.
    
    The key insight: "almost" requires 85-99% completeness with tiny fixable errors.
    "partial" means 20-75% complete with significant gaps remaining.
    
    Args:
        reasoning: The reasoning text from the LLM response
        
    Returns:
        Suggested category or None if no strong indicators found
    """
    if not isinstance(reasoning, str) or not reasoning:
        return None
    
    reasoning_lower = reasoning.lower()
    
    # ============================================================================
    # STEP 1: Check for "incorrect" indicators (highest priority for wrong answers)
    # ============================================================================
    incorrect_patterns = [
        "no valid mathematical", "no progress", "fundamentally wrong",
        "completely wrong", "totally wrong", "absolutely wrong", "utterly wrong",
        "nonsense", "gibberish", "garbage", "rubbish", "invalid approach",
        "wrong approach", "no understanding", "no solution", "failed completely",
        "no attempt", "blank", "empty", "no answer", "missing answer",
        "fundamental misunderstanding", "fundamental error", "critical error",
        "fatal error", "catastrophic error", "serious mistake", "major error",
        "completely incorrect", "totally incorrect", "absolutely incorrect",
        "not correct", "is wrong", "are wrong", "was wrong", "were wrong",
        "incorrect solution", "incorrect answer", "incorrect approach",
        "no valid", "invalid reasoning", "flawed reasoning", "unsound reasoning",
        "does not demonstrate", "failed to", "unable to", "cannot solve",
        "no meaningful progress", "no valid steps", "completely failed",
        "does not solve", "did not solve", "failed to solve", "unable to solve",
        "wrong answer", "wrong result", "wrong conclusion", "incorrect conclusion",
        "not a valid", "not an valid", "invalid solution", "invalid answer",
        "no proof", "no valid proof", "missing proof", "lacks proof",
        "does not prove", "did not prove", "failed to prove",
        "erroneous", "fallacious", "bogus", "invalid", "unsound",
        "0%", "zero percent", "nothing correct", "no correct",
        "irrelevant", "off topic", "not related", "does not address",
        "did not address", "failed to address", "missed the point",
    ]
    
    for pattern in incorrect_patterns:
        if pattern in reasoning_lower:
            return "incorrect"
    
    # ============================================================================
    # STEP 2: Check for "almost" disqualifiers - these strongly indicate NOT "almost"
    # If ANY of these are present, the solution CANNOT be "almost"
    # ============================================================================
    # IMPROVED: More comprehensive disqualifiers to catch partial cases misclassified as almost
    almost_disqualifiers = [
        # Fundamental issues - clearly not "almost"
        "fundamentally wrong", "fundamental misunderstanding", "fundamental error",
        "wrong approach", "incorrect approach", "invalid approach",
        "no understanding", "no valid mathematical", "nonsense", "gibberish",
        "blank", "empty", "no solution", "failed completely", "no attempt",
        # Major structural issues
        "major logical gap", "critical logical gap", "fatal flaw",
        "fatal error", "catastrophic error", "serious structural",
        # Very incomplete - clearly partial (less than 85%)
        "far from complete", "nowhere near complete", "long way from complete",
        "less than 50%", "under 50%", "below 50%", "less than 60%", "under 60%",
        "less than 70%", "under 70%", "less than 80%", "under 80%",
        "less than 85%", "under 85%",
        "only 10%", "only 20%", "only 30%", "only 40%", "only 50%",
        "only 60%", "only 70%", "only 80%",
        "gave up immediately", "stopped immediately", "stopped very early",
        # Incomplete indicators - these are the key differentiators
        "incomplete proof", "incomplete solution", "unfinished",
        "missing major", "missing significant", "lacks major", "lacks significant",
        "did not complete", "did not finish", "did not solve",
        "stopped early", "gave up", "only proved", "only showed",
        "partial case", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis",
        # Progress indicators that suggest partial (not almost)
        "some progress", "limited progress", "modest progress",
        "made progress", "good start", "started correctly", "began correctly",
        "good beginning", "valid beginning", "reasonable start",
        "on the right track", "correct direction", "correct approach started",
        # Significant gaps remaining
        "significant gaps", "major gaps", "large gaps", "substantial gaps",
        "considerable gaps", "important gaps", "critical gaps",
        "needs substantial work", "requires substantial work",
        "needs considerable work", "requires considerable work",
        "needs significant work", "requires significant work",
        "incomplete execution", "incomplete implementation", "unfinished work",
        "work remaining", "more work needed", "further work needed",
        # Partial result indicators
        "partial result", "partial answer", "incomplete answer",
        "only one direction", "only one part", "proved a lemma",
        "proved a useful lemma", "found an invariant", "found a correct invariant",
        "some cases only", "not all cases", "missing cases",
        # Multiple errors (almost should have only tiny errors)
        "multiple errors", "several errors", "many errors", "various errors",
        "multiple mistakes", "several mistakes", "many mistakes",
        # Did not reach conclusion
        "did not reach", "failed to reach", "did not arrive",
        "did not conclude", "failed to conclude", "no conclusion",
        # Additional disqualifiers for "almost"
        "not all steps", "missing steps", "incomplete steps",
        "stopped prematurely", "ended prematurely",
        "did not fully", "not fully developed", "not fully executed",
        "lacks completion", "lacks conclusion", "no final answer",
        "work in progress", "unfinished business", "incomplete work",
    ]
    
    has_almost_disqualifier = any(d in reasoning_lower for d in almost_disqualifiers)
    
    # ============================================================================
    # STEP 3: Check for "almost" indicators - ONLY if no disqualifiers present
    # "Almost" means: 85-99% complete, tiny fixable errors, nearly perfect
    # ============================================================================
    if not has_almost_disqualifier:
        # Strong "almost" patterns - these clearly indicate "almost"
        strong_almost_patterns = [
            # Explicit "almost" phrases
            "almost correct", "nearly correct", "almost complete", "nearly complete",
            "almost perfect", "nearly perfect", "almost solved", "nearly solved",
            "essentially correct", "practically correct", "virtually correct",
            "mostly correct", "mostly right", "essentially right", "practically right",
            # Minor error only
            "minor mistake only", "minor error only", "small error only",
            "tiny mistake only", "slight error only", "trivial error only",
            "minor issue only", "small issue only", "tiny issue only",
            "minor flaw only", "small flaw only", "tiny flaw only",
            "minor problem only", "small problem only", "tiny problem only",
            "minor gap only", "small gap only", "tiny gap only",
            # Single minor error
            "one minor", "a minor", "the minor", "single minor",
            "one small", "a small", "the small", "single small",
            "one tiny", "a tiny", "the tiny", "single tiny",
            "just a minor", "just a small", "just a tiny",
            "only a minor", "only a small", "only a tiny",
            # Specific error types that are minor
            "minor typo", "small typo", "tiny typo", "slight typo",
            "minor calculation error", "small calculation error", "tiny calculation error",
            "minor arithmetic error", "small arithmetic error", "tiny arithmetic error",
            "slight miscalculation", "tiny miscalculation", "minor miscalculation",
            "sign error", "sign mistake", "wrong sign",
            "minor oversight", "small oversight", "tiny oversight",
            "trivial error", "insignificant error", "negligible error",
            "cosmetic error", "formatting error", "notation error",
            "minor gap", "small gap", "trivial gap", "tiny gap",
            # Would be perfect/correct if...
            "would be perfect if", "would be correct if", "would be right if",
            "would be valid if", "would be complete if",
            "just needs", "only needs", "simply needs", "merely needs",
            "just requires", "only requires", "simply requires", "merely requires",
            # Easy to fix
            "easily fixed", "easily corrected", "easily remedied",
            "simple fix", "simple correction", "simple remedy",
            "quick fix", "quick correction", "quick remedy",
            "trivial fix", "trivial correction", "trivial remedy",
            "minor correction", "small correction", "tiny correction",
            "minor fix", "small fix", "tiny fix", "slight fix",
            # High completion indicators
            "85%", "90%", "95%", "99%",
            "about 85%", "about 90%", "about 95%", "about 99%",
            "roughly 85%", "roughly 90%", "roughly 95%", "roughly 99%",
            "approximately 85%", "approximately 90%", "approximately 95%", "approximately 99%",
            "nearly complete", "almost complete", "essentially complete", "practically complete",
            "nearly done", "almost done", "essentially done", "practically done",
            # Essentially proven
            "essentially proven", "practically proven", "virtually proven",
            "essentially solved", "practically solved", "virtually solved",
            "main result proven", "main result essentially proven",
            # Single minor error
            "one minor", "a minor", "the minor", "single minor",
            "one small", "a small", "the small", "single small",
            "one tiny", "a tiny", "the tiny", "single tiny",
            "just a minor", "just a small", "just a tiny",
            "only a minor", "only a small", "only a tiny",
            # Specific error types that are minor
            "minor typo", "small typo", "tiny typo", "slight typo",
            "minor calculation error", "small calculation error", "tiny calculation error",
            "minor arithmetic error", "small arithmetic error", "tiny arithmetic error",
            "slight miscalculation", "tiny miscalculation", "minor miscalculation",
            "sign error", "sign mistake", "wrong sign",
            "minor oversight", "small oversight", "tiny oversight",
            "trivial error", "insignificant error", "negligible error",
            "cosmetic error", "formatting error", "notation error",
            "minor gap", "small gap", "trivial gap", "tiny gap",
            # Would be perfect/correct if...
            "would be perfect if", "would be correct if", "would be right if",
            "would be valid if", "would be complete if",
            "just needs", "only needs", "simply needs", "merely needs",
            "just requires", "only requires", "simply requires", "merely requires",
            # Easy to fix
            "easily fixed", "easily corrected", "easily remedied",
            "simple fix", "simple correction", "simple remedy",
            "quick fix", "quick correction", "quick remedy",
            "trivial fix", "trivial correction", "trivial remedy",
            "minor correction", "small correction", "tiny correction",
            "minor fix", "small fix", "tiny fix", "slight fix",
            "minor adjustment", "small adjustment", "tiny adjustment",
            "minor refinement", "small refinement", "tiny refinement",
            # High completion percentage
            "85%", "90%", "95%", "99%", "nearly there", "very close",
            "almost there", "so close", "very nearly",
            # Correct with minor exceptions
            "correct except for", "correct apart from", "correct aside from",
            "correct with minor", "correct with small", "correct with tiny",
            "correct approach with minor", "correct method with minor",
            "correct solution with minor", "correct proof with minor",
            "fundamentally correct", "essentially correct approach",
            # Verification with minor issues
            "verification contains minor mistakes", "verification has minor",
            "minor mistakes only", "small mistakes only", "tiny mistakes only",
            "minor errors only", "small errors only", "tiny errors only",
            # Additional strong almost indicators
            "essentially solved", "practically solved", "virtually solved",
            "nearly done", "almost done", "essentially done",
            "complete solution with minor", "valid proof with minor",
            "correct result with minor", "right approach with minor",
        ]
        
        for pattern in strong_almost_patterns:
            if pattern in reasoning_lower:
                return "almost"
    
    # ============================================================================
    # STEP 4: Check for "partial" indicators
    # "Partial" means: 20-75% complete, meaningful progress but significant gaps
    # ============================================================================
    partial_patterns = [
        # Explicit partial phrases
        "partial credit", "partially correct", "partial solution",
        "partial proof", "partial progress", "partial result",
        "partial answer", "partial argument", "partial case",
        "partially valid", "partially complete", "partially worked",
        "partially successful", "partially solved", "incomplete but",
        "incomplete solution", "incomplete proof", "incomplete answer",
        "incomplete reasoning", "incomplete argument", "incomplete work",
        "unfinished solution", "unfinished proof", "unfinished work",
        # Progress indicators
        "significant progress", "meaningful progress", "valid insight",
        "key insight", "important insight", "useful insight",
        "correct lemma", "proved a lemma", "proved useful lemma",
        "identified key", "correctly identified", "identified correctly",
        "some understanding", "partial understanding",
        "demonstrates some understanding", "shows some understanding",
        "has some understanding", "contains some understanding",
        # Valid components
        "correct framework", "valid reasoning", "valid approach",
        "correct approach", "valid strategy", "correct strategy",
        "valid start", "correct start", "good start", "reasonable start",
        "valid beginning", "correct beginning", "good beginning",
        "began correctly", "started correctly", "on the right track",
        "correct direction", "valid direction",
        "some correct", "some valid", "some right",
        "partially right", "partly correct", "partly right",
        "some correct steps", "some valid steps", "valid partial",
        "partially valid solution", "incomplete but valid",
        "incomplete but sound", "incomplete but promising",
        # Not fully complete
        "not fully correct", "not completely correct", "not entirely correct",
        "not totally correct", "not fully complete", "not completely complete",
        "not entirely complete", "not totally complete",
        "not fully solved", "not completely solved", "not entirely solved",
        # Marks/credit
        "partial marks", "some credit", "some marks",
        # Gaps remaining
        "missing steps", "missing parts", "missing components",
        "needs more work", "requires completion", "requires more work",
        "needs completion", "needs further work", "requires further work",
        "did not finish", "did not complete", "did not conclude",
        "stopped before", "ended before", "gave up before",
        # Progress level
        "made progress", "some progress", "limited progress",
        "modest progress", "reasonable progress", "substantial progress",
        "good progress", "decent progress", "fair progress",
        # Demonstrates/shows/has
        "demonstrates some", "shows some", "has some", "contains some",
        "demonstrates partial", "shows partial", "has partial", "contains partial",
        # Intuition/ideas
        "correct intuition", "valid intuition", "good intuition",
        "correct idea", "valid idea", "good idea", "useful idea",
        "correct observation", "valid observation", "useful observation",
        "correct insight", "valid insight", "useful insight",
        # Not entirely wrong
        "not entirely wrong", "not completely wrong", "not totally wrong",
        "not entirely incorrect", "not completely incorrect", "not totally incorrect",
        "not nonsense", "not gibberish", "not garbage", "not rubbish",
        "not a failure", "not failed", "not useless",
        # Specific partial achievements
        "proved one direction", "proved one part", "only one direction",
        "only one part", "proved a sub", "proved special case",
        "found an invariant", "found a correct invariant",
        "some cases only", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis",
        # Percentage indicators for partial (20-75%)
        "20%", "25%", "30%", "40%", "50%", "60%", "70%", "75%",
        "about 20%", "about 25%", "about 30%", "about 40%", "about 50%",
        "about 60%", "about 70%", "about 75%",
        "roughly 20%", "roughly 25%", "roughly 30%", "roughly 40%", "roughly 50%",
        "roughly 60%", "roughly 70%", "roughly 75%",
        "approximately 20%", "approximately 25%", "approximately 30%",
        "approximately 40%", "approximately 50%", "approximately 60%",
        "approximately 70%", "approximately 75%",
        "half correct", "half complete", "halfway", "about half",
        # Attempt indicators
        "valid attempt", "legitimate attempt", "serious attempt",
        "reasonable attempt", "decent attempt", "fair attempt",
        "has merit", "has value", "has substance", "has content",
        # Additional partial indicators
        "incomplete development", "incomplete execution",
        "not finished", "not concluded", "not proven completely",
        "partially developed", "partially executed", "partially proven",
    ]
    
    for pattern in partial_patterns:
        if pattern in reasoning_lower:
            return "partial"
    
    # ============================================================================
    # STEP 5: Check for "correct" indicators
    # "Correct" means: 100% complete, flawless, no errors whatsoever
    # ============================================================================
    correct_patterns = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "100% correct", "perfectly correct", "absolutely correct",
        "flawless", "no errors", "no mistakes", "perfect solution",
        "complete solution", "complete proof", "complete answer",
        "full marks", "full credit", "full score", "full points",
        "excellent solution", "perfect work", "excellent work",
        "full understanding", "complete understanding", "thorough understanding",
        "all correct", "everything correct", "correct throughout",
        "valid solution", "valid proof", "sound solution", "sound proof",
        "sound reasoning", "correct reasoning", "valid reasoning",
        "correctly solved", "properly solved", "well done",
        "mastered", "expertly solved", "elegantly solved", "beautifully solved",
        "is correct", "are correct", "was correct", "were correct",
        "correctly derived", "correctly proved", "correctly shown",
        "correctly demonstrated", "correctly established", "correctly concluded",
        "valid derivation", "valid demonstration", "valid establishment",
        "sound argument", "valid argument", "rigorous proof",
        "rigorous solution", "rigorous argument",
        "complete and rigorous", "complete and sound", "complete and valid",
        "undoubtedly correct", "clearly correct", "obviously correct",
        "definitely correct", "certainly correct",
        "demonstrates full", "shows full", "has full", "contains full",
        "deep understanding", "complete mastery",
    ]
    
    for pattern in correct_patterns:
        if pattern in reasoning_lower:
            return "correct"
    
    # ============================================================================
    # STEP 6: No strong indicators found
    # ============================================================================
    return None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid categories with improved accuracy.
    
    Key insight: The distinction between "almost" and "partial" is critical:
    - "almost": 85-99% complete, tiny fixable errors, nearly perfect
    - "partial": 20-75% complete, meaningful progress but significant gaps
    
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
    
    # ============================================================================
    # STEP 1: Check for compound phrases (highest priority after exact match)
    # ============================================================================
    # IMPROVED: More comprehensive compound indicators with better almost vs partial distinction
    compound_indicators = {
        "partial": [
            # Core partial phrases
            "partially correct", "partial solution", "partial proof", 
            "partially valid", "partially complete", "partially worked", 
            "partially successful", "partially solved",
            "partial result", "partial answer", "partial argument", "partial case",
            "partial progress", "partial work",
            # Incomplete phrases (strong indicators of partial, not almost)
            "incomplete proof", "incomplete solution", "incomplete answer",
            "incomplete reasoning", "incomplete argument", "incomplete work",
            "incomplete execution", "incomplete implementation", "unfinished solution",
            "unfinished proof", "unfinished work", "unfinished",
            # Not fully phrases
            "not fully correct", "not completely correct", "not entirely correct",
            "not totally correct", "not fully complete", "not completely complete",
            "not entirely complete", "not totally complete",
            "not fully solved", "not completely solved", "not entirely solved",
            # Missing/remaining work
            "missing steps", "missing parts", "missing components", "missing major",
            "missing significant", "lacks major", "lacks significant",
            "needs more work", "requires completion", "requires more work",
            "needs completion", "needs further work", "requires further work",
            "work remaining", "more work needed", "further work needed",
            # Progress indicators (suggest partial, not almost)
            "some progress", "limited progress", "modest progress",
            "made progress", "good start", "started correctly", "began correctly",
            "good beginning", "valid beginning", "reasonable start",
            "on the right track", "correct direction", "correct approach started",
            # Gaps remaining
            "significant gaps", "major gaps", "large gaps", "substantial gaps",
            "considerable gaps", "important gaps", "critical gaps",
            "needs substantial work", "requires substantial work",
            "needs considerable work", "requires considerable work",
            "needs significant work", "requires significant work",
            # Stopped/gave up
            "stopped early", "gave up", "stopped before", "ended before", "gave up before",
            "did not finish", "did not complete", "did not conclude",
            "did not reach", "failed to reach", "did not arrive",
            # Only proved/showed
            "only proved", "only showed", "only one direction", "only one part",
            "only a lemma", "only proved a lemma", "proved a lemma",
            "some cases only", "not all cases", "missing cases",
            "incomplete case analysis", "partial case analysis",
            # Percentage indicators for partial (20-75%)
            "20%", "25%", "30%", "40%", "50%", "60%", "70%", "75%",
            "about 20%", "about 25%", "about 30%", "about 40%", "about 50%",
            "about 60%", "about 70%", "about 75%",
            "roughly 20%", "roughly 25%", "roughly 30%", "roughly 40%", "roughly 50%",
            "roughly 60%", "roughly 70%", "roughly 75%",
            "approximately 20%", "approximately 25%", "approximately 30%",
            "approximately 40%", "approximately 50%", "approximately 60%",
            "approximately 70%", "approximately 75%",
            "only 20%", "only 30%", "only 40%", "only 50%", "only 60%", "only 70%",
            "half correct", "half complete", "halfway", "about half",
            # Far from complete
            "far from complete", "nowhere near complete", "long way from complete",
            "less than 50%", "under 50%", "below 50%", "less than 60%", "under 60%",
            "less than 70%", "under 70%", "less than 80%", "under 80%",
            "less than 85%", "under 85%",
        ],
        "almost": [
            # Strong almost indicators - clearly NOT partial
            "almost correct", "nearly correct", "almost complete", "nearly complete", 
            "mostly correct", "mostly right", "essentially correct", "practically correct",
            "virtually correct", "almost solved", "nearly solved", "almost perfect", 
            "nearly perfect", "essentially solved", "practically solved",
            "correct except for", "correct apart from", "correct aside from",
            "correct with minor", "correct with small", "correct with tiny",
            "minor mistake only", "minor error only", "small error only",
            "tiny error only", "minor issue only", "minor flaw only",
            "tiny mistake only", "slight error only", "trivial error only",
            "small issue only", "tiny issue only", "small flaw only", "tiny flaw only",
            "small problem only", "tiny problem only", "minor problem only",
            "minor gap only", "small gap only", "tiny gap only",
            # Single minor error
            "one minor", "a minor", "the minor", "single minor",
            "one small", "a small", "the small", "single small",
            "one tiny", "a tiny", "the tiny", "single tiny",
            "just a minor", "just a small", "just a tiny",
            "only a minor", "only a small", "only a tiny",
            # Specific error types that are minor
            "minor typo", "small typo", "tiny typo", "slight typo",
            "minor calculation error", "small calculation error", "tiny calculation error",
            "minor arithmetic error", "small arithmetic error", "tiny arithmetic error",
            "slight miscalculation", "tiny miscalculation", "minor miscalculation",
            "sign error", "sign mistake", "wrong sign",
            "minor oversight", "small oversight", "tiny oversight",
            "trivial error", "insignificant error", "negligible error",
            "cosmetic error", "formatting error", "notation error",
            "minor gap", "small gap", "trivial gap", "tiny gap",
            # Would be perfect/correct if...
            "would be perfect if", "would be correct if", "would be right if",
            "would be valid if", "would be complete if",
            "just needs", "only needs", "simply needs", "merely needs",
            "just requires", "only requires", "simply requires", "merely requires",
            # Easy to fix
            "easily fixed", "easily corrected", "easily remedied",
            "simple fix", "simple correction", "simple remedy",
            "quick fix", "quick correction", "quick remedy",
            "trivial fix", "trivial correction", "trivial remedy",
            "minor correction", "small correction", "tiny correction",
            "minor fix", "small fix", "tiny fix", "slight fix",
            # High completion indicators
            "85%", "90%", "95%", "99%",
            "about 85%", "about 90%", "about 95%", "about 99%",
            "roughly 85%", "roughly 90%", "roughly 95%", "roughly 99%",
            "approximately 85%", "approximately 90%", "approximately 95%", "approximately 99%",
            "nearly complete", "almost complete", "essentially complete", "practically complete",
            "nearly done", "almost done", "essentially done", "practically done",
            # Essentially proven
            "essentially proven", "practically proven", "virtually proven",
            "essentially solved", "practically solved", "virtually solved",
            "main result proven", "main result essentially proven",
        ],
        "incorrect": [
            "not correct", "not right", "not valid", "wrong approach", "incorrect approach",
            "not solved", "unsolved", "not a valid", "not an valid", "fundamentally wrong",
            "completely wrong", "totally wrong", "absolutely wrong", "no valid",
        ]
    }
    
    # Check for compound indicators first
    for category, indicators in compound_indicators.items():
        for indicator in indicators:
            if indicator in prediction_lower:
                return category
    
    # ============================================================================
    # STEP 2: Check for "incorrect" indicators (highest priority for wrong answers)
    # ============================================================================
    incorrect_indicators = [
        "no valid mathematical", "no progress", "fundamentally wrong", 
        "completely wrong", "totally wrong", "absolutely wrong", "utterly wrong",
        "nonsense", "gibberish", "garbage", "rubbish", "invalid approach", 
        "wrong approach", "no understanding", "no solution", "failed completely",
        "no attempt", "blank", "empty", "no answer", "missing answer",
        "fundamental misunderstanding", "fundamental error", "critical error",
        "fatal error", "catastrophic error", "serious mistake", "major error",
        "completely incorrect", "totally incorrect", "absolutely incorrect",
        "not correct", "is wrong", "are wrong", "was wrong", "were wrong",
        "incorrect solution", "incorrect answer", "incorrect approach",
        "no valid", "invalid reasoning", "flawed reasoning", "unsound reasoning",
        "does not demonstrate", "failed to", "unable to", "cannot solve",
        "no meaningful progress", "no valid steps", "completely failed",
        "does not solve", "did not solve", "failed to solve", "unable to solve",
        "wrong answer", "wrong result", "wrong conclusion", "incorrect conclusion",
        "not a valid", "not an valid", "invalid solution", "invalid answer",
        "no proof", "no valid proof", "missing proof", "lacks proof",
        "does not prove", "did not prove", "failed to prove",
        "erroneous", "fallacious", "bogus", "invalid", "unsound",
        "0%", "zero percent", "nothing correct", "no correct",
        "irrelevant", "off topic", "not related", "does not address",
    ]
    for indicator in incorrect_indicators:
        if indicator in prediction_lower:
            return "incorrect"
    
    # ============================================================================
    # STEP 3: Check for "almost" disqualifiers - if present, CANNOT be "almost"
    # ============================================================================
    almost_disqualifiers = [
        # Fundamental issues
        "fundamentally wrong", "fundamental misunderstanding", "fundamental error",
        "wrong approach", "incorrect approach", "invalid approach",
        "no understanding", "no valid mathematical", "nonsense", "gibberish",
        "blank", "empty", "no solution", "failed completely", "no attempt",
        # Major structural issues
        "major logical gap", "critical logical gap", "fatal flaw",
        "fatal error", "catastrophic error", "serious structural",
        # Very incomplete (less than 85%)
        "far from complete", "nowhere near complete", "long way from complete",
        "less than 50%", "under 50%", "below 50%", "less than 60%", "under 60%",
        "less than 70%", "under 70%", "less than 80%", "under 80%",
        "less than 85%", "under 85%",
        "only 10%", "only 20%", "only 30%", "only 40%", "only 50%",
        "only 60%", "only 70%", "only 80%",
        "gave up immediately", "stopped immediately", "stopped very early",
        # Incomplete indicators (key differentiators from "almost")
        "incomplete proof", "incomplete solution", "unfinished",
        "missing major", "missing significant", "lacks major", "lacks significant",
        "did not complete", "did not finish", "did not solve",
        "stopped early", "gave up", "only proved", "only showed",
        "partial case", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis",
        # Progress indicators (suggest partial, not almost)
        "some progress", "limited progress", "modest progress",
        "made progress", "good start", "started correctly", "began correctly",
        "good beginning", "valid beginning", "reasonable start",
        "on the right track", "correct direction", "correct approach started",
        # Significant gaps
        "significant gaps", "major gaps", "large gaps", "substantial gaps",
        "considerable gaps", "important gaps", "critical gaps",
        "needs substantial work", "requires substantial work",
        "needs considerable work", "requires considerable work",
        "needs significant work", "requires significant work",
        "incomplete execution", "incomplete implementation", "unfinished work",
        "work remaining", "more work needed", "further work needed",
        # Partial result indicators
        "partial result", "partial answer", "incomplete answer",
        "only one direction", "only one part", "proved a lemma",
        "proved a useful lemma", "found an invariant", "found a correct invariant",
        "some cases only", "not all cases", "missing cases",
        # Multiple errors (almost should have only tiny errors)
        "multiple errors", "several errors", "many errors", "various errors",
        "multiple mistakes", "several mistakes", "many mistakes",
        # Did not reach conclusion
        "did not reach", "failed to reach", "did not arrive",
        "did not conclude", "failed to conclude", "no conclusion",
    ]
    has_almost_disqualifier = any(d in prediction_lower for d in almost_disqualifiers)
    
    # ============================================================================
    # STEP 4: Check for "almost" indicators - ONLY if no disqualifiers present
    # ============================================================================
    if not has_almost_disqualifier:
        almost_indicators = [
            # Explicit "almost" phrases
            "almost correct", "nearly correct", "almost complete", "nearly complete",
            "almost perfect", "nearly perfect", "almost solved", "nearly solved",
            "essentially correct", "practically correct", "virtually correct",
            "mostly correct", "mostly right", "essentially right", "practically right",
            # Minor error only
            "minor mistake only", "minor error only", "small error only",
            "tiny mistake only", "slight error only", "trivial error only",
            "minor issue only", "small issue only", "tiny issue only",
            "minor flaw only", "small flaw only", "tiny flaw only",
            "minor problem only", "small problem only", "tiny problem only",
            "minor gap only", "small gap only", "tiny gap only",
            # Single minor error
            "one minor", "a minor", "the minor", "single minor",
            "one small", "a small", "the small", "single small",
            "one tiny", "a tiny", "the tiny", "single tiny",
            "just a minor", "just a small", "just a tiny",
            "only a minor", "only a small", "only a tiny",
            # Specific error types that are minor
            "minor typo", "small typo", "tiny typo", "slight typo",
            "minor calculation error", "small calculation error", "tiny calculation error",
            "minor arithmetic error", "small arithmetic error", "tiny arithmetic error",
            "slight miscalculation", "tiny miscalculation", "minor miscalculation",
            "sign error", "sign mistake", "wrong sign",
            "minor oversight", "small oversight", "tiny oversight",
            "trivial error", "insignificant error", "negligible error",
            "cosmetic error", "formatting error", "notation error",
            "minor gap", "small gap", "trivial gap", "tiny gap",
            # Would be perfect/correct if...
            "would be perfect if", "would be correct if", "would be right if",
            "would be valid if", "would be complete if",
            "just needs", "only needs", "simply needs", "merely needs",
            "just requires", "only requires", "simply requires", "merely requires",
            # Easy to fix
            "easily fixed", "easily corrected", "easily remedied",
            "simple fix", "simple correction", "simple remedy",
            "quick fix", "quick correction", "quick remedy",
            "trivial fix", "trivial correction", "trivial remedy",
            "minor correction", "small correction", "tiny correction",
            "minor fix", "small fix", "tiny fix", "slight fix",
            "minor adjustment", "small adjustment", "tiny adjustment",
            "minor refinement", "small refinement", "tiny refinement",
            # High completion percentage
            "85%", "90%", "95%", "99%", "nearly there", "very close",
            "almost there", "so close", "very nearly",
            # Correct with minor exceptions
            "correct except for", "correct apart from", "correct aside from",
            "correct with minor", "correct with small", "correct with tiny",
            "correct approach with minor", "correct method with minor",
            "correct solution with minor", "correct proof with minor",
            "fundamentally correct", "essentially correct approach",
            # Verification with minor issues
            "verification contains minor mistakes", "verification has minor",
            "minor mistakes only", "small mistakes only", "tiny mistakes only",
            "minor errors only", "small errors only", "tiny errors only",
        ]
        for indicator in almost_indicators:
            if indicator in prediction_lower:
                return "almost"
    
    # ============================================================================
    # STEP 5: Check for "partial" indicators
    # ============================================================================
    partial_indicators = [
        # Explicit partial phrases
        "partial credit", "partially correct", "partial solution",
        "partial proof", "partial progress", "partial result",
        "partial answer", "partial argument", "partial case",
        "partially valid", "partially complete", "partially worked",
        "partially successful", "partially solved", "incomplete but",
        "incomplete solution", "incomplete proof", "incomplete answer",
        "incomplete reasoning", "incomplete argument", "incomplete work",
        "unfinished solution", "unfinished proof", "unfinished work",
        # Progress indicators
        "significant progress", "meaningful progress", "valid insight",
        "key insight", "important insight", "useful insight",
        "correct lemma", "proved a lemma", "proved useful lemma",
        "identified key", "correctly identified", "identified correctly",
        "some understanding", "partial understanding",
        "demonstrates some understanding", "shows some understanding",
        "has some understanding", "contains some understanding",
        # Valid components
        "correct framework", "valid reasoning", "valid approach",
        "correct approach", "valid strategy", "correct strategy",
        "valid start", "correct start", "good start", "reasonable start",
        "valid beginning", "correct beginning", "good beginning",
        "began correctly", "started correctly", "on the right track",
        "correct direction", "valid direction",
        "some correct", "some valid", "some right",
        "partially right", "partly correct", "partly right",
        "some correct steps", "some valid steps", "valid partial",
        "partially valid solution", "incomplete but valid",
        "incomplete but sound", "incomplete but promising",
        # Not fully complete
        "not fully correct", "not completely correct", "not entirely correct",
        "not totally correct", "not fully complete", "not completely complete",
        "not entirely complete", "not totally complete",
        "not fully solved", "not completely solved", "not entirely solved",
        # Marks/credit
        "partial marks", "some credit", "some marks",
        # Gaps remaining
        "missing steps", "missing parts", "missing components",
        "needs more work", "requires completion", "requires more work",
        "needs completion", "needs further work", "requires further work",
        "did not finish", "did not complete", "did not conclude",
        "stopped before", "ended before", "gave up before",
        # Progress level
        "made progress", "some progress", "limited progress",
        "modest progress", "reasonable progress", "substantial progress",
        "good progress", "decent progress", "fair progress",
        # Demonstrates/shows/has
        "demonstrates some", "shows some", "has some", "contains some",
        "demonstrates partial", "shows partial", "has partial", "contains partial",
        # Intuition/ideas
        "correct intuition", "valid intuition", "good intuition",
        "correct idea", "valid idea", "good idea", "useful idea",
        "correct observation", "valid observation", "useful observation",
        "correct insight", "valid insight", "useful insight",
        # Not entirely wrong
        "not entirely wrong", "not completely wrong", "not totally wrong",
        "not entirely incorrect", "not completely incorrect", "not totally incorrect",
        "not nonsense", "not gibberish", "not garbage", "not rubbish",
        "not a failure", "not failed", "not useless",
        # Specific partial achievements
        "proved one direction", "proved one part", "only one direction",
        "only one part", "proved a sub", "proved special case",
        "found an invariant", "found a correct invariant",
        "some cases only", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis",
        # Percentage indicators for partial (20-75%)
        "20%", "25%", "30%", "40%", "50%", "60%", "70%", "75%",
        "about 20%", "about 25%", "about 30%", "about 40%", "about 50%",
        "about 60%", "about 70%", "about 75%",
        "roughly 20%", "roughly 25%", "roughly 30%", "roughly 40%", "roughly 50%",
        "roughly 60%", "roughly 70%", "roughly 75%",
        "approximately 20%", "approximately 25%", "approximately 30%",
        "approximately 40%", "approximately 50%", "approximately 60%",
        "approximately 70%", "approximately 75%",
        "half correct", "half complete", "halfway", "about half",
        # Attempt indicators
        "valid attempt", "legitimate attempt", "serious attempt",
        "reasonable attempt", "decent attempt", "fair attempt",
        "has merit", "has value", "has substance", "has content",
    ]
    for indicator in partial_indicators:
        if indicator in prediction_lower:
            return "partial"
    
    # ============================================================================
    # STEP 6: Check for "correct" indicators
    # ============================================================================
    correct_indicators = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "100% correct", "perfectly correct", "absolutely correct",
        "flawless", "no errors", "no mistakes", "perfect solution",
        "complete solution", "complete proof", "complete answer",
        "full marks", "full credit", "full score", "full points",
        "excellent solution", "perfect work", "excellent work",
        "full understanding", "complete understanding", "thorough understanding",
        "all correct", "everything correct", "correct throughout",
        "valid solution", "valid proof", "sound solution", "sound proof",
        "sound reasoning", "correct reasoning", "valid reasoning",
        "correctly solved", "properly solved", "well done",
        "mastered", "expertly solved", "elegantly solved", "beautifully solved",
        "is correct", "are correct", "was correct", "were correct",
        "correctly derived", "correctly proved", "correctly shown",
        "correctly demonstrated", "correctly established", "correctly concluded",
        "valid derivation", "valid demonstration", "valid establishment",
        "sound argument", "valid argument", "rigorous proof",
        "rigorous solution", "rigorous argument",
        "complete and rigorous", "complete and sound", "complete and valid",
        "undoubtedly correct", "clearly correct", "obviously correct",
        "definitely correct", "certainly correct",
        "demonstrates full", "shows full", "has full", "contains full",
        "deep understanding", "complete mastery",
    ]
    for indicator in correct_indicators:
        if indicator in prediction_lower:
            return "correct"
    
    # ============================================================================
    # STEP 7: Handle negations and qualifiers
    # ============================================================================
    # Check for negated correct (which means incorrect)
    if ("not correct" in prediction_lower or 
        "not fully" in prediction_lower or
        "not completely" in prediction_lower or
        "not entirely" in prediction_lower or
        "not totally" in prediction_lower or
        "not accurate" in prediction_lower or
        "not right" in prediction_lower or
        "not valid" in prediction_lower or
        "not 100%" in prediction_lower or
        "not perfect" in prediction_lower or
        "not flawless" in prediction_lower):
        return "incorrect"
    
    # Check for "wrong" or "error" without "minor" qualifier
    if ("wrong" in prediction_lower or "error" in prediction_lower or "mistake" in prediction_lower):
        # Check if it's qualified as minor/small/tiny
        if not any(q in prediction_lower for q in ["minor", "small", "tiny", "slight", "trivial", "cosmetic", "insignificant", "negligible", "minimal"]):
            return "incorrect"
    
    # Check for "mostly" - handle cases not caught by compound indicators
    if "mostly" in prediction_lower:
        if "mostly wrong" in prediction_lower or "mostly incorrect" in prediction_lower:
            return "incorrect"
        if "mostly incomplete" in prediction_lower or "mostly partial" in prediction_lower:
            return "partial"
        if ("mostly correct" in prediction_lower or "mostly right" in prediction_lower) and not has_almost_disqualifier:
            return "almost"
        # Default for "mostly" without clear direction
        return "partial"
    
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
    
    # ============================================================================
    # STEP 8: Handle standalone keywords
    # ============================================================================
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
    
    # ============================================================================
    # STEP 9: Default fallback to incorrect for safety
    # ============================================================================
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
        
        instruction = f"""You are an expert grader for {domain} problems. Evaluate the student's answer and classify it into exactly ONE category.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Classification Categories (choose exactly one):

**"correct"** - PERFECT solution: 100% complete, NO errors, NO gaps, ready to submit.
- Complete proof with all steps valid
- No calculation errors, no missing cases
- Use ONLY when truly flawless

**"almost"** - Nearly perfect (85-99% complete) with ONLY tiny, easily fixable issues:
- One small calculation/sign error in otherwise correct solution
- Minor typo not affecting mathematical validity
- Missing one trivial edge case
- Core proof structure is complete
- Would be correct with just a few minutes of fixes
- Main result is essentially proven

**"partial"** - Meaningful progress (20-75% complete) with valid content but SIGNIFICANT gaps:
- Proved a useful lemma but main proof incomplete
- Correct approach started but execution stopped short
- Significant additional work needed to finish
- Main result is NOT essentially proven

**"incorrect"** - No valid mathematical progress. Wrong approach, nonsense, or empty.

## CRITICAL DECISION RULES - APPLY IN ORDER:
1. **100% flawless?** → "correct"
2. **85-99% complete with only tiny fixable errors?** → "almost"  
3. **20-75% complete with meaningful progress but significant gaps?** → "partial"
4. **Otherwise** → "incorrect"

## KEY DISTINCTION: "almost" vs "partial" (THIS IS CRITICAL)

**"ALMOST" = Nearly Done (85-99% complete):**
- Solution feels "essentially done" - like a complete solution with tiny blemishes
- Core proof structure is complete and correct
- Only tiny, easily fixable issues (minor arithmetic, small typos)
- Would be correct with just a few minutes of corrections
- Main result is essentially proven
- The student has essentially SOLVED the problem
- Examples: "correct except for one sign error", "complete proof with minor typo"

**"PARTIAL" = Incomplete (20-75% complete):**
- Solution feels "unfinished" - work-in-progress, not nearly done
- Valid mathematical content BUT major gaps remain
- Significant additional work would be needed (more than 5 minutes)
- Main result is NOT proven or only partially proven
- The student has NOT essentially solved the problem
- Examples: "proved a lemma but stopped", "correct start but incomplete execution"

**DECISION HEURISTIC - USE THESE QUESTIONS:**
1. Estimate completeness percentage:
   - 95-100% → "correct" (or "almost" if tiny errors)
   - 85-94% → "almost" (nearly done, minor fixes)
   - 20-75% → "partial" (meaningful progress but incomplete)
   - 0-20% → "incorrect"

2. Ask: "Is the main result essentially proven?"
   - YES with tiny issues → "almost"
   - NO, only partial progress → "partial"

3. Ask: "Would this take under 5 minutes to fix?"
   - YES → "almost"
   - NO → "partial"

4. Ask: "Does the solution feel 'essentially done' or 'unfinished'?"
   - Essentially done → "almost"
   - Unfinished → "partial"

## CONSERVATIVE GRADING:
When in doubt, choose the LOWER category:
- Doubt between "correct" and "almost"? → "almost"
- Doubt between "almost" and "partial"? → "partial"
- Doubt between "partial" and "incorrect"? → "incorrect"

## DISQUALIFIERS - ABSOLUTE RULES:

**NOT "correct" if ANY of:**
- Any calculation error, even small
- Any missing step, even trivial
- Any doubt in your mind

**NOT "almost" if ANY of:**
- Wrong approach or fundamental misunderstanding
- Multiple errors (more than one tiny error)
- Less than 85% completeness
- Missing major portions of the proof
- "Incomplete", "unfinished", "partial" appear in description
- "Only proved" a sub-result (main result not done)
- "Did not complete", "stopped early", "gave up"
- "Significant gaps", "major gaps" remaining
- Main result not essentially proven
- Would take more than 5 minutes to fix

**NOT "partial" if ANY of:**
- 85-99% complete with only minor issues → "almost"
- "Almost correct", "nearly correct", "essentially correct"
- "Minor error only", "small error only", "tiny error only"
- "Would be correct if...", "just needs...", "only needs..."
- "Easily fixed", "simple fix", "quick fix"
- Main result is essentially proven with tiny blemishes

## Output Format:
<json>
{{
    "reasoning": "Step-by-step analysis: 1) Completeness % estimate, 2) Main result proven? 3) Fix time estimate, 4) Category choice with justification",
    "response": "correct"
}}</json>

Response MUST be exactly: "correct", "almost", "partial", or "incorrect" (lowercase).

## FINAL VERIFICATION CHECKLIST:
Before answering, you MUST verify:
1. **Completeness %**: What percentage is complete? (0-20%→incorrect, 20-75%→partial, 85-94%→almost, 95-100%→correct)
2. **Error severity**: Are errors tiny fixable ones or significant gaps?
3. **Main result**: Is the main result essentially proven? (YES→almost/correct, NO→partial/incorrect)
4. **Fix time**: Would fixes take under 5 minutes? (YES→almost, NO→partial)
5. **Feeling**: Does it feel "essentially done" or "unfinished"?
6. **Conservative check**: Am I being too generous? If ANY doubt, use a lower category.

## COMMON MISTAKES TO AVOID:
- DON'T call something "almost" if it has significant gaps or is <85% complete
- DON'T call something "partial" if it's 85-99% complete with only tiny errors
- DON'T call something "correct" if it has ANY errors, even tiny ones
- "Almost" = nearly perfect, "Partial" = meaningful but incomplete progress
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
                            
                            # IMPROVED: Enhanced reasoning-based correction logic
                            # This helps catch cases where the response field is wrong
                            if reasoning_suggestion and reasoning_suggestion != normalized_pred:
                                # Priority 1: If reasoning suggests "incorrect", trust it (strong signal)
                                if reasoning_suggestion == "incorrect":
                                    self.log_fn(f"Reasoning suggests 'incorrect' but response says '{normalized_pred}'. Using reasoning.")
                                    prediction = reasoning_suggestion
                                
                                # Priority 2: If reasoning suggests "partial" but response says "almost",
                                # this is the most common misclassification - trust reasoning
                                elif reasoning_suggestion == "partial" and normalized_pred == "almost":
                                    # Double-check: only override if reasoning is strong
                                    has_partial_indicators = _analyze_reasoning_for_category(reasoning_text) == "partial"
                                    if has_partial_indicators:
                                        self.log_fn(f"Reasoning clearly suggests 'partial' over 'almost'. Using reasoning.")
                                        prediction = reasoning_suggestion
                                    else:
                                        prediction = normalized_pred
                                
                                # Priority 3: If reasoning suggests "almost" and response says "partial",
                                # check if reasoning has strong "almost" indicators
                                elif reasoning_suggestion == "almost" and normalized_pred == "partial":
                                    # Check for strong "almost" indicators in combined text
                                    has_strong_almost = (
                                        "nearly correct" in combined_text or 
                                        "almost correct" in combined_text or 
                                        "minor mistake only" in combined_text or
                                        "minor error only" in combined_text or
                                        "small error only" in combined_text or
                                        "tiny error only" in combined_text or
                                        "essentially correct" in combined_text or
                                        "practically correct" in combined_text or
                                        "verification contains minor" in combined_text or
                                        "correct except for" in combined_text or
                                        "correct apart from" in combined_text or
                                        "would be perfect" in combined_text or
                                        "would be correct" in combined_text or
                                        "would be right" in combined_text or
                                        "just needs" in combined_text or
                                        "only needs" in combined_text or
                                        "easily fixed" in combined_text or
                                        "simple fix" in combined_text or
                                        "minor fix" in combined_text or
                                        "85%" in combined_text or
                                        "90%" in combined_text or
                                        "95%" in combined_text
                                    )
                                    # Check for strong "partial" disqualifiers
                                    has_partial_disqualifiers = (
                                        "incomplete proof" in combined_text or
                                        "unfinished" in combined_text or
                                        "missing major" in combined_text or
                                        "significant gap" in combined_text or
                                        "stopped early" in combined_text or
                                        "gave up" in combined_text or
                                        "only proved" in combined_text or
                                        "far from complete" in combined_text or
                                        "did not complete" in combined_text or
                                        "did not finish" in combined_text or
                                        "incomplete solution" in combined_text or
                                        "partial case" in combined_text or
                                        "not all cases" in combined_text or
                                        "only 50%" in combined_text or
                                        "only 60%" in combined_text or
                                        "only 70%" in combined_text or
                                        "less than 85%" in combined_text
                                    )
                                    if has_strong_almost and not has_partial_disqualifiers:
                                        self.log_fn(f"Reasoning strongly suggests 'almost' over 'partial'. Using reasoning.")
                                        prediction = reasoning_suggestion
                                    else:
                                        prediction = normalized_pred
                                
                                # Priority 4: If reasoning suggests "correct" but response says something else,
                                # only trust if reasoning is very clear
                                elif reasoning_suggestion == "correct" and normalized_pred in ["almost", "partial", "incorrect"]:
                                    # Only trust "correct" if reasoning is very clear
                                    has_strong_correct = (
                                        "fully correct" in reasoning_text or 
                                        "completely correct" in reasoning_text or 
                                        "100% correct" in reasoning_text or
                                        "flawless" in reasoning_text or
                                        "no errors" in reasoning_text or
                                        "perfect solution" in reasoning_text
                                    )
                                    if has_strong_correct:
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
        
        # Additional edge case: if prediction is a list or other non-string type
        if not isinstance(prediction, str):
            try:
                prediction = str(prediction)
                # Re-validate after conversion
                if prediction not in valid_categories:
                    self.log_fn(f"Converted prediction '{prediction}' not valid, defaulting to 'incorrect'")
                    prediction = "incorrect"
            except Exception:
                self.log_fn(f"Could not convert prediction to string, defaulting to 'incorrect'")
                prediction = "incorrect"
        
        # Final edge case: check for common misclassifications using reasoning
        # This is a last-chance correction based on reasoning analysis
        if reasoning_suggestion and reasoning_suggestion != prediction:
            # If prediction is "almost" but reasoning strongly suggests "partial"
            if prediction == "almost" and reasoning_suggestion == "partial":
                self.log_fn("Final validation: prediction is 'almost' but reasoning suggests 'partial'. Correcting to 'partial'.")
                prediction = "partial"
            # If prediction is "partial" but reasoning strongly suggests "almost" (less common)
            elif prediction == "partial" and reasoning_suggestion == "almost":
                # Only correct if reasoning is very clear about "almost"
                if reasoning_text and (
                    "nearly correct" in reasoning_text or 
                    "almost correct" in reasoning_text or
                    "minor error only" in reasoning_text or
                    "essentially correct" in reasoning_text or
                    "practically correct" in reasoning_text or
                    "85%" in reasoning_text or
                    "90%" in reasoning_text or
                    "95%" in reasoning_text
                ):
                    self.log_fn("Final validation: prediction is 'partial' but reasoning strongly suggests 'almost'. Correcting to 'almost'.")
                    prediction = "almost"
            # If prediction is "correct" but reasoning suggests lower category
            elif prediction == "correct" and reasoning_suggestion in ["almost", "partial", "incorrect"]:
                # Be conservative - trust reasoning over response for "correct"
                self.log_fn(f"Final validation: prediction is 'correct' but reasoning suggests '{reasoning_suggestion}'. Correcting to '{reasoning_suggestion}'.")
                prediction = reasoning_suggestion
            # If prediction is "incorrect" but reasoning suggests higher category
            elif prediction == "incorrect" and reasoning_suggestion in ["correct", "almost", "partial"]:
                # Only upgrade if reasoning is very clear
                if reasoning_suggestion == "partial" and reasoning_text and (
                    "some progress" in reasoning_text or
                    "valid insight" in reasoning_text or
                    "correct lemma" in reasoning_text or
                    "meaningful progress" in reasoning_text
                ):
                    self.log_fn("Final validation: prediction is 'incorrect' but reasoning suggests 'partial'. Correcting to 'partial'.")
                    prediction = "partial"
        
        # Check for completeness percentage indicators in reasoning
        if reasoning_text and prediction in ["almost", "partial"]:
            # Check for explicit percentage indicators that might suggest different category
            has_high_completion = (
                "85%" in reasoning_text or "90%" in reasoning_text or 
                "95%" in reasoning_text or "99%" in reasoning_text or
                "nearly complete" in reasoning_text or "almost complete" in reasoning_text
            )
            has_low_completion = (
                "20%" in reasoning_text or "30%" in reasoning_text or 
                "40%" in reasoning_text or "50%" in reasoning_text or
                "60%" in reasoning_text or "70%" in reasoning_text or
                "half" in reasoning_text or "incomplete" in reasoning_text or
                "unfinished" in reasoning_text
            )
            
            if prediction == "partial" and has_high_completion and not has_low_completion:
                # Check if there are strong "almost" indicators
                if ("minor" in reasoning_text or "small" in reasoning_text or "tiny" in reasoning_text or
                    "nearly correct" in reasoning_text or "almost correct" in reasoning_text or
                    "essentially correct" in reasoning_text or "practically correct" in reasoning_text):
                    self.log_fn("Final validation: prediction is 'partial' but reasoning indicates high completion with minor issues. Correcting to 'almost'.")
                    prediction = "almost"
            elif prediction == "almost" and has_low_completion:
                self.log_fn("Final validation: prediction is 'almost' but reasoning indicates low completion. Correcting to 'partial'.")
                prediction = "partial"
        
        # Check for "incomplete" vs "complete" in reasoning for almost/partial distinction
        if reasoning_text and prediction == "almost":
            # If reasoning mentions "incomplete" or "unfinished" multiple times, likely partial
            incomplete_count = reasoning_text.count("incomplete") + reasoning_text.count("unfinished") + reasoning_text.count("missing")
            if incomplete_count >= 2:
                self.log_fn(f"Final validation: prediction is 'almost' but reasoning mentions incomplete/unfinished {incomplete_count} times. Correcting to 'partial'.")
                prediction = "partial"
        
        # IMPROVED: Additional check for "almost" vs "partial" based on key phrases
        if reasoning_text and prediction in ["almost", "partial"]:
            # Strong "almost" phrases that should override "partial"
            strong_almost_phrases = [
                "essentially correct", "practically correct", "virtually correct",
                "essentially solved", "practically solved", "virtually solved",
                "essentially proven", "practically proven", "virtually proven",
                "main result essentially proven", "main result proven",
                "would be correct if", "would be perfect if",
                "just needs", "only needs", "simply needs",
                "easily fixed", "simple fix", "quick fix",
                "minor error only", "small error only", "tiny error only",
                "one minor", "a minor", "single minor",
                "85%", "90%", "95%", "99%",
            ]
            # Strong "partial" phrases that should override "almost"
            strong_partial_phrases = [
                "incomplete proof", "incomplete solution", "unfinished",
                "missing major", "missing significant", "significant gap",
                "stopped early", "gave up", "did not complete",
                "did not finish", "only proved", "only showed",
                "partial case", "not all cases", "missing cases",
                "far from complete", "nowhere near complete",
                "less than 50%", "less than 60%", "less than 70%", "less than 80%",
                "only 50%", "only 60%", "only 70%",
                "20%", "30%", "40%", "50%", "60%", "70%",
            ]
            
            has_almost_phrase = any(p in reasoning_text for p in strong_almost_phrases)
            has_partial_phrase = any(p in reasoning_text for p in strong_partial_phrases)
            
            if prediction == "partial" and has_almost_phrase and not has_partial_phrase:
                self.log_fn("Final validation: prediction is 'partial' but reasoning has strong 'almost' phrases. Correcting to 'almost'.")
                prediction = "almost"
            elif prediction == "almost" and has_partial_phrase and not has_almost_phrase:
                self.log_fn("Final validation: prediction is 'almost' but reasoning has strong 'partial' phrases. Correcting to 'partial'.")
                prediction = "partial"
        
        # Final safety check: ensure prediction is still valid after all corrections
        if prediction not in valid_categories:
            self.log_fn(f"Prediction '{prediction}' became invalid after corrections, defaulting to 'incorrect'")
            prediction = "incorrect"
        
        return str(prediction), msg_history
