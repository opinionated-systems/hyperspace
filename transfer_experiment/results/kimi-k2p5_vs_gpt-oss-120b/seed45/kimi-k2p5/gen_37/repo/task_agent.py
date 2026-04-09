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
    
    # Try ```json code blocks (with or without json specifier)
    results = []
    # Try ```json blocks
    pattern = r'```json\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    # Also try generic ``` blocks
    if not results:
        pattern = r'```\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            match_stripped = match.strip()
            if match_stripped.startswith('{'):
                try:
                    results.append(json.loads(match_stripped))
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


def _check_almost_vs_partial_boundary(reasoning_lower: str) -> str | None:
    """Specifically check the boundary between 'almost' and 'partial' categories.
    
    This is the most common source of misclassification. Uses a strict rule-based
    approach: if ANY disqualifier is present, it CANNOT be "almost".
    
    Key insight: "almost" requires main result to be essentially proven with
    only tiny fixable errors. "partial" means main result is NOT proven.
    
    Args:
        reasoning_lower: Lowercase reasoning text
        
    Returns:
        "almost", "partial", or None if boundary is unclear
    """
    if not reasoning_lower or not isinstance(reasoning_lower, str):
        return None
    
    # ============================================================================
    # STEP 1: Check for ABSOLUTE disqualifiers - these override everything
    # If ANY of these are present, CANNOT be "almost" → must be "partial"
    # ============================================================================
    
    absolute_almost_disqualifiers = [
        # Main result NOT proven indicators (highest priority)
        "main result not proven", "main result is not proven", 
        "main result not essentially proven", "main result is not essentially proven",
        "did not prove the main", "failed to prove the main", 
        "did not establish the main", "failed to establish the main",
        "main result remains unproven", "main result not established",
        # Incomplete/unfinished (very strong indicators)
        "incomplete proof", "incomplete solution", "unfinished proof", "unfinished solution",
        "proof is incomplete", "solution is incomplete", 
        "proof incomplete", "solution incomplete",
        "unfinished work", "incomplete work", "work in progress",
        "not complete", "not finished", "not concluded",
        # Stopped early / gave up
        "stopped early", "gave up", "gave up early", "stopped prematurely",
        "ended early", "stopped before", "ended before", 
        "did not finish", "did not complete", "failed to complete", "failed to finish",
        "did not conclude", "failed to conclude", "no conclusion reached",
        # Only proved partial results
        "only proved a lemma", "only proved lemma", "proved only a lemma",
        "only one direction", "only proved one direction", "proved only one direction",
        "only one part", "only proved one part", "proved only one part",
        "only a partial", "only partial", "partial result only",
        "not all cases", "missing cases", "incomplete case analysis",
        "partial case analysis", "only some cases",
        # Far from complete
        "far from complete", "nowhere near complete", "long way from complete",
        "far from done", "nowhere near done", "long way from done",
        # Low completion percentage (under 85%)
        "less than 50%", "under 50%", "below 50%",
        "less than 60%", "under 60%", "below 60%",
        "less than 70%", "under 70%", "below 70%",
        "less than 75%", "under 75%", "below 75%",
        "less than 80%", "under 80%", "below 80%",
        "less than 85%", "under 85%", "below 85%",
        "only 10%", "only 20%", "only 30%", "only 40%", "only 50%",
        "only 60%", "only 70%", "only 80%",
        # Significant gaps remaining
        "significant gaps remain", "major gaps remain", "substantial gaps remain",
        "significant gaps remaining", "major gaps remaining", "substantial gaps remaining",
        "significant gap remains", "major gap remains", "substantial gap remains",
        "significant work remaining", "substantial work remaining", "considerable work remaining",
        "significant work remains", "substantial work remains",
        # Progress indicators (NOT completion - these indicate partial)
        "some progress", "limited progress", "modest progress",
        "made progress", "making progress", "good progress",
        "good start", "started correctly", "began correctly",
        "good beginning", "valid beginning", "reasonable start",
        "on the right track", "correct direction", "correct approach started",
        "right idea", "correct intuition", "valid intuition",
        # Work remaining indicators
        "needs more work", "requires more work", "needs further work", "requires further work",
        "needs substantial work", "requires substantial work",
        "needs considerable work", "requires considerable work",
        "needs significant work", "requires significant work",
        "work remaining", "more work needed", "further work needed",
        "work still needed", "additional work needed",
        "incomplete execution", "incomplete implementation", "unfinished work",
        # Multiple errors (almost should have only one tiny error)
        "multiple errors", "several errors", "many errors", "various errors",
        "multiple mistakes", "several mistakes", "many mistakes", "various mistakes",
        # Did not reach conclusion
        "did not reach", "failed to reach", "did not arrive",
        "did not conclude", "failed to conclude", "no conclusion",
        "did not obtain", "failed to obtain",
    ]
    
    for disqualifier in absolute_almost_disqualifiers:
        if disqualifier in reasoning_lower:
            return "partial"
    
    # ============================================================================
    # STEP 2: Check for ABSOLUTE "almost" indicators
    # These strongly suggest "almost" ONLY if no disqualifiers were found
    # ============================================================================
    
    absolute_almost_indicators = [
        # Main result proven with minor issues (highest priority)
        "main result essentially proven", "main result is essentially proven",
        "essentially proven the main", "essentially solved the main",
        "main result proven", "main result is proven",
        "main result essentially solved", "main result is essentially solved",
        # Essentially/practically/virtually correct
        "essentially correct", "practically correct", "virtually correct",
        "essentially solved", "practically solved", "virtually solved",
        "essentially proven", "practically proven", "virtually proven",
        # Nearly/almost correct
        "nearly correct", "almost correct", "mostly correct",
        "nearly solved", "almost solved", "mostly solved",
        "nearly proven", "almost proven", "mostly proven",
        # Minor error only (strongest indicator)
        "minor mistake only", "minor error only", "small error only",
        "tiny mistake only", "slight error only", "trivial error only",
        "minor issue only", "small issue only", "tiny issue only",
        "minor flaw only", "small flaw only", "tiny flaw only",
        "minor problem only", "small problem only", "tiny problem only",
        "minor gap only", "small gap only", "tiny gap only",
        # Single minor error
        "one minor error", "a minor error", "single minor error",
        "one small error", "a small error", "single small error",
        "one tiny error", "a tiny error", "single tiny error",
        "just a minor", "just a small", "just a tiny",
        "only a minor", "only a small", "only a tiny",
        "only one minor", "only one small", "only one tiny",
        # Specific minor error types
        "minor typo", "small typo", "tiny typo", "slight typo",
        "minor calculation error", "small calculation error", "tiny calculation error",
        "minor arithmetic error", "small arithmetic error", "tiny arithmetic error",
        "slight miscalculation", "tiny miscalculation", "minor miscalculation",
        "sign error", "sign mistake", "wrong sign",
        "minor oversight", "small oversight", "tiny oversight",
        "trivial error", "insignificant error", "negligible error",
        "cosmetic error", "formatting error", "notation error",
        "minor gap", "small gap", "trivial gap", "tiny gap",
        # Would be correct/perfect if
        "would be correct if", "would be perfect if", "would be right if",
        "would be valid if", "would be complete if",
        "would be correct with", "would be perfect with",
        # Easy to fix
        "easily fixed", "easily corrected", "easily remedied",
        "simple fix", "simple correction", "simple remedy",
        "quick fix", "quick correction", "quick remedy",
        "trivial fix", "trivial correction", "trivial remedy",
        "minor correction", "small correction", "tiny correction",
        "minor fix", "small fix", "tiny fix", "slight fix",
        "just needs", "only needs", "simply needs",
        # High completion percentage (85%+)
        "85% complete", "90% complete", "95% complete", "99% complete",
        "about 85%", "about 90%", "about 95%", "about 99%",
        "roughly 85%", "roughly 90%", "roughly 95%", "roughly 99%",
        "approximately 85%", "approximately 90%", "approximately 95%", "approximately 99%",
        "nearly complete", "almost complete", "essentially complete", "practically complete",
        "nearly done", "almost done", "essentially done", "practically done",
        "nearly there", "almost there", "very close",
        # Correct with minor exceptions
        "correct except for", "correct apart from", "correct aside from",
        "correct with minor", "correct with small", "correct with tiny",
        "correct approach with minor", "correct method with minor",
        "correct solution with minor", "correct proof with minor",
        "fundamentally correct", "essentially correct approach",
        # Verification with minor issues
        "verification contains minor", "verification has minor",
        "minor mistakes only", "small mistakes only", "tiny mistakes only",
        "minor errors only", "small errors only", "tiny errors only",
    ]
    
    for indicator in absolute_almost_indicators:
        if indicator in reasoning_lower:
            return "almost"
    
    # ============================================================================
    # STEP 3: Weighted scoring for unclear cases
    # ============================================================================
    
    strong_almost_indicators = {
        "essentially correct": 5, "practically correct": 5, "virtually correct": 5,
        "nearly correct": 4, "almost correct": 4, "mostly correct": 3,
        "essentially solved": 5, "practically solved": 5, "virtually solved": 5,
        "nearly solved": 4, "almost solved": 4, "mostly solved": 3,
        "essentially proven": 5, "practically proven": 5, "virtually proven": 5,
        "main result proven": 4, "main result essentially proven": 5,
        "minor error only": 5, "small error only": 5, "tiny error only": 5,
        "minor mistake only": 5, "small mistake only": 5, "tiny mistake only": 5,
        "one minor": 4, "a minor": 4, "single minor": 4,
        "one small": 4, "a small": 4, "single small": 4,
        "correct except for": 4, "correct apart from": 4, "correct aside from": 4,
        "would be correct if": 4, "would be perfect if": 4, "would be right if": 4,
        "just needs": 3, "only needs": 3, "simply needs": 3,
        "easily fixed": 3, "simple fix": 3, "quick fix": 3,
        "85%": 3, "90%": 4, "95%": 5, "99%": 5,
        "nearly complete": 4, "almost complete": 4, "essentially complete": 4,
        "nearly done": 4, "almost done": 4, "essentially done": 4,
    }
    
    strong_partial_indicators = {
        "incomplete proof": 5, "incomplete solution": 5, "unfinished": 4,
        "missing major": 5, "missing significant": 5, "significant gap": 4,
        "major gap": 4, "large gap": 3, "substantial gap": 3,
        "stopped early": 4, "gave up": 4, "did not complete": 4,
        "did not finish": 4, "did not conclude": 4,
        "only proved": 4, "only showed": 4, "only one direction": 4,
        "only one part": 4, "only a lemma": 4, "proved a lemma": 3,
        "partial case": 3, "not all cases": 3, "missing cases": 3,
        "incomplete case analysis": 4, "partial case analysis": 3,
        "far from complete": 5, "nowhere near complete": 5,
        "less than 50%": 5, "less than 60%": 4, "less than 70%": 3,
        "less than 80%": 3, "less than 85%": 3,
        "only 20%": 5, "only 30%": 5, "only 40%": 4, "only 50%": 4,
        "only 60%": 3, "only 70%": 3,
        "20%": 4, "30%": 4, "40%": 3, "50%": 3, "60%": 2, "70%": 2,
        "some progress": 2, "limited progress": 2, "modest progress": 2,
        "made progress": 2, "good start": 2, "started correctly": 2,
        "on the right track": 2, "correct direction": 2,
        "significant gaps": 3, "major gaps": 3, "large gaps": 2,
        "needs substantial work": 3, "requires substantial work": 3,
        "needs more work": 2, "requires more work": 2,
        "work remaining": 2, "more work needed": 2,
    }
    
    almost_score = 0
    partial_score = 0
    
    for indicator, weight in strong_almost_indicators.items():
        if indicator in reasoning_lower:
            almost_score += weight
    
    for indicator, weight in strong_partial_indicators.items():
        if indicator in reasoning_lower:
            partial_score += weight
    
    score_diff = almost_score - partial_score
    
    if score_diff >= 4:
        return "almost"
    elif score_diff <= -4:
        return "partial"
    elif almost_score >= 6 and partial_score < 3:
        return "almost"
    elif partial_score >= 6 and almost_score < 3:
        return "partial"
    
    return None


def _has_partial_disqualifiers(text: str) -> bool:
    """Check if text contains disqualifiers that prevent 'almost' classification.
    
    Args:
        text: Text to check (should be lowercase)
        
    Returns:
        True if text contains partial disqualifiers
    """
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    disqualifiers = [
        "incomplete proof", "incomplete solution", "unfinished",
        "only proved a lemma", "only one direction", "only one part",
        "stopped early", "gave up", "did not finish", "did not complete",
        "significant gaps remain", "major gaps remain",
        "less than 85%", "only 50%", "only 60%", "only 70%",
        "some progress", "good start", "made progress",
        "needs more work", "requires more work", "needs further work",
        "main result not proven", "main result is not proven",
        "did not prove the main", "failed to prove the main",
    ]
    
    return any(d in text_lower for d in disqualifiers)


def _has_almost_indicators(text: str) -> bool:
    """Check if text contains strong indicators for 'almost' classification.
    
    Args:
        text: Text to check (should be lowercase)
        
    Returns:
        True if text contains strong almost indicators
    """
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    indicators = [
        "nearly correct", "almost correct", 
        "minor mistake only", "minor error only", "small error only", "tiny error only",
        "essentially correct", "practically correct",
        "main result essentially proven", "main result proven",
        "correct except for", "correct apart from",
        "would be perfect", "would be correct", "would be right",
        "just needs", "only needs",
        "easily fixed", "simple fix", "minor fix",
        "85%", "90%", "95%",
        "essentially done", "nearly complete",
    ]
    
    return any(i in text_lower for i in indicators)



def _analyze_reasoning_for_category(reasoning: str) -> str | None:
    """Analyze reasoning text to detect category indicators with improved accuracy.
    
    This helps catch cases where the reasoning clearly indicates one category
    but the response field might have a different value.
    
    The key insight: "almost" requires 85-99% completeness with tiny fixable errors, main result proven.
    "partial" means 20-75% complete with significant gaps, main result NOT proven.
    
    Args:
        reasoning: The reasoning text from the LLM response
        
    Returns:
        Suggested category or None if no strong indicators found
    """
    if not isinstance(reasoning, str) or not reasoning:
        return None
    
    reasoning_lower = reasoning.lower()
    
    # First check the almost vs partial boundary specifically (most important)
    boundary_result = _check_almost_vs_partial_boundary(reasoning_lower)
    if boundary_result:
        return boundary_result
    
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
    # ENHANCED: More comprehensive disqualifiers with higher priority patterns first
    almost_disqualifiers = [
        # Main result NOT proven - highest priority
        "main result not proven", "main result is not proven",
        "did not prove the main", "failed to prove the main",
        "did not establish the main", "failed to establish the main",
        # Incomplete/unfinished - very strong indicators
        "incomplete proof", "incomplete solution", "unfinished proof", "unfinished solution",
        "proof is incomplete", "solution is incomplete", "proof incomplete", "solution incomplete",
        "unfinished work", "incomplete work", "work in progress",
        # Stopped early / gave up
        "stopped early", "gave up", "gave up early", "stopped prematurely",
        "ended early", "stopped before", "ended before",
        "did not finish", "did not complete", "failed to complete", "failed to finish",
        "did not conclude", "failed to conclude", "no conclusion reached",
        # Only partial results
        "only proved a lemma", "only proved lemma", "proved only a lemma",
        "only one direction", "only proved one direction", "proved only one direction",
        "only one part", "only proved one part", "proved only one part",
        "only a partial", "only partial", "partial result only",
        "not all cases", "missing cases", "incomplete case analysis",
        # Far from complete
        "far from complete", "nowhere near complete", "long way from complete",
        "far from done", "nowhere near done", "long way from done",
        # Low completion percentage
        "less than 50%", "under 50%", "below 50%",
        "less than 60%", "under 60%", "below 60%",
        "less than 70%", "under 70%", "below 70%",
        "less than 75%", "under 75%", "below 75%",
        "less than 80%", "under 80%", "below 80%",
        "less than 85%", "under 85%", "below 85%",
        "only 10%", "only 20%", "only 30%", "only 40%", "only 50%",
        "only 60%", "only 70%", "only 80%",
        # Fundamental issues
        "fundamentally wrong", "fundamental misunderstanding", "fundamental error",
        "wrong approach", "incorrect approach", "invalid approach",
        "no understanding", "no valid mathematical", "nonsense", "gibberish",
        "blank", "empty", "no solution", "failed completely", "no attempt",
        # Major structural issues
        "major logical gap", "critical logical gap", "fatal flaw",
        "fatal error", "catastrophic error", "serious structural",
        # Significant gaps remaining
        "significant gaps remain", "major gaps remain", "substantial gaps remain",
        "significant gaps remaining", "major gaps remaining", "substantial gaps remaining",
        "significant gap remains", "major gap remains", "substantial gap remains",
        "significant work remaining", "substantial work remaining", "considerable work remaining",
        # Progress indicators (not completion)
        "some progress", "limited progress", "modest progress",
        "made progress", "good start", "started correctly", "began correctly",
        "good beginning", "valid beginning", "reasonable start",
        "on the right track", "correct direction", "correct approach started",
        # Work remaining
        "needs more work", "requires more work", "needs further work", "requires further work",
        "needs substantial work", "requires substantial work",
        "needs considerable work", "requires considerable work",
        "needs significant work", "requires significant work",
        "work remaining", "more work needed", "further work needed",
        "incomplete execution", "incomplete implementation",
        # Partial result indicators
        "partial result", "partial answer", "incomplete answer",
        "only one direction", "only one part", "proved a lemma",
        "proved a useful lemma", "found an invariant", "found a correct invariant",
        "some cases only", "not all cases", "missing cases",
        # Multiple errors (almost should have only one tiny error)
        "multiple errors", "several errors", "many errors", "various errors",
        "multiple mistakes", "several mistakes", "many mistakes",
        # Did not reach conclusion
        "did not reach", "failed to reach", "did not arrive",
        "did not conclude", "failed to conclude", "no conclusion",
        # Additional disqualifiers
        "not all steps", "missing steps", "incomplete steps",
        "stopped prematurely", "ended prematurely",
        "did not fully", "not fully developed", "not fully executed",
        "lacks completion", "lacks conclusion", "no final answer",
        "unfinished business", "incomplete work",
    ]
    
    has_almost_disqualifier = any(d in reasoning_lower for d in almost_disqualifiers)
    
    # ============================================================================
    # STEP 3: Check for "almost" indicators - ONLY if no disqualifiers present
    # "Almost" means: 85-99% complete, tiny fixable errors, nearly perfect
    # ============================================================================
    if not has_almost_disqualifier:
        # Strong "almost" patterns - these clearly indicate "almost"
        strong_almost_patterns = [
            # Main result proven with minor issues - highest priority
            "main result essentially proven", "main result is essentially proven",
            "essentially proven the main", "essentially solved the main",
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
            "complete solution with minor", "valid proof with minor",
            "correct result with minor", "right approach with minor",
            # High completion percentage words
            "nearly there", "very close", "almost there", "so close", "very nearly",
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
    - "almost": 85-99% complete, tiny fixable errors, nearly perfect, main result essentially proven
    - "partial": 20-75% complete, meaningful progress but significant gaps, main result NOT proven
    
    Args:
        prediction: Raw prediction string from LLM
        
    Returns:
        Normalized prediction: "correct", "almost", "partial", or "incorrect"
    """
    # Handle None or non-string inputs
    if prediction is None:
        return "incorrect"
    if not isinstance(prediction, str):
        return "incorrect"
    
    # Handle empty or whitespace-only strings
    prediction_stripped = prediction.strip()
    if not prediction_stripped:
        return "incorrect"
    
    prediction_lower = prediction_stripped.lower()
    
    # Direct matches (exact match takes highest priority)
    valid_categories = ["correct", "almost", "partial", "incorrect"]
    for cat in valid_categories:
        if prediction_lower == cat:
            return cat
    
    # ============================================================================
    # STEP 1: Check for ABSOLUTE disqualifiers for "almost" - highest priority
    # If ANY of these are present, the solution CANNOT be "almost"
    # ============================================================================
    absolute_almost_disqualifiers = [
        # Main result NOT proven - highest priority
        "main result not proven", "main result is not proven", "main result not essentially proven",
        "did not prove the main", "failed to prove the main",
        "did not establish the main", "failed to establish the main",
        # Incomplete/unfinished - very strong indicators
        "incomplete proof", "incomplete solution", "unfinished proof", "unfinished solution",
        "proof is incomplete", "solution is incomplete", "proof incomplete", "solution incomplete",
        "unfinished work", "incomplete work", "work in progress",
        # Stopped early / gave up
        "stopped early", "gave up", "gave up early", "stopped prematurely",
        "ended early", "stopped before", "ended before",
        "did not finish", "did not complete", "failed to complete", "failed to finish",
        "did not conclude", "failed to conclude", "no conclusion reached",
        # Only partial results
        "only proved a lemma", "only proved lemma", "proved only a lemma",
        "only one direction", "only proved one direction", "proved only one direction",
        "only one part", "only proved one part", "proved only one part",
        "only a partial", "only partial", "partial result only",
        "not all cases", "missing cases", "incomplete case analysis",
        # Far from complete
        "far from complete", "nowhere near complete", "long way from complete",
        "far from done", "nowhere near done", "long way from done",
        # Low completion percentage
        "less than 50%", "under 50%", "below 50%",
        "less than 60%", "under 60%", "below 60%",
        "less than 70%", "under 70%", "below 70%",
        "less than 75%", "under 75%", "below 75%",
        "less than 80%", "under 80%", "below 80%",
        "less than 85%", "under 85%", "below 85%",
        "only 10%", "only 20%", "only 30%", "only 40%", "only 50%",
        "only 60%", "only 70%", "only 80%",
        # Significant gaps remaining
        "significant gaps remain", "major gaps remain", "substantial gaps remain",
        "significant gaps remaining", "major gaps remaining", "substantial gaps remaining",
        "significant gap remains", "major gap remains", "substantial gap remains",
        "significant work remaining", "substantial work remaining", "considerable work remaining",
        # Progress indicators (suggest partial, not almost)
        "some progress", "limited progress", "modest progress",
        "made progress", "good start", "started correctly", "began correctly",
        "good beginning", "valid beginning", "reasonable start",
        "on the right track", "correct direction", "correct approach started",
        # Work remaining
        "needs more work", "requires more work", "needs further work", "requires further work",
        "needs substantial work", "requires substantial work",
        "needs considerable work", "requires considerable work",
        "needs significant work", "requires significant work",
        "work remaining", "more work needed", "further work needed",
        "incomplete execution", "incomplete implementation",
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
    ]
    
    has_absolute_disqualifier = any(d in prediction_lower for d in absolute_almost_disqualifiers)
    
    # ============================================================================
    # STEP 2: Check for compound indicators (multi-word phrases)
    # ============================================================================
    compound_indicators = {
        "correct": [
            "fully correct", "completely correct", "entirely correct", "totally correct",
            "100% correct", "perfectly correct", "absolutely correct",
            "flawless", "no errors", "no mistakes", "perfect solution",
            "complete solution", "complete proof", "complete answer",
        ],
        "almost": [
            # Main result proven with minor issues - highest priority
            "main result essentially proven", "main result is essentially proven",
            "essentially proven the main", "essentially solved the main",
            # Strong almost indicators - clearly NOT partial
            "almost correct", "nearly correct", "almost complete", "nearly complete", 
            "mostly correct", "mostly right", "essentially correct", "practically correct",
            "virtually correct", "almost solved", "nearly solved", "almost perfect", 
            "nearly perfect", "essentially solved", "practically solved",
            "correct except for", "correct apart from", "correct aside from",
            "correct with minor", "correct with small", "correct with tiny",
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
            # High completion percentage
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
                # If we found "almost" indicator but have absolute disqualifier, override to partial
                if category == "almost" and has_absolute_disqualifier:
                    return "partial"
                return category
    
    # ============================================================================
    # STEP 3: Check for "incorrect" indicators (highest priority for wrong answers)
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
    # STEP 4: Check for "almost" disqualifiers - if present, CANNOT be "almost"
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
        "only 20%", "only 30%", "only 40%", "only 50%", "only 60%", "only 70%",
        # Incomplete/unfinished
        "incomplete proof", "incomplete solution", "unfinished proof", "unfinished solution",
        "unfinished work", "incomplete work", "work in progress",
        "not complete", "not finished", "not concluded",
        # Stopped early
        "stopped early", "gave up", "stopped before", "ended before",
        "did not finish", "did not complete", "did not conclude",
        # Only partial results
        "only proved", "only showed", "only one direction", "only one part",
        "only a lemma", "only proved a lemma", "proved a lemma",
        "some cases only", "not all cases", "missing cases",
        "incomplete case analysis", "partial case analysis",
        # Progress indicators (not completion)
        "some progress", "limited progress", "modest progress",
        "made progress", "good start", "started correctly", "began correctly",
        "good beginning", "valid beginning", "reasonable start",
        "on the right track", "correct direction", "correct approach started",
        # Work remaining
        "needs more work", "requires more work", "needs further work", "requires further work",
        "needs substantial work", "requires substantial work",
        "needs considerable work", "requires considerable work",
        "needs significant work", "requires significant work",
        "work remaining", "more work needed", "further work needed",
        "incomplete execution", "incomplete implementation",
        # Did not reach conclusion
        "did not reach", "failed to reach", "did not arrive",
        "did not conclude", "failed to conclude", "no conclusion",
        "did not obtain", "failed to obtain",
    ]
    
    has_almost_disqualifier = any(d in prediction_lower for d in almost_disqualifiers)
    
    # ============================================================================
    # STEP 5: Check for "almost" indicators - ONLY if no disqualifiers present
    # ============================================================================
    if not has_almost_disqualifier and not has_absolute_disqualifier:
        almost_indicators = [
            # Main result proven with minor issues - highest priority
            "main result essentially proven", "main result is essentially proven",
            "essentially proven the main", "essentially solved the main",
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
    # STEP 6: Check for "partial" indicators
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
    # STEP 7: Check for "correct" indicators
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

    def _apply_reasoning_correction(
        self, 
        normalized_pred: str, 
        reasoning_suggestion: str | None, 
        reasoning_text: str,
        combined_text: str
    ) -> str:
        """Apply reasoning-based correction to the prediction.
        
        This method uses the reasoning text to detect and correct misclassifications,
        particularly the common almost vs partial boundary errors.
        
        Args:
            normalized_pred: The normalized prediction from the response field
            reasoning_suggestion: The category suggested by reasoning analysis
            reasoning_text: The full reasoning text (lowercase)
            combined_text: Combined prediction and reasoning text
            
        Returns:
            The corrected prediction
        """
        if not reasoning_suggestion or reasoning_suggestion == normalized_pred:
            return _normalize_prediction(combined_text)
        
        reasoning_lower = reasoning_text.lower() if reasoning_text else ""
        
        # Priority 1: If reasoning suggests "incorrect", trust it (strong signal)
        if reasoning_suggestion == "incorrect":
            has_strong_incorrect = (
                "completely wrong" in reasoning_lower or 
                "totally wrong" in reasoning_lower or 
                "fundamentally wrong" in reasoning_lower or
                "nonsense" in reasoning_lower or
                "gibberish" in reasoning_lower or
                "no valid mathematical" in reasoning_lower or
                "no understanding" in reasoning_lower or
                "blank" in reasoning_lower or
                "empty" in reasoning_lower
            )
            if has_strong_incorrect:
                self.log_fn(f"Reasoning strongly suggests 'incorrect' over '{normalized_pred}'. Using reasoning.")
                return "incorrect"
            return normalized_pred
        
        # Priority 2: If reasoning suggests "partial" but response says "almost"
        # This is the most common misclassification - trust reasoning strongly
        if reasoning_suggestion == "partial" and normalized_pred == "almost":
            if _has_partial_disqualifiers(reasoning_text):
                self.log_fn(f"Partial disqualifiers found. Overriding 'almost' to 'partial'.")
                return "partial"
            # Double-check: only override if reasoning is strong
            if _analyze_reasoning_for_category(reasoning_text) == "partial":
                self.log_fn(f"Reasoning clearly suggests 'partial' over 'almost'. Using reasoning.")
                return "partial"
            return normalized_pred
        
        # Priority 3: If reasoning suggests "almost" and response says "partial"
        # Check if reasoning has strong "almost" indicators AND no disqualifiers
        if reasoning_suggestion == "almost" and normalized_pred == "partial":
            if _has_partial_disqualifiers(reasoning_text):
                # Reasoning has disqualifiers - cannot be "almost"
                self.log_fn(f"Disqualifiers found in reasoning. Keeping 'partial'.")
                return "partial"
            
            # Check for strong "almost" indicators
            if _has_almost_indicators(combined_text):
                self.log_fn(f"Reasoning strongly suggests 'almost' over 'partial'. Using reasoning.")
                return "almost"
            return normalized_pred
        
        # Priority 4: If reasoning suggests "correct" but response says something else
        # Only trust if reasoning is very clear
        if reasoning_suggestion == "correct" and normalized_pred in ["almost", "partial", "incorrect"]:
            has_strong_correct = (
                "fully correct" in reasoning_lower or 
                "completely correct" in reasoning_lower or 
                "100% correct" in reasoning_lower or
                "flawless" in reasoning_lower or
                "no errors" in reasoning_lower or
                "perfect solution" in reasoning_lower
            )
            if has_strong_correct:
                self.log_fn(f"Reasoning strongly suggests 'correct' over '{normalized_pred}'. Using reasoning.")
                return "correct"
            return normalized_pred
        
        # Default: use combined prediction
        return _normalize_prediction(combined_text)

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
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
            
            # Check for extremely long inputs that might cause issues
            max_input_length = 50000  # Reasonable limit
            if len(problem) > max_input_length or len(solution) > max_input_length or len(student_answer) > max_input_length:
                self.log_fn(f"Warning: Very long input detected. Problem: {len(problem)}, Solution: {len(solution)}, Answer: {len(student_answer)}")
                # Truncate if necessary
                problem = problem[:max_input_length] if len(problem) > max_input_length else problem
                solution = solution[:max_input_length] if len(solution) > max_input_length else solution
                student_answer = student_answer[:max_input_length] if len(student_answer) > max_input_length else student_answer
        except Exception as e:
            self.log_fn(f"Error sanitizing inputs: {e}")
            return "incorrect", [{"role": "error", "text": f"Input sanitization failed: {e}"}]
        
        # Build a more structured prompt with clear instructions
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

**"almost"** - Nearly perfect (85-99% complete) with ONLY tiny, easily fixable issues:
- One small calculation/sign error in otherwise correct solution
- Minor typo not affecting mathematical validity
- Core proof structure is complete and correct
- Would be correct with just a few minutes of fixes
- Main result is essentially proven - the student HAS solved the problem

**"partial"** - Meaningful progress (20-75% complete) with valid content but SIGNIFICANT gaps:
- Proved a useful lemma but main proof incomplete
- Correct approach started but execution stopped short
- Significant additional work needed to finish (more than 5 minutes)
- Main result is NOT essentially proven - the student has NOT solved the problem

**"incorrect"** - No valid mathematical progress. Wrong approach, nonsense, or empty.

## CRITICAL DECISION RULES - APPLY IN ORDER:
1. **100% flawless?** → "correct"
2. **85-99% complete with only tiny fixable errors AND main result essentially proven?** → "almost"  
3. **20-75% complete with meaningful progress but main result NOT essentially proven?** → "partial"
4. **Otherwise** → "incorrect"

## KEY DISTINCTION: "almost" vs "partial" (THIS IS THE MOST COMMON ERROR - READ CAREFULLY!)

**THE DEFINING QUESTION: Is the main result essentially proven?**

**"ALMOST" = Nearly Done (85-99% complete, main result ESSENTIALLY PROVEN):**
- Solution feels "essentially done" - like a complete solution with tiny blemishes
- Core proof structure is complete and correct
- Only tiny, easily fixable issues (minor arithmetic, small typos)
- Would be correct with just a few minutes of corrections
- Main result is essentially proven - the student HAS solved the problem
- Examples: "correct except for one sign error", "complete proof with minor typo"
- KEY PHRASES in reasoning: "main result essentially proven", "minor error only", "would be correct if..."
- KEY TEST: Can you say "the student has essentially solved the problem"? If YES → "almost"

**PARTIAL = Incomplete (20-75% complete, main result NOT proven):**
- Solution feels "unfinished" - work-in-progress, not nearly done
- Valid mathematical content BUT major gaps remain
- Significant additional work would be needed (more than 5 minutes)
- Main result is NOT proven or only partially proven - the student has NOT solved the problem
- Examples: "proved a lemma but stopped", "correct start but incomplete execution"
- KEY PHRASES in reasoning: "incomplete proof", "only proved a lemma", "stopped early", "needs more work"
- KEY TEST: Can you say "the student has NOT solved the problem yet"? If YES → "partial"

**CRITICAL: If you see ANY of these phrases, it CANNOT be "almost":**
- "incomplete proof", "incomplete solution", "unfinished"
- "only proved a lemma", "only one direction", "only one part"
- "stopped early", "gave up", "did not finish", "did not complete"
- "significant gaps remain", "major gaps remain"
- "less than 85%", "only 50%", "only 60%", "only 70%"
- "some progress", "good start" (these indicate partial, not almost!)
- "needs more work", "requires more work", "needs further work"

**DECISION HEURISTIC - USE THESE QUESTIONS:**
1. **Estimate completeness percentage:**
   - 95-100% → "correct" (or "almost" if tiny errors)
   - 85-94% → "almost" (nearly done, minor fixes)
   - 20-75% → "partial" (meaningful progress but incomplete)
   - 0-20% → "incorrect"

2. **Ask: "Is the main result essentially proven?" (THIS IS THE KEY QUESTION)**
   - YES with tiny issues → "almost"
   - NO, only partial progress → "partial"

3. **Ask: "Would this take under 5 minutes to fix?"**
   - YES → "almost"
   - NO → "partial"

4. **Ask: "Does the solution feel 'essentially done' or 'unfinished'?"**
   - Essentially done → "almost"
   - Unfinished → "partial"

5. **Ask: "Has the student essentially solved the problem?"**
   - YES, just needs tiny fixes → "almost"
   - NO, still has significant work → "partial"

## ADDITIONAL GUIDANCE FOR "ALMOST" vs "PARTIAL" CLASSIFICATION:

**When to choose "almost":**
- The student has essentially solved the problem - the main result is there
- Only tiny cosmetic or arithmetic errors remain
- The proof structure is complete, just needs minor polishing
- You can confidently say "this is essentially correct with minor fixes"
- Examples: sign error in final answer, minor typo in formula, small arithmetic slip
- The solution is 85-99% complete AND the main result is proven

**When to choose "partial":**
- The student has made meaningful progress but the main result is NOT proven
- Significant work remains to complete the proof
- The solution is "on the right track" but far from done
- You would need to say "this is incomplete, more work needed"
- Examples: proved a lemma but main theorem unfinished, correct start but stopped halfway, only handled some cases
- The solution is 20-75% complete OR the main result is NOT essentially proven

**CRITICAL DISTINCTION:**
- "Almost" = Student HAS essentially solved it, just needs tiny fixes (like polishing a finished piece)
- "Partial" = Student has NOT solved it, but made progress (like a draft that needs major work)

## CONSERVATIVE GRADING:
When in doubt, choose the LOWER category:
- Doubt between "correct" and "almost"? → "almost"
- Doubt between "almost" and "partial"? → "partial"
- Doubt between "partial" and "incorrect"? → "incorrect"

## PERCENTAGE-BASED DECISION MATRIX (USE THIS!):
- 95-100% complete, NO errors → "correct"
- 85-94% complete, tiny fixable errors, main result essentially proven → "almost"
- 20-75% complete, meaningful progress, main result NOT essentially proven → "partial"
- 0-20% complete, no valid progress → "incorrect"

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
- The solution is "unfinished" or "work-in-progress"
- "Needs more work", "requires more work", "needs further work"

**NOT "partial" if ANY of:**
- 85-99% complete with only minor issues AND main result essentially proven → "almost"
- "Almost correct", "nearly correct", "essentially correct" AND main result proven
- "Minor error only", "small error only", "tiny error only" AND main result proven
- "Would be correct if...", "just needs...", "only needs..." AND main result proven
- "Easily fixed", "simple fix", "quick fix" AND main result proven
- Main result is essentially proven with tiny blemishes
- The solution is "essentially done" or "nearly complete"

## Output Format:
<json>
{{
    "reasoning": "Step-by-step analysis: 1) Completeness % estimate (be specific: 50%, 90%, etc.), 2) Main result essentially proven? (YES/NO with explanation), 3) Fix time estimate (minutes), 4) Category choice with explicit justification referencing the criteria above",
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
7. **Almost vs Partial check**: Did I use "almost" but the solution has disqualifiers like "incomplete", "unfinished", "only proved", "stopped early", "needs more work"? → Change to "partial"

## COMMON MISTAKES TO AVOID:
- DON'T call something "almost" if it has significant gaps or is <85% complete
- DON'T call something "partial" if it's 85-99% complete with only tiny errors AND main result essentially proven
- DON'T call something "correct" if it has ANY errors, even tiny ones
- "Almost" = nearly perfect (main result essentially proven), "Partial" = meaningful but incomplete (main result NOT proven)
- "Almost" requires BOTH: 85-99% complete AND main result essentially proven
- "Partial" requires: main result NOT essentially proven (regardless of % if <85%)

## EXAMPLES TO GUIDE YOUR GRADING:

**Example 1 - "almost":**
Student: Complete proof with one sign error in the final step.
Analysis: 95% complete, main result essentially proven, one tiny fixable error.
Category: "almost"

**Example 2 - "partial":**
Student: Proved a useful lemma, started main proof but stopped halfway.
Analysis: 50% complete, main result NOT proven, significant work remaining.
Category: "partial"

**Example 3 - "partial" (NOT "almost"):**
Student: Correct approach, good start, but "incomplete" and "needs more work".
Analysis: 70% complete, main result NOT essentially proven, unfinished.
Category: "partial" (even if reasoning mentions "almost correct" - the key is main result NOT proven)

**Example 4 - "almost" (NOT "partial"):**
Student: "Essentially correct" solution with "minor error only" in calculation.
Analysis: 90% complete, main result essentially proven, tiny fixable error.
Category: "almost"

**Example 5 - "partial" (NOT "almost"):**
Student: "Good progress" made with "some progress" on the problem, but "incomplete proof" and "needs more work".
Analysis: 60% complete, main result NOT proven, significant gaps remain.
Category: "partial" (phrases like "good progress" and "some progress" indicate partial, NOT almost)

**Example 6 - "partial" (NOT "almost"):**
Student: "Only proved a lemma" and "only one direction" of a two-direction proof.
Analysis: 40% complete, main result NOT proven, only partial result achieved.
Category: "partial" ("only proved" phrases disqualify "almost")

**Example 7 - "almost" (NOT "partial"):**
Student: "Nearly correct" solution that is "essentially correct" with "minor mistake only".
Analysis: 88% complete, main result essentially proven, tiny fixable error.
Category: "almost" ("nearly correct" + "minor mistake only" = almost, not partial)

**Example 8 - "partial" (NOT "almost"):**
Student: "Stopped early" and "did not complete" the proof, "gave up" before finishing.
Analysis: 45% complete, main result NOT proven, unfinished work.
Category: "partial" (stopped early = partial, never almost)

**Example 9 - "partial" (NOT "almost"):**
Student: "Correct approach" with "good progress" but "needs more work" and "incomplete proof".
Analysis: 75% complete, main result NOT essentially proven, needs more work.
Category: "partial" ("needs more work" disqualifies "almost")

**Example 10 - "almost" (NOT "partial"):**
Student: "Essentially correct" with "minor error only" and "main result essentially proven".
Analysis: 92% complete, main result essentially proven, tiny fixable error.
Category: "almost" ("main result essentially proven" + "minor error only" = almost)

## MANDATORY FINAL CHECK - READ CAREFULLY:
Before outputting your answer, verify:
1. Did you use "almost" but the solution has "incomplete", "unfinished", "only proved", "stopped early", "needs more work", "requires more work"? → Change to "partial"
2. Did you use "partial" but the solution is 85-99% complete with only tiny errors AND main result essentially proven? → Change to "almost"
3. Is your reasoning consistent with your category choice? If not, fix the category.
4. Did you check if "main result essentially proven" appears in your reasoning? If NO and you chose "almost", reconsider.

## REMEMBER:
- "Almost" = 85-99% complete, main result ESSENTIALLY PROVEN, tiny fixable errors only
- "Partial" = 20-75% complete, main result NOT essentially proven, significant work remaining
- When in doubt between "almost" and "partial", choose "partial" (conservative grading)
- The KEY question is: "Has the student essentially solved the problem?"
"""
        
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
                            
                            # Use combined text for more accurate classification
                            combined_text = raw_prediction
                            if reasoning_text:
                                combined_text = f"{raw_prediction} {reasoning_text}"
                            combined_prediction = _normalize_prediction(combined_text)
                            
                            # SIMPLIFIED: Use reasoning-based correction with helper functions
                            prediction = self._apply_reasoning_correction(
                                normalized_pred, 
                                reasoning_suggestion, 
                                reasoning_text,
                                combined_text
                            )
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
        if reasoning_suggestion and reasoning_suggestion != prediction and reasoning_text:
            reasoning_lower = reasoning_text.lower()
            combined_text = f"{prediction} {reasoning_text}".lower()
            
            # Priority 1: If reasoning suggests "incorrect" but prediction says something else
            if reasoning_suggestion == "incorrect" and prediction in ["correct", "almost", "partial"]:
                has_strong_incorrect = (
                    "completely wrong" in reasoning_lower or 
                    "totally wrong" in reasoning_lower or 
                    "fundamentally wrong" in reasoning_lower or
                    "nonsense" in reasoning_lower or
                    "gibberish" in reasoning_lower or
                    "no valid mathematical" in reasoning_lower or
                    "no understanding" in reasoning_lower or
                    "blank" in reasoning_lower or
                    "empty" in reasoning_lower
                )
                if has_strong_incorrect:
                    self.log_fn(f"Reasoning strongly suggests 'incorrect' over '{prediction}'. Using reasoning.")
                    prediction = reasoning_suggestion
            
            # Priority 2: If reasoning suggests "partial" but prediction says "almost"
            elif reasoning_suggestion == "partial" and prediction == "almost":
                if _has_partial_disqualifiers(reasoning_text):
                    self.log_fn(f"Partial disqualifiers found in final check. Overriding to 'partial'.")
                    prediction = "partial"
                elif _analyze_reasoning_for_category(reasoning_text) == "partial":
                    self.log_fn(f"Reasoning strongly suggests 'partial' over 'almost'. Using reasoning.")
                    prediction = reasoning_suggestion
            
            # Priority 3: If reasoning suggests "almost" but prediction says "partial"
            elif reasoning_suggestion == "almost" and prediction == "partial":
                if _has_partial_disqualifiers(reasoning_text):
                    self.log_fn(f"Disqualifiers found in final check. Keeping 'partial'.")
                    prediction = "partial"
                elif _has_almost_indicators(combined_text):
                    self.log_fn(f"Reasoning strongly suggests 'almost' over 'partial'. Using reasoning.")
                    prediction = reasoning_suggestion
            
            # Priority 4: If reasoning suggests "correct" but prediction says something else
            elif reasoning_suggestion == "correct" and prediction in ["almost", "partial", "incorrect"]:
                has_strong_correct = (
                    "fully correct" in reasoning_lower or 
                    "completely correct" in reasoning_lower or 
                    "100% correct" in reasoning_lower or
                    "flawless" in reasoning_lower or
                    "no errors" in reasoning_lower or
                    "perfect solution" in reasoning_lower
                )
                if has_strong_correct:
                    self.log_fn(f"Reasoning strongly suggests 'correct' over '{prediction}'. Using reasoning.")
                    prediction = reasoning_suggestion
        
        # Final safety check: ensure prediction is still valid after all corrections
        valid_categories = ["correct", "almost", "partial", "incorrect"]
        
        # Handle None or non-string prediction
        if prediction is None:
            self.log_fn("Prediction is None, defaulting to 'incorrect'")
            prediction = "incorrect"
        
        # Ensure prediction is a string
        if not isinstance(prediction, str):
            try:
                prediction = str(prediction)
            except:
                self.log_fn(f"Cannot convert prediction to string, defaulting to 'incorrect'")
                prediction = "incorrect"
        
        # Strip and lowercase
        prediction = prediction.strip().lower()
        
        # Validate against valid categories
        if prediction not in valid_categories:
            self.log_fn(f"Prediction '{prediction}' not in valid categories, defaulting to 'incorrect'")
            prediction = "incorrect"
        
        return prediction, msg_history
