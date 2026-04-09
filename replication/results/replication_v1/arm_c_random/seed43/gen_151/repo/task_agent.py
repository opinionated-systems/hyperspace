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
    Also handles markdown-style ```json blocks with enhanced error recovery.
    """
    results = []
    search_from = 0
    
    # First try <json>...</json> tags
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
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    # Also try markdown code blocks with json tag
    md_search_from = 0
    while True:
        start = text.find("```json", md_search_from)
        if start == -1:
            break
        end = text.find("```", start + 7)
        if end == -1:
            break
        inner = text[start + 7:end].strip()
        md_search_from = end + 3
        
        parsed = _try_parse_json(inner)
        if parsed is not None:
            results.append(parsed)
    
    return results or None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with multiple recovery strategies.
    
    Returns the parsed dict or None if all strategies fail.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix single quotes to double quotes (common LLM error)
    try:
        # Replace single quotes around keys and string values
        fixed = re.sub(r"'([^']*?)'\s*:", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract just the JSON object if there's extra text
    try:
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            extracted = text[start:end+1]
            return json.loads(extracted)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from markdown code blocks (```json ... ```).
    
    Enhanced to handle various code block formats and common JSON errors.
    Uses _try_parse_json for robust error recovery.
    """
    # Try ```json ... ``` blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        parsed = _try_parse_json(match.strip())
        if parsed is not None:
            return parsed
    
    # Also try plain ``` blocks (without json tag)
    plain_pattern = r'```\s*(\{[\s\S]*?\})\s*```'
    plain_matches = re.findall(plain_pattern, text, re.DOTALL)
    for match in plain_matches:
        parsed = _try_parse_json(match.strip())
        if parsed is not None:
            return parsed
    
    return None


def _validate_grade(prediction: str, grading_guidelines: str) -> tuple[str, bool]:
    """Validate that the extracted grade is reasonable.
    
    Enhanced validation with multiple extraction strategies and confidence scoring.
    Returns a clean numeric grade (0-7) or "None" if invalid.
    
    Returns:
        (validated_grade, is_valid) where validated_grade is "0"-"7" or "None"
    """
    if not prediction or prediction == "None":
        return "None", False
    
    pred_clean = prediction.strip()
    pred_lower = pred_clean.lower()
    
    # Direct numeric match (0-7 for IMO problems) - most common case, highest confidence
    if pred_clean.isdigit():
        grade = int(pred_clean)
        if 0 <= grade <= 7:
            return str(grade), True
        # Handle edge case: grade might be 10 (out of 10) -> scale to 7
        if grade == 10:
            return "7", True
        return "None", False
    
    # Extract numeric grade from text patterns - ordered by confidence
    
    # Pattern 1: "X out of 7" or "X/7" - high confidence
    out_of_match = re.search(r'([0-7])\s*(?:out\s+of|/)\s*7', pred_lower)
    if out_of_match:
        return out_of_match.group(1), True
    
    # Pattern 2: "X points" or "grade of X" - high confidence
    points_match = re.search(r'(?:grade|score|award)\s*(?:of\s*)?["\']?([0-7])["\']?(?:\s*points?)?', pred_lower)
    if points_match:
        return points_match.group(1), True
    
    # Pattern 3: "X/7" format with slash
    slash_match = re.search(r'\b([0-7])\s*/\s*7\b', pred_lower)
    if slash_match:
        return slash_match.group(1), True
    
    # Pattern 4: Partial credit with explicit number
    partial_match = re.search(r'partial\s+(?:credit|score)?\s*:?\s*([0-7])', pred_lower)
    if partial_match:
        return partial_match.group(1), True
    
    # Pattern 5: Standalone numeric grades in text (be careful with this)
    # Only match if it's clearly a grade (surrounded by word boundaries or punctuation)
    numeric_match = re.search(r'(?:^|[\s:;,-])([0-7])(?:$|[\s.!;,])', pred_clean)
    if numeric_match:
        return numeric_match.group(1), True
    
    # Semantic patterns (lower confidence but useful)
    
    # Full credit patterns -> 7
    full_patterns = ['full credit', 'full marks', 'complete solution', 'perfect', '7/7', 'full score', 'entirely correct']
    if any(p in pred_lower for p in full_patterns):
        return "7", True
    
    # "Correct" without negative modifiers -> 7
    if 'correct' in pred_lower and not any(p in pred_lower for p in ['partial', 'incorrect', 'not correct', 'mostly incorrect']):
        return "7", True
    
    # Zero/incorrect patterns -> 0
    zero_patterns = ['zero', 'no credit', '0/7', 'none', 'incorrect', 'wrong', 'invalid', 'empty', 'no solution', 'blank']
    if any(p in pred_lower for p in zero_patterns):
        return "0", True
    
    # Progress indicators with implied grades
    if 'substantial progress' in pred_lower or 'mostly correct' in pred_lower:
        return "5", True
    if 'some progress' in pred_lower or 'significant progress' in pred_lower:
        return "4", True
    if 'limited progress' in pred_lower:
        return "2", True
    if 'minimal progress' in pred_lower or 'minor observation' in pred_lower:
        return "1", True
    
    # Partial without number -> default to 3 (middle of partial range)
    if 'partial' in pred_lower:
        return "3", True
    
    # If no clear grade found, return None
    return "None", False


class TaskAgent:
    """Task agent that solves IMO grading problems with enhanced reasoning and validation."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def _build_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for IMO grading with chain-of-thought reasoning."""
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        # Calculate approximate length for context
        student_len = len(student_answer) if student_answer else 0
        solution_len = len(solution) if solution else 0

        return f"""You are an expert IMO (International Mathematical Olympiad) grader with deep knowledge of mathematical problem-solving and competition grading standards.

Your task is to evaluate a student's solution to a mathematical problem and assign an appropriate grade.

## Problem Domain
{domain}

## Problem Statement
{problem}

## Official Solution (Length: ~{solution_len} chars)
{solution}

## Grading Guidelines
{grading_guidelines}

## Student's Answer (Length: ~{student_len} chars)
{student_answer}

## IMO Grading Scale Reference
- 7 points: Complete, correct solution with proper reasoning
- 6 points: Minor flaw in an otherwise correct solution
- 5 points: Significant progress with one gap or error
- 4 points: Multiple gaps but substantial progress
- 3 points: Some meaningful progress toward solution
- 2 points: Limited progress, some correct ideas
- 1 point: Minimal progress, minor relevant observation
- 0 points: No meaningful progress or completely wrong

## Instructions

1. **Analyze**: Carefully read the student's answer and compare it to the official solution.
2. **Identify**: Note any errors, missing steps, creative alternative approaches, or partial progress.
3. **Evaluate**: Consider the grading guidelines and the IMO scale above.
4. **Decide**: Assign a grade from 0-7 based on the student's demonstrated understanding and progress.
5. **Format**: Provide your detailed reasoning, then give the final grade as a single number (0-7).

Respond ONLY in the following JSON format:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis of the student's answer, comparing to the official solution, identifying errors or gaps, and explaining your evaluation...",
    "response": "X"
}}
</json>

Where X is a single digit from 0 to 7 representing the final grade. The "response" field must contain ONLY the numeric grade (0-7), nothing else."""

    def _extract_from_parsed_json(self, parsed: dict) -> tuple[str, str]:
        """Extract prediction and reasoning from a parsed JSON dict.
        
        Returns:
            (prediction, reasoning) - prediction is "None" if not found
        """
        prediction = "None"
        reasoning = ""
        
        # Try multiple possible keys for the grade/score
        for key in ["response", "grade", "score"]:
            if key in parsed:
                prediction = str(parsed[key]).strip()
                break
        
        if "reasoning" in parsed:
            reasoning = str(parsed["reasoning"])[:500]
        
        return prediction, reasoning

    def _extract_prediction(self, msg_history: list[dict]) -> tuple[str, str]:
        """Extract prediction and reasoning from message history.
        
        Enhanced extraction with multiple fallback strategies and better error handling.
        
        Returns:
            (prediction, reasoning)
        """
        prediction = "None"
        reasoning = ""
        
        try:
            if not msg_history:
                return prediction, reasoning
                
            last_msg = msg_history[-1].get("text", "") if isinstance(msg_history[-1], dict) else ""
            if not last_msg:
                return prediction, reasoning
            
            # Strategy 1: Try <json> tags first (most reliable)
            extracted = _extract_jsons(last_msg)
            if extracted:
                prediction, reasoning = self._extract_from_parsed_json(extracted[-1])
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 2: Try markdown code blocks
            md_json = _extract_json_from_markdown(last_msg)
            if md_json:
                prediction, reasoning = self._extract_from_parsed_json(md_json)
                if prediction != "None":
                    return prediction, reasoning
            
            # Strategy 3: Try to find any JSON-like object with grade/response/score
            # Use a more flexible pattern that handles nested braces
            json_pattern = r'\{[\s\S]*?"(?:response|grade|score)"[\s\S]*?\}'
            json_matches = re.findall(json_pattern, last_msg, re.DOTALL)
            for match in json_matches:
                parsed = _try_parse_json(match)
                if parsed:
                    prediction, reasoning = self._extract_from_parsed_json(parsed)
                    if prediction != "None":
                        return prediction, reasoning
            
            # Strategy 4: Look for explicit grade declarations in text
            text_patterns = [
                r'(?:final\s+)?(?:grade|score)\s*:?\s*["\']?([0-7])["\']?',
                r'(?:award|assign)\s*:?\s*["\']?([0-7])["\']?\s*(?:points?)?',
                r'(?:the\s+)?(?:grade|score)\s+(?:is|of)\s*["\']?([0-7])["\']?',
                r'\bgive\s+(?:a\s+)?(?:grade|score)\s+of\s*["\']?([0-7])["\']?',
            ]
            for pattern in text_patterns:
                match = re.search(pattern, last_msg, re.IGNORECASE)
                if match:
                    prediction = match.group(1).strip()
                    # Try to extract nearby reasoning
                    reasoning = self._extract_nearby_reasoning(last_msg, match.start())
                    return prediction, reasoning
                    
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
        
        return prediction, reasoning
    
    def _extract_nearby_reasoning(self, text: str, position: int, window: int = 200) -> str:
        """Extract reasoning text near a grade declaration.
        
        Looks for sentences before and after the grade declaration.
        """
        try:
            # Get text window around the match
            start = max(0, position - window)
            end = min(len(text), position + window)
            context = text[start:end]
            
            # Look for reasoning keywords
            reasoning_indicators = [
                'because', 'since', 'as', 'therefore', 'thus', 'hence',
                'analysis', 'evaluation', 'assessment', 'reasoning'
            ]
            
            # Find sentences with reasoning indicators
            sentences = re.split(r'[.!?]+', context)
            reasoning_sentences = []
            for sent in sentences:
                sent_lower = sent.lower()
                if any(indicator in sent_lower for indicator in reasoning_indicators):
                    reasoning_sentences.append(sent.strip())
            
            if reasoning_sentences:
                return ' '.join(reasoning_sentences)[:500]
            
            # Fallback: return the context window
            return context[:500]
        except Exception:
            return ""

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with enhanced reasoning and validation.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        instruction = self._build_prompt(inputs)

        # Log prompt statistics for debugging
        prompt_len = len(instruction)
        self.log_fn(f"Building prompt for problem (length: {prompt_len} chars)")

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
            # Log usage info if available
            usage = info.get("usage", {})
            if usage:
                self.log_fn(f"LLM usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                          f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                          f"Total: {usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            self.log_fn(f"LLM call failed: {e}")
            # Even on LLM failure, try to make an intelligent guess based on student answer
            return self._intelligent_fallback(inputs), []

        # Extract prediction from message history
        prediction, reasoning = self._extract_prediction(msg_history)
        
        # Validate the grade
        validated_grade, is_valid = _validate_grade(prediction, "")
        
        # Log extraction results
        if reasoning:
            self.log_fn(f"Reasoning: {reasoning[:200]}...")
        self.log_fn(f"Extracted: {prediction}, Validated: {validated_grade}, Valid: {is_valid}")
        
        # Fallback extraction from full response if needed
        if not is_valid and response:
            validated_grade, is_valid = self._fallback_extraction(response, inputs)

        return str(validated_grade), msg_history
    
    def _fallback_extraction(self, response: str, inputs: dict) -> tuple[str, bool]:
        """Extract grade from full response when primary extraction fails.
        
        Uses multiple strategies to find a valid grade.
        """
        # Strategy 1: Find any numeric grade (0-7) in the response
        numeric_matches = re.findall(r'\b([0-7])\b', response)
        if numeric_matches:
            # Use the last occurrence (usually the final decision)
            grade = numeric_matches[-1]
            self.log_fn(f"Fallback: Found grade {grade} in response")
            return grade, True
        
        # Strategy 2: Use content-based heuristic
        return self._intelligent_fallback(inputs), True
    
    def _intelligent_fallback(self, inputs: dict) -> str:
        """Make an intelligent grade guess based on student answer content.
        
        Analyzes the student answer to determine if it's empty, minimal, or substantial.
        """
        student_answer = inputs.get("student_answer", "")
        if not student_answer:
            self.log_fn("Fallback: No student answer, defaulting to 0")
            return "0"
        
        student_clean = student_answer.strip()
        student_len = len(student_clean)
        
        # Empty or nearly empty answer
        if student_len < 10:
            self.log_fn("Fallback: Empty/minimal answer, defaulting to 0")
            return "0"
        
        # Very short answer (just a number or few words)
        if student_len < 50:
            self.log_fn("Fallback: Very short answer, defaulting to 1")
            return "1"
        
        # Short answer (likely incomplete)
        if student_len < 200:
            self.log_fn("Fallback: Short answer, defaulting to 2")
            return "2"
        
        # Medium answer (some work shown)
        if student_len < 500:
            self.log_fn("Fallback: Medium answer, defaulting to 3")
            return "3"
        
        # Longer answer (substantial work)
        if student_len < 1000:
            self.log_fn("Fallback: Substantial answer, defaulting to 4")
            return "4"
        
        # Very long answer (likely complete attempt)
        self.log_fn("Fallback: Extensive answer, defaulting to 5")
        return "5"
