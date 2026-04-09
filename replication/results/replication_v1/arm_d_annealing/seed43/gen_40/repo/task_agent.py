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


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Fallback JSON extraction using regex for malformed responses."""
    results = []
    # Try to find JSON objects in code blocks
    json_pattern = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_pattern:
        try:
            results.append(json.loads(json_pattern.group(1)))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON-like structure with "response" key
    response_pattern = re.search(r'"response"\s*:\s*"([^"]*)"', text)
    if response_pattern and not results:
        results.append({"response": response_pattern.group(1)})
    
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.max_retries = 3

    def _build_grading_prompt(self, inputs: dict) -> str:
        """Build a structured prompt for the grading task with chain-of-thought."""
        domain = inputs.get("domain", "unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        return f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem Statement
{problem}

## Correct Solution
{solution}

## Grading Guidelines
{guidelines}

## Student's Answer
{student_answer}

## Instructions
1. First, analyze the student's answer step by step. Compare it against the correct solution.
2. Identify any errors, omissions, or alternative valid approaches.
3. Consider the grading guidelines carefully.
4. Provide your reasoning for the grade you will assign.
5. Finally, provide your grade/assessment in the JSON format below.

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed step-by-step analysis and reasoning...",
    "response": "Your final grade/assessment (e.g., 'Correct', 'Partial', 'Incorrect', or a numeric score)",
    "confidence": 0.95
}}
</json>

The confidence field should be a number between 0.0 and 1.0 indicating your confidence in the grade assigned.

Ensure your response is valid JSON wrapped in <json>...</json> tags."""

    def _extract_prediction(self, text: str) -> tuple[str, str, float]:
        """Extract prediction, reasoning, and confidence from response text.
        
        Returns:
            (prediction, reasoning, confidence) tuple
        """
        prediction = "None"
        reasoning = ""
        confidence = 0.5  # Default neutral confidence
        
        # Try primary extraction method
        extracted = _extract_jsons(text)
        if extracted is None:
            # Fallback to regex extraction
            extracted = _extract_json_with_regex(text)
        
        if extracted:
            last_json = extracted[-1]
            if "response" in last_json:
                prediction = str(last_json["response"])
            if "reasoning" in last_json:
                reasoning = str(last_json["reasoning"])
            if "confidence" in last_json:
                try:
                    confidence = float(last_json["confidence"])
                    # Clamp confidence to [0, 1] range
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.5
        
        return prediction, reasoning, confidence

    def _calculate_confidence_score(self, prediction: str, reasoning: str) -> float:
        """Calculate a confidence score based on prediction clarity and reasoning quality.
        
        Args:
            prediction: The extracted prediction string
            reasoning: The extracted reasoning string
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.5  # Base confidence
        
        # Boost confidence for clear binary predictions
        clear_correct = ["correct", "right", "true", "yes", "1", "100%"]
        clear_incorrect = ["incorrect", "wrong", "false", "no", "0", "0%"]
        pred_lower = prediction.lower()
        
        if any(term in pred_lower for term in clear_correct + clear_incorrect):
            score += 0.2
        
        # Boost confidence for detailed reasoning
        if len(reasoning) > 200:
            score += 0.15
        if len(reasoning) > 500:
            score += 0.1
        
        # Boost for structured reasoning (indicates careful analysis)
        structure_indicators = ["step", "first", "second", "third", "finally", "conclusion"]
        if any(indicator in reasoning.lower() for indicator in structure_indicators):
            score += 0.05
        
        return min(1.0, score)

    def forward(self, inputs: dict) -> tuple[str, list[dict], dict]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history, metadata) where metadata includes confidence score
        """
        instruction = self._build_grading_prompt(inputs)
        
        msg_history = []
        prediction = "None"
        reasoning = ""
        confidence = 0.0
        
        # Retry loop for robust extraction
        for attempt in range(self.max_retries):
            try:
                response, msg_history, info = get_response_from_llm(
                    msg=instruction,
                    model=self.model,
                    msg_history=msg_history if attempt > 0 else [],
                )
                
                last_text = msg_history[-1]["text"] if msg_history else ""
                prediction, reasoning, llm_confidence = self._extract_prediction(last_text)
                
                # Combine LLM confidence with calculated confidence
                calculated_confidence = self._calculate_confidence_score(prediction, reasoning)
                confidence = (llm_confidence + calculated_confidence) / 2
                
                if prediction != "None":
                    self.log_fn(f"Successfully extracted prediction: {prediction}")
                    self.log_fn(f"Confidence: {confidence:.2f}")
                    if reasoning:
                        self.log_fn(f"Reasoning: {reasoning[:200]}...")
                    break
                else:
                    self.log_fn(f"Attempt {attempt + 1}: Failed to extract prediction, retrying...")
                    # Add feedback for retry
                    if attempt < self.max_retries - 1:
                        instruction = f"""Your previous response did not contain valid JSON with a 'response' field.

Please ensure you wrap your response in <json>...</json> tags with the following format:
<json>
{{
    "reasoning": "Your analysis...",
    "response": "Your grade...",
    "confidence": 0.95
}}
</json>

Original task:
{self._build_grading_prompt(inputs)}"""
                    
            except Exception as e:
                self.log_fn(f"Error in attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    prediction = f"Error: {e}"
        
        metadata = {
            "confidence": confidence,
            "reasoning": reasoning,
            "attempts": attempt + 1,
        }
        
        return str(prediction), msg_history, metadata

    def forward_batch(self, inputs_list: list[dict]) -> list[tuple[str, list[dict], dict]]:
        """Process multiple grading tasks in batch.
        
        Args:
            inputs_list: List of input dicts, each with domain, problem, solution, 
                        grading_guidelines, student_answer
                        
        Returns:
            List of (prediction, msg_history, metadata) tuples
        """
        results = []
        for i, inputs in enumerate(inputs_list):
            self.log_fn(f"Processing batch item {i + 1}/{len(inputs_list)}")
            result = self.forward(inputs)
            results.append(result)
        return results

    def get_grading_summary(self, prediction: str, confidence: float, reasoning: str) -> dict:
        """Generate a structured summary of a grading result.
        
        Args:
            prediction: The grade/assessment assigned
            confidence: Confidence score (0.0-1.0)
            reasoning: The reasoning behind the grade
            
        Returns:
            Dictionary with structured grading summary
        """
        # Determine grade category
        pred_lower = prediction.lower()
        if any(term in pred_lower for term in ["correct", "right", "true", "yes", "1", "100%"]):
            category = "correct"
        elif any(term in pred_lower for term in ["incorrect", "wrong", "false", "no", "0", "0%"]):
            category = "incorrect"
        elif any(term in pred_lower for term in ["partial", "partially", "half"]):
            category = "partial"
        else:
            category = "unknown"
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            "grade": prediction,
            "category": category,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "reasoning_summary": reasoning[:500] if reasoning else "",
            "requires_review": confidence < 0.5 or category == "unknown",
        }
