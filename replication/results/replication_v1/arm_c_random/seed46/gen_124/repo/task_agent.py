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


class JSONExtractor:
    """Robust JSON extraction from LLM responses with multiple fallback strategies."""

    @staticmethod
    def extract_from_tags(text: str) -> list[dict]:
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
        return results

    @staticmethod
    def find_json_objects_by_brace_depth(s: str) -> list[str]:
        """Find potential JSON objects by tracking brace depth."""
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                depth = 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects

    @staticmethod
    def extract_by_brace_depth(text: str) -> list[dict]:
        """Extract JSON objects by parsing brace depth."""
        results = []
        for obj_str in JSONExtractor.find_json_objects_by_brace_depth(text):
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results

    @staticmethod
    def extract_by_regex(text: str) -> list[dict]:
        """Extract JSON objects using regex for simpler cases."""
        results = []
        pattern = r'\{[^{}]*"response"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group())
                if "response" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results

    @staticmethod
    def extract_full_text(text: str) -> list[dict]:
        """Try to parse the entire text as a single JSON object."""
        results = []
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "response" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass
        return results

    @staticmethod
    def extract_response_pattern(text: str) -> list[dict]:
        """Final fallback: extract response key-value pattern."""
        results = []
        response_pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
        match = re.search(response_pattern, text)
        if match:
            results.append({"response": match.group(1)})
        return results

    @classmethod
    def extract(cls, text: str, method: str = "auto") -> list[dict] | None:
        """Extract JSON using specified method or all methods in sequence.

        Args:
            text: The text to extract JSON from
            method: "auto" to try all methods, or specific method name

        Returns:
            List of extracted JSON objects, or None if none found
        """
        # Track which methods were attempted for debugging
        attempted_methods = []
        
        if method == "tags":
            results = cls.extract_from_tags(text)
            attempted_methods.append("tags")
        elif method == "brace_depth":
            results = cls.extract_by_brace_depth(text)
            attempted_methods.append("brace_depth")
        elif method == "regex":
            results = cls.extract_by_regex(text)
            attempted_methods.append("regex")
        elif method == "full_text":
            results = cls.extract_full_text(text)
            attempted_methods.append("full_text")
        elif method == "pattern":
            results = cls.extract_response_pattern(text)
            attempted_methods.append("pattern")
        elif method == "auto":
            # Try all methods in order of reliability with logging
            methods_to_try = [
                ("tags", cls.extract_from_tags),
                ("brace_depth", cls.extract_by_brace_depth),
                ("regex", cls.extract_by_regex),
                ("full_text", cls.extract_full_text),
                ("pattern", cls.extract_response_pattern),
            ]
            
            results = None
            for method_name, method_func in methods_to_try:
                attempted_methods.append(method_name)
                results = method_func(text)
                if results:
                    logger.debug(f"JSON extraction succeeded with method: {method_name}")
                    break
        else:
            raise ValueError(f"Unknown extraction method: {method}")

        if not results:
            logger.debug(f"JSON extraction failed after trying methods: {attempted_methods}")
            
        return results or None


# Backward-compatible function aliases
def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks."""
    return JSONExtractor.extract(text, method="tags")


def _extract_json_fallback(text: str) -> list[dict] | None:
    """Fallback JSON extraction for unwrapped JSON objects."""
    return (
        JSONExtractor.extract(text, method="brace_depth")
        or JSONExtractor.extract(text, method="regex")
        or JSONExtractor.extract(text, method="full_text")
        or JSONExtractor.extract(text, method="pattern")
    )


class TaskAgent:
    """Task agent that solves IMO grading problems with robust JSON extraction."""

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
        # Extract key fields for better structured prompting
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader evaluating student solutions to competition mathematics problems.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER TO EVALUATE:
{student_answer}

Your task is to carefully evaluate the student's answer against the official solution and grading guidelines.

Think step by step:
1. Understand what the problem is asking
2. Review the official solution approach
3. Analyze the student's answer for correctness and completeness
4. Check if the student followed the grading guidelines
5. Determine if the answer is correct, partially correct, or incorrect

Respond in JSON format with the following schema:
<json>
{{
    "response": "Your evaluation here - be specific about what is correct/incorrect and why"
}}
</json>"""

        try:
            response, msg_history, info = get_response_from_llm(
                msg=instruction,
                model=self.model,
                msg_history=[],
            )
        except Exception as e:
            self.log_fn(f"Error calling LLM: {e}")
            return "Error: LLM call failed", [{"role": "system", "text": f"Error: {e}"}]

        # Check if we got a valid response
        if not msg_history or len(msg_history) < 2:
            self.log_fn("Warning: Empty or incomplete message history from LLM")
            return "Error: No response from LLM", msg_history if msg_history else [{"role": "system", "text": "No response"}]

        # Extract prediction from JSON using primary method
        prediction = "None"
        extraction_method = "primary"
        raw_response = msg_history[-1].get("text", "")
        
        if not raw_response or not raw_response.strip():
            self.log_fn("Warning: Empty response text from LLM")
            return "Error: Empty response from LLM", msg_history
        
        try:
            extracted = _extract_jsons(raw_response)
            if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                prediction = extracted[-1]["response"]
                self.log_fn(f"Primary extraction succeeded, response type: {type(prediction).__name__}")
            else:
                # Try fallback extraction
                self.log_fn("Primary extraction failed, trying fallback...")
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded, response type: {type(prediction).__name__}")
                else:
                    self.log_fn("Warning: No valid JSON with 'response' key found in LLM output")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            # Try fallback on exception
            try:
                extracted = _extract_json_fallback(raw_response)
                if extracted and len(extracted) > 0 and "response" in extracted[-1]:
                    prediction = extracted[-1]["response"]
                    extraction_method = "fallback"
                    self.log_fn(f"Fallback extraction succeeded after exception, response type: {type(prediction).__name__}")
            except Exception as fallback_e:
                self.log_fn(f"Fallback extraction also failed: {fallback_e}")

        self.log_fn(f"Extraction method used: {extraction_method}")
        
        # Convert prediction to string, handling various types
        if prediction is None:
            prediction = "None"
        elif isinstance(prediction, (list, dict)):
            prediction = json.dumps(prediction)
        else:
            prediction = str(prediction)
            
        return prediction, msg_history
