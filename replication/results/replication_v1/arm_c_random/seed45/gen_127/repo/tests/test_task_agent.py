"""
Tests for the task agent module.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_agent import TaskAgent, _extract_jsons, _extract_response_fallback


def test_extract_jsons_basic():
    """Test basic JSON extraction from <json> tags."""
    text = '<json>{"response": "Test answer"}</json>'
    result = _extract_jsons(text)
    assert result is not None
    assert len(result) == 1
    assert result[0]["response"] == "Test answer"


def test_extract_jsons_multiple():
    """Test extraction of multiple JSON objects."""
    text = '''<json>{"response": "First"}</json>
Some text in between
<json>{"response": "Second"}</json>'''
    result = _extract_jsons(text)
    assert result is not None
    assert len(result) == 2
    assert result[0]["response"] == "First"
    assert result[1]["response"] == "Second"


def test_extract_jsons_invalid():
    """Test handling of invalid JSON."""
    text = '<json>{invalid json}</json>'
    result = _extract_jsons(text)
    assert result is None


def test_extract_jsons_no_tags():
    """Test handling of text without <json> tags."""
    text = 'Just plain text without JSON tags'
    result = _extract_jsons(text)
    assert result is None


def test_extract_fallback_code_block():
    """Test fallback extraction from code blocks."""
    text = '```json\n{"response": "Code block answer"}\n```'
    result = _extract_response_fallback(text)
    assert result == "Code block answer"


def test_extract_fallback_pattern():
    """Test fallback extraction from response pattern."""
    text = '"response": "Pattern answer"'
    result = _extract_response_fallback(text)
    assert result == "Pattern answer"


def test_extract_fallback_none():
    """Test fallback returns None when no pattern found."""
    text = 'Just some random text without any JSON'
    result = _extract_response_fallback(text)
    assert result is None


def test_task_agent_validation():
    """Test input validation in TaskAgent."""
    agent = TaskAgent()
    
    # Test missing fields
    is_valid, error = agent._validate_inputs({})
    assert not is_valid
    assert "Missing required fields" in error
    
    # Test valid inputs
    valid_inputs = {
        "domain": "math",
        "problem": "2+2=?",
        "solution": "4",
        "grading_guidelines": "Correct if answer is 4",
        "student_answer": "4"
    }
    is_valid, error = agent._validate_inputs(valid_inputs)
    assert is_valid
    assert error == ""


def test_task_agent_stats():
    """Test statistics tracking in TaskAgent."""
    agent = TaskAgent()
    
    # Initial stats
    stats = agent.get_stats()
    assert stats["call_count"] == 0
    assert stats["success_count"] == 0
    assert stats["error_count"] == 0
    assert stats["success_rate"] == 0.0


def test_task_agent_build_instruction():
    """Test instruction building in TaskAgent."""
    agent = TaskAgent()
    inputs = {
        "domain": "math",
        "problem": "2+2=?",
        "solution": "4",
        "grading_guidelines": "Correct if answer is 4",
        "student_answer": "4"
    }
    instruction = agent._build_instruction(inputs)
    
    # Check that all inputs are included
    assert "math" in instruction
    assert "2+2" in instruction
    assert "4" in instruction
    assert "<json>" in instruction


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_extract_jsons_basic,
        test_extract_jsons_multiple,
        test_extract_jsons_invalid,
        test_extract_jsons_no_tags,
        test_extract_fallback_code_block,
        test_extract_fallback_pattern,
        test_extract_fallback_none,
        test_task_agent_validation,
        test_task_agent_stats,
        test_task_agent_build_instruction,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
