"""
Test script to verify the improvements made to the codebase.
"""

import sys
sys.path.insert(0, '/workspaces/hyperagents/replication/results/replication_v1/arm_a_full/seed43/gen_17/repo')

from task_agent import _extract_jsons, _extract_json_fallback, _validate_and_normalize_prediction
from agent.tools.code_analysis_tool import tool_function as analyze_code


def test_json_extraction():
    """Test improved JSON extraction."""
    print("Testing JSON extraction...")
    
    # Test 1: Normal JSON tags
    text1 = '<json>{"reasoning": "test", "response": "7"}</json>'
    result1 = _extract_jsons(text1)
    assert result1 is not None and len(result1) == 1
    assert result1[0]["response"] == "7"
    print("  ✓ Normal JSON tags work")
    
    # Test 2: JSON with extra text inside tags
    text2 = '<json>Here is the JSON: {"reasoning": "test", "response": "5"}</json>'
    result2 = _extract_jsons(text2)
    assert result2 is not None and len(result2) == 1
    assert result2[0]["response"] == "5"
    print("  ✓ JSON with extra text inside tags works")
    
    # Test 3: Multiple JSON blocks
    text3 = '<json>{"a": 1}</json><json>{"b": 2}</json>'
    result3 = _extract_jsons(text3)
    assert result3 is not None and len(result3) == 2
    print("  ✓ Multiple JSON blocks work")
    
    print("JSON extraction tests passed!\n")


def test_prediction_validation():
    """Test improved prediction validation."""
    print("Testing prediction validation...")
    
    # Test IMO scoring
    guidelines_imo = "Score from 0 to 7"
    assert _validate_and_normalize_prediction("7", guidelines_imo) == "7"
    assert _validate_and_normalize_prediction("score: 5", guidelines_imo) == "5"
    assert _validate_and_normalize_prediction("grade: 3", guidelines_imo) == "3"
    print("  ✓ IMO scoring validation works")
    
    # Test Correct/Incorrect
    guidelines_bool = "Answer should be Correct or Incorrect"
    assert _validate_and_normalize_prediction("correct", guidelines_bool) == "Correct"
    assert _validate_and_normalize_prediction("right", guidelines_bool) == "Correct"
    assert _validate_and_normalize_prediction("incorrect", guidelines_bool) == "Incorrect"
    assert _validate_and_normalize_prediction("wrong", guidelines_bool) == "Incorrect"
    print("  ✓ Boolean validation works")
    
    # Test None variations
    assert _validate_and_normalize_prediction("none", "") == "None"
    assert _validate_and_normalize_prediction("N/A", "") == "None"
    assert _validate_and_normalize_prediction("null", "") == "None"
    print("  ✓ None normalization works")
    
    # Test letter grades
    guidelines_letter = "Grade: A, B, C, D, or F"
    assert _validate_and_normalize_prediction("a", guidelines_letter) == "A"
    assert _validate_and_normalize_prediction("B+", guidelines_letter) == "B+"
    print("  ✓ Letter grade validation works")
    
    print("Prediction validation tests passed!\n")


def test_code_analysis():
    """Test the new code analysis tool."""
    print("Testing code analysis tool...")
    
    # Test valid code
    valid_code = '''
def hello():
    """Say hello."""
    return "Hello, World!"

x = hello()
'''
    result = analyze_code(code=valid_code)
    assert "✅ Code is syntactically valid" in result
    print("  ✓ Valid code analysis works")
    
    # Test code with issues
    code_with_issues = '''
def empty():
    pass

def long_function():
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    x = 8
    x = 9
    x = 10
    x = 11
    x = 12
    x = 13
    x = 14
    x = 15
    x = 16
    x = 17
    x = 18
    x = 19
    x = 20
    x = 21
    x = 22
    x = 23
    x = 24
    x = 25
    x = 26
    x = 27
    x = 28
    x = 29
    x = 30
    x = 31
    x = 32
    x = 33
    x = 34
    x = 35
    x = 36
    x = 37
    x = 38
    x = 39
    x = 40
    x = 41
    x = 42
    x = 43
    x = 44
    x = 45
    x = 46
    x = 47
    x = 48
    x = 49
    x = 50
    x = 51
    x = 52
    x = 53
    x = 54
    x = 55
    return x
'''
    result = analyze_code(code=code_with_issues)
    assert "⚠️ Warnings" in result
    print("  ✓ Code with issues is detected")
    
    # Test invalid code
    invalid_code = "def broken(:\n    pass"
    result = analyze_code(code=invalid_code)
    assert "❌ Code is INVALID" in result
    print("  ✓ Invalid code detection works")
    
    print("Code analysis tests passed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running improvement tests")
    print("=" * 50 + "\n")
    
    test_json_extraction()
    test_prediction_validation()
    test_code_analysis()
    
    print("=" * 50)
    print("All tests passed! ✨")
    print("=" * 50)
