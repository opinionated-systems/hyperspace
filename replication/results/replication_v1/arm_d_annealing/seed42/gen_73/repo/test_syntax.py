#!/usr/bin/env python3
"""Test script to verify task_agent.py syntax and basic functionality."""

import ast
import sys

def test_syntax():
    """Test that task_agent.py is syntactically valid."""
    with open("task_agent.py", "r") as f:
        source = f.read()
    
    try:
        ast.parse(source)
        print("✓ Syntax check passed: task_agent.py is valid Python")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

def test_import():
    """Test that task_agent.py can be imported."""
    try:
        import task_agent
        print("✓ Import check passed: task_agent module loads successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_class_instantiation():
    """Test that TaskAgent can be instantiated."""
    try:
        from task_agent import TaskAgent
        agent = TaskAgent()
        print("✓ Instantiation check passed: TaskAgent can be created")
        return True
    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        return False

def test_normalize_prediction():
    """Test the _normalize_prediction method with various inputs."""
    try:
        from task_agent import TaskAgent
        agent = TaskAgent()
        
        test_cases = [
            ("correct", "Correct"),
            ("Correct", "Correct"),
            ("incorrect", "Incorrect"),
            ("Incorrect", "Incorrect"),
            ("partially correct", "Partially correct"),
            ("Partially correct", "Partially correct"),
            ("almost correct", "Almost correct"),
            ("Almost correct", "Almost correct"),
            ("nearly correct", "Almost correct"),
            ("minor errors", "Almost correct"),
            ("wrong", "Incorrect"),
            ("right", "Correct"),
            ("nearly right", "Almost correct"),
            ("mostly right", "Almost correct"),
            ("slight errors", "Almost correct"),
            ("trivial errors", "Almost correct"),
            ("partial solution", "Partially correct"),
            ("major gaps", "Partially correct"),
            ("some correct parts", "Partially correct"),
            ("partial success", "Partially correct"),
            ("fundamentally wrong", "Incorrect"),
            ("not correct", "Incorrect"),
            ("not right", "Incorrect"),
            ("not valid", "Incorrect"),
            ("perfect", "Correct"),
            ("complete solution", "Correct"),
            ("fully correct", "Correct"),
        ]
        
        all_passed = True
        for input_val, expected in test_cases:
            result = agent._normalize_prediction(input_val)
            if result == expected:
                print(f"  ✓ '{input_val}' -> '{result}'")
            else:
                print(f"  ✗ '{input_val}' -> '{result}' (expected '{expected}')")
                all_passed = False
        
        if all_passed:
            print("✓ All normalization tests passed")
        return all_passed
    except Exception as e:
        print(f"✗ Normalization test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_extraction():
    """Test the JSON extraction functions."""
    try:
        from task_agent import _extract_jsons, _clean_json_string
        
        # Test JSON cleaning
        test_cases_clean = [
            ('{"key": "value",}', '{"key": "value"}'),  # trailing comma
            ("{'key': 'value'}", '{"key": "value"}'),  # single quotes
            ('{"key": "value", // comment\n}', '{"key": "value"\n}'),  # comment
            ('{"key": "value", /* comment */}', '{"key": "value"}'),  # block comment
        ]
        
        all_passed = True
        for input_val, expected_substr in test_cases_clean:
            result = _clean_json_string(input_val)
            if expected_substr in result or result == expected_substr:
                print(f"  ✓ Cleaned JSON correctly")
            else:
                print(f"  ✗ Clean failed: '{input_val}' -> '{result}'")
                all_passed = False
        
        # Test JSON extraction from various formats
        test_cases_extract = [
            ('<json>{"a": 1}</json>', [{"a": 1}]),
            ('```json\n{"b": 2}\n```', [{"b": 2}]),
            ('Some text {"c": 3} more text', [{"c": 3}]),
            ('No JSON here', None),
        ]
        
        for input_val, expected in test_cases_extract:
            result = _extract_jsons(input_val)
            if result == expected:
                print(f"  ✓ Extracted JSON correctly")
            else:
                print(f"  ✗ Extract failed: got {result}, expected {expected}")
                all_passed = False
        
        if all_passed:
            print("✓ All JSON extraction tests passed")
        return all_passed
    except Exception as e:
        print(f"✗ JSON extraction test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running task_agent.py tests...\n")
    
    results = []
    results.append(test_syntax())
    results.append(test_import())
    results.append(test_class_instantiation())
    results.append(test_normalize_prediction())
    results.append(test_json_extraction())
    
    print("\n" + "="*50)
    if all(results):
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. ✗")
        sys.exit(1)
