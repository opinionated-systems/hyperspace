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
            # Correct variations
            ("correct", "Correct"),
            ("Correct", "Correct"),
            ("right", "Correct"),
            ("true", "Correct"),
            ("yes", "Correct"),
            ("pass", "Correct"),
            ("accepted", "Correct"),
            ("full marks", "Correct"),
            ("full credit", "Correct"),
            ("valid", "Correct"),
            ("accurate", "Correct"),
            ("perfect", "Correct"),
            ("complete", "Correct"),
            ("exact", "Correct"),
            ("precise", "Correct"),
            ("proper", "Correct"),
            ("appropriate", "Correct"),
            ("satisfactory", "Correct"),
            ("successful", "Correct"),
            ("complete solution", "Correct"),
            ("complete answer", "Correct"),
            ("fully correct", "Correct"),
            ("entirely correct", "Correct"),
            ("completely correct", "Correct"),
            
            # Incorrect variations
            ("incorrect", "Incorrect"),
            ("Incorrect", "Incorrect"),
            ("wrong", "Incorrect"),
            ("false", "Incorrect"),
            ("no", "Incorrect"),
            ("fail", "Incorrect"),
            ("rejected", "Incorrect"),
            ("zero", "Incorrect"),
            ("0", "Incorrect"),
            ("invalid", "Incorrect"),
            ("inaccurate", "Incorrect"),
            ("error", "Incorrect"),
            ("not correct", "Incorrect"),
            ("not right", "Incorrect"),
            ("not valid", "Incorrect"),
            ("not accurate", "Incorrect"),
            ("not true", "Incorrect"),
            ("fails", "Incorrect"),
            ("failed", "Incorrect"),
            ("failure", "Incorrect"),
            ("unsuccessful", "Incorrect"),
            ("not accepted", "Incorrect"),
            ("denied", "Incorrect"),
            ("not pass", "Incorrect"),
            ("not passed", "Incorrect"),
            ("flawed", "Incorrect"),
            ("erroneous", "Incorrect"),
            ("mistaken", "Incorrect"),
            
            # Partially correct variations
            ("partially correct", "Partially correct"),
            ("Partially correct", "Partially correct"),
            ("partial credit", "Partially correct"),
            ("partially right", "Partially correct"),
            ("incomplete", "Partially correct"),
            ("partial", "Partially correct"),
            ("significant gaps", "Partially correct"),
            ("partial solution", "Partially correct"),
            ("partial answer", "Partially correct"),
            ("some correct", "Partially correct"),
            ("partly correct", "Partially correct"),
            ("partly right", "Partially correct"),
            ("half correct", "Partially correct"),
            ("half right", "Partially correct"),
            ("mixed results", "Partially correct"),
            ("some progress", "Partially correct"),
            ("partial success", "Partially correct"),
            ("incomplete solution", "Partially correct"),
            ("missing parts", "Partially correct"),
            
            # Almost correct variations
            ("almost correct", "Almost correct"),
            ("Almost correct", "Almost correct"),
            ("nearly correct", "Almost correct"),
            ("mostly correct", "Almost correct"),
            ("minor errors", "Almost correct"),
            ("small mistakes", "Almost correct"),
            ("almost right", "Almost correct"),
            ("nearly right", "Almost correct"),
            ("mostly right", "Almost correct"),
            ("slight error", "Almost correct"),
            ("trivial error", "Almost correct"),
            ("minor mistake", "Almost correct"),
            ("small error", "Almost correct"),
            ("tiny error", "Almost correct"),
            ("minimal error", "Almost correct"),
            ("nearly perfect", "Almost correct"),
            ("almost perfect", "Almost correct"),
            ("mostly perfect", "Almost correct"),
            ("close to correct", "Almost correct"),
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

def test_extract_jsons():
    """Test the _extract_jsons function with various inputs."""
    try:
        from task_agent import _extract_jsons, _try_parse_json
        
        test_cases = [
            # Test <json> blocks
            ('<json>{"assessment": "Correct"}</json>', [{"assessment": "Correct"}]),
            ('<json>{"assessment": "Correct", "score": 100}</json>', [{"assessment": "Correct", "score": 100}]),
            
            # Test markdown code blocks
            ('```json\n{"assessment": "Incorrect"}\n```', [{"assessment": "Incorrect"}]),
            ('```\n{"assessment": "Partially correct"}\n```', [{"assessment": "Partially correct"}]),
            
            # Test inline JSON with nested braces (the key improvement)
            ('Some text {"outer": {"inner": "value"}} more text', [{"outer": {"inner": "value"}}]),
            ('{"a": {"b": {"c": 1}}}', [{"a": {"b": {"c": 1}}}]),
            
            # Test JSON with escaped quotes
            ('{"key": "value with \\"quotes\\""}', [{"key": 'value with "quotes"'}]),
            
            # Test multiple JSON objects
            ('<json>{"a": 1}</json> <json>{"b": 2}</json>', [{"a": 1}, {"b": 2}]),
            
            # Test no JSON
            ('Just plain text without any JSON', None),
            ('', None),
        ]
        
        all_passed = True
        for input_val, expected in test_cases:
            result = _extract_jsons(input_val)
            if result == expected:
                print(f"  ✓ Extracted JSON correctly from input")
            else:
                print(f"  ✗ Input: {repr(input_val[:50])}...")
                print(f"    Expected: {expected}")
                print(f"    Got: {result}")
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
    results.append(test_extract_jsons())
    
    print("\n" + "="*50)
    if all(results):
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. ✗")
        sys.exit(1)
