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
            # Basic cases
            ("correct", "Correct"),
            ("Correct", "Correct"),
            ("incorrect", "Incorrect"),
            ("Incorrect", "Incorrect"),
            ("partially correct", "Partially correct"),
            ("Partially correct", "Partially correct"),
            ("almost correct", "Almost correct"),
            ("Almost correct", "Almost correct"),
            # Variations
            ("nearly correct", "Almost correct"),
            ("mostly correct", "Almost correct"),
            ("minor errors", "Almost correct"),
            ("small mistakes", "Almost correct"),
            ("almost right", "Almost correct"),
            ("nearly right", "Almost correct"),
            ("partial credit", "Partially correct"),
            ("partially right", "Partially correct"),
            ("incomplete", "Partially correct"),
            ("partial solution", "Partially correct"),
            ("significant gaps", "Partially correct"),
            ("major gaps", "Partially correct"),
            ("wrong answer", "Incorrect"),
            ("fundamentally wrong", "Incorrect"),
            ("invalid solution", "Incorrect"),
            ("major error", "Incorrect"),
            ("zero marks", "Incorrect"),
            ("valid solution", "Correct"),
            ("full marks", "Correct"),
            ("full credit", "Correct"),
            ("accurate", "Correct"),
            # Edge cases - ensure "correct" in "incorrect" doesn't match
            ("The solution is incorrect", "Incorrect"),
            ("This is not correct", "Incorrect"),
            # Standalone values
            ("wrong", "Incorrect"),
            ("right", "Correct"),
            ("true", "Correct"),
            ("false", "Incorrect"),
            ("pass", "Correct"),
            ("fail", "Incorrect"),
            ("accepted", "Correct"),
            ("rejected", "Incorrect"),
            ("yes", "Correct"),
            ("no", "Incorrect"),
            ("0", "Incorrect"),
            ("zero", "Incorrect"),
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

if __name__ == "__main__":
    print("Running task_agent.py tests...\n")
    
    results = []
    results.append(test_syntax())
    results.append(test_import())
    results.append(test_class_instantiation())
    results.append(test_normalize_prediction())
    
    print("\n" + "="*50)
    if all(results):
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. ✗")
        sys.exit(1)
