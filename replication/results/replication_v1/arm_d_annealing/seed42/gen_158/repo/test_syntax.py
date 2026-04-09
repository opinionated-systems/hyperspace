#!/usr/bin/env python3
"""Quick syntax check for task_agent.py"""

import ast
import sys

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✓ {filepath} has valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

if __name__ == "__main__":
    files_to_check = [
        "task_agent.py",
        "meta_agent.py",
    ]
    
    all_valid = True
    for filepath in files_to_check:
        if not check_syntax(filepath):
            all_valid = False
    
    sys.exit(0 if all_valid else 1)
