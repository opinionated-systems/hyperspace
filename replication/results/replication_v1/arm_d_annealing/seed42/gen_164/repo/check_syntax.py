#!/usr/bin/env python3
"""Check syntax of all Python files in the repo."""

import ast
import os
import sys

def check_file(path):
    """Check syntax of a single Python file."""
    try:
        with open(path, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"OK: {path}")
        return True
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {path}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {path}: {type(e).__name__}: {e}")
        return False

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = script_dir
    all_ok = True
    
    for root, dirs, files in os.walk(repo_path):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                if not check_file(path):
                    all_ok = False
    
    if all_ok:
        print("\nAll files have valid syntax!")
        return 0
    else:
        print("\nSome files have syntax errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
