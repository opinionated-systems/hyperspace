#!/usr/bin/env python3
"""Test script to verify syntax of all Python files in the repo."""

import ast
import os
import sys

def check_file(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✓ {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: SyntaxError at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"✗ {filepath}: {type(e).__name__}: {e}")
        return False

def main():
    repo_path = "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_156/repo"
    all_valid = True
    
    for root, dirs, files in os.walk(repo_path):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if not check_file(filepath):
                    all_valid = False
    
    if all_valid:
        print("\n✓ All files have valid syntax!")
        return 0
    else:
        print("\n✗ Some files have syntax errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
