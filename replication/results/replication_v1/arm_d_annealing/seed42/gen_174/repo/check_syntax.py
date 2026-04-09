#!/usr/bin/env python3
"""Check syntax of task_agent.py"""
import ast
import sys

try:
    with open("task_agent.py", "r") as f:
        source = f.read()
    
    # Try to parse the source
    tree = ast.parse(source)
    print("✓ Syntax check passed: task_agent.py is valid Python")
    print(f"  File has {len(source)} characters and {len(source.splitlines())} lines")
    
    # Check for the TaskAgent class
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    print(f"  Classes found: {classes}")
    
    # Check for functions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    print(f"  Functions found: {functions}")
    
    sys.exit(0)
except SyntaxError as e:
    print(f"✗ Syntax error at line {e.lineno}, column {e.offset}: {e.msg}")
    print(f"  Text: {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
