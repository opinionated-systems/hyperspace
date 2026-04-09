#!/usr/bin/env python3
"""Test that task_agent.py is syntactically correct."""

import ast
import sys

with open('task_agent.py', 'r') as f:
    source = f.read()

try:
    ast.parse(source)
    print("✓ task_agent.py is syntactically correct")
    sys.exit(0)
except SyntaxError as e:
    print(f"✗ Syntax error in task_agent.py: {e}")
    sys.exit(1)
