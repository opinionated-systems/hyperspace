import ast
import sys

with open('task_agent.py', 'r') as f:
    code = f.read()

try:
    ast.parse(code)
    print("Syntax OK")
    sys.exit(0)
except SyntaxError as e:
    print(f"Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")
    sys.exit(1)
