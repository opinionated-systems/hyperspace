#!/usr/bin/env python3
"""Test script to verify syntax of modified files."""

import ast
import sys

def test_file_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✓ {filepath} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath} - Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"✗ {filepath} - Error: {e}")
        return False

files_to_test = [
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/agent/tools/bash_tool.py",
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/task_agent.py",
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/meta_agent.py",
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/agent/agentic_loop.py",
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/agent/llm_client.py",
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/agent/tools/editor_tool.py",
    "/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_42/repo/agent/tools/registry.py",
]

all_ok = True
for filepath in files_to_test:
    if not test_file_syntax(filepath):
        all_ok = False

if all_ok:
    print("\n✓ All files have valid syntax!")
    sys.exit(0)
else:
    print("\n✗ Some files have syntax errors!")
    sys.exit(1)
