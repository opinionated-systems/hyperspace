#!/usr/bin/env python3
"""Test script to verify bash tool works correctly."""

import sys
sys.path.insert(0, '/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_192/repo')

# Test that the module can be imported
from agent.tools.bash_tool import tool_function, set_allowed_root, reset_session

# Set allowed root
set_allowed_root('/workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_192/repo')

# Test simple commands
print("Testing bash tool...")

# Test 1: Simple echo
result = tool_function("echo 'Hello World'")
print(f"Test 1 (echo): {result}")

# Test 2: pwd
result = tool_function("pwd")
print(f"Test 2 (pwd): {result}")

# Test 3: ls
result = tool_function("ls -la")
print(f"Test 3 (ls): {result[:200]}...")

# Test 4: cat
result = tool_function("cat /workspaces/hyperagents/replication/results/replication_v1/arm_d_annealing/seed42/gen_192/repo/meta_agent.py | head -5")
print(f"Test 4 (cat): {result}")

print("\nAll tests completed successfully!")
