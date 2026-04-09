#!/usr/bin/env python3
"""
Test runner for the agent package.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py task_agent  # Run specific test module
    python run_tests.py -v          # Run with verbose output
"""

import sys
import os
import importlib
import argparse

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_MODULES = [
    "tests.test_task_agent",
    "tests.test_tools",
    "tests.test_utils",
    "tests.test_llm_client",
]


def run_test_module(module_name: str, verbose: bool = False) -> tuple[int, int]:
    """Run a single test module and return (passed, failed) counts."""
    try:
        module = importlib.import_module(module_name)
        
        # Find all test functions
        test_functions = [
            getattr(module, name)
            for name in dir(module)
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        
        passed = 0
        failed = 0
        
        for test_func in test_functions:
            try:
                test_func()
                if verbose:
                    print(f"  ✓ {test_func.__name__}")
                passed += 1
            except AssertionError as e:
                if verbose:
                    print(f"  ✗ {test_func.__name__}: {e}")
                failed += 1
            except Exception as e:
                if verbose:
                    print(f"  ✗ {test_func.__name__}: Unexpected error: {e}")
                failed += 1
        
        return passed, failed
        
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return 0, 1


def main():
    parser = argparse.ArgumentParser(description="Run agent package tests")
    parser.add_argument(
        "module",
        nargs="?",
        help="Specific test module to run (e.g., 'task_agent', 'tools', 'utils')"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine which modules to run
    if args.module:
        # Map short names to full module names
        module_map = {
            "task_agent": "tests.test_task_agent",
            "tools": "tests.test_tools",
            "utils": "tests.test_utils",
            "llm_client": "tests.test_llm_client",
        }
        if args.module in module_map:
            modules_to_run = [module_map[args.module]]
        else:
            print(f"Unknown test module: {args.module}")
            print(f"Available modules: {', '.join(module_map.keys())}")
            sys.exit(1)
    else:
        modules_to_run = TEST_MODULES
    
    # Run tests
    total_passed = 0
    total_failed = 0
    
    print("=" * 60)
    print("Running Agent Package Tests")
    print("=" * 60)
    
    for module_name in modules_to_run:
        short_name = module_name.replace("tests.test_", "")
        print(f"\n{short_name}:")
        
        passed, failed = run_test_module(module_name, verbose=args.verbose)
        total_passed += passed
        total_failed += failed
        
        if not args.verbose:
            status = "✓" if failed == 0 else "✗"
            print(f"  {status} {passed} passed, {failed} failed")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
