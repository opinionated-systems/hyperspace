"""
PWD tool: returns the current working directory.
"""

import os


def tool_info():
    return {
        "name": "pwd",
        "description": "Return the current working directory.",
        "input_schema": {},
    }


def tool_function():
    return os.getcwd()
