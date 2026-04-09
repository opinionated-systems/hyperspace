"""
Agent tools package.

This package provides tools for the agentic loop:
- bash: Execute bash commands
- editor: View, create, and edit files
- file_stats: Get file statistics (lines, words, characters, size)
- git: Git operations
- search: Search for files and content

All tools follow the same interface:
- tool_info() -> dict: Returns tool metadata (name, description, input_schema)
- tool_function(**kwargs) -> str: Executes the tool and returns a string result
"""
