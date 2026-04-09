#!/usr/bin/env python3
"""Clear Python cache files to ensure fresh code is loaded."""

import os
import shutil

def clear_pycache(root_dir):
    """Remove all __pycache__ directories and .pyc files."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove __pycache__ directories
        if '__pycache__' in dirnames:
            cache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(cache_path)
                print(f"Removed: {cache_path}")
            except Exception as e:
                print(f"Error removing {cache_path}: {e}")
            dirnames.remove('__pycache__')
        
        # Remove .pyc files
        for filename in filenames:
            if filename.endswith('.pyc'):
                pyc_path = os.path.join(dirpath, filename)
                try:
                    os.remove(pyc_path)
                    print(f"Removed: {pyc_path}")
                except Exception as e:
                    print(f"Error removing {pyc_path}: {e}")

if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    clear_pycache(repo_dir)
    print("Cache cleared successfully!")
