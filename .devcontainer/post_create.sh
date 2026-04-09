#!/bin/bash

# Suppress 'detected dubious ownership' errors for git commands.
git config --global --add safe.directory '*'

# Install project dependencies
pip install poetry
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install
