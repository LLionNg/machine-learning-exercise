#!/bin/bash
# Grade a submission locally using the project's uv environment.
# Copies submissions/<username>/solution.py next to the canonical test file in a
# temporary directory and runs pytest via `uv run`, so the originals are never
# touched.

if [ ! -f "test_solution.py" ]; then
    echo "Error: test_solution.py not found. Run this from the challenge directory."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

read -p "Enter your GitHub username: " USERNAME
SUBMISSION="submissions/$USERNAME/solution.py"

if [ ! -f "$SUBMISSION" ]; then
    echo "Error: '$SUBMISSION' not found."
    echo "Create it first:"
    echo "  mkdir -p submissions/$USERNAME"
    echo "  cp solution_template.py submissions/$USERNAME/solution.py"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
cp "$SUBMISSION" "$TEMP_DIR/solution.py"
cp "test_solution.py" "$TEMP_DIR/"

echo "Running tests for '$USERNAME'..."
( cd "$REPO_ROOT" && uv run python -m pytest "$TEMP_DIR" -v )
TEST_EXIT_CODE=$?

rm -rf "$TEMP_DIR"
exit $TEST_EXIT_CODE
