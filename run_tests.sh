#!/bin/bash

# Script to run comprehensive test suite for Bayesian A/B Testing Playground

echo "====================================================================="
echo "Running Bayesian A/B Testing Playground Test Suite"
echo "====================================================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "ERROR: pytest not found. Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run tests with coverage
echo "Running tests with coverage..."
echo "---------------------------------------------------------------------"
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

TEST_EXIT_CODE=$?

echo ""
echo "====================================================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
    echo "====================================================================="
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
    echo "To view: open htmlcov/index.html"
else
    echo "✗ Some tests failed"
    echo "====================================================================="
    exit 1
fi

echo ""
