#!/bin/bash
# Quick setup script for testing environment

echo "Setting up testing environment for trk-algorithms..."

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-cov matplotlib pandas

# Optionally install the package in development mode
echo "Installing package in development mode..."
pip install -e .

echo ""
echo "Setup complete! You can now run tests with:"
echo "  pytest                    # Run all tests"
echo "  pytest -v                 # Verbose output"
echo "  pytest --cov=trk_algorithms  # With coverage"
echo "  make test                 # Using Makefile"
echo ""
echo "Or install dev dependencies with:"
echo "  pip install -e '.[dev]'"
