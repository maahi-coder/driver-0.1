#!/bin/bash

# Driver Cognitive Monitoring System Startup Script

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$PROJECT_DIR/venv"

echo "=========================================="
echo "   Driver Cognitive Monitoring System"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Checking dependencies..."
    pip install -r "$PROJECT_DIR/requirements.txt" > /dev/null
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Run the application
echo "Starting application..."
"$VENV_DIR/bin/python" "$PROJECT_DIR/run.py"
