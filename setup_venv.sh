#!/bin/bash

# The name of the virtual environment directory
VENV_DIR="comfy-env"

# The Python version to use
PYTHON_VERSION="3.13"

# --- Script Start ---

echo "Checking for Python ${PYTHON_VERSION}..."

# Attempt to find the python3.13 executable
PYTHON_EXEC=$(command -v "python${PYTHON_VERSION}")

if [ -z "$PYTHON_EXEC" ]; then
    echo "Error: python${PYTHON_VERSION} not found in your PATH." >&2
    echo "Please install Python ${PYTHON_VERSION} or make sure it is in your system's PATH." >&2
    exit 1
fi

echo "Found Python at: ${PYTHON_EXEC}"

# Check if the virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '${VENV_DIR}' already exists."
else
    echo "Creating virtual environment '${VENV_DIR}' with ${PYTHON_EXEC}..."
    # Create the virtual environment using the specified Python version
    "$PYTHON_EXEC" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create the virtual environment." >&2
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate the virtual environment
echo "Activating virtual environment: ${VENV_DIR}"
echo "To activate, run 'source "${VENV_DIR}/bin/activate"' in your terminal."
echo "To deactivate, simply type 'deactivate' in your terminal."
