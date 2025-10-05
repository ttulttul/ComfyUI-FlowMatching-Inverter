#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_PATH}/bin/python"
PIP_BIN="${VENV_PATH}/bin/pip"

if [[ ! -d "${VENV_PATH}" ]]; then
    python3 -m venv "${VENV_PATH}"
fi

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

"${PYTHON_BIN}" -m pip install --upgrade pip
if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    "${PIP_BIN}" install --extra-index-url "${PIP_EXTRA_INDEX_URL}" -r "${ROOT_DIR}/requirements.txt"
else
    "${PIP_BIN}" install -r "${ROOT_DIR}/requirements.txt"
fi

"${PYTHON_BIN}" -m pytest "$@"