#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="clinc150-project"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
FRESH_RUN=false

usage() {
  echo "Usage: $0 [--fresh|-f]"
  echo
  echo "Options:"
  echo "  --fresh, -f   Remove existing .venv and recreate everything from scratch"
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --fresh|-f)
      FRESH_RUN=true
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      ;;
  esac
done

echo "==> Bootstrapping ${PROJECT_NAME}"

# 1. Fresh reset if requested
if [ "$FRESH_RUN" = true ]; then
  echo "==> Fresh run requested"
  if [ -d "$VENV_DIR" ]; then
    echo "==> Removing existing virtual environment at ${VENV_DIR}"
    rm -rf "$VENV_DIR"
  fi
fi

# 2. Create venv if it does not exist
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at ${VENV_DIR}"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "==> Reusing existing virtual environment at ${VENV_DIR}"
fi

# 3. Activate venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 4. Upgrade pip/setuptools/wheel
echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

# 5. Install uv inside the virtual environment
echo "==> Installing uv inside ${VENV_DIR}"
python -m pip install --upgrade uv

# 6. Initialize pyproject.toml if it doesn't exist
if [ ! -f "pyproject.toml" ]; then
  echo "==> Creating pyproject.toml"
  cat > pyproject.toml <<'EOF'
[project]
name = "clinc150-project"
version = "0.1.0"
description = "Intent classification and out-of-scope detection on CLINC150 using PyTorch"
requires-python = ">=3.11"
dependencies = [
  "torch",
  "datasets",
  "scikit-learn",
  "pandas",
  "numpy",
  "matplotlib",
  "tqdm",
  "jupyter",
  "ipykernel",
]

[dependency-groups]
dev = [
  "pytest",
  "ruff",
]
EOF
else
  echo "==> pyproject.toml already exists, leaving it unchanged"
fi

# 7. Sync environment from pyproject
echo "==> Syncing environment with uv"
uv sync --frozen

# 8. Export the portable locked dependencies and local project reference
echo "==> Generating requirements.txt"
uv export --frozen --no-hashes --output-file requirements.txt

echo
echo "==> Done."
echo "To activate the environment later, run:"
echo "   source ${VENV_DIR}/bin/activate"
echo
echo "Then start Jupyter if needed with:"
echo "   jupyter notebook"