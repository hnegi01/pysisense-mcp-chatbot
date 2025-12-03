#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# PySisense repo URL
PYSISENSE_REPO_URL="https://github.com/sisense/pysisense"

# Target branch to pull from
BRANCH="main"

# ------------------------------------------------------------
# Script start
# ------------------------------------------------------------
# Get the directory of this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Refreshing PySisense tool registry ==="
echo "Repo root         : ${REPO_ROOT}"
echo "Expected origin   : ${PYSISENSE_REPO_URL}"
echo "Target branch     : ${BRANCH}"
echo

cd "${REPO_ROOT}"

# ------------------------------------------------------------
# 1) Ensure 'origin' remote exists and points to PySisense repo
# ------------------------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[1/3] Checking git remotes..."

  origin_url="$(git remote get-url origin 2>/dev/null || echo "")"
  if [[ -z "${origin_url}" ]]; then
    echo "  No 'origin' remote found. Adding origin -> ${PYSISENSE_REPO_URL}"
    git remote add origin "${PYSISENSE_REPO_URL}"
    origin_url="${PYSISENSE_REPO_URL}"
  else
    echo "  Current origin: ${origin_url}"
    if [[ "${origin_url}" != "${PYSISENSE_REPO_URL}" ]]; then
      echo "  WARNING: origin does NOT match expected URL:"
      echo "           expected: ${PYSISENSE_REPO_URL}"
      echo "           current : ${origin_url}"
      echo "  Proceeding anyway (using existing origin)."
    fi
  fi

  echo
else
  echo "ERROR: This directory is not a git repository."
  echo "Clone ${PYSISENSE_REPO_URL} first, then run this script from that repo."
  exit 1
fi

# ------------------------------------------------------------
# 2) Pull latest code from origin/main
# ------------------------------------------------------------
echo "[2/3] Pulling latest changes from origin/${BRANCH}..."

current_branch="$(git rev-parse --abbrev-ref HEAD || echo 'UNKNOWN')"
echo "  Current branch: ${current_branch}"
echo "  Git status (before pull):"
git status -sb || true
echo

git fetch origin
# Try to checkout main
if [[ "${current_branch}" != "${BRANCH}" ]]; then
  echo "  Checking out branch ${BRANCH}..."
  git checkout "${BRANCH}" || {
    echo "  WARNING: could not checkout ${BRANCH}, staying on ${current_branch}"
  }
fi

git pull --rebase origin "${BRANCH}" || {
  echo "  WARNING: git pull --rebase failed. Check your local changes."
}

echo
echo "  Git status (after pull):"
git status -sb || true
echo

# ------------------------------------------------------------
# 3) Build registry + examples
# ------------------------------------------------------------
echo "[3/3] Building tool registry and examples..."

echo "  → Running: python -m scripts.01_build_registry_from_sdk"
python -m scripts.01_build_registry_from_sdk

echo "  → Running: python -m scripts.02_add_llm_examples_to_registry"
python -m scripts.02_add_llm_examples_to_registry

echo
echo "Generated files:"
echo "  - config/tools.registry.json"
echo "  - config/tools.registry.with_examples.json"
echo
echo "=== Done: registry refreshed and examples regenerated ==="
