#!/usr/bin/env bash
set -euo pipefail

PYSISENSE_REPO_URL="https://github.com/sisense/pysisense"
BRANCH="main"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_SRC_DIR="${REPO_ROOT}/docs_src"

echo "=== Refreshing PySisense tool registry ==="
echo "Repo root     : ${REPO_ROOT}"
echo "docs_src path : ${DOCS_SRC_DIR}"
echo "Upstream repo : ${PYSISENSE_REPO_URL}"
echo "Target branch : ${BRANCH}"
echo

# 1) Ensure docs_src exists and is a git clone of sisense/pysisense
if [[ ! -d "${DOCS_SRC_DIR}/.git" ]]; then
  echo "[1/3] docs_src is missing or not a git repo. Cloning upstream into docs_src..."
  rm -rf "${DOCS_SRC_DIR}"
  git clone "${PYSISENSE_REPO_URL}" "${DOCS_SRC_DIR}"
  echo
else
  echo "[1/3] docs_src exists. Updating it from upstream..."
  echo
fi

# 2) Pull latest upstream inside docs_src (do NOT touch root repo)
echo "[2/3] Pulling latest changes in docs_src from origin/${BRANCH}..."
cd "${DOCS_SRC_DIR}"

origin_url="$(git remote get-url origin 2>/dev/null || echo "")"
if [[ "${origin_url}" != "${PYSISENSE_REPO_URL}" ]]; then
  echo "  WARNING: docs_src origin does not match expected upstream."
  echo "  expected: ${PYSISENSE_REPO_URL}"
  echo "  current : ${origin_url}"
  echo "  Fixing origin to expected upstream."
  git remote set-url origin "${PYSISENSE_REPO_URL}"
fi

git fetch origin
git checkout "${BRANCH}"
git pull --rebase origin "${BRANCH}"
echo

# 3) Build registry + examples from the chatbot repo root
echo "[3/3] Building tool registry and examples..."
cd "${REPO_ROOT}"

python -m scripts.01_build_registry_from_sdk
python -m scripts.02_add_llm_examples_to_registry

echo
echo "Generated files:"
echo "  - config/tools.registry.json"
echo "  - config/tools.registry.with_examples.json"
echo
echo "=== Done: registry refreshed and examples regenerated ==="