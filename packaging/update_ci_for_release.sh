#!/bin/bash
set -euo pipefail

VERSION="${1:?Usage: $0 <release-version> (e.g. 0.12)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="${REPO_ROOT}/.github/workflows"

# 1. Update test-infra @main refs and test-infra-ref on ALL workflow files
for f in "${WORKFLOW_DIR}"/*.yaml; do
    echo "Updating test-infra refs in $f ..."
    sed -i "s|pytorch/test-infra/\(.*\)@main|pytorch/test-infra/\1@release/${VERSION}|g" "$f"
    sed -i "s|test-infra-ref: main|test-infra-ref: release/${VERSION}|g" "$f"
done

# 2. Update python/ffmpeg/cuda versions in wheel install-and-test jobs only.
#    This must NOT touch install-and-test-third-party-interface or build-docs.
WHEEL_FILES=(
    "${WORKFLOW_DIR}/linux_wheel.yaml"
    "${WORKFLOW_DIR}/linux_aarch64_wheel.yaml"
    "${WORKFLOW_DIR}/linux_cuda_wheel.yaml"
    "${WORKFLOW_DIR}/linux_cuda_aarch64_wheel.yaml"
    "${WORKFLOW_DIR}/macos_wheel.yaml"
    "${WORKFLOW_DIR}/windows_wheel.yaml"
    "${WORKFLOW_DIR}/windows_cuda_wheel.yaml"
)

for f in "${WHEEL_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "WARNING: $f not found, skipping"
        continue
    fi
    echo "Updating matrix versions in $f ..."
    sed -i "s|python-version: \[.*\]|python-version: ['3.10', '3.11', '3.12', '3.13', '3.14']|g" "$f"
    sed -i "s|ffmpeg-version-for-tests: \[.*\]|ffmpeg-version-for-tests: ['4.4.2', '5.1.2', '6.1.1', '7.0.1', '8.0']|g" "$f"
    sed -i "s|cuda-version: \[.*\]|cuda-version: ['12.6', '13.0', '13.2']|g" "$f"
done

echo "Done! Updated workflow files for release/${VERSION}"
