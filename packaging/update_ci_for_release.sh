#!/bin/bash
set -euo pipefail

BRANCH="${1:?Usage: $0 <test-infra-release-branch> (e.g. release/2.12)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="${REPO_ROOT}/.github/workflows"

WHEEL_FILES=(
    "${WORKFLOW_DIR}/linux_wheel.yaml"
    "${WORKFLOW_DIR}/linux_aarch64_wheel.yaml"
    "${WORKFLOW_DIR}/linux_cuda_wheel.yaml"
    "${WORKFLOW_DIR}/linux_cuda_aarch64_wheel.yaml"
    "${WORKFLOW_DIR}/macos_wheel.yaml"
    "${WORKFLOW_DIR}/windows_wheel.yaml"
    "${WORKFLOW_DIR}/windows_cuda_wheel.yaml"
)

# Only the `install-and-test` job of each file: from its header up to the next
# top-level job header.
INSTALL_AND_TEST_JOB='/^  install-and-test:$/,/^  [^ #].*:$/'

for f in "${WHEEL_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "WARNING: $f not found, skipping"
        continue
    fi

    # 1. Update test-infra @main refs and test-infra-ref
    echo "Updating test-infra refs in $f ..."
    sed -i "s|pytorch/test-infra/\(.*\)@main|pytorch/test-infra/\1@${BRANCH}|g" "$f"
    sed -i "s|test-infra-ref: main|test-infra-ref: ${BRANCH}|g" "$f"

    # 2. Update python/ffmpeg/cuda versions in the install-and-test job.
    #    The `python-version` sed below only expands the *test* matrix: wheels
    #    are py3-none and are always built with a single Python (the plural
    #    `python-versions` input of the generate-matrix jobs, left untouched).
    echo "Updating matrix versions in $f ..."
    sed -i "${INSTALL_AND_TEST_JOB} s|python-version: \[.*\]|python-version: ['3.11', '3.12', '3.13', '3.14', '3.15']|" "$f"
    sed -i "${INSTALL_AND_TEST_JOB} s|ffmpeg-version-for-tests: \[.*\]|ffmpeg-version-for-tests: ['4', '5', '6', '7', '8', '9']|" "$f"
    sed -i "${INSTALL_AND_TEST_JOB} s|cuda-version: \[.*\]|cuda-version: ['13.2', '13.4']|" "$f"
done

echo "Done! Updated workflow files to point to test-infra branch '${BRANCH}'"
