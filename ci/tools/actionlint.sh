#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script runs static checks on GitHub Actions workflows via actionlint.
#
# Unlike other linters, actionlint always examines all workflow files under .github/.
# It runs blazing fast and takes well under a second.
#
# The script outputs progress to stdout, and one actionlint diagnostic per
# violation (file, line, column, rule name) followed by self-serve fix instructions.
# It exits with 0 when no findings, 1 when actionlint reports any violation, and
# non-zero codes from `set -e` if the download or checksum verification fails.
#
# Usage:
#   # Run from the repository root with no arguments.
#   $ ci/tools/actionlint.sh
#
# See .github/actionlint.yaml for additional configurations.

set -euo pipefail

# Pin a specific release for reproducible CI behavior. The checksum is the one
# published in actionlint_${VERSION}_checksums.txt on the release page.
ACTIONLINT_VERSION="1.7.12"
ACTIONLINT_SHA256="8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
ACTIONLINT_ARCHIVE="actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz"
ACTIONLINT_URL="https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/${ACTIONLINT_ARCHIVE}"

ACTIONLINT="$(command -v actionlint || true)"

if [ -z "$ACTIONLINT" ]; then
    echo "INFO: actionlint not found in PATH. Downloading v${ACTIONLINT_VERSION}..."
    download_dir="$(mktemp -d)"
    trap 'rm -rf "$download_dir"' EXIT
    curl -sSfL "$ACTIONLINT_URL" -o "${download_dir}/${ACTIONLINT_ARCHIVE}"
    echo "${ACTIONLINT_SHA256}  ${download_dir}/${ACTIONLINT_ARCHIVE}" | sha256sum --check --quiet
    tar -xzf "${download_dir}/${ACTIONLINT_ARCHIVE}" -C "$download_dir" actionlint
    ACTIONLINT="${download_dir}/actionlint"
fi

# actionlint delegates `run:` blocks to `shellcheck` when it is on PATH.
# GitHub-hosted runners with `ubuntu-latest` have shellcheck preinstalled.
if ! command -v shellcheck &> /dev/null; then
    echo "WARNING: shellcheck not found in PATH. 'run:' blocks will not be checked."
fi

echo "INFO: Running actionlint on GitHub Actions workflows..."
if ! "$ACTIONLINT" -color; then
    echo "================================================================="
    echo "ERROR: actionlint found issues in .github/."
    echo "================================================================="
    echo "actionlint has no autofix. Either correct the workflow, or, if the"
    echo "report is a false positive:"
    echo "1. For an unknown runner label, add it to .github/actionlint.yaml."
    echo "2. For a shellcheck report, add '# shellcheck disable=SC<code>' to"
    echo "   the offending 'run:' block."
    echo "3. Otherwise, add an 'ignore:' pattern to .github/actionlint.yaml."
    echo "   See https://github.com/rhysd/actionlint/blob/main/docs/config.md"
    echo "================================================================="
    exit 1
fi
echo "INFO: actionlint check passed successfully."
