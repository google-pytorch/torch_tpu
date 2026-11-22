#!/usr/bin/env bash
# Run ON the TPU VM. Waits until no vm_branch_tests.sh run is active, then
# runs one. Usage: vm_queue_branch.sh BRANCH FILTER [PYCHECK]
set -u
B=${1:?}; F=${2:?}; C=${3:-}
exec > ~/queue_$B.log 2>&1
while pgrep -f "bash .*vm_branch_tests.sh" >/dev/null || pgrep -f "bazel (build|test)" >/dev/null; do sleep 30; done
exec bash ~/vm_branch_tests.sh "$B" "$F" "$C"
