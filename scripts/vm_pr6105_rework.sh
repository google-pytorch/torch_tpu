#!/usr/bin/env bash
# Run ON the TPU VM, in the background. Waits for the running vm_full_suite.sh
# to finish (by PID), checks out the PR #6105 branch plus ~/pr6105_rework.patch,
# builds a wheel, runs the max_unpool tests against it, and finally restarts
# vm_suite_chain.sh with the given REV LABEL pairs so the chip keeps working
# whether or not the build and tests succeed.
# Usage: nohup bash ~/vm_pr6105_rework.sh [REV LABEL ...] > ~/pr6105_rework.log 2>&1 &
# Writes ~/pr6105_rework.summary (ends with "ALL DONE") and ~/results_pr6105_rework.txt.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
SUM=~/pr6105_rework.summary; RES=~/results_pr6105_rework.txt; BLOG=~/build_pr6105_rework.log
PATCH=~/pr6105_rework.patch
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
restart_chain() {
  # Keep the test logs, then hand the chain a clean output tree: the PR branch
  # and the chain's branches use different XLA pins, and stale outputs from
  # both pins filled the 100 GB disk on 2026-11-05.
  mkdir -p ~/pr6105_rework_testlogs
  for t in ops_unit_test_tpu-v6e errors_test_tpu ops_test_tpu-v6e; do
    find -L ~/torch_tpu/bazel-testlogs/tests/$t -name test.log 2>/dev/null | while read -r f; do
      cp "$f" ~/pr6105_rework_testlogs/${t}_$(echo "$f" | md5sum | cut -c1-6).log
    done
  done
  (cd ~/torch_tpu && bazel clean >> ~/pr6105_rework.log 2>&1); df -h / | tail -1 | tee -a "$SUM"
  if [ $# -ge 2 ]; then
    log "restarting chain: $*"
    nohup bash ~/vm_suite_chain.sh "$@" >> ~/suite_chain.log 2>&1 &
  fi
}
CHAIN_ARGS=("$@")
trap 'restart_chain "${CHAIN_ARGS[@]}"' EXIT

rm -f "$SUM" "$RES"
while pid=$(pgrep -of "bash .*vm_full_suite.sh "); do
  log "waiting for vm_full_suite pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done

cd ~/torch_tpu || exit 1
log "checkout max-unpool2d + patch"
if ! { git fetch -q origin max-unpool2d && git checkout -q -f -B max-unpool2d-rework FETCH_HEAD \
       && git clean -fdq -e 'bazel-*' && git apply "$PATCH"; }; then
  log "ALL DONE (checkout or patch failed)"; exit 1
fi
git log --oneline -1 | tee -a "$SUM"; git status --short | tee -a "$SUM"

log "build wheel"
nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe \
  --local_resources=cpu=HOST_CPUS*0.8 > "$BLOG" 2>&1
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT" >> "$BLOG"; tail -1 "$BLOG" | tee -a "$SUM"
if [ "$BUILD_EXIT" -ne 0 ]; then
  grep -E "error:" "$BLOG" | head -20 | tee -a "$SUM"
  log "ALL DONE (build failed)"; exit 1
fi
WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
UV=$(command -v uv || echo ~/.local/bin/uv)
$UV pip install -q --python ~/env_fix/bin/python --reinstall --no-deps "$WHL" 2>&1 | tail -1
PY=~/env_fix/bin/python
SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true

log "tests"
{
  echo "--- pytest tests/ops_unit_test.py -k unpool"
  PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/ops_unit_test.py -k unpool 2>&1 | filt | tail -3
  echo "--- bazel, as CI runs it (errors_test needs --test_mode=tpu, which only the bazel target passes)"
  # ops_test does not expand globs in --test_filter; it needs the TestCase.test_name prefix form.
  for TF in "//tests:ops_unit_test_tpu-v6e *unpool*" "//tests:errors_test_tpu *unpool*" \
            "//tests:ops_test_tpu-v6e TestOps.test_max_unpool"; do
    set -- $TF
    bazel test "$1" --test_filter="$2" --config=no_rbe --local_test_jobs=1 \
      --cache_test_results=no --test_output=errors 2>&1 \
      | grep -E "^//|PASSED|FAILED|Executed|ERROR|NO STATUS|no test" | tail -4
    LOGDIR=${1#//}; LOGDIR=bazel-testlogs/${LOGDIR/:/\/}
    echo "   tests run: $(find -L "$LOGDIR" -name test.log 2>/dev/null | xargs grep -h "^Ran " | awk '{s+=$2} END {print s+0}')"
  done
} > "$RES" 2>&1
cat "$RES" | tee -a "$SUM"
log "ALL DONE"
