#!/usr/bin/env bash
# Run ON the TPU VM in a gap between chain runs (or after stopping the chain).
# Validates the three prepared branches on current main:
#   no-rbe-keeps-sandbox        : bazel build --config=no_rbe //csrc/ops:xla_utils must PASS
#   torch-headers-virtual-includes : same target under the unchanged no_rbe (standalone) must PASS
#   im2col-error-coverage       : errors_test im2col cases + error_coverage_test must PASS
# Also records that plain origin/main FAILS under --config=no_rbe (the bug).
# Usage: nohup bash ~/vm_test_prepared_branches.sh [REV LABEL ...] > ~/prepared.log 2>&1 &
# Writes ~/prepared.summary (ends with ALL DONE); restarts the chain with the given args.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
SUM=~/prepared.summary; CHAIN_ARGS=("$@")
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
finish() {
  (cd ~/torch_tpu && bazel clean >> ~/prepared.log 2>&1); df -h / | tail -1 | tee -a "$SUM"
  if [ ${#CHAIN_ARGS[@]} -ge 2 ]; then
    log "restarting chain: ${CHAIN_ARGS[*]}"
    nohup bash ~/vm_suite_chain.sh "${CHAIN_ARGS[@]}" >> ~/suite_chain.log 2>&1 &
  fi
}
trap finish EXIT
run() {  # LABEL CMD...
  local label=$1; shift
  log "$label: $*"
  nice -n 10 "$@" > ~/prepared_$label.log 2>&1; local rc=$?
  echo "RESULT $label exit=$rc" | tee -a "$SUM"
  [ $rc -ne 0 ] && grep -E "error:|FAILED|FAIL:" ~/prepared_$label.log | head -4 | cut -c1-200 | tee -a "$SUM"
  return $rc
}
rm -f "$SUM"
for pat in "^bash /home/joinal/vm_suite_chain.sh" "^bash /home/joinal/vm_full_suite.sh" "bin/bazel (build|test) "; do
  P=$(pgrep -f "$pat"); [ -n "$P" ] && { log "stopping $pat ($P)"; kill $P; }
done
sleep 5
echo "##### chain run stopped $(date -u +%H:%M:%S) to test prepared branches; re-queued after" >> ~/suite_chain.log
cd ~/torch_tpu || exit 1
git fetch -q upstream && git fetch -q origin
CPU="--local_resources=cpu=HOST_CPUS*0.8"
log "baseline: upstream/main under --config=no_rbe (expected to FAIL)"
git checkout -q -f upstream/main && git clean -fdq -e 'bazel-*'
run main_no_rbe bazel build --config=no_rbe $CPU //csrc/ops:xla_utils
log "branch no-rbe-keeps-sandbox (expected PASS)"
git checkout -q -f origin/no-rbe-keeps-sandbox && git clean -fdq -e 'bazel-*'
run sandbox_fix bazel build --config=no_rbe $CPU //csrc/ops:xla_utils
run sandbox_fix_test bazel test --config=no_rbe $CPU //tests:errors_test_tpu --test_filter='TpuVsGpuErrorTest.test_col2im*' --test_output=errors
log "branch torch-headers-virtual-includes (expected PASS under standalone)"
git checkout -q -f origin/torch-headers-virtual-includes && git clean -fdq -e 'bazel-*'
run virtual_includes bazel build --config=no_rbe $CPU //csrc/ops:xla_utils
run virtual_includes_layering bazel build --config=no_rbe $CPU --features=layering_check //csrc/ops:xla_utils //csrc/eager:op_dispatcher
log "branch im2col-error-coverage (needs a buildable base: uses the no_rbe sandbox fix on top)"
git checkout -q -f origin/im2col-error-coverage && git clean -fdq -e 'bazel-*' && git cherry-pick -n origin/no-rbe-keeps-sandbox
run im2col_errors bazel test --config=no_rbe $CPU //tests:errors_test_tpu --test_filter='TpuVsGpuErrorTest.test_im2col*' --test_output=errors
run im2col_cov bazel test --config=no_rbe $CPU //tests:error_coverage_test --test_output=errors
git checkout -q -f -- . ; git reset -q --hard
log "ALL DONE"
