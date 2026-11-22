#!/usr/bin/env bash
# Run ON the TPU VM in the background. Builds a wheel from upstream main, runs
# scalar_cache_bench.py and the scalar cache tests, then applies
# ~/scalar_cache_sweep.patch, rebuilds, and runs them again. Finally cleans
# the bazel output tree and restarts vm_suite_chain.sh with the given REV
# LABEL pairs.
# Usage: nohup bash ~/vm_scalar_cache_bench.sh [REV LABEL ...] > ~/scalar_bench.log 2>&1 &
# Writes ~/scalar_bench.summary (ends with ALL DONE), ~/bench_main.txt, ~/bench_sweep.txt.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
SUM=~/scalar_bench.summary
CHAIN_ARGS=("$@")
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
restart_chain() {
  (cd ~/torch_tpu && bazel clean >> ~/scalar_bench.log 2>&1); df -h / | tail -1 | tee -a "$SUM"
  if [ ${#CHAIN_ARGS[@]} -ge 2 ]; then
    log "restarting chain: ${CHAIN_ARGS[*]}"
    nohup bash ~/vm_suite_chain.sh "${CHAIN_ARGS[@]}" >> ~/suite_chain.log 2>&1 &
  fi
}
trap restart_chain EXIT
PY=~/env_fix/bin/python
UV=$(command -v uv || echo ~/.local/bin/uv)

build_and_install() {  # LABEL
  log "build wheel ($1)"
  nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe \
    --local_resources=cpu=HOST_CPUS*0.8 > ~/build_scalar_$1.log 2>&1
  local rc=$?; echo "BUILD_EXIT=$rc" >> ~/build_scalar_$1.log; tail -1 ~/build_scalar_$1.log | tee -a "$SUM"
  [ $rc -ne 0 ] && { grep -E "error:" ~/build_scalar_$1.log | head -10 | tee -a "$SUM"; return 1; }
  WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
  $UV pip install -q --python $PY --reinstall --no-deps "$WHL" 2>&1 | tail -1
  SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
  cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true
}
run_bench() {  # LABEL
  log "bench ($1)"
  {
    PYTHONPATH=. $PY -I ~/scalar_cache_bench.py "$1" 2>&1 | filt
    echo "--- pytest accelerator_api_test -k scalar"
    PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/accelerator_api_test.py -k "scalar" 2>&1 | filt | tail -4
  } > ~/bench_$1.txt 2>&1
  cat ~/bench_$1.txt | tee -a "$SUM"
}

rm -f "$SUM"
while pid=$(pgrep -of "^bash /home/joinal/vm_full_suite.sh"); do
  log "waiting for vm_full_suite pid $pid"; while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
cd ~/torch_tpu || exit 1
log "checkout upstream/main"
git fetch -q upstream main && git checkout -q -f -B scalar-cache-bench upstream/main && git clean -fdq -e 'bazel-*' \
  && git log --oneline -1 | tee -a "$SUM" || { log "ALL DONE (checkout failed)"; exit 1; }
build_and_install main && run_bench main
log "apply sweep patch"
git apply ~/scalar_cache_sweep.patch && git status --short | tee -a "$SUM" || { log "ALL DONE (patch failed)"; exit 1; }
build_and_install sweep && run_bench sweep
log "ALL DONE"
