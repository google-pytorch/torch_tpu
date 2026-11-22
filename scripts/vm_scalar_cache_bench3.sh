#!/usr/bin/env bash
# Run ON the TPU VM in the background. Stops any chain run, then on BASE_REV
# (the last upstream revision that builds on this VM) builds and benchmarks:
#   clear  = BASE_REV + ~/p_clear.patch   (the landed #6037 full clear)
#   sweep  = clear   + ~/p_sweep.patch    (erase only unreferenced entries)
#   twogen = clear   + ~/p_twogen.patch   (two generations)
# Finally cleans the bazel tree and restarts vm_suite_chain.sh with the REV
# LABEL pairs that follow BASE_REV.
# Usage: nohup bash ~/vm_scalar_cache_bench3.sh BASE_REV [REV LABEL ...] > ~/scalar_bench3.log 2>&1 &
# Writes ~/scalar_bench3.summary (ends with ALL DONE) and ~/bench_{clear,sweep,twogen}.txt.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
BASE_REV=${1:?base rev}; shift
SUM=~/scalar_bench3.summary
CHAIN_ARGS=("$@")
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
finish() {
  (cd ~/torch_tpu && bazel clean >> ~/scalar_bench3.log 2>&1); df -h / | tail -1 | tee -a "$SUM"
  if [ ${#CHAIN_ARGS[@]} -ge 2 ]; then
    log "restarting chain: ${CHAIN_ARGS[*]}"
    nohup bash ~/vm_suite_chain.sh "${CHAIN_ARGS[@]}" >> ~/suite_chain.log 2>&1 &
  fi
}
trap finish EXIT
PY=~/env_fix/bin/python
UV=$(command -v uv || echo ~/.local/bin/uv)

build_install_bench() {  # LABEL
  log "build wheel ($1)"
  nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe \
    --local_resources=cpu=HOST_CPUS*0.8 > ~/build_scalar_$1.log 2>&1
  local rc=$?; echo "BUILD_EXIT=$rc" >> ~/build_scalar_$1.log; tail -1 ~/build_scalar_$1.log | tee -a "$SUM"
  [ $rc -ne 0 ] && { grep -E "error:" ~/build_scalar_$1.log | head -5 | cut -c1-200 | tee -a "$SUM"; return 1; }
  WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
  $UV pip install -q --python $PY --reinstall --no-deps "$WHL" 2>&1 | tail -1
  SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
  cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true
  log "bench ($1)"
  {
    PYTHONPATH=. $PY -I ~/scalar_cache_bench.py "$1" 2>&1 | filt
    echo "--- pytest accelerator_api_test -k scalar"
    PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/accelerator_api_test.py -k "scalar" 2>&1 | filt | tail -4
  } > ~/bench_$1.txt 2>&1
  cat ~/bench_$1.txt | tee -a "$SUM"
}

rm -f "$SUM"
for pat in "^bash /home/joinal/vm_suite_chain.sh" "^bash /home/joinal/vm_full_suite.sh" "bin/bazel (build|test) "; do
  P=$(pgrep -f "$pat"); [ -n "$P" ] && { log "stopping $pat ($P)"; kill $P; }
done
sleep 5
for pat in "^bash /home/joinal/vm_full_suite.sh" "bin/bazel (build|test) "; do
  P=$(pgrep -f "$pat"); [ -n "$P" ] && kill -KILL $P
done
echo "##### chain run stopped $(date -u +%H:%M:%S) for the scalar-cache benchmark (base $BASE_REV); re-queued after it" >> ~/suite_chain.log

cd ~/torch_tpu || exit 1
log "checkout $BASE_REV + clear"
git fetch -q upstream && git checkout -q -f -B scalar-cache-bench "$BASE_REV" && git clean -fdq -e 'bazel-*' \
  && git apply ~/p_clear.patch && git log --oneline -1 | tee -a "$SUM" || { log "ALL DONE (checkout failed)"; exit 1; }
build_install_bench clear
log "apply sweep"
git apply ~/p_sweep.patch && git status --short | tee -a "$SUM" && build_install_bench sweep
log "reset to clear, apply twogen"
git checkout -q -f -- . && git apply ~/p_clear.patch && git apply ~/p_twogen.patch && git status --short | tee -a "$SUM" && build_install_bench twogen
log "ALL DONE"
