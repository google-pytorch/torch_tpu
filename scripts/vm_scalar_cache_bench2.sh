#!/usr/bin/env bash
# Run ON the TPU VM in the background, after vm_scalar_cache_bench.sh was
# started. Waits for that job (by PID), stops the chain run it restarts (which
# is seconds old at that point), builds upstream main plus
# ~/scalar_cache_twogen.patch, runs the benchmark as "twogen", cleans the bazel
# tree and restarts the chain with the given REV LABEL pairs.
# Usage: nohup bash ~/vm_scalar_cache_bench2.sh [REV LABEL ...] > ~/scalar_bench2.log 2>&1 &
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
SUM=~/scalar_bench2.summary
CHAIN_ARGS=("$@")
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
finish() {
  (cd ~/torch_tpu && bazel clean >> ~/scalar_bench2.log 2>&1); df -h / | tail -1 | tee -a "$SUM"
  if [ ${#CHAIN_ARGS[@]} -ge 2 ]; then
    log "restarting chain: ${CHAIN_ARGS[*]}"
    nohup bash ~/vm_suite_chain.sh "${CHAIN_ARGS[@]}" >> ~/suite_chain.log 2>&1 &
  fi
}
trap finish EXIT
PY=~/env_fix/bin/python
UV=$(command -v uv || echo ~/.local/bin/uv)

rm -f "$SUM"
while pid=$(pgrep -of "^bash /home/joinal/vm_scalar_cache_bench.sh"); do
  log "waiting for bench job pid $pid"; while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
# The bench job restarts the chain on exit; stop that run before it gets far.
sleep 20
for pat in "^bash /home/joinal/vm_suite_chain.sh" "^bash /home/joinal/vm_full_suite.sh" "bin/bazel (build|test) "; do
  P=$(pgrep -f "$pat"); [ -n "$P" ] && { log "stopping $pat ($P)"; kill $P; }
done
sleep 5
for pat in "^bash /home/joinal/vm_full_suite.sh" "bin/bazel (build|test) "; do
  P=$(pgrep -f "$pat"); [ -n "$P" ] && kill -KILL $P
done
echo "##### chain run stopped $(date -u +%H:%M:%S) for the twogen benchmark; re-queued after it" >> ~/suite_chain.log
cd ~/torch_tpu || exit 1
log "checkout upstream/main + twogen patch"
git checkout -q -f -B scalar-cache-bench upstream/main && git clean -fdq -e 'bazel-*' && git apply ~/scalar_cache_twogen.patch \
  && git status --short | tee -a "$SUM" || { log "ALL DONE (checkout/patch failed)"; exit 1; }
log "build wheel (twogen)"
nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe \
  --local_resources=cpu=HOST_CPUS*0.8 > ~/build_scalar_twogen.log 2>&1
rc=$?; echo "BUILD_EXIT=$rc" >> ~/build_scalar_twogen.log; tail -1 ~/build_scalar_twogen.log | tee -a "$SUM"
[ $rc -ne 0 ] && { grep -E "error:" ~/build_scalar_twogen.log | head -10 | tee -a "$SUM"; log "ALL DONE (build failed)"; exit 1; }
WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
$UV pip install -q --python $PY --reinstall --no-deps "$WHL" 2>&1 | tail -1
SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true
log "bench (twogen)"
{
  PYTHONPATH=. $PY -I ~/scalar_cache_bench.py twogen 2>&1 | filt
  echo "--- pytest accelerator_api_test -k scalar"
  PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/accelerator_api_test.py -k "scalar" 2>&1 | filt | tail -4
} > ~/bench_twogen.txt 2>&1
cat ~/bench_twogen.txt | tee -a "$SUM"
log "ALL DONE"
