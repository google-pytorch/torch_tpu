#!/usr/bin/env bash
# Run ON the VM, detached. Waits for the running max-unpool2d build, tests it,
# then builds the merged fix-scalar-buffer-cache-bound branch and tests that.
cd ~/torch_tpu; export PATH=$HOME/bin:$PATH TMPDIR=/mnt/disks/build/tmp
PY=~/env_fix/bin/python
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a ~/rebuild.log; }
build() { nice -n 19 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_cpu_resources=HOST_CPUS*0.6 > "$1" 2>&1; echo "BUILD_EXIT=$?" >> "$1"; tail -1 "$1" >> ~/rebuild.log; }

log "phase1: waiting for max-unpool2d build"
while ! grep -q BUILD_EXIT ~/build2.log; do sleep 30; done
tail -1 ~/build2.log >> ~/rebuild.log
if grep -q "BUILD_EXIT=0" ~/build2.log; then
  log "phase1: testing max-unpool2d"
  ~/vm_test_unpool.sh > ~/results_unpool.txt 2>&1
  grep -E "^#####|passed|failed|error|^(cpu|tpu) " ~/results_unpool.txt >> ~/rebuild.log
else
  log "phase1: BUILD FAILED"; grep -nE "^ERROR|error:" ~/build2.log | head -20 >> ~/rebuild.log
fi

log "phase2: building merged fix-scalar-buffer-cache-bound"
git fetch -q fork fix-scalar-buffer-cache-bound && git checkout -q -B fix-scalar-buffer-cache-bound FETCH_HEAD && git log --oneline -1 >> ~/rebuild.log
build ~/build3.log
if grep -q "BUILD_EXIT=0" ~/build3.log; then
  log "phase2: testing"
  WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1); echo "$WHL" >> ~/rebuild.log
  ~/env_fix/bin/pip install -q --force-reinstall --no-deps "$WHL" 2>&1 | tail -1
  { PYTHONPATH=. $PY -m pytest -q tests/scalar_buffer_cache_test.py -p no:cacheprovider 2>&1 | filt | tail -3; $PY ~/scalar_leak_check.py 2>&1 | filt; } > ~/results_fix2.txt 2>&1
  cat ~/results_fix2.txt >> ~/rebuild.log
else
  log "phase2: BUILD FAILED"; grep -nE "^ERROR|error:" ~/build3.log | head -20 >> ~/rebuild.log
fi
log "REBUILD ALL DONE"
