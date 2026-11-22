#!/usr/bin/env bash
# Run ON the TPU VM. Builds the cpu-view-args-split-compile branch wheel,
# installs it into ~/env_fix, runs the CPU-view check script, the pytest
# split_compile tests and the bazel target. Summary in ~/summary_cpuview.log.
set -uo pipefail
BRANCH=cpu-view-args-split-compile
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCH_TPU_TIER2_COMPILATION_CACHE=0
cd ~/torch_tpu
SUM=~/summary_cpuview.log; BLOG=~/build_cpuview.log; PY=~/env_fix/bin/python
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
rm -f "$SUM"
log "checkout $BRANCH"
git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" FETCH_HEAD && git log --oneline -1 | tee -a "$SUM"
log "build"
nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_resources=cpu=HOST_CPUS*0.8 > "$BLOG" 2>&1
BUILD_EXIT=$?; echo "BUILD_EXIT=$BUILD_EXIT" | tee -a "$SUM"
if [ "$BUILD_EXIT" -ne 0 ]; then grep -E "error:" "$BLOG" | tee -a "$SUM"; log "ALL DONE (build failed)"; exit 1; fi
WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
uv pip install -q --python $PY --reinstall --no-deps "$WHL" 2>&1 | tail -1
SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true
log "tests"
{
  echo "--- check_cpu_view_args.py"
  PYTHONPATH=. $PY ~/checks/check_cpu_view_args.py 2>&1 | filt
  echo "--- pytest split_compile_test"
  PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/compile/split_compile_test.py 2>&1 | filt | tail -3
  echo "--- bazel"
  bazel test //tests/compile:split_compile_test --config=no_rbe --local_test_jobs=1 --cache_test_results=no --test_output=errors 2>&1 | grep -E "^//|PASSED|FAILED|Executed|ERROR|NO STATUS" | tail -4
} 2>&1 | tee -a "$SUM"
log "ALL DONE"
