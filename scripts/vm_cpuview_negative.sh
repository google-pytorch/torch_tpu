#!/usr/bin/env bash
# Run ON the TPU VM. Builds two reference points for PR 6106 and runs the
# CPU-view check script on each: the previous contiguous_format revision
# (c32405eea, expect apullin's f1-f4 to FAIL) and the branch base main
# (f7172418b, expect the original repro cases to FAIL and f1-f4 to pass).
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
export TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCH_TPU_TIER2_COMPILATION_CACHE=0
cd ~/torch_tpu
SUM=~/summary_cpuview_negative.log; PY=~/env_fix/bin/python
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing|based on|TFRT|Built on|C API)|UserWarning|See https|compilation_cache|get_attr_node|expected = f'; }
rm -f "$SUM"
git fetch -q origin cpu-view-args-split-compile
for REV in "c32405eea contiguous_format-revision" "f7172418b base-main"; do
  set -- $REV; SHA=$1; LABEL=$2
  log "checkout $LABEL ($SHA)"
  git checkout -q --detach "$SHA" && git log --oneline -1 | tee -a "$SUM"
  log "build"
  nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_resources=cpu=HOST_CPUS*0.8 > ~/build_$LABEL.log 2>&1
  BUILD_EXIT=$?; echo "BUILD_EXIT=$BUILD_EXIT" | tee -a "$SUM"
  if [ "$BUILD_EXIT" -ne 0 ]; then grep -E "error:" ~/build_$LABEL.log | head | tee -a "$SUM"; continue; fi
  WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
  uv pip install -q --python $PY --reinstall --no-deps "$WHL" 2>&1 | tail -1
  log "check_cpu_view_args.py on $LABEL"
  PYTHONPATH=. $PY ~/checks/check_cpu_view_args.py 2>&1 | filt | tee -a "$SUM"
done
log "ALL DONE"
