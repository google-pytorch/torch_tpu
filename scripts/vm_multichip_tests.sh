#!/usr/bin/env bash
# Run ON the multi-chip TPU VM after vm_multichip_setup.sh and the main build.
# For each fork branch: check it out, incremental bazel build of the wheel,
# install into ~/env_fix, run that branch's tests. Results in ~/results_<branch>.txt
# and a one-line summary per branch in ~/multichip_summary.log.
#   usage: vm_multichip_tests.sh [branch ...]   (default: all five + the 3 PR branches)
set -uo pipefail
export PATH=$HOME/bin:$PATH TMPDIR=/mnt/disks/build/tmp
cd ~/torch_tpu
GEN=$(~/env/bin/python -c "import torch_tpu, torch; print(torch.tpu.get_device_properties().name if hasattr(torch,'tpu') else '?')" 2>/dev/null | tail -1)
CHIPS=$(~/env/bin/python -c "import torch, torch_tpu; print(torch.accelerator.device_count())" 2>/dev/null | tail -1)
V=$(case "$GEN" in *v5*lite*|*v5e*) echo v5lite;; *v6e*) echo v6e;; *) echo v6e;; esac)
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a ~/multichip_summary.log; }
log "device: $GEN x$CHIPS, test variant: tpu-$V"

# env_fix: the wheel is built for the hermetic default Python (3.14, MODULE.bazel) and only imports
# with the torch pinned in requirements/requirements_3_14.txt; uv supplies that interpreter.
export PATH=$HOME/.local/bin:$PATH
if [ ! -d ~/env_fix ]; then
  [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  TORCH_PIN=$(grep -oE "^torch==[0-9.]+\+cpu" requirements/requirements_3_14.txt | head -1 | cut -d= -f3)
  uv venv -q --python 3.14 ~/env_fix
  uv pip install -q --python ~/env_fix/bin/python "torch==${TORCH_PIN:-2.11.0+cpu}" --index-url https://download.pytorch.org/whl/cpu
  # The wheel is installed --no-deps, so libtpu (the PjRT plugin) must be installed explicitly.
  uv pip install -q --python ~/env_fix/bin/python "libtpu>=0.0.50" "transformers>=5" accelerate captum pytest scikit-learn absl-py parameterized numpy portpicker expecttest hypothesis etils pyzmq attrs cattrs
fi
PY=~/env_fix/bin/python

build_and_install() {
  nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_cpu_resources=HOST_CPUS*0.8 > ~/build_$1.log 2>&1
  local rc=$?; echo "BUILD_EXIT=$rc" >> ~/build_$1.log
  [ $rc -eq 0 ] || { log "$1: BUILD FAILED"; grep -nE "^ERROR|error:" ~/build_$1.log | head -10 >> ~/multichip_summary.log; return 1; }
  WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
  uv pip install -q --python ~/env_fix/bin/python --reinstall --no-deps "$WHL" 2>&1 | tail -1
  # The wheel omits the test-only helpers the in-repo tests import; copy them from the source tree.
  SP=$(~/env_fix/bin/python -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
  cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true
}
pyt() { PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider "$@" 2>&1 | filt | tail -4; }
bzt() { bazel test "$@" --config=no_rbe --local_test_jobs=1 --cache_test_results=no --test_output=errors 2>&1 | grep -E "^//|PASSED|FAILED|Executed|ERROR" | tail -6; }

run_branch() {
  local b=$1; local out=~/results_$b.txt
  log "$b: checkout"
  git fetch -q origin "$b" && git checkout -q -B "$b" FETCH_HEAD || { log "$b: checkout failed"; return; }
  git log --oneline -1 | tee -a ~/multichip_summary.log
  log "$b: build"
  build_and_install "$b" || return
  log "$b: tests"
  {
    echo "device $GEN x$CHIPS"
    case $b in
      fix-scalar-buffer-cache-bound)
        pyt tests/scalar_buffer_cache_test.py
        $PY ~/scalar_leak_check.py 2>&1 | filt
        echo "--- multi-device: one torchrun worker per chip"
        ( eval $($PY -m torch.tpu.distributed 2>/dev/null | sed "s/^/export /"); PYTHONPATH=. $PY -m torch.distributed.run --nproc_per_node=$WORLD_SIZE ~/scalar_leak_check.py 2>&1 | grep "^\[rank" | sort ) ;;
      max-unpool2d)
        pyt tests/ops_unit_test.py -k "unpool"
        bzt //tests:ops_test_tpu-$V --test_filter="TestOps.test_max_unpool" ;;
      cpu-view-args-split-compile)
        pyt tests/compile/split_compile_test.py -k "cpu_view_argument"
        # tests/compile has no run_on_accelerators variants; the target carries its plain name.
        bzt //tests/compile:split_compile_test --test_filter="*test_cpu_view_argument_to_tpu_partition*" ;;
      missing-ops-batch)
        pyt tests/ops_unit_test.py -k "piecewise or smooth_l1 or huber or logaddexp or nansum or std_mean or trace or addbmm"
        bzt //tests:ops_test_tpu-$V --test_filter="TestOps.test_logaddexp TestOps.test_nansum TestOps.test_std_mean TestOps.test_trace TestOps.test_addbmm TestOps.test_smooth_l1_loss TestOps.test_huber_loss" ;;
      dynamism-denylist-triage)
        F="TestOps.test_eq TestOps.test_maximum TestOps.test_ne TestOps.test_neg TestOps.test_pow TestOps.test_remainder TestOps.test_round TestOps.test_sign TestOps.test_sinh TestOps.test_sub TestOps.test_trunc"
        bzt //tests:ops_test_dynamic_tpu-$V --test_filter="$F"
        bzt //tests:ops_test_dynamic_compiled_tpu-$V --test_filter="$F" ;;
      loader-import-error-context)
        # The loader test inspects the package layout, so run it the way CI does (bazel, repo tree), not against the wheel.
        bzt //tests:_versioned_so_loader_test ;;
      docs-memory-apis)
        $PY - <<'EOF' 2>&1 | filt
import torch, torch_tpu
x = torch.ones(1024, 1024, device="tpu"); torch.accelerator.synchronize()
for f in ["memory_allocated", "max_memory_allocated", "memory_reserved", "get_memory_info"]:
    print(f, getattr(torch.accelerator, f)())
print("hbm_bytes_per_device", torch.tpu.get_device_properties().hbm_bytes_per_device)
EOF
        ;;
      explainability-example)
        uv pip install -q --python ~/env_fix/bin/python captum "transformers>=5" accelerate 2>&1 | tail -1
        $PY examples/explainability/captum_attribution.py 2>&1 | filt | tail -8
        TORCH_TPU_DEFER_AND_FUSE=1 $PY examples/explainability/captum_attribution.py 2>&1 | filt | tail -8
        $PY examples/explainability/captum_attribution.py --compile 2>&1 | filt | tail -8 ;;
    esac
  } > "$out" 2>&1
  grep -E "passed|failed|error|PASSED|FAILED|Executed|warm|rel err|peak|live alloc|memory_|hbm_" "$out" | head -20 | tee -a ~/multichip_summary.log
}

BRANCHES=${*:-"fix-scalar-buffer-cache-bound max-unpool2d cpu-view-args-split-compile missing-ops-batch dynamism-denylist-triage loader-import-error-context docs-memory-apis explainability-example"}
for b in $BRANCHES; do run_branch "$b"; done
log "ALL DONE"
