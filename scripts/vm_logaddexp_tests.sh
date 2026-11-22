#!/usr/bin/env bash
# Run ON the TPU VM after vm_multichip_setup.sh finished its main build.
# Builds the logaddexp branch wheel, installs into ~/env_fix, runs its tests.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
cd ~/torch_tpu
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a ~/logaddexp_summary.log; }
filt() { grep -vE '^(I0000|W0000|E0000|num_|peak_|Loading|WARNING|Successfully|Registered|Device|Initializing)|UserWarning|See https|compilation_cache'; }
V=$(~/env/bin/python -c "import torch, torch_tpu; print(torch.tpu.get_device_properties().name)" 2>/dev/null | tail -1)
V=$(case "$V" in *v5*lite*|*v5e*) echo v5lite;; *) echo v6e;; esac)

if [ ! -d ~/env_fix ]; then
  [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  TORCH_PIN=$(grep -oE "^torch==[0-9.]+\+cpu" requirements/requirements_3_14.txt | head -1 | cut -d= -f3)
  uv venv -q --python 3.14 ~/env_fix
  uv pip install -q --python ~/env_fix/bin/python "torch==${TORCH_PIN:-2.11.0+cpu}" --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python ~/env_fix/bin/python "libtpu>=0.0.51" pytest absl-py parameterized numpy portpicker expecttest hypothesis etils pyzmq attrs cattrs safetensors pyyaml scikit-learn
fi
PY=~/env_fix/bin/python

log "checkout logaddexp"
git fetch -q origin logaddexp && git checkout -q -B logaddexp FETCH_HEAD && git log --oneline -1 | tee -a ~/logaddexp_summary.log
log "build"
nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_resources=cpu=HOST_CPUS*0.8 > ~/build_logaddexp.log 2>&1
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT" >> ~/build_logaddexp.log; tail -1 ~/build_logaddexp.log | tee -a ~/logaddexp_summary.log
if [ "$BUILD_EXIT" -ne 0 ]; then
  grep -E "error:" ~/build_logaddexp.log | tee -a ~/logaddexp_summary.log
  log "ALL DONE (build failed)"; exit 1
fi
WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
uv pip install -q --python ~/env_fix/bin/python --reinstall --no-deps "$WHL" 2>&1 | tail -1
SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true

log "tests"
{
  echo "--- semantics vs CPU"
  PYTHONPATH=. $PY - <<'PYEOF' 2>&1 | filt
import torch, torch_tpu
inf = float("inf")
a = torch.tensor([0.0, 1.0, -1.0, 100.0, -100.0, inf, -inf, inf, float("nan")])
b = torch.tensor([0.0, -1.0, 1.0, -100.0, 100.0, inf, -inf, -inf, 1.0])
cpu = torch.logaddexp(a, b); tpu = torch.logaddexp(a.to("tpu"), b.to("tpu")).cpu()
print("cpu", cpu.tolist()); print("tpu", tpu.tolist())
print("equal_nan allclose:", torch.allclose(cpu, tpu, equal_nan=True))
try: torch.logaddexp(torch.tensor([1, 2], device="tpu"), torch.tensor([1, 2], device="tpu")); print("int: no error (BAD)")
except NotImplementedError as e: print("int ->", e)
PYEOF
  echo "--- pytest"
  PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/ops_unit_test.py -k logaddexp 2>&1 | filt | tail -3
  echo "--- bazel, as CI runs it"
  for T in "//tests:ops_unit_test_tpu-$V" "//tests:errors_test_tpu" "//tests:ops_test_tpu-$V"; do
    bazel test "$T" --test_filter="*logaddexp*" --config=no_rbe --local_test_jobs=1 --cache_test_results=no --test_output=errors 2>&1 | grep -E "^//|PASSED|FAILED|Executed|ERROR|NO STATUS|no test" | tail -4
  done
} > ~/results_logaddexp.txt 2>&1
cat ~/results_logaddexp.txt | tee -a ~/logaddexp_summary.log
log "ALL DONE"
