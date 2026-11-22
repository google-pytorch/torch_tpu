#!/usr/bin/env bash
# Run ON the TPU VM after vm_multichip_setup.sh finished its main build.
# Usage: vm_branch_tests.sh BRANCH FILTER [PYCHECK]
#   BRANCH  - fork branch to fetch and build
#   FILTER  - bazel --test_filter glob and pytest -k expression, e.g. "logaddexp2"
#   PYCHECK - optional python file to run against the installed wheel first
# Builds the branch wheel, installs into ~/env_fix, runs its tests.
# Writes ~/results_BRANCH.txt and ~/summary_BRANCH.log (ends with "ALL DONE").
set -uo pipefail
BRANCH=${1:?branch}; FILTER=${2:?filter}; PYCHECK=${3:-}
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
cd ~/torch_tpu
SUM=~/summary_$BRANCH.log; RES=~/results_$BRANCH.txt; BLOG=~/build_$BRANCH.log
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$SUM"; }
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

rm -f "$SUM" "$RES"
log "checkout $BRANCH"
git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" FETCH_HEAD && git log --oneline -1 | tee -a "$SUM"
log "build"
nice -n 10 bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe --local_resources=cpu=HOST_CPUS*0.8 > "$BLOG" 2>&1
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT" >> "$BLOG"; tail -1 "$BLOG" | tee -a "$SUM"
if [ "$BUILD_EXIT" -ne 0 ]; then
  grep -E "error:" "$BLOG" | tee -a "$SUM"
  log "ALL DONE (build failed)"; exit 1
fi
WHL=$(ls -t bazel-bin/ci/wheel/torch_tpu-*.whl | head -1)
uv pip install -q --python ~/env_fix/bin/python --reinstall --no-deps "$WHL" 2>&1 | tail -1
SP=$($PY -c "import torch_tpu, os; print(os.path.dirname(torch_tpu.__file__))" 2>/dev/null | tail -1)
cp src/torch_tpu/_internal/utils/test_utils.py src/torch_tpu/_internal/utils/test_fixtures.py "$SP/_internal/utils/" 2>/dev/null || true

log "tests"
{
  if [ -n "$PYCHECK" ] && [ -f "$PYCHECK" ]; then
    echo "--- semantics vs CPU ($PYCHECK)"
    PYTHONPATH=. $PY "$PYCHECK" 2>&1 | filt
  fi
  echo "--- pytest"
  PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/ops_unit_test.py -k "$FILTER" 2>&1 | filt | tail -3
  echo "--- bazel, as CI runs it"
  for T in "//tests:ops_unit_test_tpu-$V" "//tests:errors_test_tpu" "//tests:ops_test_tpu-$V"; do
    bazel test "$T" --test_filter="*$FILTER*" --config=no_rbe --local_test_jobs=1 --cache_test_results=no --test_output=errors 2>&1 | grep -E "^//|PASSED|FAILED|Executed|ERROR|NO STATUS|no test" | tail -4
  done
} > "$RES" 2>&1
cat "$RES" | tee -a "$SUM"
log "ALL DONE"
