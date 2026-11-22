#!/usr/bin/env bash
# Run ON the TPU VM, CPU only (TORCH_DEVICE_BACKEND_AUTOLOAD=0 keeps torch_tpu
# from grabbing the chip while bazel TPU tests run). Runs Captum's full
# tests/attr suite on upstream main and on each PR branch, recording per-test
# results for diffing. Usage: [MAIN_REF=rev] vm_captum_tests.sh BRANCH...
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0
LOG=~/captum_suite.log
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$LOG"; }
rm -f "$LOG"
[ -d ~/captum ] || git clone -q https://github.com/meta-pytorch/captum.git ~/captum
cd ~/captum
git remote get-url fork >/dev/null 2>&1 || git remote add fork https://github.com/joinalahmed/captum.git
# Captum's development branch is master; its main is years stale.
git fetch -q origin master && git fetch -q fork
# "main" is the merge-base of the first branch with upstream master unless
# MAIN_REF overrides it, so the diff is against the exact code the branch
# was built on.
MAIN_REF=${MAIN_REF:-$(git merge-base origin/master "fork/${1:-master}")}
if [ ! -d ~/env_captum ]; then
  uv venv -q --python 3.12 ~/env_captum
  uv pip install -q --python ~/env_captum/bin/python torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python ~/env_captum/bin/python pytest pytest-xdist parameterized numpy scikit-learn matplotlib
fi
PY=~/env_captum/bin/python
for B in main "$@"; do
  REF=fork/$B; [ "$B" = main ] && REF=$MAIN_REF
  git checkout -q --detach "$REF" && log "captum $B: $(git log --oneline -1)"
  uv pip install -q --python $PY -e . 2>&1 | tail -1
  nice -n 15 $PY -m pytest tests/attr -q -p no:cacheprovider -n 8 -rA --timeout=0 2>/dev/null \
    > ~/captum_${B//\//_}.log 2>&1 || nice -n 15 $PY -m pytest tests/attr -q -p no:cacheprovider -n 8 -rA > ~/captum_${B//\//_}.log 2>&1
  grep -E "^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) " ~/captum_${B//\//_}.log | sed -E 's/ - .*//' | sort -u > ~/captum_${B//\//_}.results
  tail -1 ~/captum_${B//\//_}.log | tee -a "$LOG"
done
log "ALL DONE"
