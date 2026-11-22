#!/usr/bin/env bash
# Run ON the TPU VM. Runs every single-chip TPU test target in the tree at a
# given revision, the way presubmit does (ci_tpu_v6 tag set minus multi-chip
# and RBE), and records one PASSED/FAILED/... line per target so two runs can
# be diffed. Usage: vm_full_suite.sh REV LABEL
#   REV    - commit, tag or origin/BRANCH to test
#   LABEL  - name for ~/fullsuite_LABEL.{log,results}
set -uo pipefail
REV=${1:?rev}; LABEL=${2:?label}
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
cd ~/torch_tpu
LOG=~/fullsuite_$LABEL.log; RES=~/fullsuite_$LABEL.results
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$LOG"; }
rm -f "$LOG" "$RES"
case "$REV" in origin/*) git fetch -q origin "${REV#origin/}" && REV=FETCH_HEAD;; esac
# -f: a previous run may have left generated files (docs/api_reference.*)
# modified; a stray edit would silently taint every later run.
git checkout -q -f --detach "$REV" && log "checkout $LABEL: $(git log --oneline -1)" \
  || { echo "CHECKOUT FAILED" | tee -a "$LOG"; log "ALL DONE"; exit 1; }
git clean -fdq -e bazel-\* && git status --short | tee -a "$LOG"

# Single-chip TPU tests only: the ci_tpu_v6 (or ci_tpu_v5 on a v5e host)
# positive set without the :4/:8 multi-chip tags, v5p and v7x. 'manual'
# targets are excluded by //... itself. On a multi-chip host the tests run
# one per chip through upstream's run_under wrapper.
ACCEL=$(curl -sf -H Metadata-Flavor:Google \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null)
CHIPS=${CHIPS:-$(ls /dev/accel* 2>/dev/null | wc -l)}
[ "$CHIPS" -ge 1 ] 2>/dev/null || CHIPS=1
case "$ACCEL" in
  v5litepod*) GEN=v5; GEN_TAGS="requires-tpu-v5lite,-requires-tpu-v6e,-fails-on-tpu-v5";;
  *)          GEN=v6; GEN_TAGS="requires-tpu-v6e,requires-tpu-v5lite,-fails-on-tpu-v6";;
esac
TAGS="requires-tpu,$GEN_TAGS,-requires-tpu-v5lite:4,-requires-tpu-v5lite:8,-requires-tpu-v5p,-requires-tpu-v5p:4,-requires-tpu-v7x,-requires-tpu-v7x:4,-notest,-nobuild,-requires-newest-deps"
RUN_UNDER=()
if [ "$CHIPS" -gt 1 ]; then
  RUN_UNDER=(--run_under="$PWD/ci/tools/parallel_accelerator_execute.sh"
    --test_env=TORCH_TPU_ACCELERATOR_COUNT=$CHIPS)
fi
log "host: accelerator=${ACCEL:-unknown} gen=$GEN chips=$CHIPS"

# //tests/... holds the Python op, error, compile and API suites. Everything
# else under //... (examples, experimental kernels) does not fit the 100 GB
# boot disk and is unrelated.
TARGETS="//tests/..."
log "build test targets"
bazel build --config=tpu_envvars --config=no_rbe --local_resources=cpu=HOST_CPUS*0.8 \
  --experimental_disk_cache_gc_max_size=8G \
  --build_tag_filters="$TAGS" -- $TARGETS > ~/fullsuite_${LABEL}_build.log 2>&1
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT" | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
[ "$BUILD_EXIT" -eq 0 ] || { log "ALL DONE (build failed)"; exit 1; }

# The C++ unit tests under //csrc/ops and //csrc/common are not run: they
# build XLA again under a different configuration transition (~30 GB), which
# does not fit next to the wheel build on a 100 GB boot disk.
# Every py_test gets a runfiles tree of ~100 MB of symlinks (one 4 KB block
# each), so ~500 test targets need ~50 GB that the disk does not have. The
# tests run in chunks and the runfiles trees are deleted after each chunk;
# Bazel recreates the ones the next chunk needs.
log "test"
OUT=$(bazel info output_path 2>/dev/null)
# //tests/torch (device-generic upstream PyTorch tests) refuse to run below
# torch 2.14, and the run_upstream_example_* targets need the torchtpu-shared
# GCS bucket mounted; both fail identically on every revision here, so they
# are left out (124 targets, ~25 min per run).
mapfile -t ALL < <(bazel query "tests($TARGETS)" 2>/dev/null \
  | grep -v -E '^//tests/torch[:/]|run_upstream_example')
log "test targets under $TARGETS: ${#ALL[@]}"
: > ~/fullsuite_${LABEL}_test.log
TEST_EXIT=0; CHUNK=${CHUNK:-40}
for ((i = 0; i < ${#ALL[@]}; i += CHUNK)); do
  bazel test --config=tpu_envvars --config=no_rbe --keep_going \
    --local_test_jobs=$CHIPS "${RUN_UNDER[@]}" \
    --experimental_disk_cache_gc_max_size=8G \
    --build_tag_filters="$TAGS" --test_tag_filters="$TAGS" \
    --test_timeout=600,1200,2400,4800 \
    --cache_test_results=no --test_output=errors --test_summary=short \
    -- "${ALL[@]:i:CHUNK}" >> ~/fullsuite_${LABEL}_test.log 2>&1
  rc=$?; [ $rc -eq 0 ] || [ $rc -eq 4 ] || TEST_EXIT=$rc
  log "chunk $i exit=$rc $(df -h / | tail -1 | awk '{print $4" free"}')"
  sudo rm -rf "$OUT"/k8-opt/bin/tests
done
echo "TEST_EXIT=$TEST_EXIT" | tee -a "$LOG"

# One line per target: "//pkg:target STATUS". The short summary lists every
# target with its status at the end of each chunk (terse lists only
# failures, which makes two runs undiffable).
grep -E "^//[^ ]+ +(PASSED|FAILED|TIMEOUT|FLAKY|NO STATUS|\(cached\) PASSED)" ~/fullsuite_${LABEL}_test.log \
  | sed -E 's/ +/ /g; s/ in [0-9.]+s.*//; s/ \(cached\)//' | sort -u > "$RES"
echo "targets: $(wc -l < "$RES")  passed: $(grep -c ' PASSED' "$RES")  failed: $(grep -c ' FAILED' "$RES")  other: $(grep -vc ' PASSED\| FAILED' "$RES")" | tee -a "$LOG"
grep -v ' PASSED' "$RES" | tee -a "$LOG"
log "ALL DONE"
