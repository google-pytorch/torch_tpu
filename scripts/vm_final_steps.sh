#!/usr/bin/env bash
# Run ON the VM after the branch queue drains: rerun trace with its test fix,
# then run the TPU-only errors test for the branches that added cases there.
set -u
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
while pgrep -f "vm_queue_branch.sh|bash .*vm_branch_tests.sh" >/dev/null; do sleep 30; done
bash ~/vm_branch_tests.sh trace test_trace ~/checks/check_trace.py
cd ~/torch_tpu
T=$(bazel query 'attr(name, "^tpu_errors_test(_tpu(-v6e)?)?$", //tests:all)' 2>/dev/null | grep -E "tpu-v6e$|_tpu$|tpu_errors_test$" | head -1)
echo "tpu_errors target: $T" > ~/summary_tpu_errors.log
for pair in "nansum nansum" "smooth_l1_loss smooth_l1"; do
  set -- $pair; B=$1; F=$2
  git fetch -q origin "$B" && git checkout -q -B "$B" FETCH_HEAD
  echo "##### $B $(git log --oneline -1) $(date -u +%H:%M:%S)" >> ~/summary_tpu_errors.log
  bazel test "$T" --test_filter="*$F*" --config=no_rbe --local_test_jobs=1 --cache_test_results=no --test_output=errors 2>&1 | grep -E "^//|PASSED|FAILED|Executed|ERROR|NO STATUS|no test" | tail -4 >> ~/summary_tpu_errors.log
done
echo "##### ALL DONE $(date -u +%H:%M:%S)" >> ~/summary_tpu_errors.log
