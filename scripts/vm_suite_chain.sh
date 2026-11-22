#!/usr/bin/env bash
# Run ON the TPU VM. Waits for any running vm_full_suite.sh to finish (by PID,
# never by pattern-matching other waiters), then runs vm_full_suite.sh for each
# REV LABEL pair in turn, so the chip never idles between runs.
# Usage: vm_suite_chain.sh REV1 LABEL1 [REV2 LABEL2 ...]
set -uo pipefail
LOG=~/suite_chain.log
echo "##### chain start $(date -u +%H:%M:%S): $*" >> "$LOG"
while pid=$(pgrep -of "bash .*vm_full_suite.sh "); do
  echo "waiting for vm_full_suite pid $pid $(date -u +%H:%M:%S)" >> "$LOG"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
while [ $# -ge 2 ]; do
  echo "##### run $2 ($1) $(date -u +%H:%M:%S)" >> "$LOG"
  bash ~/vm_full_suite.sh "$1" "$2" >> "$LOG" 2>&1
  shift 2
done
echo "##### chain ALL DONE $(date -u +%H:%M:%S)" >> "$LOG"
