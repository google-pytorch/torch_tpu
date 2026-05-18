#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Crash-resilient pytest runner.

The TPU compiler aborts the host process via SIGABRT on certain ops. A single
pytest invocation can therefore lose all subsequent tests when one crashes.
This runner:

  1. Collects test nodeids via `pytest --collect-only -q` using the test
     targets + filter args.
  2. Splits the collected set into chunks of N tests.
  3. Invokes pytest once per chunk, passing ONLY the per-chunk nodeids
     (not the original test-target paths — pytest rejects mixed specs).
     Each chunk gets a UPSTREAM_RESULTS_LOG env var pointing at its own
     JSONL file. The plugin appends + flushes one line per test, so even
     a SIGABRT mid-session preserves results for previously-completed tests
     (junit XML is only flushed at session end and is therefore lost).
  4. On chunk crash, sleeps briefly so libtpu can release /dev/vfio, then
     continues with the next chunk.
  5. Aggregates all per-chunk JSONL files and prints final counts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def collect_tests(test_targets: list[str], filter_args: list[str]) -> list[str]:
    """Run pytest --collect-only to get the full nodeid list.

    Pytest's `--collect-only -q` emits nodeids relative to its detected rootdir
    (e.g. `test/test_ops.py::TestX::test_y`). Since we invoke per-chunk pytest
    from a different cwd, we rebase those relative nodeids to absolute paths
    using the first test target's absolute prefix.
    """
    cmd = (
        ["pytest", "--collect-only", "-q", "--no-header"]
        + filter_args
        + test_targets
    )
    print(f"[collect] {shlex.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"collect failed (exit {result.returncode})")
    raw_nodeids = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(("=", "<", " "))
    ]

    # Derive absolute prefix from the first abs test target. Example:
    #   target  = "/tmp/pytorch/test/test_ops.py::TestCommonPRIVATEUSE1"
    #   nodeid  = "test/test_ops.py::TestCommonPRIVATEUSE1::test_X"
    #   prefix  = "/tmp/pytorch/"  → absolute = "/tmp/pytorch/test/test_ops.py::TestCommonPRIVATEUSE1::test_X"
    prefix = ""
    for t in test_targets:
        path_part = t.split("::", 1)[0]
        if path_part.startswith("/"):
            rel_root = raw_nodeids[0].split("/", 1)[0] if raw_nodeids else ""
            # path_part ends with /<rel_root>/... ; strip that suffix to get prefix.
            marker = "/" + rel_root + "/"
            idx = path_part.find(marker)
            if idx >= 0:
                prefix = path_part[: idx + 1]
                break
    if prefix:
        return [
            n if n.startswith("/") else prefix + n for n in raw_nodeids
        ]
    return raw_nodeids


def run_chunk(
    chunk_index: int,
    nodeids: list[str],
    junit_out: Path,
    results_log: Path,
    filter_args: list[str],
) -> int:
    """Run pytest on one chunk; return its exit code."""
    cmd = [
        "pytest",
        "-ra",
        "--tb=short",
        "--timeout=60",
        "--timeout-method=signal",
        f"--junitxml={junit_out}",
        "-o", "junit_family=xunit2",
    ] + filter_args + nodeids
    env = dict(os.environ, UPSTREAM_RESULTS_LOG=str(results_log))
    print(
        f"[chunk {chunk_index}] {len(nodeids)} tests → {results_log.name}",
        flush=True,
    )
    return subprocess.run(cmd, env=env).returncode


def aggregate_results(log_files: list[Path]) -> tuple[Counter[str], float, int]:
    """Sum up outcomes across all per-chunk JSONL files."""
    counts: Counter[str] = Counter()
    total_time = 0.0
    seen_nodeids: set[str] = set()
    for p in log_files:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["nodeid"] in seen_nodeids:
                    continue
                seen_nodeids.add(rec["nodeid"])
                counts[rec["outcome"]] += 1
                total_time += rec.get("duration", 0.0)
    return counts, total_time, len(seen_nodeids)


def merge_junits(chunk_files: list[Path], out_path: Path) -> dict[str, float]:
    """Merge per-chunk junit XMLs into one and return aggregate counts."""
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    root = ET.Element("testsuites")
    for p in chunk_files:
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            tree = ET.parse(p)
        except ET.ParseError:
            continue
        chunk_root = tree.getroot()
        suites = chunk_root.findall("testsuite") or [chunk_root]
        for s in suites:
            totals["tests"] += int(s.get("tests", 0))
            totals["failures"] += int(s.get("failures", 0))
            totals["errors"] += int(s.get("errors", 0))
            totals["skipped"] += int(s.get("skipped", 0))
            totals["time"] += float(s.get("time", 0))
            root.append(s)
    root.set("tests", str(totals["tests"]))
    root.set("failures", str(totals["failures"]))
    root.set("errors", str(totals["errors"]))
    root.set("skipped", str(totals["skipped"]))
    root.set("time", f"{totals['time']:.3f}")
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    passed = (
        totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    )
    return {**totals, "passed": passed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--post-crash-sleep", type=float, default=20.0)
    parser.add_argument("--junit-out", default="upstream_results.xml")
    parser.add_argument("--junit-dir", default="/tmp/upstream_junits")
    parser.add_argument(
        "--test-target",
        action="append",
        required=True,
        dest="test_targets",
        help="Pytest path/nodeid target for collection (repeatable)",
    )
    parser.add_argument(
        "filter_args",
        nargs=argparse.REMAINDER,
        help="Pytest filter args (-k, -p, etc.). Pass after a `--` separator.",
    )
    args = parser.parse_args()

    filter_args = [a for a in args.filter_args if a != "--"]
    junit_dir = Path(args.junit_dir)
    junit_dir.mkdir(parents=True, exist_ok=True)

    nodeids = collect_tests(args.test_targets, filter_args)
    print(f"[collect] {len(nodeids)} tests selected", flush=True)
    if not nodeids:
        print("No tests collected.", file=sys.stderr)
        return 5

    chunks: list[list[str]] = [
        nodeids[i : i + args.chunk_size]
        for i in range(0, len(nodeids), args.chunk_size)
    ]
    log_files: list[Path] = []
    crash_count = 0
    for i, chunk in enumerate(chunks, start=1):
        junit_out = junit_dir / f"chunk_{i:04d}.xml"
        results_log = junit_dir / f"chunk_{i:04d}.jsonl"
        log_files.append(results_log)
        exit_code = run_chunk(i, chunk, junit_out, results_log, filter_args)
        # subprocess.run returns negative numbers for terminating signals
        # (e.g. -6 for SIGABRT) on POSIX. Bash convention is 128+signum.
        if exit_code < 0 or exit_code >= 128:
            crash_count += 1
            print(
                f"[chunk {i}] crashed with exit {exit_code}; "
                f"sleeping {args.post_crash_sleep}s for /dev/vfio release",
                flush=True,
            )
            time.sleep(args.post_crash_sleep)
        elif exit_code not in (0, 1):
            print(f"[chunk {i}] pytest exit {exit_code}", flush=True)

    counts, total_time, total = aggregate_results(log_files)
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)
    print("==== upstream test summary ====")
    print(f"chunks:         {len(chunks)}")
    print(f"chunks crashed: {crash_count}")
    print(f"total recorded: {total}")
    print(f"passed:         {passed}")
    print(f"failed:         {failed}")
    print(f"skipped:        {skipped}")
    print(f"time (s):       {total_time:.1f}")
    print("===============================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
