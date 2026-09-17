#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Partitions Bazel test targets into balanced shards using LPT scheduling."""

import argparse
from collections.abc import Iterable, Mapping, Sequence
import pathlib
import re
import sys


def _iter_rule_blocks(text: str) -> Iterable[str]:
  """Yields top-level Starlark rule blocks balanced by parentheses."""
  block: list[str] = []
  depth = 0
  for line in text.splitlines():
    clean = re.sub(r'(["\'])(?:\\.|(?!\1).)*\1|#.*', "", line)
    depth += clean.count("(") - clean.count(")")
    if depth > 0 or block:
      block.append(line)
      if depth <= 0:
        yield "\n".join(block)
        block = []
        depth = 0


def load_build_metadata(
    targets: Iterable[str],
    workspace_root: pathlib.Path = pathlib.Path("."),
) -> tuple[dict[str, int], dict[str, set[str]]]:
  """Reads rule weights and tag sets from package BUILD files."""
  pkgs = {t.split(":", 1)[0].lstrip("@/") for t in targets if ":" in t}
  weights: dict[str, int] = {}
  rule_tags: dict[str, set[str]] = {}
  for pkg in pkgs:
    path = next(
        (
            root / pkg / fname
            for root in (workspace_root, workspace_root.parent)
            for fname in ("BUILD.bazel", "BUILD")
            if (root / pkg / fname).is_file()
        ),
        None,
    )
    if not path:
      continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for block in _iter_rule_blocks(text):
      name_m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block)
      if not name_m:
        continue
      rule_name = name_m.group(1)
      tags_m = re.search(r"tags\s*=\s*\[([^\]]*)\]", block, re.DOTALL)
      tags = (
          set(re.findall(r'["\']([^"\']+)["\']', tags_m.group(1)))
          if tags_m
          else set()
      )
      rule_tags.setdefault(rule_name, set()).update(tags)

      sc_m = re.search(r"shard_count\s*=\s*(\d+)", block)
      shard_count = int(sc_m.group(1)) if sc_m else 1
      is_long = bool(
          re.search(r'timeout\s*=\s*["\'](long|eternal)["\']', block)
          or re.search(r'size\s*=\s*["\'](large|enormous)["\']', block)
      )
      is_multichip = bool(
          re.search(r"(requires-[a-z0-9-]+:(8|4)|exclusive)", block)
      )
      weight = 6
      if is_long or shard_count >= 10:
        weight = max(120 if is_long else 6, shard_count * 4)
      elif is_multichip:
        weight = 36
      weights[rule_name] = max(weights.get(rule_name, 0), weight)
  return weights, rule_tags


def is_target_compatible(
    target: str,
    bazel_config: str = "",
    rule_tags: Mapping[str, set[str]] | None = None,
) -> bool:
  """Returns True if target is a runnable test compatible with bazel_config."""
  name = target.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
  if "test" not in name or name.endswith(
      ("_build_test", "_test_utils", "_test_data", "_testing")
  ):
    return False
  base = re.sub(r"_tpu-[a-z0-9_]+$", "", name)
  tags = set()
  if rule_tags:
    tags = rule_tags.get(name, set()) | rule_tags.get(base, set())
  if "notest" in tags or "nobuild" in tags:
    return False
  if "presubmit" in bazel_config and "nopresubmit" in tags:
    return False

  for gen, other_gens in (
      ("tpu_v5", ("_cpu", "_gpu", "_cuda", "_tpu-v6", "_tpu-v7")),
      ("tpu_v6", ("_cpu", "_gpu", "_cuda", "_tpu-v5", "_tpu-v7")),
      ("tpu_v7", ("_cpu", "_gpu", "_cuda", "_tpu-v5", "_tpu-v6")),
  ):
    if gen in bazel_config:
      num = gen[-1]
      if any(x in name for x in other_gens) or f"fails-on-tpu-v{num}" in tags:
        return False
      return (
          f"_tpu-v{num}" in name
          or f"presubmit-v{num}" in tags
          or "requires-tpu" in tags
      )
  if "cpu" in bazel_config:
    if any(x in name for x in ("_tpu-", "_gpu", "_cuda")):
      return False
    if "requires-tpu" in tags or "requires-gpu" in tags:
      return False
  return True


def estimate_target_weight(
    target: str,
    build_weights: Mapping[str, int] | None = None,
) -> int:
  """Returns estimated wall-clock weight in seconds on an 8-chip TPU runner."""
  name = target.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
  base = re.sub(r"_tpu-[a-z0-9_]+$", "", name)
  weight = (
      max(build_weights.get(name, 0), build_weights.get(base, 0))
      if build_weights
      else 0
  )
  if (
      "distributed" in target
      or "collective" in name
      or name.endswith(("_tpu-v5lite_8", "_8tpu", "_4tpu"))
  ):
    weight = max(weight, 36)
  return weight if weight > 0 else 6


def partition_targets(
    targets: Iterable[str],
    shard_index: int,
    shard_count: int,
    workspace_root: pathlib.Path = pathlib.Path("."),
    bazel_config: str = "",
) -> list[str]:
  """Partitions targets into `shard_count` balanced buckets via LPT."""
  if shard_count < 1 or not (0 <= shard_index < shard_count):
    raise ValueError(
        f"Invalid shard arguments: shard_index={shard_index},"
        f" shard_count={shard_count}"
    )

  raw_unique = sorted({t.strip() for t in targets if t.strip()})
  build_weights, rule_tags = load_build_metadata(raw_unique, workspace_root)
  unique = [
      t for t in raw_unique if is_target_compatible(t, bazel_config, rule_tags)
  ]
  if shard_count == 1:
    return unique

  buckets: list[list[str]] = [[] for _ in range(shard_count)]
  loads: list[int] = [0] * shard_count
  for target in sorted(
      unique, key=lambda t: (-estimate_target_weight(t, build_weights), t)
  ):
    idx = min(range(shard_count), key=lambda i: (loads[i], len(buckets[i]), i))
    buckets[idx].append(target)
    loads[idx] += estimate_target_weight(target, build_weights)

  return sorted(buckets[shard_index])


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shard-index", type=int, required=True)
  parser.add_argument("--shard-count", type=int, required=True)
  parser.add_argument("--bazel-config", type=str, default="")
  parser.add_argument("--input-file", type=pathlib.Path, default=None)
  parser.add_argument("--output-file", type=pathlib.Path, default=None)
  args = parser.parse_args(argv)

  raw = (
      args.input_file.read_text(encoding="utf-8")
      if args.input_file
      else sys.stdin.read()
  )
  selected = partition_targets(
      raw.splitlines(),
      args.shard_index,
      args.shard_count,
      bazel_config=args.bazel_config,
  )
  output = "\n".join(selected) + ("\n" if selected else "")

  if args.output_file:
    args.output_file.write_text(output, encoding="utf-8")
  else:
    sys.stdout.write(output)
  return 0


if __name__ == "__main__":
  sys.exit(main())
