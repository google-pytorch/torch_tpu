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

"""Worker binary for the `split_gpu_goldens` Starlark build rule.

Decompresses monolithic .gz archives, merges architecture overrides (such as
Hopper), and writes individual <test_case_name>.plist files into the output
directory declared by Starlark.
"""

import argparse
import collections
from collections.abc import Sequence
import gzip
import pathlib
import plistlib
import typing

# (test_case_name, variant_name, dtype_key)
_GoldenKey: typing.TypeAlias = tuple[str, str, str]

# A single test case's golden data tree:
# variant_name -> dtype_key -> list of samples
_OpGoldenTree: typing.TypeAlias = dict[str, dict[str, list[typing.Any]]]


def _load_golden_archives(
    files: Sequence[pathlib.Path],
) -> tuple[dict[str, _OpGoldenTree], dict[_GoldenKey, str]]:
  """Reads golden archives, accumulating samples and recording source files."""
  tree: collections.defaultdict[str, _OpGoldenTree] = collections.defaultdict(
      lambda: collections.defaultdict(dict)
  )
  sources: dict[_GoldenKey, str] = {}
  for gz_path in sorted(files):
    with gzip.open(gz_path, "rb") as f:
      data = plistlib.load(
          typing.cast(typing.IO[bytes], f), fmt=plistlib.FMT_BINARY
      )
    for test_case_name, variant_to_dt in data.items():
      for variant_name, dt_to_samples in variant_to_dt.items():
        for dt_key, samples in dt_to_samples.items():
          sources[(test_case_name, variant_name, dt_key)] = gz_path.name
          existing = tree[test_case_name][variant_name].setdefault(dt_key, [])
          existing.extend(samples)
  return tree, sources


def _merge_goldens(
    base_tree: dict[str, _OpGoldenTree],
    base_sources: dict[_GoldenKey, str],
    hopper_tree: dict[str, _OpGoldenTree],
    hopper_sources: dict[_GoldenKey, str],
) -> dict[str, _OpGoldenTree]:
  """Merges base and Hopper trees, rejecting cross-architecture key overlap."""
  overlap = set(base_sources) & set(hopper_sources)
  if overlap:
    key = sorted(overlap)[0]
    test_case_name, variant_name, dt_key = key
    raise ValueError(
        f"{hopper_sources[key]} and {base_sources[key]} both hold goldens for"
        f" {test_case_name} (variant {variant_name}, dtype {dt_key}), so it is"
        " ambiguous which one applies. Each test method must record its"
        " goldens on exactly one GPU architecture; check whether"
        f" {test_case_name} is tagged with"
        " @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)."
    )
  for test_case_name, variant_to_dt in hopper_tree.items():
    for variant_name, dt_to_samples in variant_to_dt.items():
      for dt_key, samples in dt_to_samples.items():
        base_tree[test_case_name][variant_name].setdefault(dt_key, []).extend(
            samples
        )
  return base_tree


def _write_per_test_case_goldens(
    all_ops: dict[str, _OpGoldenTree],
    output_dir: pathlib.Path,
) -> int:
  """Writes individual <test_case_name>.plist files to output_dir."""
  output_dir.mkdir(parents=True, exist_ok=True)
  for test_case_name, op_tree in all_ops.items():
    out_file = output_dir / f"{test_case_name}.plist"
    with open(out_file, "wb") as f:
      plistlib.dump(op_tree, f, fmt=plistlib.FMT_BINARY)
  return len(all_ops)


def split_goldens(
    base_input_files: Sequence[pathlib.Path | str],
    output_dir: pathlib.Path | str,
    hopper_input_files: Sequence[pathlib.Path | str] = (),
) -> int:
  """Reads input golden .gz files and writes per-test-case .plist files.

  Args:
    base_input_files: List of paths to base architecture .gz golden files.
    output_dir: Target directory where `<test_case_name>.plist` files are
      written.
    hopper_input_files: Optional list of paths to Hopper architecture override
      .gz golden files.

  Returns:
    The total number of test case plist files written.
  """
  base_paths = [
      pathlib.Path(f) for f in base_input_files if pathlib.Path(f).is_file()
  ]
  hopper_paths = [
      pathlib.Path(f) for f in hopper_input_files if pathlib.Path(f).is_file()
  ]

  base_tree, base_sources = _load_golden_archives(base_paths)
  hopper_tree, hopper_sources = _load_golden_archives(hopper_paths)
  all_ops = _merge_goldens(base_tree, base_sources, hopper_tree, hopper_sources)
  return _write_per_test_case_goldens(all_ops, pathlib.Path(output_dir))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--output_dir",
      required=True,
      type=pathlib.Path,
      help="Directory to write per-test-case .plist files.",
  )
  parser.add_argument(
      "--base_input_files",
      nargs="+",
      required=True,
      type=pathlib.Path,
      help="Paths to base architecture golden .gz files.",
  )
  parser.add_argument(
      "--hopper_input_files",
      nargs="*",
      default=[],
      type=pathlib.Path,
      help="Paths to Hopper architecture override golden .gz files.",
  )
  args = parser.parse_args()
  count = split_goldens(
      base_input_files=args.base_input_files,
      output_dir=args.output_dir,
      hopper_input_files=args.hopper_input_files,
  )
  print(f"Wrote {count} per-test-case golden files to {args.output_dir}.")


if __name__ == "__main__":
  main()
