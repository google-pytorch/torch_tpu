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

import collections.abc
import os
import pathlib
import subprocess
import sys
import time
from absl import app
from absl import flags

_BENCHMARK_CONFIGS = flags.DEFINE_list(
    "benchmark_configs",
    default=["1:1024", "16:1024", "16:2048"],
    help=(
        "Comma-separated list of benchmark configurations, each in the format"
        " 'batch_size:prefill_seq_len'."
    ),
)


def main(argv: collections.abc.Sequence[str]) -> None:
  forward_args = []
  for arg in argv[1:]:
    if not arg.startswith(("--batch_size=", "--prefill_seq_len=")):
      forward_args.append(arg)

  configs = []
  for config_str in _BENCHMARK_CONFIGS.value:
    try:
      bs_str, seq_len_str = config_str.split(":")
      configs.append((int(bs_str), int(seq_len_str)))
    except ValueError:
      print(
          f"Invalid benchmark configuration format: '{config_str}'. "
          "Expected 'batch_size:prefill_seq_len'.",
          file=sys.stderr,
      )
      sys.exit(1)

  test_srcdir = os.environ.get("TEST_SRCDIR")
  if not test_srcdir:
    print("TEST_SRCDIR not set, cannot find child binary", file=sys.stderr)
    sys.exit(1)

  google3_dir = pathlib.Path(test_srcdir) / "google3"
  child_binary = (
      google3_dir
      / "third_party/py/torch_tpu/examples/qwen3/qwen3_dynamic_kv_cache"
  )

  success = True
  for i, (batch_size, prefill_seq_len) in enumerate(configs):
    print("\n==================================================")
    print(
        f"Spawning subprocess for batch_size={batch_size},"
        f" prefill_seq_len={prefill_seq_len}"
    )
    print("==================================================")

    cmd = (
        [str(child_binary)]
        + forward_args
        + [
            f"--batch_size={batch_size}",
            f"--prefill_seq_len={prefill_seq_len}",
        ]
    )
    print(f"Running command: {' '.join(cmd)}")

    try:
      # Use subprocess.run and let stdout/stderr flow to console
      subprocess.run(cmd, check=True)
    except FileNotFoundError:
      print(f"Child binary not found at: {child_binary}", file=sys.stderr)
      sys.exit(1)
    except subprocess.CalledProcessError:
      print(
          f"Subprocess failed for config {batch_size}:{prefill_seq_len}",
          file=sys.stderr,
      )
      success = False

    if i < len(configs) - 1:
      print("Sleeping for 10 seconds to allow TPU resource release...")
      time.sleep(10)

  if not success:
    sys.exit("One or more benchmark configurations failed.")


def _parse_flags(argv):
  return flags.FLAGS(argv, known_only=True)


if __name__ == "__main__":
  app.run(main, flags_parser=_parse_flags)
