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

"""Example of a simple pointwise addition and ReLU fusion on XLA (TPU)."""

import os
import sys
from typing import Final

# Inject flags and env var to libtpu. It sometimes needs flags, sometimes envvar.
LLO_DUMP_TO: Final[str] = "/tmp/llo_dump"
flags = [
    f"--xla_jf_dump_to={LLO_DUMP_TO}",
    "--xla_jf_dump_llo_text=true",
]
sys.argv.extend(flags)
os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)

XLA_DUMP_TO: Final[str] = "/tmp/xla_dump"
os.environ["XLA_FLAGS"] = f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"

from absl import app  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top


@torch.compile(backend="tpu")
def fwd(x, y):
  return torch.nn.functional.relu(torch.add(x, y))


def dump_dir(directory):
  entries = os.listdir(directory) if os.path.exists(directory) else []
  print(f"\n=== Files in {directory} (Total: {len(entries)}) ===")

  # Print all files first
  for f in sorted(entries):
    print(f)

  # Print contents of all files
  for f in sorted(entries):
    full_path = os.path.join(directory, f)
    if os.path.isfile(full_path):
      print(f"\n=== File: {f} ===")
      with open(full_path, "r") as file:
        print(file.read())


def main(argv):
  del argv

  x = torch.randn(1024).to("tpu")
  y = torch.randn(1024).to("tpu")

  res = fwd(x, y)
  _ = res.cpu()

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print("Success.")
  return 0


if __name__ == "__main__":
  app.run(main)
