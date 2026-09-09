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

"""Example of quantizing and packing in a single operation."""

import os
import sys
from typing import Final, Tuple

LLO_DUMP_TO: Final[str] = "/tmp/llo_dump"
sys.argv.extend(
    [f"--xla_jf_dump_to={LLO_DUMP_TO}", "--xla_jf_dump_llo_text=true"]
)
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_TO} --xla_jf_dump_llo_text=true"
)

XLA_DUMP_TO: Final[str] = "/tmp/xla_dump"
os.environ["XLA_FLAGS"] = f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"

from absl import app  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

# pylint: disable=protected-access
torch._logging.set_logs(aot_graphs=True)


# Copied from qat_linear.py
def quantize(values):
  """Returns signs and a scale as a quantized representation of the values."""
  scale = abs(values).mean()
  signs = torch.sign(values)
  signs[signs == 0.0] = 1.0
  return signs, scale


# Copied from qat_linear.py
def _pack32(values):
  """Packs a vector of 32 quantized values (signs) into a uint32"""
  # Omit size and dtype asserts to avoid compilation errors on TPU

  bits = (values == 1.0).to(torch.uint32)
  bit_patterns = (
      2 ** torch.arange(32, dtype=torch.int64, device=values.device)
  ).to(torch.uint32)
  return torch.sum(bits * bit_patterns, dim=0, dtype=torch.uint32)


# Copied from qat_linear.py
def pack(values):
  return torch.vmap(_pack32)(values)


def quant_and_pack(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  signs, scale = quantize(values)
  packed = pack(signs)
  return packed, scale


@torch.compile(backend="tpu")
def compiled_quant_and_pack(values):
  return quant_and_pack(values)


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

  x = torch.randn(1024, 32, device="tpu")

  packed, scale = compiled_quant_and_pack(x)

  # Force execution
  _ = packed.cpu()
  _ = scale.cpu()

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print("Success.")
  return 0


if __name__ == "__main__":
  app.run(main)
