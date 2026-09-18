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

"""Example of using Pallas JAX op for popcount in quantized sum."""

import os
import random
import sys
from typing import Callable, Final

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

# pylint: disable=protected-access
torch._logging.set_logs(aot_graphs=True)


# Copied from qat_linear.py
def _pack32(values):
  """Packs a vector of 32 quantized values (signs) into a uint32"""
  assert values.size() == (32,)
  assert values.dtype == torch.float32
  assert torch.all((values == 1.0) | (values == -1.0))

  # Translate values from {-1, 1} to {0, 1}
  bits = (values == 1.0).to(torch.uint32)

  # Create all bit patterns from 0000, 0001, 0010, etc.
  bit_patterns = (
      2 ** torch.arange(32, dtype=torch.int64, device=values.device)
  ).to(torch.uint32)

  # Summing the bit patterns is a bitwise_or reduction on a vector.
  return torch.sum(bits * bit_patterns, dim=0, dtype=torch.uint32)


def create_popcount_op() -> Callable:
  import jax

  def jax_popcount(packed_vector: jax.Array) -> jax.Array:
    """Returns the number of active bits in an Array."""
    return jax.lax.population_count(packed_vector)

  # jax_op takes name and fn as positional-only arguments.
  return torch.tpu.pallas.jax_op(
      "test_example::popcount",
      jax_popcount,
  )


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

  popcount_op = create_popcount_op()

  def quantized_sum(packed_values, scale, total_bits):
    popcnt = popcount_op(packed_values)
    total_popcnt = popcnt.to(torch.int32).sum()
    return (2 * total_popcnt - total_bits) * scale

  num_ones = 7
  num_zeros = 25

  values = ([1.0] * num_ones) + ([-1.0] * num_zeros)
  random.shuffle(values)

  scale = 42.42
  expected_sum = sum(v * scale for v in values)

  # Create tensor directly on TPU and pack it
  data = torch.tensor(values, dtype=torch.float32, device="tpu")
  packed = _pack32(data)

  # Call quantized_sum
  quantized_sum_val = quantized_sum(packed, scale, total_bits=32)

  print("Quantized sum: ", quantized_sum_val.item())
  print("Expected sum: ", expected_sum)

  # The above print outs will be:
  # Quantized sum:  -763.5599975585938
  # Expected sum:  -763.5600000000001

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print("Success.")
  return 0


if __name__ == "__main__":
  app.run(main)
