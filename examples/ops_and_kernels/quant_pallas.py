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

"""Example of quantizing and packing using Pallas."""

import os
import sys
from typing import Final, Tuple

# Inject flags and env var to libtpu.
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
import torch

torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access


# --- Pallas Implementation ---
# This section requires touching the JAX library.

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


def quant_and_pack_pallas_kernel(x_ref, packed_ref, scale_ref):
  x = x_ref[...]
  scale = jnp.mean(jnp.abs(x))
  scale_ref[...] = jnp.expand_dims(scale, 0)

  bits = (x >= 0.0).astype(jnp.int32)
  bit_patterns = jnp.left_shift(jnp.int32(1), jnp.arange(32, dtype=jnp.int32))
  packed = jnp.sum(bits * bit_patterns, axis=1, dtype=jnp.int32)
  packed_ref[...] = packed.astype(jnp.uint32)


def quant_and_pack_jax(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
  packed_shape = jax.ShapeDtypeStruct((x.shape[0],), jnp.uint32)
  scale_shape = jax.ShapeDtypeStruct((1,), jnp.float32)
  quant_and_pack_entry_func = pl.pallas_call(
      quant_and_pack_pallas_kernel, out_shape=(packed_shape, scale_shape)
  )
  return quant_and_pack_entry_func(x)


# --- The rest of this code is pure PyTorch, with no reference to JAX modules.

quant_and_pack = torch.tpu.pallas.jax_op(
    "testing_op::quant_and_pack",
    quant_and_pack_jax,
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

  device = torch.device("tpu")
  x = torch.tensor([[-7.0, 7.0, 1.0, -1.0] * 8], device=device)

  # Run Pallas
  packed, scale = quant_and_pack(x)

  # Force execution
  _ = packed.cpu()
  _ = scale.cpu()

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print(f"Expected scale: 4.0. Actual: {scale.item()=}")
  print(f"Expected packed: 0x66666666. Actual: {packed.item()=:#08x}")
  return 0


if __name__ == "__main__":
  app.run(main)
