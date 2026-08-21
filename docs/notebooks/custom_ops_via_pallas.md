---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

+++

# Custom Ops via Pallas

This notebook is the advanced continuation of our Pallas integration series.

### 🔗 Building on Foundations
Before diving into this notebook, you should be familiar with the basic integration patterns covered in **[Custom Kernels with Pallas](custom_pallas_kernel.ipynb)** (including `pallas.jax_op` wrappers, static shapes, and compiling/autograd registration).

While the introductory guide focused on the syntax of wrapping simple math (like addition), this advanced notebook focuses on **TPU hardware execution performance and register-level memory optimizations**.

+++

## 1. The Performance Goal: Register-Level Fusion

TorchTPU exposes lower-level control of TPUs via Pallas, the domain-specific
language for writing custom kernels. Like the Triton CUDA kernel,
Pallas allows fine-grained control of the TPU memory hierarchy (HBM, VMEM, and registers).

### The Case Study: DeepSeek-style Activation Quantization
Suppose we want to explore speeding up the blockwise activation binarization and register-level bit-packing. This case study is inspired by the **[DeepSeek v3 activation quantization kernel in Triton](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py#L10)**.

This example will only scratch the surface of Pallas. Refer to the
[JAX documentation on Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
for a more complete introduction.

The DeepSeek `act_quant` kernel takes higher precision activations and both
quantizes them down to fp8 blockwise (with an f32 scaling factor per block), and
packs them to one float per byte.

In our previous composition design (**[Extending Modules via Composition](extending_modules_and_functions_via_composition.ipynb)**), we created two separate passes: one for binarization (quantize) and one for packing (pack). Saving the binarized outputs to High-Bandwidth Memory (HBM) only to read them back immediately for packing creates a severe memory bottleneck.

To optimize this, we write a single merged Pallas kernel that loads the activations, calculates the scale binarization, shifts, and packs the bits **entirely within the TPU registers**, performing only a single memory read and write.

Since writing kernels is about performance, you want to understand the performance of an unoptimized baseline first. This baseline also acts as a test of numerical correctness.

The unoptimized baseline in Python is implemented in **[Custom Ops via Pallas: Python Baseline](quant.ipynb)**. In a realistic example, you would profile the baseline first. For this case study, we assume you have done so and decided to write a Pallas kernel.

+++

## 2. Pallas Implementation

Look at the implementation below. Notice that:
*   The wrapper around the JAX function is similar to the quantized sum example (**[Custom Ops via JAX Kernels](quantized_sum.ipynb)**), using `torch_tpu._internal.pallas.jax_op`.
*   The JAX code itself calls into Pallas to implement the register-level packing kernel.

```{code-cell} ipython3
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
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"
)

import torch
import torch_tpu._internal.pallas

torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access
```

```{code-cell} ipython3
# --- Pallas Implementation ---
# This section requires JAX.

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
```

```{code-cell} ipython3
# --- The rest of this code is pure PyTorch, with no reference to JAX modules.

# pylint: disable=protected-access
quant_and_pack = torch_tpu._internal.pallas.jax_op(
    "testing_op::quant_and_pack",
    quant_and_pack_jax,
)
```

```{code-cell} ipython3
def dump_dir(directory):
  import os

  entries = os.listdir(directory) if os.path.exists(directory) else []
  print(f"\n=== Files in {directory} (Total: {len(entries)}) ===")

  # Only print filenames to keep execution fast
  for f in sorted(entries):
    print(f)

  print(
      "\n💡 Tip: All compilation files are written to disk. You can read any"
      " file interactively using: open(os.path.join(directory,"
      " filename)).read()"
  )
```

```{code-cell} ipython3
def main():
  tpu_available = False
  try:
    _ = torch.randn(1, device="tpu")
    tpu_available = True
  except Exception:
    pass

  if not tpu_available:
    print(
        "TPU not available. Run this notebook on a TPU VM to execute Pallas"
        " kernel."
    )
    return 0

  device = torch.device("tpu")
  x = torch.tensor([[-7.0, 7.0, 1.0, -1.0] * 8], device=device)

  # Run Pallas
  packed, scale = quant_and_pack(x)

  # Force execution
  _ = packed.cpu()
  _ = scale.cpu()

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print(f"\n=========================================")
  print(f"Expected scale: 4.0. Actual: {scale.item()=}")
  print(f"Expected packed: 0x66666666. Actual: {packed.item()=:#08x}")
  print(f"=========================================")
  print("Success.")
  return 0

main()
```
