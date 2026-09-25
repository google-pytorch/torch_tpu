---
torch_tpu_version: 0.1.1.dev20260922
---

# Introduction to Custom Ops

You started this journey learning to extend PyTorch functionality by composing
existing functionality.

Then, with `torch.autograd.Function`, you learned the first technique to gain
increased control of PyTorch's core engines.

You learned some necessary background information on ops and compile, and now
you are ready to start creating your own custom ops that live alongside the
existing ATen ops.

### Custom ops via Python

Custom ops in Python are a well covered topic in the
[PyTorch docs](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html).

These custom ops have all the advantages of PyTorch's ATen ops: they compose
well with `torch.compile` to avoid a graph break, ultimately leading to better
performance.

### Custom ops via C++ HLO kernels

Stay tuned for a forthcoming guide to implementing custom ops via HLO kernels in
C++. That said, since `jax.lax` is a nearly 1:1 mapping to StableHLO ops, the
next section on kernels in JAX may give you the ability to generate the HLO ops
you want, in pure Python, without diving into C++ build details.

## Custom ops via JAX kernels (without Pallas)

JAX is a different framework from TorchTPU but it ultimately compiles down to
the same HLO as TorchTPU. In some cases, JAX may give you some additional
control over the HLO. For example, JAX provides the
[population_count op](https://docs.jax.dev/en/latest/_autosummary/jax.lax.population_count.html),
which lowers directly to the
[StableHLO popcnt op](https://openxla.org/stablehlo/spec#popcnt). This makes JAX
a solution to implement some custom ops.

As a toy problem, suppose you want to implement via JAX a custom op that
performs a quantized sum for the one-bit format developed in
**[extending_modules_and_functions_via_composition.py](?file=extending_modules_and_functions_via_composition.py)**.
You decide to take advantage of popcount as a "sum" function for the one bit
format. The sum of a tensor in this quantization format can be inferred from the
number of bits set *without conversion to a native PyTorch floating point
format*.

Each bit represents scale, and each non-bit represents -scale. So the sum is
simply

```
(number of bits set) * scale + (number of non-bits set) * (-scale)
= (number of bits set) * scale - (number of non-bits set) * scale
= ((number of bits set) - (number of non-bits set)) * scale
= ((number of bits set) - (total number of bits - number of bits set)) * scale
= (2 * (number of bits set) - (total number of bits)) * scale
```

Below is the sample implementation. Notice that:

*   `torch.tpu.pallas.jax_op` wraps JAX inside a PyTorch custom op.
*   The caller of the PyTorch custom op has no visibility into the fact that the
    op is implemented in terms of JAX.

Using the logging techniques demonstrated previously, you can even trace down
the [`popcnt`](https://openxla.org/stablehlo/spec#popcnt) in the dump file
`module_0039.tt_jit_custom_kernel.before_optimizations.txt`.

```
%test_example__popcount_0xb92b15e71d7bc5ec.1 (packed_vector.1: u32[]) -> u32[] {
  %packed_vector.1 = u32[] parameter(0), sharding={replicated}, metadata={op_name="packed_vector"}
  ROOT %population_count.1 = u32[] popcnt(%packed_vector.1), metadata={op_name="jit(jax_popcount)/population_count" stack_frame_id=10}
}
```

`jax.lax.population_count` is not differentiable so you would only use it for
inference. For a more realistic `jax.lax` op you might want to expose, consider
`jax.lax.approx_max_k`.

For a canonical example of implementing a custom op using a JAX kernel,
including use of `jax.jvp`/`jax.grad` to automatically create a backward
function, see the unit test `torch_tpu/tests/pallas/pallas_test.py`,
specifically `test_jax_dot_grad_for_backwards()`.

```python
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
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"
)

from absl import app as absl_app  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

# pylint: disable=protected-access
torch._logging.set_logs(aot_graphs=True)
```

```python
# Copied from qat_linear.py
def pack32(values):
  """Packs a vector of 32 quantized values (signs) into a uint32"""
  assert values.size() == (32,)
  assert values.dtype == torch.float32
  assert torch.all((values == 1.0) | (values == -1.0))

  # Translate values from {-1, 1} to {0, 1}
  bits = (values == 1.0).to(torch.uint32)

  # Create all bit patterns from 0000, 0001, 0010, etc. on the same device.
  bit_patterns = 2 ** torch.arange(
      32, dtype=torch.uint32, device=values.device
  )

  # Summing the bit patterns is a bitwise_or reduction on a vector.
  return torch.sum(bits * bit_patterns, dim=0, dtype=torch.uint32)
```

```python
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
```

```python
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

```python
def main():
  # Verify device is TPU before running
  tpu_available = False
  try:
    _ = torch.randn(1, device="tpu")
    tpu_available = True
  except Exception:
    pass

  if not tpu_available:
    print(
        "TPU not available. Run this notebook on a TPU VM to execute"
        " quantized sum."
    )
    return 0

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
  packed = pack32(data)

  # Call quantized_sum
  quantized_sum_val = quantized_sum(packed, scale, total_bits=32)

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print("\n=========================================")
  print("Quantized sum: ", quantized_sum_val.item())
  print("Expected sum: ", expected_sum)
  print("=========================================")
  print("Success.")
  return 0

main()
```
