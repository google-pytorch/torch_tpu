# Pallas kernel verification

`torch.tpu.pallas.verifier` (implemented in `torch_tpu._internal.pallas.verify`)
finds concurrency, memory-safety, and integration bugs in Pallas TPU kernels and
in the PyTorch programs that call them through `@jax_op`. Everything runs on
CPU: no TPU is needed.

This is an experimental API and is subject to change.

## Quick start

```python
import jax
import jax.numpy as jnp
from torch.tpu.pallas import verifier

# A TorchTPU @jax_op, with example arguments like torch.library.opcheck.
report = verifier.verify(my_jax_op, (x, y))
report.raise_for_errors()  # Raises VerificationError listing every finding.

# Any PyTorch callable, e.g. an nn.Module that calls @jax_op kernels.
assert verifier.verify(model, (batch,)).passed

# A raw JAX function, called with jax.Array or jax.ShapeDtypeStruct arguments.
spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
print(verifier.verify(my_pallas_fn, (spec,)))
```

`verify(target, args, kwargs=None, *, mode=..., options=None)` detects what
`target` is:

*   **`@jax_op`**: op-contract, TPU-lowering, model-checking, autograd-parity,
    and CPU-interpret checks.
*   **Raw JAX function**: model-checking and CPU-interpret checks.
*   **Other callables**: traced into an FX graph whose buffer donation and
    aliasing are checked; every `@jax_op` call in the graph is then verified for
    the shapes and dtypes it is called with.

`mode=VerificationMode.STATIC` skips everything that executes kernels (the CPU
interpret sweep and autograd parity), which is much faster.

## Checks

Findings have a severity: errors fail the report, warnings and infos do not.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| Category                  | Engines | Meaning                                                 |
| ------------------------- | ------- | ------------------------------------------------------- |
| `verification_incomplete` | any     | A part of the program that could not be fully verified. |

<!-- mdformat on -->

## Tests

The tests in `tests/pallas/pallas_verify_*_test.py` run on CPU, including
multi-chip kernels on virtual CPU devices.
