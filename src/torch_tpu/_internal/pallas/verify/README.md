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
`VerificationOptions` controls the autograd check, the unrolling bound, the
solver time budget, the TPU generation of the lowering check, and the interpret
sweep.

The lowering check lowers a `@jax_op` for TPU on the host, as TorchTPU does the
first time the op runs. Ops without a TPU-typed mesh are lowered for
`VerificationOptions.tpu_version` (when `None`, inferred from a bound/input TPU
mesh or callable `supported_on` predicate, or `torch.tpu.TpuVersion.V7` by
default); ops with a TPU mesh are lowered for the device kind of their mesh
devices, so build the mesh with `torch.tpu.pallas.get_pallas_mesh`.

## Checks

Findings have a severity: errors fail the report, warnings and infos do not.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| Category                   | Engines                 | Meaning                                                                                                  |
| -------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------- |
| `use_after_donate`         | `fx_graph`, `autograd`  | A tensor (or a view of it) is used after being donated to a `@jax_op`, or saved for backward.            |
| `aliased_donor_inputs`     | `contracts`, `fx_graph` | Two arguments of one call share storage while one of them is donated.                                    |
| `donated_parameter`        | `contracts`, `fx_graph` | A module parameter, buffer, or captured tensor is donated and would be corrupted.                        |
| `zero_output_silent_skip`  | `contracts`             | A `@jax_op` without outputs, which TorchTPU skips entirely.                                              |
| `partition_spec_mismatch`  | `contracts`             | Mesh or `PartitionSpec` errors: unknown or repeated axes, rank or divisibility mismatches.               |
| `fake_kernel_mismatch`     | `contracts`             | A custom `register_fake` output shape, dtype, or pytree disagrees with the `JaxCallable` runtime output. |
| `lowering_failure`         | `lowering`              | The kernel fails Pallas TPU lowering, so it cannot run on TPU.                                           |
| `autograd_parity_mismatch` | `autograd`              | A registered custom backward disagrees with the forward kernel.                                          |
| `verification_incomplete`  | any                     | A part of the program that could not be fully verified.                                                  |

<!-- mdformat on -->

## Multi-chip kernels

A `@jax_op` with a `mesh` (`jax.sharding.Mesh` or `jax.sharding.AbstractMesh`)
or a raw `jax.shard_map` callable is checked for every device of the mesh. The
CPU interpret sweep runs it on virtual CPU devices, which must be created before
JAX initializes, for example with
`XLA_FLAGS=--xla_force_host_platform_device_count=8` or
`jax.config.update("jax_num_cpu_devices", 8)`.

## TPU hosts

`VerificationMode.STATIC` only traces and lowers kernels, which needs no JAX
devices (provided user `supported_on` predicates do not query live backends).
`VerificationMode.FULL` runs kernels on JAX CPU devices; unless `JAX_PLATFORMS`
restricts it, creating them makes JAX initialize all of its backends, including
TPU when libtpu is installed. In a process that uses the TPU through TorchTPU,
use `VerificationMode.STATIC`, or run the full verification in a separate
process such as a unit test.

## Tests

The tests in `tests/pallas/pallas_verify_*_test.py` run on CPU, including
multi-chip kernels on virtual CPU devices.
