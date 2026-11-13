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
report.raise_for_errors()  # Pass fail_on_incomplete=True to reject bounded/skipped checks.

# Any PyTorch callable, e.g. an nn.Module that calls @jax_op kernels.
assert verifier.verify(model, (batch,)).passed

# A raw JAX function, called with jax.Array or jax.ShapeDtypeStruct arguments.
spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
print(verifier.verify(my_pallas_fn, (spec,)))
```

`verify(target, args, kwargs=None, *, mode=..., options=None)` detects what
`target` is:

*   **`@jax_op`**: op-contract, TPU-lowering, model-checking, and
    autograd-parity checks.
*   **Raw JAX function**: TPU-lowering and model-checking checks.
*   **Other callables**: traced into an FX graph (including nested `torch.cond`,
    `while_loop`, and `scan` subgraphs plus joint forward+backward autograd
    graphs) whose buffer donation and aliasing are checked; every `@jax_op` call
    in the graph is then verified for the shapes, dtypes, and concrete small
    integer/boolean inputs it is called with.

`mode=VerificationMode.STATIC` skips everything that executes kernels (such as
autograd parity), which is much faster. `VerificationOptions` controls the
autograd check, the unrolling bound (`max_unrolled_iterations`), and the TPU
generation of the lowering and static hardware checks (`tpu_version`).

The lowering check lowers a `@jax_op` or raw JAX Pallas function to Mosaic TPU
MLIR on the host via `jax.export` (up to `mosaic-serde`, including Pallas's
`func_op.verify()` MLIR dialect checks, without running `libtpu` layout or LLO
passes). Pure non-Pallas raw JAX functions skip lowering. When a kernel has a
concrete TPU-typed mesh (built with `torch.tpu.pallas.get_pallas_mesh` or a TPU
`AbstractMesh`), its mesh `device_kind` governs contracts, lowering, and static
hardware checks; otherwise `VerificationOptions.tpu_version` is used (or, when
`None`, inferred from a bound/input TPU mesh or callable `supported_on`
predicate, defaulting to `torch.tpu.TpuVersion.V7`).

## Checks

Findings have a severity: errors fail the report, warnings and infos do not
(unless `raise_for_errors(fail_on_incomplete=True)` is used to fail on
`verification_incomplete` findings).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| Category                        | Engines                 | Meaning                                                                                                                                                              |
| ------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_after_donate`              | `fx_graph`, `autograd`  | A tensor (or a view of it) is used after being donated to a `@jax_op`, or saved for backward across donation.                                                        |
| `aliased_donor_inputs`          | `contracts`, `fx_graph` | Two arguments of one call share storage while one of them is donated.                                                                                                |
| `donated_parameter`             | `contracts`, `fx_graph` | A module parameter, buffer, or captured tensor is donated and would be corrupted.                                                                                    |
| `zero_output_silent_skip`       | `contracts`             | A `@jax_op` without outputs, which TorchTPU skips entirely.                                                                                                          |
| `partition_spec_mismatch`       | `contracts`             | Mesh or `PartitionSpec` errors: unknown or repeated axes, rank or divisibility mismatches.                                                                           |
| `fake_kernel_mismatch`          | `contracts`             | A custom `register_fake` output shape, dtype, or pytree disagrees with the `JaxCallable` runtime output.                                                             |
| `lowering_failure`              | `lowering`              | The kernel fails Pallas TPU lowering, so it cannot run on TPU.                                                                                                       |
| `autograd_parity_mismatch`      | `autograd`              | A registered custom backward disagrees with the forward kernel.                                                                                                      |
| `unwritten_output`              | `model_check`           | Part of an output is never written, is clobbered by an unwritten parallel tile, or is accumulated into before being initialized.                                     |
| `out_of_bounds_access`          | `model_check`           | A static ref slice, DMA slice, or `BlockSpec` `index_map` is out of bounds.                                                                                          |
| `invalid_device_id`             | `model_check`           | A remote DMA or signal targets a device or core outside the mesh, or uses an invalid axis name or `device_id_type`.                                                  |
| `low_precision_accumulation`    | `model_check`           | A low-precision `dot_general` without `preferred_element_type=jnp.float32`, or cross-step accumulation into a 16-bit `Ref` (warning).                                |
| `hardware_constraint_violation` | `model_check`           | Rules missed by `jax.export`: TensorCore HBM/CMEM/ANY scratch, SparseCore sync `ANY` loads/stores, non-consecutive revisits, DMA sem rules, or invalid wait counts.  |
| `verification_incomplete`       | any                     | A part of the program that could not be fully verified (see limitations).                                                                                            |

<!-- mdformat on -->

## Multi-chip kernels

A `@jax_op` with a `mesh` (`jax.sharding.Mesh` or `jax.sharding.AbstractMesh`)
or a raw `jax.shard_map` callable is checked for every device of the mesh.

## TPU hosts

`VerificationMode.STATIC` only traces and lowers kernels, which needs no JAX
devices (provided user `supported_on` predicates do not query live backends).
`VerificationMode.FULL` runs autograd parity on JAX CPU devices; unless
`JAX_PLATFORMS` restricts it, creating them makes JAX initialize all of its
backends, including TPU when libtpu is installed. In a process that uses the TPU
through TorchTPU, use `VerificationMode.STATIC`, or run the full verification in
a separate process such as a unit test.

## How the model checker works

`jaxpr_model.py` traces the kernel without initializing a backend, locates every
`pallas_call` and `mpmd_map` (`pl.core_map` / `pl.kernel`), and unrolls each
kernel across mesh devices, grid points, and control-flow primitives (`scan` /
`fori_loop`, `while_loop`, `cond` / `pl.when`, and `run_scoped`). Concrete
integer and Boolean scalars are propagated per unrolled step to resolve
`BlockSpec` `index_map`s, `Ref` slices, DMA/semaphore operands, remote
`device_id`s, and coordinate-compressed output write coverage.

## Limitations

*   Model checking is bounded: loop iterations and scan steps past
    `max_unrolled_iterations` (32 by default) are not unrolled, and static
    bounds, device IDs, semaphore rules, and output coverage are checked across
    up to 4096 grid points.
*   Scalar-prefetch block indices and predicates are resolved at the provided
    example inputs when small concrete integer or boolean arrays/tensors are
    supplied, whereas abstract `ShapeDtypeStruct` scalar-prefetch inputs and
    values computed from runtime data inside the kernel (dynamic slice offsets
    or sizes, semaphore array indices, remote device ids, loop trip counts,
    branch conditions) are not resolved; the affected checks are skipped or
    approximated.
*   Iterations of `parallel` grid axes are modeled as if each ran on its own
    core, with its own scratch memory and semaphores.
*   VMEM and SMEM capacity are not checked.
*   Every bounded or skipped check is reported as `verification_incomplete`.

## Tests

The tests in `tests/pallas/pallas_verify_*_test.py` run on CPU, including
multi-chip kernels on virtual CPU devices.
