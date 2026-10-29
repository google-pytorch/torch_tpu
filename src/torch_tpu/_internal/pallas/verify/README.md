# Pallas kernel verification

`torch_tpu._internal.pallas.verify` finds concurrency, memory-safety, and
integration bugs in Pallas TPU kernels and in the PyTorch programs that call
them through `@jax_op`. Everything runs on CPU: no TPU is needed.

This is an internal API that may change without notice.

## Quick start

```python
import jax
import jax.numpy as jnp
from torch_tpu._internal.pallas.verify import verifier

# A TorchTPU @jax_op, with example arguments like torch.library.opcheck.
report = verifier.verify(my_jax_op, (x, y))
report.raise_for_errors()  # Raises VerificationError listing every finding.

# Any PyTorch callable, e.g. an nn.Module that calls @jax_op kernels. Reports
# are truthy if and only if they contain no errors.
assert verifier.verify(model, (batch,))

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
first time the op runs. Ops without a mesh are lowered for
`VerificationOptions.tpu_device_kind` (`"TPU v5 lite"` by default); ops with a
mesh are lowered for the device kind of their mesh devices, so build the mesh
with `torch.tpu.pallas.get_pallas_mesh`.

## Checks

Findings have a severity: errors fail the report, warnings and infos do not.

Category                        | Engines                    | Meaning
------------------------------- | -------------------------- | -------
`use_after_donate`              | `fx_graph`                 | A tensor (or a view of it) is used after being donated to a `@jax_op`.
`aliased_donor_inputs`          | `contracts`, `fx_graph`    | Two arguments of one call share storage while one of them is donated.
`donated_parameter`             | `contracts`, `fx_graph`    | A module parameter, buffer, or captured tensor is donated and would be corrupted.
`zero_output_silent_skip`       | `contracts`                | A `@jax_op` without outputs, which TorchTPU skips entirely.
`partition_spec_mismatch`       | `contracts`                | Mesh or `PartitionSpec` errors: unknown or repeated axes, rank or divisibility mismatches.
`lowering_failure`              | `lowering`                 | The kernel fails Pallas TPU lowering, so it cannot run on TPU.
`autograd_parity_mismatch`      | `autograd`                 | A registered custom backward disagrees with the forward kernel.
`data_race`                     | `model_check`, `interpret` | Two accesses to overlapping memory, at least one a write, can happen at the same time.
`concurrent_dma_sem_sharing`    | `model_check`              | A race enabled by concurrent DMAs that signal the same semaphore.
`unthrottled_remote_overwrite`  | `model_check`              | A remote DMA can overwrite a buffer its receiver is still using.
`parallel_grid_conflict`        | `model_check`              | Grid iterations along a `parallel` axis write the same shared memory.
`semaphore_deadlock`            | `model_check`              | No schedule lets every semaphore wait complete.
`semaphore_nonzero_exit`        | `model_check`, `interpret` | A semaphore is not back at zero when the kernel exits.
`unwritten_output`              | `model_check`              | Part of an output is never written and returns uninitialized memory.
`out_of_bounds_access`          | `model_check`, `interpret` | A static ref slice, DMA slice, or BlockSpec index map is out of bounds.
`invalid_device_id`             | `model_check`              | A remote DMA or signal targets a device outside the mesh.
`low_precision_accumulation`    | `model_check`              | A bfloat16 matmul without `preferred_element_type=jnp.float32` (warning).
`hardware_constraint_violation` | `model_check`              | Hardware usage rules, such as barrier semaphores without `collective_id`.
`nan_or_uninit_read`            | `interpret`                | NaN-poisoned uninitialized memory reaches an output (NaNs that zero-filled memory avoids).
`schedule_divergence`           | `interpret`                | Outputs differ across DMA schedules or grid orders.
`verification_incomplete`       | any                        | A part of the program that could not be fully verified (see limitations).

Races come with a counterexample: the prefix of a concrete schedule, ordered by
time, that leads to the conflicting accesses.

## Multi-chip kernels

A `@jax_op` with a `mesh` (typically wrapping `jax.shard_map`) is checked for
every device of the mesh. The CPU interpret sweep runs it on virtual CPU
devices, which must be created before JAX initializes, for example with
`XLA_FLAGS=--xla_force_host_platform_device_count=8` or
`jax.config.update("jax_num_cpu_devices", 8)`.

## TPU hosts

`VerificationMode.STATIC` only traces and lowers kernels, which needs no JAX
devices. `VerificationMode.FULL` runs kernels on JAX CPU devices; unless
`JAX_PLATFORMS` restricts it, creating them makes JAX initialize all of its
backends, including TPU when libtpu is installed. In a process that uses the TPU
through TorchTPU, use `VerificationMode.STATIC`, or run the full verification in
a separate process such as a unit test.

## How the model checker works

`jaxpr_model.py` traces the kernel and unrolls every `pallas_call` for each
device and grid point into events: vector loads and stores on hyper-rectangular
regions of buffers, the implicit pipeline copies of `BlockSpec` operands, DMA
starts and waits, and semaphore signals and waits.

`smt_checker.py` gives every event an integer timestamp and encodes program
order, DMA causality (start < read done <= write done), and counting-semaphore
semantics (a wait completes only once the signals before it cover its amount
plus what earlier waits consumed) as difference and pseudo-Boolean constraints.
If no schedule satisfies them, the kernel deadlocks. For each pair of
conflicting accesses it asks whether some schedule lets them overlap.

The constraints are decided by `idl_solver.py`, a small conflict-driven clause
learning solver for integer difference logic with pseudo-Boolean constraints. It
only needs the Python standard library, and it keeps learned clauses between the
race queries of a kernel.

## Limitations

*   Model checking is bounded: grid points, loop iterations, and scan steps past
    `max_unrolled_iterations` (32 by default) are not checked. Races are still
    checked among the checked iterations, but semaphores that the dropped
    iterations also use are not checked for their values at kernel exit, and a
    deadlock that may involve them is only reported as incomplete. Output
    coverage and bounds use up to 4096 grid points.
*   Values that depend on runtime data (scalar-prefetch block indices, semaphore
    array indices, remote device ids, loop trip counts, branch conditions) are
    not resolved; the affected checks are skipped or approximated. Semaphore
    operations whose execution depends on runtime data disable the deadlock and
    race checks.
*   Iterations of `parallel` grid axes are modeled as if each ran on its own
    core, with its own scratch memory and semaphores, so reads of per-core
    scratch that depend on another iteration's writes are not reported.
*   VMEM and SMEM capacity are not checked.
*   The dynamic checks of kernels with DMAs or semaphore signals between devices
    need JAX 0.11.2 or newer: the CPU interpreter of older releases (such as
    0.9) fails on them, so they are skipped there. Other interpreter errors that
    do not identify a kernel bug also skip the interpret checks.
*   Every bounded or skipped check is reported as `verification_incomplete`.

## Tests

The tests in `tests/pallas/pallas_verify_*_test.py` run on CPU, including
multi-chip kernels on virtual CPU devices.
