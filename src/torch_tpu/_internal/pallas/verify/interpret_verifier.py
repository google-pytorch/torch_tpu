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

"""CPU TPU-interpret sweeps over DMA schedules, seeds, and poisoned memory.

`verify_with_cpu_interpret` executes a TorchTPU `@jax_op` or a raw JAX function
on CPU virtual devices under `pltpu.force_tpu_interpret_mode` across multiple
`(DmaScheduleMode, seed)` schedules:

* Checks the interpreter's internal `pltpu.races` tracker after each schedule
  for dynamically observed data races (`DATA_RACE`) and its kernel-exit
  semaphore validation for non-zero semaphore counts (`SEMAPHORE_NONZERO_EXIT`).
* Runs with `uninitialized_memory="nan"` (which poisons floating-point buffers
  with quiet NaNs, integer buffers with `iinfo.max`, and boolean buffers with
  `True`) and compares against a `uninitialized_memory="zero"` control run so
  genuine uninitialized reads or unwritten output regions (`NAN_OR_UNINIT_READ`)
  are distinguished from intentional kernel-computed values.
* Compares output tensors across all `(DmaScheduleMode, seed)` combinations to
  detect schedule-dependent non-determinism (`SCHEDULE_DIVERGENCE`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import dataclasses
import inspect
import sys
import threading
from typing import Any, cast

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier

_INTERPRET = findings.Phase.INTERPRET

InterpretSweepConfig = findings.InterpretSweepConfig

# Serializes CPU TPU interpret runs, which share process-global state in JAX's
# `interpret_pallas_call` module (`_shared_memory`, `races`, `dma_id_counter`).
_INTERPRET_LOCK = threading.RLock()

# Per-schedule watchdog timeout so a runtime semaphore wait deadlock aborts the
# shared memory barrier rather than hanging the process indefinitely.
_DEFAULT_INTERPRET_RUN_TIMEOUT_SECONDS = 10.0

_OPTIONAL_INTERPRET_PARAMS = frozenset({"on_race"})


def make_interpret_params(**params: Any) -> pltpu.InterpretParams:
  """Builds `pltpu.InterpretParams`, dropping optional fields this JAX lacks.

  For example `on_race` only exists in newer JAX releases; without it races
  are still recorded and checked after the run.

  Args:
    **params: `pltpu.InterpretParams` fields.

  Returns:
    The interpret parameters.
  """
  supported = {f.name for f in dataclasses.fields(pltpu.InterpretParams)}
  unknown = set(params) - supported - _OPTIONAL_INTERPRET_PARAMS
  if unknown:
    raise TypeError(f"unexpected InterpretParams field(s): {sorted(unknown)!r}")
  return pltpu.InterpretParams(
      **{k: v for k, v in params.items() if k in supported}
  )


# The CPU interpreter of older JAX releases (0.9 for example) fails on DMAs and
# semaphore signals between devices and then fails every later interpret run
# in the process, so kernels with such operations only run on newer releases.
MIN_JAX_FOR_REMOTE_INTERPRET = (0, 11, 2)
MIN_JAX_FOR_REMOTE_INTERPRET_STR = ".".join(
    str(v) for v in MIN_JAX_FOR_REMOTE_INTERPRET
)


def remote_interpret_supported() -> bool:
  """Returns whether this JAX can interpret operations between devices."""
  return tuple(jax.__version_info__[:3]) >= MIN_JAX_FOR_REMOTE_INTERPRET


@contextlib.contextmanager
def _force_primitive_interpret_params(
    params: pltpu.InterpretParams,
) -> Iterator[None]:
  """Overrides `interpret` on pre-constructed `pallas_call` / `mpmd_map` binds.

  `pl.pallas_call(...)` and `@pl.kernel(...)` evaluate
  `config.pallas_tpu_interpret_mode_context_manager.value` when the wrapper is
  constructed rather than when it is called. Because
  `pltpu.force_tpu_interpret_mode`
  is part of `@jax.jit`'s cache key, re-tracing inside the context manager
  re-invokes `pallas_call_p.bind` / `mpmd_map_p.bind`, where this hook supplies
  `params`.

  Args:
    params: Interpret parameters to force on every Pallas primitive bind.

  Yields:
    None.
  """
  saved_binds: list[tuple[Any, Any]] = []
  for mod_name, prim_attr in (
      ("jax._src.pallas.pallas_call", "pallas_call_p"),
      ("jax._src.pallas.mpmd", "mpmd_map_p"),
  ):
    mod = sys.modules.get(mod_name)
    prim = getattr(mod, prim_attr, None) if mod is not None else None
    orig_bind = getattr(prim, "bind", None) if prim is not None else None
    if prim is None or orig_bind is None:
      continue

    def _make_bind(bound_orig: Any) -> Any:
      def _wrapped_bind(*args: Any, **kwargs: Any) -> Any:
        if "interpret" in kwargs:
          kwargs = dict(kwargs, interpret=params)
        return bound_orig(*args, **kwargs)

      return _wrapped_bind

    saved_binds.append((prim, orig_bind))
    prim.bind = _make_bind(orig_bind)
  try:
    yield
  finally:
    for prim, orig_bind in reversed(saved_binds):
      prim.bind = orig_bind


def _prepare_jax_arg_on_cpu(
    val: jax.Array | jax.ShapeDtypeStruct,
    cpu0: jax.Device,
    cpu_mesh: jax.sharding.Mesh | None,
    *,
    seed: int,
) -> jax.Array:
  arr = _uncommitted_cpu_array(val, cpu0, seed=seed)
  sharding = getattr(val, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding) and cpu_mesh is not None:
    return jax.device_put(
        arr, jax.sharding.NamedSharding(cpu_mesh, sharding.spec)
    )
  return arr


def _prepare_cpu_inputs(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
) -> tuple[
    Any,
    list[Any],
    dict[str, Any],
    jax.sharding.Mesh | None,
]:
  """Prepares `(raw_fn, jax_args, jax_kwargs, cpu_mesh)`."""
  jax_callable = fx_verifier.extract_jax_callable(op_or_fn)
  arg_leaves = jax.tree.leaves((tuple(example_args), dict(example_kwargs)))
  has_named_sharding = jax_callable is not None or any(
      isinstance(getattr(leaf, "sharding", None), jax.sharding.NamedSharding)
      for leaf in arg_leaves
  )
  raw_fn, mesh_or_cpu_mesh = fx_verifier.unwrap_raw_jax_fn(
      op_or_fn,
      prefer_cpu_mesh=True,
      preserve_axis_types=has_named_sharding,
  )
  if mesh_or_cpu_mesh is None:
    for leaf in arg_leaves:
      sharding = getattr(leaf, "sharding", None)
      if isinstance(sharding, jax.sharding.NamedSharding):
        if fx_verifier.is_mock_tpu_mesh(sharding.mesh):
          cpu_candidate = fx_verifier.make_cpu_mesh_for(
              sharding.mesh, preserve_axis_types=True
          )
          if cpu_candidate is not None:
            raw_fn = fx_verifier.rebind_function_meshes(
                raw_fn, cpu_candidate, clone_closures=True
            )
            mesh_or_cpu_mesh = cpu_candidate
          else:
            mesh_or_cpu_mesh = sharding.mesh
        else:
          mesh_or_cpu_mesh = sharding.mesh
        break

  if mesh_or_cpu_mesh is not None and fx_verifier.is_mock_tpu_mesh(
      mesh_or_cpu_mesh
  ):
    num = mesh_or_cpu_mesh.size
    raise ValueError(
        f"the {num}-device mesh needs {num} virtual CPU devices but only"
        f" {len(jax.devices('cpu'))} exist; before JAX initializes, set"
        f" XLA_FLAGS=--xla_force_host_platform_device_count={num} or call"
        f' jax.config.update("jax_num_cpu_devices", {num})'
    )
  cpu_mesh = cast(jax.sharding.Mesh | None, mesh_or_cpu_mesh)

  if jax_callable is not None:
    # pylint: disable=protected-access
    norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
        example_args, dict(example_kwargs)
    )
    # pylint: enable=protected-access
    norm_args = fx_verifier.replace_mock_meshes_in_pytree(norm_args, cpu_mesh)
    norm_kwargs = fx_verifier.replace_mock_meshes_in_pytree(
        norm_kwargs, cpu_mesh
    )
    in_specs = jax_callable.input_partition_specs
  else:
    norm_args = fx_verifier.replace_mock_meshes_in_pytree(
        list(example_args), cpu_mesh
    )
    norm_kwargs = fx_verifier.replace_mock_meshes_in_pytree(
        dict(example_kwargs), cpu_mesh
    )
    in_specs = None

  cpu0 = jax.devices("cpu")[0]
  seed_counter = 0

  def _convert_tree(val: Any, spec: Any) -> Any:
    nonlocal seed_counter
    if isinstance(val, torch.Tensor):
      if cpu_mesh is not None:
        return fx_verifier.build_sharded_cpu_array(
            val, cpu_mesh, spec, distinguish_ranks=True
        )
      np_val, jax_dtype = fx_verifier.tensor_to_numpy(val)
      return jax.device_put(jnp.asarray(np_val, dtype=jax_dtype), cpu0)
    if isinstance(val, (jax.Array, jax.ShapeDtypeStruct)):
      cur_seed = seed_counter
      seed_counter += 1
      return _prepare_jax_arg_on_cpu(val, cpu0, cpu_mesh, seed=cur_seed)
    if isinstance(val, (tuple, list)):
      converted = [
          _convert_tree(
              elem,
              spec[j]
              if isinstance(spec, (tuple, list))
              and not isinstance(spec, jax.sharding.PartitionSpec)
              and j < len(spec)
              else spec,
          )
          for j, elem in enumerate(val)
      ]
      if isinstance(val, tuple) and hasattr(val, "_fields"):
        return type(val)(*converted)
      return type(val)(converted)
    if isinstance(val, Mapping):
      return {
          k: _convert_tree(
              v, spec.get(k) if isinstance(spec, Mapping) else spec
          )
          for k, v in val.items()
      }
    return val

  jax_args: list[Any] = []
  with fx_verifier.cpu_interpret_mesh_context(cpu_mesh):
    for i, arg in enumerate(norm_args):
      if jax_callable is not None and fx_verifier.is_static_positional_arg(
          jax_callable, i
      ):
        jax_args.append(arg)
        continue
      spec = in_specs[i] if in_specs is not None and i < len(in_specs) else None
      jax_args.append(_convert_tree(arg, spec))

    jax_kwargs: dict[str, Any] = {
        k: _convert_tree(val, None) for k, val in norm_kwargs.items()
    }

  return raw_fn, jax_args, jax_kwargs, cpu_mesh


def _uncommitted_cpu_array(
    value: jax.Array | jax.ShapeDtypeStruct, cpu0: jax.Device, *, seed: int
) -> jax.Array:
  """Returns `value`, or example values for its shape, on the host CPU.

  The array is uncommitted, so that `jax.jit` and `jax.shard_map` in the
  function under test may place it on any CPU devices.

  Args:
    value: A concrete array, or an abstract argument to fill with deterministic
      example values.
    cpu0: The CPU device to create the array on.
    seed: Seed of the example values.

  Returns:
    The array.
  """
  if isinstance(value, jax.Array):
    host_value = np.asarray(value)
  elif jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
    with jax.default_device(cpu0):
      base_key = jax.random.key(seed)
      return (
          jnp.broadcast_to(base_key, value.shape) if value.shape else base_key
      )
  else:
    rng = np.random.default_rng(seed)
    if jnp.issubdtype(value.dtype, jnp.floating) or jnp.issubdtype(
        value.dtype, jnp.complexfloating
    ):
      # Positive values keep functions like log and sqrt finite.
      host_value = rng.uniform(0.5, 1.5, value.shape)
    else:
      # Integer and boolean abstract inputs are frequently scalar-prefetch block
      # indices, dynamic slice offsets, or semaphore wait counts, so default to
      # zeros (matching `_materialize_fake_arg` in `verifier.py`).
      host_value = np.zeros(value.shape)
  with jax.default_device(cpu0):
    return jnp.asarray(host_value, dtype=value.dtype)


def _extract_innermost_exception_text(msg: str) -> str:
  """Strips embedded callback tracebacks, keeping the exception message."""
  tail = msg.split("with exception:\n\n")[-1].strip()
  if "Traceback (most recent call last):" not in tail:
    return tail
  lines = tail.splitlines()
  last_header_idx = 0
  for idx, line in enumerate(lines):
    if (
        line
        and not line.startswith(" ")
        and not line.startswith("Traceback (most recent call last):")
        and not line.startswith("During handling of the above exception")
        and not line.startswith("The above exception was the direct cause")
    ):
      last_header_idx = idx
  return "\n".join(lines[last_header_idx:]).strip()


def _classify_interpret_exception(
    exc: Exception,
    op_name: str,
    schedule_desc: str,
) -> findings.VerificationFinding:
  """Maps a CPU TPU interpret exception onto a structured `VerificationFinding`.

  Exceptions that do not name a kernel bug may come from features that the
  interpreter of the installed JAX does not support (for example, JAX 0.9
  fails multi-device remote DMAs with a KeyError), so they are reported as
  incomplete verification rather than as errors.

  Args:
    exc: The exception raised by the interpret run.
    op_name: Name of the op in findings.
    schedule_desc: Description of the schedule that was run.

  Returns:
    The finding.
  """
  msg = str(exc).strip()
  exc_text = _extract_innermost_exception_text(msg)
  lower = exc_text.lower()
  if "race detected" in lower:
    cat = findings.FindingCategory.DATA_RACE
  elif "out of bounds" in lower or "out-of-bounds" in lower:
    cat = findings.FindingCategory.OUT_OF_BOUNDS_ACCESS
  elif "semaphore wait timed out" in lower:
    cat = findings.FindingCategory.SEMAPHORE_DEADLOCK
  elif "semaphore" in lower and (
      "non-zero" in lower
      or "nonzero" in lower
      or "at kernel exit" in lower
      or "!= 0" in lower
  ):
    cat = findings.FindingCategory.SEMAPHORE_NONZERO_EXIT
  elif (
      "revisited block" in lower
      or "revisiting block" in lower
      or "advanced indexers are not supported on tpu" in lower
  ):
    cat = findings.FindingCategory.HARDWARE_CONSTRAINT_VIOLATION
  elif (
      "unrecognized axes in device_id" in lower
      or "unsupported device id type" in lower
      or "semaphore_signal with device_id" in lower
  ):
    cat = findings.FindingCategory.INVALID_DEVICE_ID
  else:
    last_line = exc_text.splitlines()[-1].strip() if exc_text else ""
    exc_name = type(exc).__name__
    if not last_line:
      exc_summary = exc_name
    elif last_line.startswith(f"{exc_name}:"):
      exc_summary = last_line
    else:
      exc_summary = f"{exc_name}: {last_line}"
    return findings.VerificationFinding(
        category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=findings.Severity.WARNING,
        phase=_INTERPRET,
        op_name=op_name,
        message=(
            "skipped CPU interpret runs because the interpreter raised"
            f" ({schedule_desc}) {exc_summary}; the kernel"
            " may be invalid or use features that the interpreter of this JAX"
            " version does not support"
        ),
    )
  return findings.VerificationFinding(
      category=cat,
      severity=findings.Severity.ERROR,
      phase=_INTERPRET,
      op_name=op_name,
      message=f"CPU TPU interpret ({schedule_desc}) raised: {exc_text or msg}",
  )


# Upstream JAX spells `SharedMemory`'s exit check with a transposed 'pa'; check
# the corrected spelling first in case newer JAX releases fix it.
_SEM_NONZERO_ATTR_CANDIDATES = (
    "get_semaphores_with_nonzero_count",
    "get_semp" + "ahores_with_nonzero_count",
)


@dataclasses.dataclass(frozen=True)
class _InterpretRunOutcome:
  """Result of a single `_interpret_run` invocation."""

  leaves: list[np.ndarray] | None
  exc: Exception | None
  race_found: bool
  nonzero_sems: tuple[str, ...]


def _interpret_run(
    raw_fn: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    params: pltpu.InterpretParams,
    cpu_device: jax.Device | None,
    cpu_mesh: jax.sharding.Mesh | None = None,
    *,
    timeout_seconds: float = _DEFAULT_INTERPRET_RUN_TIMEOUT_SECONDS,
) -> _InterpretRunOutcome:
  """Runs `raw_fn` once in TPU interpret mode.

  Args:
    raw_fn: The JAX function to run.
    args: Positional arguments.
    kwargs: Keyword arguments.
    params: Interpret-mode parameters.
    cpu_device: Default device for single-device functions; None for functions
      that place their inputs on a CPU mesh themselves.
    cpu_mesh: Concrete CPU mesh when running multi-device kernels.
    timeout_seconds: Per-run watchdog timeout to abort hanging semaphore waits.

  Returns:
    An `_InterpretRunOutcome` capturing output leaves (or the raised exception)
    along with any races or non-zero semaphores observed before state reset.
  """
  with _INTERPRET_LOCK:
    pltpu.reset_tpu_interpret_mode_state()
    interpret_mod = inspect.getmodule(pltpu.reset_tpu_interpret_mode_state)
    memory_mod = getattr(interpret_mod, "memory", None)
    shm_cls = getattr(memory_mod, "SharedMemory", None)
    sem_nonzero_attr: str | None = None
    orig_get_nonzero: Any = None
    if shm_cls is not None:
      for candidate in _SEM_NONZERO_ATTR_CANDIDATES:
        fn = getattr(shm_cls, candidate, None)
        if callable(fn):
          sem_nonzero_attr = candidate
          orig_get_nonzero = fn
          break
    nonzero_sems: list[str] = []

    def _abort_on_timeout() -> None:
      shm = getattr(interpret_mod, "_shared_memory", None)
      if shm is not None and hasattr(shm, "set_failed"):
        shm.set_failed(
            TimeoutError(
                "semaphore wait timed out in CPU TPU interpret mode (possible"
                " semaphore deadlock)"
            ),
            top_level=False,
        )
        clean_up = getattr(shm, "clean_up_barrier", None)
        if clean_up is not None and hasattr(clean_up, "abort"):
          clean_up.abort()

    watchdog: threading.Timer | None = None
    patched_shm = False
    try:
      if shm_cls is not None and sem_nonzero_attr is not None:

        def _recording_get_nonzero(self_shm: Any, device_id: int) -> Any:
          res = orig_get_nonzero(self_shm, device_id)
          for sem, gci in res:
            count = sem.count_by_core[gci]
            nonzero_sems.append(
                f"semaphore {sem.id} has non-zero count {count} for device"
                f" {device_id} (global core {gci}) at kernel exit"
            )
          return res

        setattr(shm_cls, sem_nonzero_attr, _recording_get_nonzero)
        patched_shm = True

      watchdog = threading.Timer(timeout_seconds, _abort_on_timeout)
      watchdog.daemon = True
      watchdog.start()

      default_cpu = (
          cpu_device if cpu_device is not None else jax.devices("cpu")[0]
      )
      dev_ctx = jax.default_device(default_cpu)
      mesh_ctx = fx_verifier.cpu_interpret_mesh_context(cpu_mesh)
      leaves: list[np.ndarray] | None = None
      run_exc: Exception | None = None
      with (
          mesh_ctx,
          dev_ctx,
          pltpu.force_tpu_interpret_mode(params),
          _force_primitive_interpret_params(params),
      ):
        try:
          # Copy array inputs so that kernels that write their inputs do not
          # affect later runs.
          run_args = jax.tree.map(
              lambda a: jnp.copy(a) if isinstance(a, jax.Array) else a,
              list(args),
          )
          run_kwargs = jax.tree.map(
              lambda v: jnp.copy(v) if isinstance(v, jax.Array) else v,
              dict(kwargs),
          )
          out = raw_fn(*run_args, **run_kwargs)
          leaves = [
              np.asarray(leaf.astype(jnp.float32))
              if jnp.issubdtype(leaf.dtype, jnp.floating)
              and not np.issubdtype(leaf.dtype, np.floating)
              else np.asarray(leaf)
              for leaf in jax.tree.leaves(out)
              if isinstance(leaf, jax.Array)
          ]
          jax.effects_barrier()
        except Exception as exc:  # pylint: disable=broad-exception-caught
          run_exc = exc
        races_state = getattr(interpret_mod, "races", None)
        race_found = races_state is not None and bool(races_state.races_found)
      return _InterpretRunOutcome(
          leaves=leaves,
          exc=run_exc,
          race_found=race_found,
          nonzero_sems=tuple(nonzero_sems),
      )
    finally:
      if watchdog is not None:
        watchdog.cancel()
      if patched_shm and shm_cls is not None and sem_nonzero_attr is not None:
        setattr(shm_cls, sem_nonzero_attr, orig_get_nonzero)
      pltpu.reset_tpu_interpret_mode_state()


def _uninitialized_memory_message(
    leaves: Sequence[np.ndarray], control: Sequence[np.ndarray]
) -> str | None:
  """Describes output elements of `leaves` that zero-filled `control` avoids."""
  for out_idx, (arr, ctrl) in enumerate(zip(leaves, control)):
    if arr.shape != ctrl.shape or arr.dtype != ctrl.dtype:
      continue
    if np.issubdtype(arr.dtype, np.inexact):
      both_nan = np.isnan(arr) & np.isnan(ctrl)
      extra = np.isnan(arr) & ~np.isnan(ctrl)
      if extra.any():
        return (
            f"output[{out_idx}] contains {int(extra.sum())}/{arr.size} NaN"
            " value(s) that disappear when uninitialized memory is zero-filled,"
            " so the kernel reads uninitialized VMEM or unmasked padding"
        )
      diff = ~both_nan & (arr != ctrl)
    else:
      diff = arr != ctrl
    if diff.any():
      return (
          f"output[{out_idx}] has {int(diff.sum())}/{arr.size} element(s)"
          " that change when uninitialized memory is zero-filled instead of"
          " poisoned, so the kernel reads uninitialized VMEM or unmasked"
          " padding"
      )
  return None


def _schedule_divergence_message(
    baseline_leaves: Sequence[np.ndarray],
    cur_leaves: Sequence[np.ndarray],
    baseline_schedule: str,
    schedule_desc: str,
    *,
    rtol: float,
    atol: float,
) -> str | None:
  """Describes the first output discrepancy between two schedules, if any."""
  if len(baseline_leaves) != len(cur_leaves):
    return (
        f"output array count differs across schedules ({baseline_schedule}"
        f" produced {len(baseline_leaves)} array(s) vs {schedule_desc} with"
        f" {len(cur_leaves)}), indicating schedule-dependent behavior"
    )
  for out_idx, (ref_arr, cur_arr) in enumerate(
      zip(baseline_leaves, cur_leaves)
  ):
    if ref_arr.shape != cur_arr.shape or ref_arr.dtype != cur_arr.dtype:
      return (
          f"output[{out_idx}] shape/dtype differs across schedules"
          f" ({baseline_schedule} had {ref_arr.shape} {ref_arr.dtype} vs"
          f" {schedule_desc} with {cur_arr.shape} {cur_arr.dtype}),"
          " indicating schedule- or grid-order-dependent behavior"
      )
    if ref_arr.dtype == np.bool_:
      if not np.array_equal(ref_arr, cur_arr):
        mismatches = int(np.sum(ref_arr != cur_arr))
        return (
            f"output[{out_idx}] differs across schedules"
            f" ({baseline_schedule} vs {schedule_desc};"
            f" {mismatches} boolean element(s) differ),"
            " indicating schedule- or grid-order-dependent behavior"
        )
      continue
    if np.issubdtype(ref_arr.dtype, np.integer):
      if not np.array_equal(ref_arr, cur_arr):
        mismatches = int(np.sum(ref_arr != cur_arr))
        max_diff = int(
            np.max(np.abs(ref_arr.astype(object) - cur_arr.astype(object)))
        )
        return (
            f"output[{out_idx}] differs across schedules"
            f" ({baseline_schedule} vs {schedule_desc};"
            f" {mismatches} integer element(s) differ,"
            f" max_abs_diff={max_diff}), indicating schedule-"
            " or grid-order-dependent behavior"
        )
      continue
    if not np.allclose(
        ref_arr,
        cur_arr,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
      nan_mismatch = not np.array_equal(np.isnan(ref_arr), np.isnan(cur_arr))
      inf_mismatch = not np.array_equal(
          np.isinf(ref_arr), np.isinf(cur_arr)
      ) or not np.array_equal(
          ref_arr[np.isinf(ref_arr)], cur_arr[np.isinf(cur_arr)]
      )
      if nan_mismatch or inf_mismatch:
        diff_detail = "non-finite mismatch"
      else:
        promoted_dtype = (
            np.complex128 if np.iscomplexobj(ref_arr) else np.float64
        )
        ref_f64 = ref_arr.astype(promoted_dtype)
        cur_f64 = cur_arr.astype(promoted_dtype)
        finite_mask = np.isfinite(ref_f64) & np.isfinite(cur_f64)
        if finite_mask.any():
          max_diff = float(
              np.max(np.abs(ref_f64[finite_mask] - cur_f64[finite_mask]))
          )
          diff_detail = f"max_abs_diff={max_diff:.4e}"
        else:
          diff_detail = "non-finite mismatch"
      return (
          f"output[{out_idx}] differs across schedules"
          f" ({baseline_schedule} vs {schedule_desc};"
          f" {diff_detail}), indicating schedule-"
          " or grid-order-dependent behavior"
      )
  return None


def verify_with_cpu_interpret(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    config: InterpretSweepConfig = InterpretSweepConfig(),
) -> findings.VerificationReport:
  """Sweeps a single-chip or distributed Pallas op on CPU TPU interpret mode.

  Args:
    op_or_fn: Target `@jax_op` or raw JAX/Pallas callable to sweep.
    example_args: Positional example tensors, arrays, or `ShapeDtypeStruct`s.
    example_kwargs: Keyword arguments for `op_or_fn`. When None, defaults to an
      empty mapping.
    config: Sweep configuration controlling DMA schedule modes, grid seeds,
      uninitialized-memory poisoning, and comparison tolerances.

  Returns:
    A `VerificationReport` recording findings observed across the sweep.
  """
  kwargs = dict(example_kwargs or {})
  op_name = fx_verifier.get_op_display_name(op_or_fn)
  detected: list[findings.VerificationFinding] = []

  try:
    raw_fn, base_args, base_kwargs, cpu_mesh = _prepare_cpu_inputs(
        op_or_fn, example_args, kwargs
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
            severity=findings.Severity.WARNING,
            phase=_INTERPRET,
            op_name=op_name,
            message=f"skipped CPU interpret runs: {exc}",
        )
    )
    return findings.report_of(None, detected)

  cpu_device = None if cpu_mesh is not None else jax.devices("cpu")[0]
  poison_uninitialized = (
      config.uninitialized_memory is findings.UninitMemoryMode.NAN
  )
  baseline_schedule: str | None = None
  baseline_leaves: list[np.ndarray] | None = None
  completed_any_schedule = False
  reported_race = False
  reported_nonzero_sem = False
  reported_incomplete = False
  reported_exc_categories: set[findings.FindingCategory] = set()
  reported_nan = not poison_uninitialized
  reported_divergence = False

  def _record_race_flag(schedule_desc: str) -> None:
    nonlocal reported_race
    if not reported_race:
      reported_race = True
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.DATA_RACE,
              severity=findings.Severity.ERROR,
              phase=_INTERPRET,
              op_name=op_name,
              message=(
                  "vector-clock race detector flagged a data race under"
                  f" CPU TPU interpret ({schedule_desc})"
              ),
          )
      )

  def _record_nonzero_sems(sems: Sequence[str], schedule_desc: str) -> None:
    nonlocal reported_nonzero_sem
    if (
        sems
        and not reported_nonzero_sem
        and findings.FindingCategory.SEMAPHORE_NONZERO_EXIT
        not in reported_exc_categories
    ):
      reported_nonzero_sem = True
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.SEMAPHORE_NONZERO_EXIT,
              severity=findings.Severity.ERROR,
              phase=_INTERPRET,
              op_name=op_name,
              message=f"CPU TPU interpret ({schedule_desc}): {sems[0]}",
          )
      )

  def _record_exc_finding(finding: findings.VerificationFinding) -> None:
    nonlocal reported_nonzero_sem
    if finding.category in reported_exc_categories:
      return
    if (
        finding.category is findings.FindingCategory.SEMAPHORE_NONZERO_EXIT
        and reported_nonzero_sem
    ):
      return
    reported_exc_categories.add(finding.category)
    if finding.category is findings.FindingCategory.SEMAPHORE_NONZERO_EXIT:
      reported_nonzero_sem = True
    detected.append(finding)

  for mode in config.schedule_modes:
    checked_uninit_for_mode = False
    for seed in config.seeds:
      schedule_desc = f"dma={mode.value}, seed={seed}"
      schedule_params = dict(
          dma_execution_mode=(
              "eager" if mode is findings.DmaScheduleMode.EAGER else "on_wait"
          ),
          detect_races=True,
          on_race="raise",
          out_of_bounds_reads="raise",
          random_seed=seed,
      )
      params = make_interpret_params(
          uninitialized_memory=config.uninitialized_memory.value,
          **schedule_params,
      )
      outcome = _interpret_run(
          raw_fn, base_args, base_kwargs, params, cpu_device, cpu_mesh
      )
      if outcome.exc is not None:
        finding = _classify_interpret_exception(
            outcome.exc, op_name, schedule_desc
        )
        if finding.category is findings.FindingCategory.VERIFICATION_INCOMPLETE:
          if outcome.race_found:
            _record_race_flag(schedule_desc)
          _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)
          # Retry without `detect_races` in case the exception came from a
          # limitation in `race_detection_state` (such as strided slices) rather
          # than the kernel interpreter itself.
          fallback = _interpret_run(
              raw_fn,
              base_args,
              base_kwargs,
              make_interpret_params(
                  uninitialized_memory=config.uninitialized_memory.value,
                  **{**schedule_params, "detect_races": False},
              ),
              cpu_device,
              cpu_mesh,
          )
          _record_nonzero_sems(fallback.nonzero_sems, schedule_desc)
          if fallback.exc is None:
            assert fallback.leaves is not None
            np_leaves = fallback.leaves
            if not reported_incomplete:
              reported_incomplete = True
              detected.append(finding)
          else:
            fallback_finding = _classify_interpret_exception(
                fallback.exc, op_name, schedule_desc
            )
            if (
                fallback_finding.category
                is not findings.FindingCategory.VERIFICATION_INCOMPLETE
            ):
              _record_exc_finding(fallback_finding)
              continue
            if baseline_leaves is not None and not reported_divergence:
              reported_divergence = True
              detected.append(
                  findings.VerificationFinding(
                      category=findings.FindingCategory.SCHEDULE_DIVERGENCE,
                      severity=findings.Severity.ERROR,
                      phase=_INTERPRET,
                      op_name=op_name,
                      message=(
                          f"schedule ({schedule_desc}) raised"
                          f" {type(fallback.exc).__name__} after"
                          f" ({baseline_schedule}) succeeded, indicating"
                          " schedule- or grid-order-dependent behavior"
                      ),
                  )
              )
              continue
            return findings.report_of(
                _INTERPRET if (completed_any_schedule or detected) else None,
                (*detected, finding),
            )
        elif finding.category is findings.FindingCategory.DATA_RACE:
          if not reported_race:
            reported_race = True
            detected.append(finding)
          _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)
          # Re-run without `detect_races` so uninitialized-memory, semaphore,
          # and schedule-divergence checks still inspect this schedule.
          fallback = _interpret_run(
              raw_fn,
              base_args,
              base_kwargs,
              make_interpret_params(
                  uninitialized_memory=config.uninitialized_memory.value,
                  **{**schedule_params, "detect_races": False},
              ),
              cpu_device,
              cpu_mesh,
          )
          _record_nonzero_sems(fallback.nonzero_sems, schedule_desc)
          if fallback.exc is not None:
            fallback_finding = _classify_interpret_exception(
                fallback.exc, op_name, schedule_desc
            )
            if fallback_finding.category not in (
                findings.FindingCategory.DATA_RACE,
                findings.FindingCategory.VERIFICATION_INCOMPLETE,
            ):
              _record_exc_finding(fallback_finding)
            continue
          assert fallback.leaves is not None
          np_leaves = fallback.leaves
        else:
          if outcome.race_found:
            _record_race_flag(schedule_desc)
          _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)
          _record_exc_finding(finding)
          continue
      else:
        assert outcome.leaves is not None
        np_leaves = outcome.leaves
        if outcome.race_found:
          _record_race_flag(schedule_desc)
        _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)

      completed_any_schedule = True

      has_float_nan = any(
          np.issubdtype(a.dtype, np.inexact) and np.isnan(a).any()
          for a in np_leaves
      )
      if (
          not reported_nan
          and np_leaves
          and (not checked_uninit_for_mode or has_float_nan)
      ):
        checked_uninit_for_mode = True
        # Compare poisoned memory against zero-filled memory on the same
        # schedule so intentional kernel-computed values or input-propagated
        # NaNs are not misattributed to uninitialized memory.
        control_outcome = _interpret_run(
            raw_fn,
            base_args,
            base_kwargs,
            make_interpret_params(
                uninitialized_memory=findings.UninitMemoryMode.ZERO.value,
                **{**schedule_params, "detect_races": False},
            ),
            cpu_device,
            cpu_mesh,
        )
        control = (
            control_outcome.leaves if control_outcome.exc is None else None
        )
        message = (
            None
            if control is None
            else _uninitialized_memory_message(np_leaves, control)
        )
        if message is not None:
          reported_nan = True
          detected.append(
              findings.VerificationFinding(
                  category=findings.FindingCategory.NAN_OR_UNINIT_READ,
                  severity=findings.Severity.ERROR,
                  phase=_INTERPRET,
                  op_name=op_name,
                  message=f"{message} ({schedule_desc})",
              )
          )

      if baseline_leaves is None:
        baseline_leaves = np_leaves
        baseline_schedule = schedule_desc
      elif not reported_divergence:
        assert baseline_schedule is not None
        div_msg = _schedule_divergence_message(
            baseline_leaves,
            np_leaves,
            baseline_schedule,
            schedule_desc,
            rtol=config.rtol,
            atol=config.atol,
        )
        if div_msg is not None:
          reported_divergence = True
          detected.append(
              findings.VerificationFinding(
                  category=findings.FindingCategory.SCHEDULE_DIVERGENCE,
                  severity=findings.Severity.ERROR,
                  phase=_INTERPRET,
                  op_name=op_name,
                  message=div_msg,
              )
          )

  return findings.report_of(_INTERPRET, detected)
