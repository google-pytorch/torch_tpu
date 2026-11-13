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

* Enables `detect_races=True` (`on_race="raise"`) and inspects the interpreter's
  internal vector-clock `races` tracker after each schedule for dynamically
  observed data races (`DATA_RACE`) and its kernel-exit semaphore validation for
  non-zero semaphore counts (`SEMAPHORE_NONZERO_EXIT`).
* Runs with `uninitialized_memory="nan"` (which poisons VMEM scratch and block
  buffers at allocation time: quiet NaNs for floats, `iinfo.max` for integers,
  and `True` for booleans) and compares race-free runs against a
  `uninitialized_memory="zero"` control run so genuine uninitialized reads,
  unmasked padding, or first-visit unwritten output blocks
  (`NAN_OR_UNINIT_READ`) are distinguished from intentional kernel values
  (later-visit unwritten output blocks are checked statically in `MODEL_CHECK`).
* Compares output tensors across all `(DmaScheduleMode, seed)` combinations to
  detect schedule-dependent non-determinism (`SCHEDULE_DIVERGENCE`).
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
import contextlib
import dataclasses
import inspect
import threading
import time
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

# Shared with `fx_verifier` so autograd-parity and schedule-sweep CPU TPU
# interpret runs serialize access to JAX's process-global interpreter state.
_INTERPRET_LOCK = fx_verifier.INTERPRET_LOCK
_force_primitive_interpret_params = fx_verifier.force_primitive_interpret_params

# Per-schedule stall timeout: aborts only when a semaphore or barrier wait
# makes no forward progress across the interpreter for this duration.
_DEFAULT_INTERPRET_TIMEOUT_SECONDS = 10.0
_DEFAULT_INTERPRET_RUN_TIMEOUT_SECONDS = _DEFAULT_INTERPRET_TIMEOUT_SECONDS

_OPTIONAL_INTERPRET_PARAMS = frozenset({"on_race"})

_STANDARD_NUMPY_FLOAT_DTYPES = (
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
)

_MISSING = object()
_WATCHDOG_STATE = threading.local()


def make_interpret_params(**params: Any) -> pltpu.InterpretParams:
  """Builds `pltpu.InterpretParams`, dropping optional fields this JAX lacks.

  For example `on_race` only exists in newer JAX releases; without it races
  are still recorded and checked after the run.

  Args:
    **params: `pltpu.InterpretParams` fields.

  Returns:
    The interpret parameters.
  """
  supported = {
      field.name for field in dataclasses.fields(pltpu.InterpretParams)
  }
  unknown = set(params) - supported - _OPTIONAL_INTERPRET_PARAMS
  if unknown:
    raise TypeError(f"unexpected InterpretParams field(s): {sorted(unknown)!r}")
  return pltpu.InterpretParams(
      **{k: v for k, v in params.items() if k in supported}
  )


# The CPU interpreter of older JAX releases (0.9 for example) fails on DMAs and
# semaphore signals between devices and then fails every later interpret run
# in the process, so kernels with such operations only run on newer releases.
MIN_JAX_FOR_REMOTE_INTERPRET = (0, 10, 2)
MIN_JAX_FOR_REMOTE_INTERPRET_STR = ".".join(
    str(v) for v in MIN_JAX_FOR_REMOTE_INTERPRET
)


def remote_interpret_supported() -> bool:
  """Returns whether this JAX can interpret operations between devices."""
  return tuple(jax.__version_info__[:3]) >= MIN_JAX_FOR_REMOTE_INTERPRET


def _is_prng_key_dtype(dtype: Any) -> bool:
  """Returns whether `dtype` is a typed JAX PRNG key dtype."""
  return hasattr(jax.dtypes, "prng_key") and bool(
      jnp.issubdtype(dtype, jax.dtypes.prng_key)
  )


def _prepare_jax_arg_on_cpu(
    jax_arg: jax.Array | jax.ShapeDtypeStruct,
    cpu0: jax.Device,
    cpu_mesh: jax.sharding.Mesh | None,
    *,
    seed: int,
) -> tuple[jax.Array, bool]:
  """Materializes a concrete or abstract JAX input as a CPU `jax.Array`.

  Args:
    jax_arg: Concrete `jax.Array` or abstract `jax.ShapeDtypeStruct` input.
    cpu0: Host CPU device used to materialize uncommitted array data.
    cpu_mesh: Virtual CPU mesh used to re-shard `NamedSharding` inputs. When
      None, returns an uncommitted array on `cpu0` without mesh resharding.
    seed: Deterministic RNG seed for synthesizing abstract input values.

  Returns:
    A `(cpu_array, synthesized_int)` pair where `synthesized_int` is True when
    an abstract integer or boolean `ShapeDtypeStruct` had to be zero-filled.
  """
  arr, synthesized_int = _uncommitted_cpu_array(jax_arg, cpu0, seed=seed)
  sharding = getattr(jax_arg, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding) and cpu_mesh is not None:
    arr = jax.device_put(
        arr, jax.sharding.NamedSharding(cpu_mesh, sharding.spec)
    )
  return arr, synthesized_int


def _prepare_cpu_inputs(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
) -> tuple[
    Any,
    list[Any],
    dict[str, Any],
    jax.sharding.Mesh | None,
    bool,
]:
  """Prepares CPU-backed callable, arguments, and mesh for TPU interpret runs.

  Args:
    op_or_fn: Target `@jax_op`, `JaxCallable`, or raw JAX callable.
    example_args: Positional example tensors, arrays, or `ShapeDtypeStruct`s.
    example_kwargs: Keyword example arguments for `op_or_fn`.

  Returns:
    A 5-tuple `(raw_fn, jax_args, jax_kwargs, cpu_mesh,
    has_synthesized_int_inputs)` containing the unwrapped JAX function (rebound
    to `cpu_mesh` when applicable), converted positional CPU arguments,
    converted keyword CPU arguments, the active virtual CPU `Mesh` (or None for
    single-device callables without a mesh), and whether any abstract integer or
    boolean `ShapeDtypeStruct` input was synthesized with placeholder zeros.
  """
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
  has_synthesized_int_inputs = False

  def _is_data_leaf(x: Any) -> bool:
    return isinstance(x, (torch.Tensor, jax.Array, jax.ShapeDtypeStruct))

  def _is_spec_leaf(s: Any) -> bool:
    return s is None or isinstance(s, jax.sharding.PartitionSpec)

  def _convert_leaf(leaf: Any, leaf_spec: Any) -> Any:
    nonlocal seed_counter, has_synthesized_int_inputs
    if isinstance(leaf, torch.Tensor):
      if cpu_mesh is not None:
        return fx_verifier.build_sharded_cpu_array(
            leaf, cpu_mesh, leaf_spec, distinguish_ranks=True
        )
      np_val, jax_dtype = fx_verifier.tensor_to_numpy(leaf)
      return jax.device_put(jnp.asarray(np_val, dtype=jax_dtype), cpu0)
    if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)):
      cur_seed = seed_counter
      seed_counter += 1
      arr, synth_int = _prepare_jax_arg_on_cpu(
          leaf, cpu0, cpu_mesh, seed=cur_seed
      )
      if synth_int:
        has_synthesized_int_inputs = True
      return arr
    return leaf

  def _convert_tree(val: Any, spec: Any) -> Any:
    if _is_spec_leaf(spec):
      return jax.tree.map(
          lambda leaf: _convert_leaf(leaf, spec), val, is_leaf=_is_data_leaf
      )
    try:
      return jax.tree.map(
          lambda s, sub: jax.tree.map(
              lambda leaf: _convert_leaf(leaf, s), sub, is_leaf=_is_data_leaf
          ),
          spec,
          val,
          is_leaf=_is_spec_leaf,
      )
    except ValueError:
      if isinstance(val, (tuple, list)) and isinstance(spec, (tuple, list)):
        converted = [
            _convert_tree(elem, spec[j] if j < len(spec) else None)
            for j, elem in enumerate(val)
        ]
        if isinstance(val, tuple) and hasattr(val, "_fields"):
          return type(val)(*converted)
        return type(val)(converted)
      return jax.tree.map(
          lambda leaf: _convert_leaf(leaf, None), val, is_leaf=_is_data_leaf
      )

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

  return raw_fn, jax_args, jax_kwargs, cpu_mesh, has_synthesized_int_inputs


def _uncommitted_cpu_array(
    value: jax.Array | jax.ShapeDtypeStruct, cpu0: jax.Device, *, seed: int
) -> tuple[jax.Array, bool]:
  """Returns `value`, or example values for its shape, on the host CPU.

  The array is uncommitted (except for concrete typed PRNG keys placed directly
  on `cpu0`), so that `jax.jit` and `jax.shard_map` in the function under test
  may place it on any CPU devices.

  Args:
    value: A concrete array, or an abstract argument to fill with deterministic
      example values.
    cpu0: The CPU device to create the array on.
    seed: Seed of the example values.

  Returns:
    A `(cpu_array, synthesized_int)` pair where `synthesized_int` indicates
    whether `value` was an abstract integer or boolean `ShapeDtypeStruct`.
  """
  if isinstance(value, jax.Array):
    if _is_prng_key_dtype(value.dtype):
      return jax.device_put(value, cpu0), False
    host_value = np.asarray(value)
    synthesized_int = False
  elif _is_prng_key_dtype(value.dtype):
    with jax.default_device(cpu0):
      base_key = jax.random.key(seed)
      out_key = (
          jnp.broadcast_to(base_key, value.shape) if value.shape else base_key
      )
      return out_key, False
  else:
    rng = np.random.default_rng(seed)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
      host_value = rng.uniform(0.5, 1.5, value.shape) + 1j * rng.uniform(
          0.5, 1.5, value.shape
      )
      synthesized_int = False
    elif jnp.issubdtype(value.dtype, jnp.floating):
      # Positive values keep functions like log and sqrt finite.
      host_value = rng.uniform(0.5, 1.5, value.shape)
      synthesized_int = False
    else:
      host_value = np.zeros(value.shape)
      synthesized_int = True
  with jax.default_device(cpu0):
    return jnp.asarray(host_value, dtype=value.dtype), synthesized_int


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


def _format_exc_summary(exc: Exception) -> str:
  """Formats a compact single-line `ExceptionType: detail` summary."""
  exc_text = _extract_innermost_exception_text(str(exc).strip())
  last_line = exc_text.splitlines()[-1].strip() if exc_text else ""
  exc_name = type(exc).__name__
  if not last_line:
    return exc_name
  if last_line.startswith(f"{exc_name}:"):
    return last_line
  return f"{exc_name}: {last_line}"


def _is_unsupported_nan_poisoning_error(exc: Exception) -> bool:
  """Returns whether `exc` is unsupported `uninitialized_memory='nan'`."""
  exc_text = _extract_innermost_exception_text(str(exc))
  if isinstance(exc, NotImplementedError) and exc_text.startswith("nan + "):
    return True
  last_line = exc_text.splitlines()[-1].strip() if exc_text else ""
  return last_line.startswith("NotImplementedError: nan + ")


def classify_interpret_exception(
    exc: Exception,
    op_name: str,
    schedule_desc: str,
    *,
    has_revisit_any_output: bool = False,
    revisit_verified_full_overwrite: bool = False,
    phase: findings.Phase = _INTERPRET,
) -> findings.VerificationFinding:
  """Maps a CPU TPU interpret exception onto a structured `VerificationFinding`.

  Exceptions that do not name a kernel bug may come from features that the
  interpreter of the installed JAX does not support (for example, JAX 0.9
  fails multi-device remote DMAs with a KeyError), so they are reported as
  incomplete verification rather than as errors.

  Args:
    exc: The exception raised by the interpret run.
    op_name: Name of the op in findings.
    schedule_desc: Description of the schedule or check context that was run.
    has_revisit_any_output: Whether any extracted kernel model uses
      `pl.RevisitMode.ANY` on an output `BlockSpec` (unsupported by JAX's CPU
      TPU interpreter).
    revisit_verified_full_overwrite: Whether `Phase.MODEL_CHECK` already
      verified that revisited output blocks perform safe full-tile overwrites.
    phase: Verification `Phase` attributed to the returned finding.

  Returns:
    The classified `VerificationFinding`.
  """
  msg = str(exc).strip()
  exc_text = _extract_innermost_exception_text(msg)
  lower = exc_text.lower()
  if "race detected" in lower:
    category = findings.FindingCategory.DATA_RACE
  elif "out of bounds" in lower or "out-of-bounds" in lower:
    category = findings.FindingCategory.OUT_OF_BOUNDS_ACCESS
  elif "semaphore wait timed out" in lower:
    category = findings.FindingCategory.SEMAPHORE_DEADLOCK
  elif "semaphore" in lower and (
      "non-zero" in lower
      or "nonzero" in lower
      or "at kernel exit" in lower
      or "!= 0" in lower
  ):
    category = findings.FindingCategory.SEMAPHORE_NONZERO_EXIT
  elif "revisited block" in lower or "revisiting block" in lower:
    if has_revisit_any_output:
      return findings.VerificationFinding(
          category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
          severity=findings.Severity.INFO,
          phase=phase,
          op_name=op_name,
          message=(
              "skipped CPU interpret runs because JAX CPU TPU interpret does"
              f" not support RevisitMode.ANY ({schedule_desc})"
          ),
      )
    if revisit_verified_full_overwrite:
      return findings.VerificationFinding(
          category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
          severity=findings.Severity.INFO,
          phase=phase,
          op_name=op_name,
          message=(
              "skipped CPU interpret runs at non-consecutive output block"
              f" revisit already checked by model_check ({schedule_desc})"
          ),
      )
    category = findings.FindingCategory.HARDWARE_CONSTRAINT_VIOLATION
  elif "advanced indexers are not supported on tpu" in lower:
    category = findings.FindingCategory.HARDWARE_CONSTRAINT_VIOLATION
  elif (
      "unrecognized axes in device_id" in lower
      or "unsupported device id type" in lower
      or "semaphore_signal with device_id" in lower
  ):
    category = findings.FindingCategory.INVALID_DEVICE_ID
  else:
    exc_summary = _format_exc_summary(exc)
    prefix = (
        "skipped CPU interpret runs"
        if phase is _INTERPRET
        else f"skipped {phase.value} check"
    )
    return findings.VerificationFinding(
        category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=findings.Severity.WARNING,
        phase=phase,
        op_name=op_name,
        message=(
            f"{prefix} because the interpreter raised ({schedule_desc})"
            f" {exc_summary}; the kernel may be invalid or use features that"
            " the interpreter of this JAX version does not support"
        ),
    )
  return findings.VerificationFinding(
      category=category,
      severity=findings.Severity.ERROR,
      phase=phase,
      op_name=op_name,
      message=f"CPU TPU interpret ({schedule_desc}) raised: {exc_text or msg}",
  )


_classify_interpret_exception = classify_interpret_exception

# Check corrected spelling first, then upstream JAX's misspelled method name in
# `jax/_src/pallas/mosaic/interpret/shared_memory.py` (split across `+` so typo
# analyzers do not reject the intentional upstream spelling).
_SEM_NONZERO_ATTR_CANDIDATES = (
    "get_semaphores_with_nonzero_count",
    "get_sempa" + "hores_with_nonzero_count",
)

# `SharedMemory` buffer methods that constitute genuine forward progress during
# kernel execution (excluding `check_failed`, which a deadlocked
# `Semaphore.wait` polls every 0.1s).
_SHM_PROGRESS_METHODS = (
    "allocate_buffer",
    "deallocate_buffer",
    "get_buffer_content",
    "store_buffer_content",
    "swap_buffer_content",
)


@dataclasses.dataclass
class _WatchdogSession:
  """Mutable stall-tracking state shared between hooks and watchdog thread."""

  lock: threading.Lock
  cond: threading.Condition
  timeout_seconds: float
  clock: Callable[[], float]
  last_progress: float
  nonzero_sems: MutableSequence[str] | None = None
  stopped: bool = False
  active_sem_waits: int = 0
  wait_stall_since: float | None = None
  last_barrier_waiters: tuple[int, int] = (0, 0)
  aborted_shm_id: int | None = None
  timeout_exc: TimeoutError | None = None


@contextlib.contextmanager
def _watchdog_session_context(
    timeout_seconds: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    nonzero_sems: MutableSequence[str] | None = None,
) -> Iterator[_WatchdogSession]:
  """Acquires interpreter hooks and runs the stall watchdog (re-entrant)."""
  existing: _WatchdogSession | None = getattr(_WATCHDOG_STATE, "session", None)
  if existing is not None:
    with existing.lock:
      prev_timeout = existing.timeout_seconds
      prev_clock = existing.clock
      prev_nonzero = existing.nonzero_sems
      existing.timeout_seconds = timeout_seconds
      existing.clock = clock
      existing.nonzero_sems = nonzero_sems
      now = clock()
      existing.last_progress = now
      existing.wait_stall_since = now if existing.active_sem_waits > 0 else None
      existing.timeout_exc = None
      existing.cond.notify_all()
    try:
      yield existing
    finally:
      with existing.lock:
        existing.timeout_seconds = prev_timeout
        existing.clock = prev_clock
        existing.nonzero_sems = prev_nonzero
        existing.cond.notify_all()
    return

  with _INTERPRET_LOCK:
    interpret_mod = inspect.getmodule(pltpu.reset_tpu_interpret_mode_state)
    memory_mod = getattr(interpret_mod, "memory", None)
    shm_cls = getattr(memory_mod, "SharedMemory", None)
    sem_cls = getattr(memory_mod, "Semaphore", None)
    sem_nonzero_attr: str | None = None
    orig_get_nonzero: Any = None
    if shm_cls is not None:
      for candidate in _SEM_NONZERO_ATTR_CANDIDATES:
        fn = getattr(shm_cls, candidate, None)
        if callable(fn):
          sem_nonzero_attr = candidate
          orig_get_nonzero = fn
          break

    w_lock = threading.Lock()
    w_cond = threading.Condition(w_lock)
    session = _WatchdogSession(
        lock=w_lock,
        cond=w_cond,
        timeout_seconds=timeout_seconds,
        clock=clock,
        last_progress=clock(),
        nonzero_sems=nonzero_sems,
    )

    def _mark_progress() -> None:
      with session.lock:
        now = session.clock()
        session.last_progress = now
        if session.wait_stall_since is not None:
          session.wait_stall_since = now

    def _wrap_progress_method(orig_fn: Any) -> Any:
      def _wrapped(*m_args: Any, **m_kwargs: Any) -> Any:
        _mark_progress()
        try:
          return orig_fn(*m_args, **m_kwargs)
        finally:
          _mark_progress()

      return _wrapped

    def _watchdog_loop() -> None:
      while True:
        try:
          with session.lock:
            if session.stopped:
              return
            cur_timeout = session.timeout_seconds
            poll_interval = min(0.05, max(0.005, cur_timeout / 4.0))
            session.cond.wait(timeout=poll_interval)
            if session.stopped:
              return
            cur_timeout = session.timeout_seconds
            shm = getattr(interpret_mod, "_shared_memory", None)
            if shm is not None and id(shm) == session.aborted_shm_id:
              continue
            barrier_obj = (
                getattr(shm, "barrier", None) if shm is not None else None
            )
            cleanup_obj = (
                getattr(shm, "clean_up_barrier", None)
                if shm is not None
                else None
            )
            cur_barrier_waiters = (
                int(getattr(barrier_obj, "n_waiting", 0) or 0),
                int(getattr(cleanup_obj, "n_waiting", 0) or 0),
            )
            now = session.clock()
            if cur_barrier_waiters != session.last_barrier_waiters:
              session.last_barrier_waiters = cur_barrier_waiters
              session.last_progress = now
              if session.wait_stall_since is not None:
                session.wait_stall_since = now
            has_waiters = (
                session.active_sem_waits > 0
                or cur_barrier_waiters[0] > 0
                or cur_barrier_waiters[1] > 0
            )
            if not has_waiters or shm is None or not hasattr(shm, "set_failed"):
              session.wait_stall_since = None
              continue
            if session.wait_stall_since is None:
              session.wait_stall_since = now
            should_abort = (
                now - max(session.last_progress, session.wait_stall_since)
                >= cur_timeout
            )
            if not should_abort:
              continue
            timeout_exc = TimeoutError(
                "semaphore wait timed out in CPU TPU interpret mode (possible"
                " semaphore deadlock)"
            )
            session.timeout_exc = timeout_exc
            session.aborted_shm_id = id(shm)
            session.wait_stall_since = None
            # Pass `top_level=False` because `top_level=True` waits on
            # `clean_up_barrier` (`shared_memory.py`), which would block this
            # external watchdog thread instead of waking stalled device threads.
            shm.set_failed(timeout_exc, top_level=False)
            if (
                barrier_obj is not None
                and hasattr(barrier_obj, "abort")
                and cur_barrier_waiters[0] > 0
            ):
              barrier_obj.abort()
            # Only abort `clean_up_barrier` when no semaphore waiter remains;
            # waking an active semaphore waiter lets `fail_on_exception` call
            # `set_failed(top_level=True)` and satisfy `clean_up_barrier`
            # cleanly.
            if (
                cleanup_obj is not None
                and hasattr(cleanup_obj, "abort")
                and cur_barrier_waiters[1] > 0
                and session.active_sem_waits == 0
            ):
              cleanup_obj.abort()
            for sem_dict_attr in ("sem", "fixed_id_sem"):
              sem_dict = getattr(shm, sem_dict_attr, None)
              if isinstance(sem_dict, dict):
                for sem_obj in list(sem_dict.values()):
                  cv = getattr(sem_obj, "cv", None)
                  if cv is not None:
                    with cv:
                      cv.notify_all()
        except Exception:  # pylint: disable=broad-exception-caught
          # Never let an unexpected interpreter attribute mismatch silently
          # terminate the daemon watchdog thread mid-run.
          continue

    saved_attrs: list[tuple[Any, str, Any]] = []

    def _patch_cls_attr(target_cls: Any, attr_name: str, wrapper: Any) -> None:
      # Save via `__dict__` rather than `getattr` so methods inherited from
      # `GenericSharedMemory` are deleted rather than copied onto subclasses on
      # restore.
      orig_raw = target_cls.__dict__.get(attr_name, _MISSING)
      saved_attrs.append((target_cls, attr_name, orig_raw))
      setattr(target_cls, attr_name, wrapper)

    watchdog_thread: threading.Thread | None = None
    try:
      if shm_cls is not None and sem_nonzero_attr is not None:

        def _recording_get_nonzero(self_shm: Any, device_id: int) -> Any:
          nonzero_semaphores = orig_get_nonzero(self_shm, device_id)
          with session.lock:
            out_sems = session.nonzero_sems
          if out_sems is not None:
            for sem, global_core_index in nonzero_semaphores:
              count = sem.count_by_core[global_core_index]
              out_sems.append(
                  f"semaphore {sem.id} has non-zero count {count} for device"
                  f" {device_id} (global core {global_core_index}) at kernel"
                  " exit"
              )
          return nonzero_semaphores

        _patch_cls_attr(shm_cls, sem_nonzero_attr, _recording_get_nonzero)

      if shm_cls is not None:
        for method_name in _SHM_PROGRESS_METHODS:
          orig_method = getattr(shm_cls, method_name, None)
          if callable(orig_method):
            _patch_cls_attr(
                shm_cls, method_name, _wrap_progress_method(orig_method)
            )

      if sem_cls is not None:
        for method_name in ("signal", "enqueue_task"):
          orig_method = getattr(sem_cls, method_name, None)
          if callable(orig_method):
            _patch_cls_attr(
                sem_cls, method_name, _wrap_progress_method(orig_method)
            )
        orig_sem_wait = getattr(sem_cls, "wait", None)
        if callable(orig_sem_wait):

          def _tracked_sem_wait(
              self_sem: Any, *w_args: Any, **w_kwargs: Any
          ) -> Any:
            with session.lock:
              enter_now = session.clock()
              session.last_progress = enter_now
              session.active_sem_waits += 1
              session.wait_stall_since = enter_now
            try:
              return orig_sem_wait(self_sem, *w_args, **w_kwargs)
            finally:
              with session.lock:
                exit_now = session.clock()
                session.active_sem_waits = max(0, session.active_sem_waits - 1)
                session.last_progress = exit_now
                session.wait_stall_since = (
                    exit_now if session.active_sem_waits > 0 else None
                )

          _patch_cls_attr(sem_cls, "wait", _tracked_sem_wait)

      watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
      watchdog_thread.start()
      _WATCHDOG_STATE.session = session
      yield session
    finally:
      _WATCHDOG_STATE.session = None
      with session.lock:
        session.stopped = True
        session.cond.notify_all()
      if watchdog_thread is not None:
        watchdog_thread.join(timeout=1.0)
      for target_cls, attr_name, orig_raw in reversed(saved_attrs):
        if orig_raw is _MISSING:
          if attr_name in target_cls.__dict__:
            delattr(target_cls, attr_name)
        else:
          setattr(target_cls, attr_name, orig_raw)


@contextlib.contextmanager
def interpret_watchdog_context(
    timeout_seconds: float = _DEFAULT_INTERPRET_TIMEOUT_SECONDS,
) -> Iterator[None]:
  """Installs the CPU TPU-interpret semaphore/barrier stall watchdog.

  Acquires `INTERPRET_LOCK`, installs `SharedMemory` and `Semaphore` progress
  and wait wrappers, and runs a daemon watchdog thread that aborts stalled
  semaphore or barrier waits after `timeout_seconds` without forward progress.
  Safely re-entrant on the same thread when already active.

  Args:
    timeout_seconds: Maximum duration in seconds that a semaphore or barrier
      wait may stall without buffer or semaphore progress before being aborted
      with `TimeoutError`.

  Yields:
    None.
  """
  with _watchdog_session_context(
      timeout_seconds=timeout_seconds,
      clock=time.monotonic,
      nonzero_sems=None,
  ) as session:
    try:
      yield
    except Exception as exc:
      with session.lock:
        timeout_exc = session.timeout_exc
      if timeout_exc is not None and (
          isinstance(exc, threading.BrokenBarrierError)
          or "BrokenBarrierError" in str(exc)
      ):
        raise timeout_exc from exc
      raise


def _leaf_to_numpy(leaf: jax.Array) -> np.ndarray:
  """Converts a `jax.Array` output leaf into a comparable NumPy array."""
  if _is_prng_key_dtype(leaf.dtype):
    return np.asarray(jax.random.key_data(leaf))
  if (
      jnp.issubdtype(leaf.dtype, jnp.floating)
      and leaf.dtype not in _STANDARD_NUMPY_FLOAT_DTYPES
  ):
    return np.asarray(leaf.astype(jnp.float32))
  if jnp.issubdtype(leaf.dtype, jnp.integer) and not np.issubdtype(
      leaf.dtype, np.integer
  ):
    return np.asarray(leaf.astype(jnp.int32))
  return np.asarray(leaf)


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
    timeout_seconds: float = _DEFAULT_INTERPRET_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
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
    timeout_seconds: Stall timeout to abort semaphore or barrier waits that make
      no forward progress across the interpreter.
    clock: Monotonic clock callable returning seconds (injectable for
      deterministic watchdog testing).

  Returns:
    An `_InterpretRunOutcome` capturing output leaves (or the raised exception)
    along with any races or non-zero semaphores observed before state reset.
  """
  with _INTERPRET_LOCK:
    pltpu.reset_tpu_interpret_mode_state()
    interpret_mod = inspect.getmodule(pltpu.reset_tpu_interpret_mode_state)
    nonzero_sems: list[str] = []
    try:
      with _watchdog_session_context(
          timeout_seconds=timeout_seconds,
          clock=clock,
          nonzero_sems=nonzero_sems,
      ) as session:
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
            # Copy array inputs so `donate_argnums` on outer or nested `jax.jit`
            # wrappers cannot invalidate buffers reused by later sweep runs.
            run_args = jax.tree.map(
                lambda arg: (
                    jnp.copy(arg)
                    if isinstance(arg, jax.Array)
                    and not _is_prng_key_dtype(arg.dtype)
                    else arg
                ),
                list(args),
            )
            run_kwargs = jax.tree.map(
                lambda value: (
                    jnp.copy(value)
                    if isinstance(value, jax.Array)
                    and not _is_prng_key_dtype(value.dtype)
                    else value
                ),
                dict(kwargs),
            )
            out = raw_fn(*run_args, **run_kwargs)
            leaves = [
                _leaf_to_numpy(leaf)
                for leaf in jax.tree.leaves(out)
                if isinstance(leaf, jax.Array)
            ]
            jax.effects_barrier()
          except Exception as exc:  # pylint: disable=broad-exception-caught
            with session.lock:
              timeout_exc = session.timeout_exc
            run_exc = (
                timeout_exc
                if timeout_exc is not None
                and (
                    isinstance(exc, threading.BrokenBarrierError)
                    or "BrokenBarrierError" in str(exc)
                )
                else exc
            )
          with session.lock:
            timeout_exc = session.timeout_exc
          if run_exc is None and timeout_exc is not None:
            run_exc = timeout_exc
          races_state = getattr(interpret_mod, "races", None)
          race_found = races_state is not None and bool(races_state.races_found)
        return _InterpretRunOutcome(
            leaves=leaves,
            exc=run_exc,
            race_found=race_found,
            nonzero_sems=tuple(nonzero_sems),
        )
    finally:
      pltpu.reset_tpu_interpret_mode_state()


def _uninitialized_memory_message(
    leaves: Sequence[np.ndarray], control: Sequence[np.ndarray]
) -> str | None:
  """Describes output elements of `leaves` that zero-filled `control` avoids.

  Args:
    leaves: Output leaves from the NaN-poisoned interpret run.
    control: Output leaves from the zero-filled control interpret run.

  Returns:
    A diagnostic message describing memory-poisoning discrepancies, or None if
    all output elements match the zero-filled control run.
  """
  if len(leaves) != len(control):
    return (
        f"output array count changes ({len(leaves)} vs {len(control)}) when"
        " uninitialized memory is zero-filled instead of poisoned"
    )
  for out_idx, (arr, ctrl) in enumerate(zip(leaves, control, strict=True)):
    if arr.shape != ctrl.shape or arr.dtype != ctrl.dtype:
      continue
    if np.issubdtype(arr.dtype, np.inexact):
      both_nan = np.isnan(arr) & np.isnan(ctrl)
      extra = np.isnan(arr) & ~np.isnan(ctrl)
      if extra.any():
        return (
            f"output[{out_idx}] contains {int(extra.sum())}/{arr.size} NaN"
            " value(s) that disappear when uninitialized memory is zero-filled,"
            " so the kernel reads uninitialized VMEM or unmasked padding, or"
            " leaves output blocks unwritten"
        )
      diff = ~both_nan & (arr != ctrl)
    else:
      diff = arr != ctrl
    if diff.any():
      return (
          f"output[{out_idx}] has {int(diff.sum())}/{arr.size} element(s)"
          " that change when uninitialized memory is zero-filled instead of"
          " poisoned, so the kernel reads uninitialized VMEM or unmasked"
          " padding, or leaves output blocks unwritten"
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
  """Describes the first output discrepancy between two schedules, if any.

  Args:
    baseline_leaves: Output array leaves from the baseline schedule run.
    cur_leaves: Output array leaves from the current schedule run.
    baseline_schedule: Human-readable label of the baseline schedule.
    schedule_desc: Human-readable label of the current schedule.
    rtol: Relative tolerance for floating-point/complex comparisons.
    atol: Absolute tolerance for floating-point/complex comparisons.

  Returns:
    A diagnostic message describing the first mismatch across output leaves, or
    None when all output arrays match within the specified tolerances.
  """
  if len(baseline_leaves) != len(cur_leaves):
    return (
        f"output array count differs across schedules ({baseline_schedule}"
        f" produced {len(baseline_leaves)} array(s) vs {schedule_desc} with"
        f" {len(cur_leaves)}), indicating schedule-dependent behavior"
    )
  for out_idx, (ref_arr, cur_arr) in enumerate(
      zip(baseline_leaves, cur_leaves, strict=True)
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
    if np.issubdtype(ref_arr.dtype, np.integer) or jnp.issubdtype(
        ref_arr.dtype, jnp.integer
    ):
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
    config: InterpretSweepConfig | None = None,
    has_revisit_any_output: bool = False,
    revisit_verified_full_overwrite: bool = False,
) -> findings.VerificationReport:
  """Sweeps a single-chip or distributed Pallas op on CPU TPU interpret mode.

  Args:
    op_or_fn: Target `@jax_op` or raw JAX/Pallas callable to sweep.
    example_args: Positional example tensors, arrays, or `ShapeDtypeStruct`s.
    example_kwargs: Keyword arguments for `op_or_fn`. When None, defaults to an
      empty mapping.
    config: Sweep configuration controlling DMA schedule modes, grid seeds,
      uninitialized-memory poisoning, and comparison tolerances. When None,
      defaults to `InterpretSweepConfig()`.
    has_revisit_any_output: Whether any extracted kernel model uses
      `pl.RevisitMode.ANY` on an output `BlockSpec` (which JAX's CPU TPU
      interpreter does not support).
    revisit_verified_full_overwrite: Whether `Phase.MODEL_CHECK` already
      verified that revisited output blocks perform safe full-tile overwrites.

  Returns:
    A `VerificationReport` recording findings observed across the sweep.
  """
  if config is None:
    config = InterpretSweepConfig()
  kwargs = dict(example_kwargs or {})
  op_name = fx_verifier.get_op_display_name(op_or_fn)
  detected: list[findings.VerificationFinding] = []

  try:
    (
        raw_fn,
        base_args,
        base_kwargs,
        cpu_mesh,
        has_synthesized_int_inputs,
    ) = _prepare_cpu_inputs(op_or_fn, example_args, kwargs)
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
  effective_uninit_mode = config.uninitialized_memory
  poison_uninitialized = effective_uninit_mode is findings.UninitMemoryMode.NAN
  baseline_schedule: str | None = None
  baseline_leaves: list[np.ndarray] | None = None
  completed_any_schedule = False
  reported_race = False
  reported_nonzero_sem = False
  reported_incomplete = False
  reported_control_incomplete = False
  reported_exc_categories: set[findings.FindingCategory] = set()
  reported_nan = not poison_uninitialized
  reported_divergence = False
  deadlock_confirmed = False

  def _run_schedule(**run_params: Any) -> _InterpretRunOutcome:
    nonlocal effective_uninit_mode, reported_nan
    out = _interpret_run(
        raw_fn,
        base_args,
        base_kwargs,
        make_interpret_params(
            uninitialized_memory=effective_uninit_mode.value,
            **run_params,
        ),
        cpu_device,
        cpu_mesh,
    )
    if (
        out.exc is not None
        and effective_uninit_mode is findings.UninitMemoryMode.NAN
        and _is_unsupported_nan_poisoning_error(out.exc)
    ):
      exc_detail = _extract_innermost_exception_text(str(out.exc))
      effective_uninit_mode = findings.UninitMemoryMode.ZERO
      reported_nan = True
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
              severity=findings.Severity.INFO,
              phase=_INTERPRET,
              op_name=op_name,
              message=(
                  "fell back to zero-filled uninitialized memory in CPU TPU"
                  " interpret because this JAX version does not support NaN"
                  " memory poisoning for all kernel buffer dtypes"
                  f" ({exc_detail})"
              ),
          )
      )
      out = _interpret_run(
          raw_fn,
          base_args,
          base_kwargs,
          make_interpret_params(
              uninitialized_memory=effective_uninit_mode.value,
              **run_params,
          ),
          cpu_device,
          cpu_mesh,
      )
    return out

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

  def _classify_for_sweep(
      exc: Exception, schedule_desc: str
  ) -> findings.VerificationFinding:
    classified = classify_interpret_exception(
        exc,
        op_name,
        schedule_desc,
        has_revisit_any_output=has_revisit_any_output,
        revisit_verified_full_overwrite=revisit_verified_full_overwrite,
    )
    if (
        has_synthesized_int_inputs
        and classified.category is findings.FindingCategory.OUT_OF_BOUNDS_ACCESS
    ):
      return findings.VerificationFinding(
          category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
          severity=findings.Severity.INFO,
          phase=_INTERPRET,
          op_name=op_name,
          message=(
              "skipped CPU interpret bounds failure because an abstract"
              " integer or boolean ShapeDtypeStruct input was zero-filled;"
              " pass concrete jax.Array inputs to verify runtime bounds"
          ),
      )
    return classified

  for mode in config.schedule_modes:
    if deadlock_confirmed:
      break
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
      outcome = _run_schedule(**schedule_params)
      schedule_had_race = outcome.race_found
      if outcome.exc is not None:
        finding = _classify_for_sweep(outcome.exc, schedule_desc)
        if finding.category is findings.FindingCategory.VERIFICATION_INCOMPLETE:
          if finding.severity is findings.Severity.INFO:
            if not reported_incomplete:
              reported_incomplete = True
              detected.append(finding)
            deadlock_confirmed = True
            break
          if outcome.race_found:
            _record_race_flag(schedule_desc)
          _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)
          # Retry without `detect_races` in case the exception came from a
          # limitation in `race_detection_state` (such as strided slices) rather
          # than the kernel interpreter itself.
          fallback = _run_schedule(**{**schedule_params, "detect_races": False})
          _record_nonzero_sems(fallback.nonzero_sems, schedule_desc)
          if fallback.exc is None:
            assert fallback.leaves is not None
            np_leaves = fallback.leaves
            if not reported_incomplete:
              reported_incomplete = True
              detected.append(
                  findings.VerificationFinding(
                      category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                      severity=findings.Severity.WARNING,
                      phase=_INTERPRET,
                      op_name=op_name,
                      message=(
                          "skipped CPU interpret race detection on"
                          f" ({schedule_desc}) because detect_races=True raised"
                          f" {_format_exc_summary(outcome.exc)}"
                      ),
                  )
              )
          else:
            fallback_finding = _classify_for_sweep(fallback.exc, schedule_desc)
            if (
                fallback_finding.category
                is not findings.FindingCategory.VERIFICATION_INCOMPLETE
            ):
              _record_exc_finding(fallback_finding)
              if (
                  fallback_finding.category
                  is findings.FindingCategory.SEMAPHORE_DEADLOCK
              ):
                deadlock_confirmed = True
                break
              continue
            if not reported_incomplete:
              reported_incomplete = True
              detected.append(fallback_finding)
            if fallback_finding.severity is findings.Severity.INFO:
              deadlock_confirmed = True
              break
            continue
        elif finding.category is findings.FindingCategory.DATA_RACE:
          schedule_had_race = True
          if not reported_race:
            reported_race = True
            detected.append(finding)
          _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)
          # Re-run without `detect_races` so post-race bounds, semaphore, and
          # schedule-divergence checks still inspect this schedule.
          fallback = _run_schedule(**{**schedule_params, "detect_races": False})
          _record_nonzero_sems(fallback.nonzero_sems, schedule_desc)
          if fallback.exc is not None:
            fallback_finding = _classify_for_sweep(fallback.exc, schedule_desc)
            if (
                fallback_finding.category
                is findings.FindingCategory.VERIFICATION_INCOMPLETE
            ):
              if not reported_incomplete:
                reported_incomplete = True
                detected.append(fallback_finding)
              if fallback_finding.severity is findings.Severity.INFO:
                deadlock_confirmed = True
                break
            elif (
                fallback_finding.category
                is not findings.FindingCategory.DATA_RACE
            ):
              _record_exc_finding(fallback_finding)
              if (
                  fallback_finding.category
                  is findings.FindingCategory.SEMAPHORE_DEADLOCK
              ):
                deadlock_confirmed = True
                break
            continue
          assert fallback.leaves is not None
          np_leaves = fallback.leaves
        else:
          if outcome.race_found:
            _record_race_flag(schedule_desc)
          _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)
          _record_exc_finding(finding)
          if finding.category is findings.FindingCategory.SEMAPHORE_DEADLOCK:
            deadlock_confirmed = True
            break
          continue
      else:
        assert outcome.leaves is not None
        np_leaves = outcome.leaves
        if outcome.race_found:
          _record_race_flag(schedule_desc)
        _record_nonzero_sems(outcome.nonzero_sems, schedule_desc)

      completed_any_schedule = True

      if not reported_nan and not schedule_had_race and np_leaves:
        # Compare poisoned memory against zero-filled memory on the exact same
        # `(dma_execution_mode, random_seed)` schedule so intentional values,
        # input-propagated NaNs, and schedule-dependent int/bool/comparison
        # uninitialized reads on later seeds are all classified accurately.
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
        if control_outcome.exc is not None:
          if not reported_control_incomplete:
            reported_control_incomplete = True
            ctrl_summary = _format_exc_summary(control_outcome.exc)
            detected.append(
                findings.VerificationFinding(
                    category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                    severity=findings.Severity.WARNING,
                    phase=_INTERPRET,
                    op_name=op_name,
                    message=(
                        "could not compare poisoned memory against zero-filled"
                        f" control run ({schedule_desc}) because the control"
                        f" run raised {ctrl_summary}"
                    ),
                )
            )
        else:
          assert control_outcome.leaves is not None
          message = _uninitialized_memory_message(
              np_leaves, control_outcome.leaves
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

  return findings.report_of(
      _INTERPRET if completed_any_schedule else None, detected
  )
