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

"""CPU TPU-interpret sweeps over DMA schedules, seeds, and NaN-poisoned memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import inspect
from typing import Any

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier

_INTERPRET = findings.Phase.INTERPRET


@dataclasses.dataclass(frozen=True)
class InterpretSweepConfig:
  """Configuration for CPU TPU interpret schedule sweeping."""

  schedule_modes: tuple[findings.DmaScheduleMode, ...] = (
      findings.DmaScheduleMode.EAGER,
      findings.DmaScheduleMode.ON_WAIT,
  )
  seeds: tuple[int, ...] = (0, 17)
  uninitialized_memory: findings.UninitMemoryMode = (
      findings.UninitMemoryMode.NAN
  )
  rtol: float = 1e-4
  atol: float = 1e-4


def make_interpret_params(**params: Any) -> pltpu.InterpretParams:
  """Builds `pltpu.InterpretParams`, dropping fields this JAX lacks.

  For example `on_race` only exists in newer JAX releases; without it races
  are still recorded and checked after the run.

  Args:
    **params: `pltpu.InterpretParams` fields.

  Returns:
    The interpret parameters.
  """
  supported = {f.name for f in dataclasses.fields(pltpu.InterpretParams)}
  return pltpu.InterpretParams(
      **{k: v for k, v in params.items() if k in supported}
  )


# The CPU interpreter of older JAX releases (0.9 for example) fails on DMAs and
# semaphore signals between devices and then fails every later interpret run
# in the process, so kernels with such operations only run on newer releases.
_MIN_JAX_FOR_REMOTE_INTERPRET = (0, 11, 2)


def remote_interpret_supported() -> bool:
  """Returns whether this JAX can interpret operations between devices."""
  return tuple(jax.__version_info__[:3]) >= _MIN_JAX_FOR_REMOTE_INTERPRET


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
  """Prepares `(raw_fn, jax_args, jax_kwargs, cpu_mesh, input_had_nan)`."""
  jax_callable = fx_verifier.extract_jax_callable(op_or_fn)
  raw_fn, cpu_mesh = fx_verifier.unwrap_raw_jax_fn(
      op_or_fn, prefer_cpu_mesh=True
  )

  if jax_callable is not None:
    if jax_callable.mesh is not None and fx_verifier.is_mock_tpu_mesh(cpu_mesh):
      num = jax_callable.mesh.size
      raise ValueError(
          f"the {num}-device mesh needs {num} virtual CPU devices but only"
          f" {len(jax.devices('cpu'))} exist; before JAX initializes, set"
          f" XLA_FLAGS=--xla_force_host_platform_device_count={num} or call"
          f' jax.config.update("jax_num_cpu_devices", {num})'
      )
    # pylint: disable=protected-access
    norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
        example_args, dict(example_kwargs)
    )
    # pylint: enable=protected-access
    norm_args = fx_verifier.replace_mock_meshes_in_pytree(norm_args, cpu_mesh)
    norm_kwargs = fx_verifier.replace_mock_meshes_in_pytree(
        norm_kwargs, cpu_mesh
    )
    static_argnums = set(jax_callable.static_argnums)
    in_specs = jax_callable.input_partition_specs
  else:
    norm_args = list(example_args)
    norm_kwargs = dict(example_kwargs)
    static_argnums = set()
    in_specs = None

  cpu0 = jax.devices("cpu")[0]
  input_had_nan = False
  jax_args: list[Any] = []
  spec_idx = 0

  for i, arg in enumerate(norm_args):
    if i in static_argnums:
      jax_args.append(arg)
      continue
    spec = (
        in_specs[spec_idx]
        if in_specs is not None and spec_idx < len(in_specs)
        else None
    )
    spec_idx += 1
    if isinstance(arg, torch.Tensor):
      if arg.is_floating_point() and bool(torch.isnan(arg).any().item()):
        input_had_nan = True
      if cpu_mesh is not None:
        jax_args.append(
            fx_verifier.build_sharded_cpu_array(
                arg, cpu_mesh, spec, distinguish_ranks=True
            )
        )
      else:
        np_val, jax_dtype = fx_verifier.tensor_to_numpy(arg)
        jax_args.append(
            jax.device_put(jnp.asarray(np_val, dtype=jax_dtype), cpu0)
        )
    elif isinstance(arg, (jax.Array, jax.ShapeDtypeStruct)):
      if isinstance(arg, jax.Array):
        if jnp.issubdtype(arg.dtype, jnp.floating) and bool(
            jnp.isnan(arg).any()
        ):
          input_had_nan = True
      jax_args.append(_uncommitted_cpu_array(arg, cpu0, seed=i))
    else:
      jax_args.append(arg)

  jax_kwargs: dict[str, Any] = {}
  for k, val in norm_kwargs.items():
    if isinstance(val, torch.Tensor):
      if val.is_floating_point() and bool(torch.isnan(val).any().item()):
        input_had_nan = True
      if cpu_mesh is not None:
        jax_kwargs[k] = fx_verifier.build_sharded_cpu_array(
            val, cpu_mesh, None, distinguish_ranks=True
        )
      else:
        np_val, jax_dtype = fx_verifier.tensor_to_numpy(val)
        jax_kwargs[k] = jax.device_put(
            jnp.asarray(np_val, dtype=jax_dtype), cpu0
        )
    elif isinstance(val, (jax.Array, jax.ShapeDtypeStruct)):
      jax_kwargs[k] = _uncommitted_cpu_array(
          val, cpu0, seed=len(norm_args) + len(jax_kwargs)
      )
    else:
      jax_kwargs[k] = val

  return raw_fn, jax_args, jax_kwargs, cpu_mesh, input_had_nan


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
  else:
    rng = np.random.default_rng(seed)
    if jnp.issubdtype(value.dtype, jnp.floating):
      # Positive values keep functions like log and sqrt finite.
      host_value = rng.uniform(0.5, 1.5, value.shape)
    elif jnp.issubdtype(value.dtype, jnp.bool_):
      host_value = rng.integers(0, 2, value.shape).astype(np.bool_)
    else:
      host_value = rng.integers(0, 8, value.shape)
  with jax.default_device(cpu0):
    return jnp.asarray(host_value, dtype=value.dtype)


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
  lower = msg.lower()
  if "race detected" in lower:
    cat = findings.FindingCategory.DATA_RACE
  elif "out of bounds" in lower or "out-of-bounds" in lower:
    cat = findings.FindingCategory.OUT_OF_BOUNDS_ACCESS
  elif "semaphore" in lower and (
      "non-zero" in lower
      or "nonzero" in lower
      or "exit" in lower
      or "!= 0" in lower
  ):
    cat = findings.FindingCategory.SEMAPHORE_NONZERO_EXIT
  else:
    # Interpreter failures inside callbacks end with the original traceback.
    last_line = msg.splitlines()[-1] if msg else ""
    return findings.VerificationFinding(
        category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
        severity=findings.Severity.WARNING,
        phase=_INTERPRET,
        op_name=op_name,
        message=(
            "skipped CPU interpret runs because the interpreter raised"
            f" ({schedule_desc}) {type(exc).__name__}: {last_line}; the kernel"
            " may be invalid or use features that the interpreter of this JAX"
            " version does not support"
        ),
    )
  return findings.VerificationFinding(
      category=cat,
      severity=findings.Severity.ERROR,
      phase=_INTERPRET,
      op_name=op_name,
      message=f"CPU TPU interpret ({schedule_desc}) raised: {msg}",
  )


def _interpret_run(
    raw_fn: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    params: pltpu.InterpretParams,
    cpu_device: jax.Device | None,
) -> tuple[list[np.ndarray], bool]:
  """Runs `raw_fn` once in TPU interpret mode.

  Args:
    raw_fn: The JAX function to run.
    args: Positional arguments.
    kwargs: Keyword arguments.
    params: Interpret-mode parameters.
    cpu_device: Default device for single-device functions; None for functions
      that place their inputs on a CPU mesh themselves.

  Returns:
    The array outputs as NumPy arrays, and whether the race detector fired.
  """
  pltpu.reset_tpu_interpret_mode_state()
  dev_ctx = (
      contextlib.nullcontext()
      if cpu_device is None
      else jax.default_device(cpu_device)
  )
  try:
    # Copy array inputs so that kernels that write their inputs do not affect
    # later runs.
    run_args = [jnp.copy(a) if isinstance(a, jax.Array) else a for a in args]
    run_kwargs = {
        k: jnp.copy(v) if isinstance(v, jax.Array) else v
        for k, v in kwargs.items()
    }
    with dev_ctx, pltpu.force_tpu_interpret_mode(params):
      out = raw_fn(*run_args, **run_kwargs)
      leaves = [
          np.asarray(leaf, dtype=np.float32)
          if leaf.dtype == jnp.bfloat16
          else np.asarray(leaf)
          for leaf in jax.tree.leaves(out)
          if isinstance(leaf, jax.Array)
      ]
      jax.effects_barrier()
      interpret_mod = inspect.getmodule(pltpu.reset_tpu_interpret_mode_state)
      races_state = getattr(interpret_mod, "races", None)
      race_found = races_state is not None and bool(races_state.races_found)
    return leaves, race_found
  finally:
    pltpu.reset_tpu_interpret_mode_state()


def _uninitialized_nan_message(
    leaves: Sequence[np.ndarray], control: Sequence[np.ndarray]
) -> str | None:
  """Describes NaNs of `leaves` that zero-filled memory (`control`) avoids."""
  for out_idx, (arr, ctrl) in enumerate(zip(leaves, control)):
    if not np.issubdtype(arr.dtype, np.floating):
      continue
    extra = np.isnan(arr) & ~np.isnan(ctrl)
    if extra.any():
      return (
          f"output[{out_idx}] contains {int(extra.sum())}/{arr.size} NaN"
          " value(s) that disappear when uninitialized memory is zero-filled,"
          " so the kernel reads uninitialized VMEM or unmasked padding"
      )
  return None


def verify_with_cpu_interpret(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    config: InterpretSweepConfig = InterpretSweepConfig(),
) -> findings.VerificationReport:
  """Sweeps a single-chip or distributed Pallas op on CPU TPU interpret mode."""
  kwargs = dict(example_kwargs or {})
  op_name = fx_verifier.get_op_display_name(op_or_fn)
  detected: list[findings.VerificationFinding] = []

  try:
    raw_fn, base_args, base_kwargs, cpu_mesh, input_had_nan = (
        _prepare_cpu_inputs(op_or_fn, example_args, kwargs)
    )
  except ValueError as exc:
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
            severity=findings.Severity.WARNING,
            phase=_INTERPRET,
            op_name=op_name,
            message=f"skipped CPU interpret runs: {exc}",
        )
    )
    return findings.VerificationReport(findings=tuple(detected))

  cpu_device = None if cpu_mesh is not None else jax.devices("cpu")[0]
  poison_uninitialized = (
      config.uninitialized_memory is findings.UninitMemoryMode.NAN
  )
  baseline_schedule: str | None = None
  baseline_leaves: list[np.ndarray] | None = None
  reported_race = False
  reported_nan = input_had_nan or not poison_uninitialized
  reported_divergence = False

  for mode in config.schedule_modes:
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
      try:
        np_leaves, race_found = _interpret_run(
            raw_fn, base_args, base_kwargs, params, cpu_device
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        finding = _classify_interpret_exception(exc, op_name, schedule_desc)
        if finding.category is findings.FindingCategory.VERIFICATION_INCOMPLETE:
          # Every other schedule would fail the same way.
          return findings.VerificationReport(findings=(*detected, finding))
        if (
            finding.category is not findings.FindingCategory.DATA_RACE
            or not reported_race
        ):
          if finding.category is findings.FindingCategory.DATA_RACE:
            reported_race = True
          detected.append(finding)
        break

      if race_found and not reported_race:
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

      if not reported_nan and any(
          np.issubdtype(a.dtype, np.floating) and np.isnan(a).any()
          for a in np_leaves
      ):
        # NaNs can also come from the computation itself, so they are only
        # attributed to uninitialized memory if zero-filling it avoids them.
        try:
          control, _ = _interpret_run(
              raw_fn,
              base_args,
              base_kwargs,
              make_interpret_params(
                  uninitialized_memory=findings.UninitMemoryMode.ZERO.value,
                  **schedule_params,
              ),
              cpu_device,
          )
        except Exception:  # pylint: disable=broad-exception-caught
          control = None  # The NaNs cannot be attributed.
        message = (
            None
            if control is None
            else _uninitialized_nan_message(np_leaves, control)
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
        for out_idx, (ref_arr, cur_arr) in enumerate(
            zip(baseline_leaves, np_leaves)
        ):
          if not np.allclose(
              ref_arr,
              cur_arr,
              rtol=config.rtol,
              atol=config.atol,
              equal_nan=True,
          ):
            reported_divergence = True
            max_diff = float(np.nanmax(np.abs(ref_arr - cur_arr)))
            detected.append(
                findings.VerificationFinding(
                    category=findings.FindingCategory.SCHEDULE_DIVERGENCE,
                    severity=findings.Severity.ERROR,
                    phase=_INTERPRET,
                    op_name=op_name,
                    message=(
                        f"output[{out_idx}] differs across schedules"
                        f" ({baseline_schedule} vs {schedule_desc};"
                        f" max_abs_diff={max_diff:.4e}), indicating schedule-"
                        " or grid-order-dependent behavior"
                    ),
                )
            )
            break

  return findings.VerificationReport.of(_INTERPRET, detected)
