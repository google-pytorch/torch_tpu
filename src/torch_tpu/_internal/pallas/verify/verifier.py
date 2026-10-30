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

"""Single entry point for verifying Pallas kernels and PyTorch programs.

Example:

  from torch_tpu._internal.pallas.verify import verifier

  report = verifier.verify(my_jax_op, (x, y))
  report.raise_for_errors()

`verify` accepts a TorchTPU `@jax_op`, a raw JAX function (called with
`jax.Array` or `jax.ShapeDtypeStruct` arguments), or any PyTorch callable such
as an `nn.Module` that calls `@jax_op` kernels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import enum
from typing import Any

import jax
import torch
import torch.fx as fx
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier
from torch_tpu._internal.pallas.verify import interpret_verifier
from torch_tpu._internal.pallas.verify import smt_checker


class VerificationMode(enum.Enum):
  """Which verification engines `verify` runs."""

  # Static checks plus dynamic checks that run the kernels on CPU in TPU
  # interpret mode (schedule sweeps and autograd parity). Creating the JAX CPU
  # devices initializes the other JAX backends too, including TPU if available.
  FULL = "full"
  # Only checks that never execute the kernels: contracts, TPU lowering, FX
  # graph analysis, and model checking. They need no JAX devices, so they also
  # suit processes that use the TPU through TorchTPU.
  STATIC = "static"


@dataclasses.dataclass(frozen=True)
class VerificationOptions:
  """Advanced knobs for `verify`.

  Attributes:
    check_autograd: Whether to compare a registered custom backward against the
      forward kernel (FULL mode only).
    max_unrolled_iterations: Bound on unrolled grid points, loop iterations, and
      scan steps in the model checker. Anything beyond it is reported as
      incomplete.
    solver_timeout_seconds: Time budget for each model-checking query.
    tpu_device_kind: JAX device kind of the TPU generation that ops without a
      mesh are lowered for, such as "TPU v5 lite", "TPU v6 lite", or "TPU7x".
    interpret: Configuration of the CPU TPU-interpret schedule sweep.
  """

  check_autograd: bool = True
  max_unrolled_iterations: int = 32
  solver_timeout_seconds: float = 5.0
  tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND
  interpret: interpret_verifier.InterpretSweepConfig = dataclasses.field(
      default_factory=interpret_verifier.InterpretSweepConfig
  )


def verify(
    target: Callable[..., Any],
    args: Sequence[Any] | torch.Tensor | jax.Array = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    mode: VerificationMode = VerificationMode.FULL,
    options: VerificationOptions | None = None,
) -> findings.VerificationReport:
  """Verifies a Pallas kernel or a PyTorch program that calls Pallas kernels.

  The kind of `target` is detected automatically:

  * A TorchTPU `@jax_op` (or the `JaxCallable` behind it) gets op-contract,
    TPU-lowering, model-checking, autograd-parity, and CPU-interpret checks.
  * A JAX function called with `jax.Array` / `jax.ShapeDtypeStruct` arguments
    gets model-checking and CPU-interpret checks.
  * Any other callable (for example an `nn.Module`) is traced into an FX graph
    whose buffer donation and aliasing are checked; every `@jax_op` it calls
    is then verified for the argument shapes it is called with.

  Args:
    target: The op, JAX function, or PyTorch callable to verify.
    args: Example positional arguments, like `torch.library.opcheck`. A single
      tensor may be passed instead of a one-element tuple.
    kwargs: Example keyword arguments.
    mode: Which engines to run.
    options: Advanced configuration.

  Returns:
    The findings of all engines that ran. The report is truthy if and only if
    no error was found; `raise_for_errors()` raises with a formatted summary.

  Raises:
    TypeError: If `target` is not callable or `args` is not a sequence.
    ValueError: If a PyTorch program cannot be traced into an FX graph.
  """
  if not callable(target):
    raise TypeError(f"target must be callable, got {type(target).__name__}")
  call_args = _as_arg_tuple(args)
  call_kwargs = dict(kwargs or {})
  options = options or VerificationOptions()
  if fx_verifier.extract_jax_callable(target) is not None or _is_jax_program(
      target, call_args, call_kwargs
  ):
    return _verify_kernel(target, call_args, call_kwargs, mode, options)
  return _verify_torch_program(target, call_args, call_kwargs, mode, options)


def _as_arg_tuple(args: Any) -> tuple[Any, ...]:
  if isinstance(args, (torch.Tensor, jax.Array, jax.ShapeDtypeStruct)):
    return (args,)
  if isinstance(args, (tuple, list)):
    return tuple(args)
  raise TypeError(
      "args must be a tuple or list of example arguments, got"
      f" {type(args).__name__}"
  )


def _is_jax_program(
    target: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> bool:
  """Returns whether `target` should be traced with JAX rather than FX."""
  if hasattr(target, "lower") and hasattr(target, "trace"):
    return True  # A `jax.jit`-wrapped function.
  leaves = jax.tree.leaves((tuple(args), dict(kwargs)))
  return any(
      isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) for leaf in leaves
  )


def _verify_kernel(
    target: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    mode: VerificationMode,
    options: VerificationOptions,
) -> findings.VerificationReport:
  """Verifies a `@jax_op` or raw JAX function."""
  is_op = fx_verifier.extract_jax_callable(target) is not None
  reports: list[findings.VerificationReport] = []
  if is_op:
    contracts = fx_verifier.verify_op_contracts(target, args, kwargs)
    reports.append(contracts)
    if contracts.has_category(findings.FindingCategory.PARTITION_SPEC_MISMATCH):
      # The kernel cannot be traced with inconsistent shardings.
      return findings.VerificationReport.merge(*reports)
    reports.append(
        fx_verifier.verify_tpu_lowering(
            target, args, kwargs, tpu_device_kind=options.tpu_device_kind
        )
    )

  model_report, models = smt_checker.check_kernel_concurrency_with_models(
      target,
      args,
      kwargs,
      config=smt_checker.ModelCheckConfig(
          max_unrolled_iterations=options.max_unrolled_iterations,
          solver_timeout_seconds=options.solver_timeout_seconds,
      ),
  )
  reports.append(model_report)
  if mode is VerificationMode.FULL:
    # Both dynamic checks run the kernels in TPU interpret mode on CPU.
    # Kernels that could not be traced might use operations between devices.
    interpret_unsupported = (
        not interpret_verifier.remote_interpret_supported()
        and (models is None or any(m.uses_remote_operations for m in models))
    )
    unsupported_reason = (
        f"the CPU interpreter of JAX {jax.__version__} cannot run DMAs or"
        " semaphore signals between devices, which the kernel"
        f" {'may use' if models is None else 'uses'}; use JAX 0.11.2 or"
        " newer to run them"
    )
    if is_op and options.check_autograd:
      if interpret_unsupported:
        reports.append(
            _skipped_dynamic_check(
                target,
                findings.Phase.AUTOGRAD,
                findings.Severity.WARNING,
                "skipped the autograd parity check because"
                f" {unsupported_reason}",
            )
        )
      else:
        reports.append(fx_verifier.verify_autograd_parity(target, args, kwargs))
    if model_report.has_category(findings.FindingCategory.SEMAPHORE_DEADLOCK):
      reports.append(
          _skipped_dynamic_check(
              target,
              findings.Phase.INTERPRET,
              findings.Severity.INFO,
              "skipped CPU interpret runs because a semaphore deadlock was"
              " proven and they would hang",
          )
      )
    elif interpret_unsupported:
      reports.append(
          _skipped_dynamic_check(
              target,
              findings.Phase.INTERPRET,
              findings.Severity.WARNING,
              f"skipped CPU interpret runs because {unsupported_reason}",
          )
      )
    else:
      reports.append(
          interpret_verifier.verify_with_cpu_interpret(
              target, args, kwargs, config=options.interpret
          )
      )
  return findings.VerificationReport.merge(*reports)


def _skipped_dynamic_check(
    target: Any,
    phase: findings.Phase,
    severity: findings.Severity,
    message: str,
) -> findings.VerificationReport:
  """Returns a report noting that a dynamic check did not run."""
  return findings.VerificationReport(
      findings=(
          findings.VerificationFinding(
              category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
              severity=severity,
              phase=phase,
              op_name=fx_verifier.get_op_display_name(target),
              message=message,
          ),
      )
  )


def _materialize_fake_arg(val: Any) -> Any:
  """Turns an FX node's FakeTensor metadata into a concrete CPU tensor."""
  if isinstance(val, fx.Node):
    meta_val = val.meta.get("val")
    if isinstance(meta_val, torch.Tensor):
      if meta_val.is_floating_point():
        return torch.ones(meta_val.shape, dtype=meta_val.dtype, device="cpu")
      return torch.zeros(meta_val.shape, dtype=meta_val.dtype, device="cpu")
    return meta_val
  if isinstance(val, (tuple, list)):
    return type(val)(_materialize_fake_arg(x) for x in val)
  return val


def _signature(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return (tuple(value.shape), value.dtype)
  if isinstance(value, (tuple, list)):
    return tuple(_signature(v) for v in value)
  return value


def _verify_torch_program(
    target: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    mode: VerificationMode,
    options: VerificationOptions,
) -> findings.VerificationReport:
  """Verifies an FX-traceable PyTorch callable and the kernels it calls."""
  try:
    gm = fx_verifier.trace_to_fx_graph(target, args, kwargs)
  except Exception as exc:
    raise ValueError(
        f"could not trace {fx_verifier.get_op_display_name(target)} into an"
        f" FX graph ({type(exc).__name__}: {exc}); to verify a raw JAX"
        " function, pass jax.Array or jax.ShapeDtypeStruct arguments"
    ) from exc
  reports = [fx_verifier.verify_fx_graph(gm, args, kwargs)]

  seen_invocations: set[Any] = set()
  for node in gm.graph.nodes:
    if node.op != "call_function":
      continue
    if fx_verifier.extract_jax_callable(node.target) is None:
      continue
    op_def = fx_verifier.resolve_custom_op_def(node.target)
    op_handle = (
        op_def
        if op_def is not None
        else fx_verifier.extract_jax_callable(node.target)
    )
    mat_args = tuple(_materialize_fake_arg(a) for a in node.args)
    mat_kwargs = {k: _materialize_fake_arg(v) for k, v in node.kwargs.items()}
    key = (
        fx_verifier.get_op_display_name(op_handle),
        _signature(mat_args),
        tuple(sorted((k, _signature(v)) for k, v in mat_kwargs.items())),
    )
    if key in seen_invocations:
      continue
    seen_invocations.add(key)
    reports.append(
        _verify_kernel(op_handle, mat_args, mat_kwargs, mode, options)
    )
  return findings.VerificationReport.merge(*reports)
