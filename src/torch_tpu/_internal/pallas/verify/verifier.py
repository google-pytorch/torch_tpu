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

  from torch.tpu.pallas import verifier

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
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.pallas.verifier.VerificationMode is experimental and subject to"
    " change."
)
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


@annotations.experimental(
    "torch.tpu.pallas.verifier.VerificationOptions is experimental and subject"
    " to change."
)
@dataclasses.dataclass(frozen=True)
class VerificationOptions:
  """Advanced knobs for `verify`.

  Attributes:
    check_autograd: Whether to compare a registered custom backward against the
      forward kernel (FULL mode only).
    tpu_version: TPU generation used across contract, lowering, and
      model-checking phases. An explicit non-CPU TPU mesh on the kernel or its
      example arguments takes precedence (since TPU lowering targets that mesh's
      device kind); otherwise `tpu_version` is used when set, else `verify`
      infers the generation from an ambient mesh or `supported_on` predicates,
      falling back to `torch.tpu.TpuVersion.V7`.
  """

  check_autograd: bool = True
  tpu_version: findings.TpuVersion | None = None

  __tt_api_stages__ = {
      name: annotations.ApiStageInfo(
          stage=annotations.Stage.EXPERIMENTAL,
          reason=(
              f"torch.tpu.pallas.verifier.VerificationOptions.{name} is"
              " experimental and subject to change."
          ),
      )
      for name in (
          "check_autograd",
          "tpu_version",
      )
  }

  def __post_init__(self) -> None:
    if not isinstance(self.check_autograd, bool):
      raise TypeError(
          "check_autograd must be a bool, got"
          f" {type(self.check_autograd).__name__}"
      )
    if self.tpu_version is not None and not isinstance(
        self.tpu_version, findings.TpuVersion
    ):
      raise TypeError(
          "tpu_version must be a torch.tpu.TpuVersion or None, got"
          f" {type(self.tpu_version).__name__}"
      )


@annotations.forbid_in_torch_compile(
    "verification traces and runs kernels outside a compiled graph."
)
@annotations.experimental(
    "torch.tpu.pallas.verifier.verify is experimental and subject to change."
)
def verify(
    target: Callable[..., Any],
    args: Sequence[Any] | torch.Tensor | jax.Array | jax.ShapeDtypeStruct = (),
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
    The findings of all engines that ran. `report.passed` is `True` if and only
    if no error was found; `raise_for_errors()` raises with a formatted summary.

  Raises:
    TypeError: If `target` is not callable, `args` is not a sequence, or `mode`
      / `options` has an invalid type.
    ValueError: If a PyTorch program cannot be traced into an FX graph.
  """
  if not callable(target):
    raise TypeError(f"target must be callable, got {type(target).__name__}")
  if not isinstance(mode, VerificationMode):
    raise TypeError(
        f"mode must be a VerificationMode, got {type(mode).__name__}"
    )
  if options is not None and not isinstance(options, VerificationOptions):
    raise TypeError(
        f"options must be a VerificationOptions, got {type(options).__name__}"
    )
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
  if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
    return tuple(args)
  raise TypeError(
      f"args must be a sequence of example arguments, got {type(args).__name__}"
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
  effective_tpu_version = fx_verifier.infer_tpu_version(
      target, options.tpu_version, args, kwargs
  )
  is_op = fx_verifier.extract_jax_callable(target) is not None
  reports: list[findings.VerificationReport] = []
  if is_op:
    contracts = fx_verifier.verify_op_contracts(
        target, args, kwargs, tpu_version=effective_tpu_version
    )
    reports.append(contracts)
    if contracts.findings_by_category(
        findings.FindingCategory.PARTITION_SPEC_MISMATCH
    ):
      # The kernel cannot be traced with inconsistent shardings.
      return findings.merge_reports(*reports)
    reports.append(
        fx_verifier.verify_tpu_lowering(
            target, args, kwargs, tpu_version=effective_tpu_version
        )
    )

    if mode is VerificationMode.FULL and options.check_autograd:
      reports.append(fx_verifier.verify_autograd_parity(target, args, kwargs))
  return findings.merge_reports(*reports)


def _materialize_fake_arg(
    val: Any, placeholder_env: Mapping[fx.Node, Any]
) -> Any:
  """Turns an FX node's FakeTensor metadata into a concrete CPU tensor."""
  if isinstance(val, fx.Node):
    if val in placeholder_env:
      concrete = placeholder_env[val]
      if isinstance(concrete, torch.nn.Parameter):
        return torch.nn.Parameter(
            concrete.detach().cpu().clone(),
            requires_grad=concrete.requires_grad,
        )
      if isinstance(concrete, torch.Tensor):
        return concrete.detach().cpu().clone()
      return concrete
    meta_val = val.meta.get("val")
    if isinstance(meta_val, torch.Tensor):
      numel = int(meta_val.numel())
      if numel == 0:
        if meta_val.dtype in fx_verifier.FLOAT8_TORCH_DTYPES:
          return torch.zeros(
              meta_val.shape, dtype=torch.float32, device="cpu"
          ).to(meta_val.dtype)
        return torch.zeros(meta_val.shape, dtype=meta_val.dtype, device="cpu")
      if (
          meta_val.dtype in fx_verifier.FLOAT8_TORCH_DTYPES
          or meta_val.is_floating_point()
      ):
        return (
            torch.linspace(0.5, 1.5, numel, dtype=torch.float32, device="cpu")
            .reshape(meta_val.shape)
            .to(meta_val.dtype)
        )
      return torch.zeros(meta_val.shape, dtype=meta_val.dtype, device="cpu")
    return meta_val
  if isinstance(val, (tuple, list)):
    return type(val)(_materialize_fake_arg(x, placeholder_env) for x in val)
  if isinstance(val, Mapping):
    return {
        k: _materialize_fake_arg(v, placeholder_env) for k, v in val.items()
    }
  return val


def _signature(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return ("tensor", tuple(value.shape), value.dtype)
  if isinstance(value, (tuple, list)):
    return tuple(_signature(v) for v in value)
  if isinstance(value, Mapping):
    return tuple(sorted((str(k), _signature(v)) for k, v in value.items()))
  try:
    hash(value)
    return value
  except TypeError:
    return (type(value).__qualname__, repr(value))


def _verify_torch_program(
    target: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    mode: VerificationMode,
    options: VerificationOptions,
) -> findings.VerificationReport:
  """Verifies an FX-traceable PyTorch callable and the kernels it calls."""
  try:
    gm = fx_verifier.trace_to_fx_graph(
        target, args, kwargs, tpu_version=options.tpu_version
    )
  except Exception as exc:
    raise ValueError(
        f"could not trace {fx_verifier.get_op_display_name(target)} into an"
        f" FX graph ({type(exc).__name__}: {exc}); to verify a raw JAX"
        " function, pass jax.Array or jax.ShapeDtypeStruct arguments"
    ) from exc
  reports = [
      fx_verifier.verify_fx_graph(
          gm, args, kwargs, tpu_version=options.tpu_version
      )
  ]

  placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
  placeholder_env = fx_verifier.match_placeholders_to_inputs(
      placeholders, args, kwargs
  )

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
    mat_args = tuple(
        _materialize_fake_arg(a, placeholder_env) for a in node.args
    )
    mat_kwargs = {
        k: _materialize_fake_arg(v, placeholder_env)
        for k, v in node.kwargs.items()
    }
    key = (
        id(op_handle),
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
  return findings.merge_reports(*reports)
