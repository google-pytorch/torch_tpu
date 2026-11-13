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

from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
import dataclasses
import enum
import hashlib
import math
import operator
from typing import Any

import jax
import torch
import torch.fx as fx
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier
from torch_tpu._internal.pallas.verify import jaxpr_model
from torch_tpu._internal.pallas.verify import smt_checker
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.pallas.verifier.VerificationMode is experimental and subject to"
    " change."
)
class VerificationMode(enum.Enum):
  """Which verification engines `verify` runs."""

  # Static checks plus dynamic checks that run the kernels on CPU in TPU
  # interpret mode (such as autograd parity). Creating the JAX CPU devices
  # initializes the other JAX backends too, including TPU if available.
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
    max_unrolled_iterations: Bound on loop iterations, scan steps, and recorded
      event-trace grid points in the model checker. Anything beyond it is
      reported as incomplete.
    solver_timeout_seconds: Time budget for each model-checking query.
    tpu_version: TPU generation used across contract, lowering, and
      model-checking phases. An explicit non-CPU TPU mesh on the kernel or its
      example arguments takes precedence (since TPU lowering targets that mesh's
      device kind); otherwise `tpu_version` is used when set, else `verify`
      infers the generation from an ambient mesh or `supported_on` predicates,
      falling back to `torch.tpu.TpuVersion.V7`.
  """

  check_autograd: bool = True
  max_unrolled_iterations: int = jaxpr_model.DEFAULT_MAX_UNROLLED_ITERATIONS
  solver_timeout_seconds: float = smt_checker.DEFAULT_SOLVER_TIMEOUT_SECONDS
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
          "max_unrolled_iterations",
          "solver_timeout_seconds",
          "tpu_version",
      )
  }

  def __post_init__(self) -> None:
    if not isinstance(self.check_autograd, bool):
      raise TypeError(
          "check_autograd must be a bool, got"
          f" {type(self.check_autograd).__name__}"
      )
    if not isinstance(self.max_unrolled_iterations, int) or isinstance(
        self.max_unrolled_iterations, bool
    ):
      raise TypeError(
          "max_unrolled_iterations must be an int, got"
          f" {type(self.max_unrolled_iterations).__name__}"
      )
    if self.max_unrolled_iterations < 1:
      raise ValueError(
          "max_unrolled_iterations must be an integer >= 1, got"
          f" {self.max_unrolled_iterations!r}"
      )
    if not isinstance(self.solver_timeout_seconds, (int, float)) or isinstance(
        self.solver_timeout_seconds, bool
    ):
      raise TypeError(
          "solver_timeout_seconds must be a float or int, got"
          f" {type(self.solver_timeout_seconds).__name__}"
      )
    if (
        not math.isfinite(self.solver_timeout_seconds)
        or self.solver_timeout_seconds < 0
    ):
      raise ValueError(
          "solver_timeout_seconds must be a non-negative finite number, got"
          f" {self.solver_timeout_seconds!r}"
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
    TPU-lowering, model-checking, and autograd-parity checks.
  * A JAX function called with `jax.Array` / `jax.ShapeDtypeStruct` arguments
    gets TPU-lowering and model-checking checks.
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


def _drop_duplicate_trace_incomplete(
    model_report: findings.VerificationReport,
    lowering_report: findings.VerificationReport,
) -> findings.VerificationReport:
  """Drops fallback model-check trace warnings when lowering reported an error."""
  if not lowering_report.findings_by_category(
      findings.FindingCategory.LOWERING_FAILURE
  ):
    return model_report
  kept = [
      f
      for f in model_report.findings
      if not (
          f.category is findings.FindingCategory.VERIFICATION_INCOMPLETE
          and f.severity is findings.Severity.WARNING
      )
  ]
  if len(kept) == len(model_report.findings):
    return model_report
  return findings.VerificationReport(
      findings=tuple(kept), checked_phases=model_report.checked_phases
  )


def _verify_kernel(
    target: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None,
    mode: VerificationMode,
    options: VerificationOptions,
    *,
    concrete_arg_ids: AbstractSet[int] | None = None,
) -> findings.VerificationReport:
  """Verifies a `@jax_op` or raw JAX function."""
  mesh_tpu_version = fx_verifier.extract_explicit_mesh_tpu_version(
      target, args, kwargs
  )
  reports: list[findings.VerificationReport] = []
  if mesh_tpu_version is not None:
    effective_tpu_version = mesh_tpu_version
    if (
        options.tpu_version is not None
        and options.tpu_version is not mesh_tpu_version
    ):
      reports.append(
          findings.VerificationReport(
              findings=(
                  findings.VerificationFinding(
                      category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                      severity=findings.Severity.INFO,
                      phase=None,
                      op_name=fx_verifier.get_op_display_name(target),
                      message=(
                          f"explicit TPU mesh targets {mesh_tpu_version.name}"
                          " and overrides VerificationOptions.tpu_version="
                          f"{options.tpu_version.name}"
                      ),
                  ),
              )
          )
      )
  else:
    effective_tpu_version = fx_verifier.infer_tpu_version(
        target, options.tpu_version, args, kwargs
    )

  is_op = fx_verifier.extract_jax_callable(target) is not None
  lowering_report: findings.VerificationReport | None = None
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

  model_report, models = smt_checker.check_kernel_concurrency_with_models(
      target,
      args,
      kwargs,
      config=smt_checker.ModelCheckConfig(
          max_unrolled_iterations=options.max_unrolled_iterations,
          solver_timeout_seconds=options.solver_timeout_seconds,
          tpu_version=effective_tpu_version,
      ),
      concrete_arg_ids=concrete_arg_ids,
  )
  # If tracing failed in `Phase.MODEL_CHECK` and was already classified as a
  # specific bounds, device-id, or hardware-constraint error, avoid reporting
  # the same trace exception a second time as `LOWERING_FAILURE`.
  trace_error_already_classified = models is None and bool(
      model_report.findings_by_severity(findings.Severity.ERROR)
  )
  if not trace_error_already_classified and (
      is_op or models is None or bool(models)
  ):
    lowering_report = fx_verifier.verify_tpu_lowering(
        target, args, kwargs, tpu_version=effective_tpu_version
    )
    reports.append(lowering_report)
  if lowering_report is not None and models is None:
    model_report = _drop_duplicate_trace_incomplete(
        model_report, lowering_report
    )
  reports.append(model_report)
  if mode is VerificationMode.FULL and is_op and options.check_autograd:
    op_def = fx_verifier.resolve_custom_op_def(target)
    has_backward = (
        op_def is not None and getattr(op_def, "_backward_fn", None) is not None
    )
    if model_report.findings_by_category(
        findings.FindingCategory.SEMAPHORE_DEADLOCK
    ):
      if has_backward:
        reports.append(
            _skipped_dynamic_check(
                target,
                findings.Phase.AUTOGRAD,
                findings.Severity.INFO,
                "skipped the autograd parity check because a semaphore"
                " deadlock was proven and running the forward kernel would"
                " hang",
            )
        )
    elif model_report.findings_by_category(
        findings.FindingCategory.INVALID_DEVICE_ID
    ):
      if has_backward:
        reports.append(
            _skipped_dynamic_check(
                target,
                findings.Phase.AUTOGRAD,
                findings.Severity.INFO,
                "skipped the autograd parity check because an invalid remote"
                " device or core id was found in the kernel",
            )
        )
    elif models is not None and any(
        model.uses_done_count_regular_dma for model in models
    ):
      if has_backward:
        reports.append(
            _skipped_dynamic_check(
                target,
                findings.Phase.AUTOGRAD,
                findings.Severity.INFO,
                "skipped the autograd parity check because the CPU TPU"
                " interpreter increments DMA semaphores by byte count rather"
                " than transfer completion count on REGULAR semaphores",
            )
        )
    else:
      reports.append(fx_verifier.verify_autograd_parity(target, args, kwargs))
  return findings.merge_reports(*reports)


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
      ),
      checked_phases=(),
  )


def _clone_cpu_concrete_tensor(tensor: torch.Tensor) -> torch.Tensor:
  """Clones `tensor` onto CPU, handling packed `float4_e2m1fn_x2` safely."""
  detached = tensor.detach().cpu()
  if detached.dtype == torch.float4_e2m1fn_x2:
    return detached.view(torch.uint8).clone().view(torch.float4_e2m1fn_x2)
  return detached.clone()


def _materialize_fake_arg(
    val: Any,
    placeholder_env: Mapping[fx.Node, Any],
    *,
    concrete_tensor_ids: MutableSet[int] | None = None,
) -> Any:
  """Turns an FX node's FakeTensor metadata into a concrete CPU tensor."""
  if isinstance(val, fx.Node):
    if val in placeholder_env:
      concrete = placeholder_env[val]
      if isinstance(concrete, torch.nn.Parameter):
        out_p = torch.nn.Parameter(
            _clone_cpu_concrete_tensor(concrete),
            requires_grad=concrete.requires_grad,
        )
        if concrete_tensor_ids is not None:
          concrete_tensor_ids.add(id(out_p))
        return out_p
      if isinstance(concrete, torch.Tensor):
        out_t = _clone_cpu_concrete_tensor(concrete)
        if concrete_tensor_ids is not None:
          concrete_tensor_ids.add(id(out_t))
        return out_t
      return concrete
    meta_val = val.meta.get("val")
    if isinstance(meta_val, torch.Tensor):
      numel = int(meta_val.numel())
      if meta_val.dtype == torch.float4_e2m1fn_x2:
        fill_byte = 0x00 if numel == 0 else 0x22
        return torch.full(
            meta_val.shape, fill_byte, dtype=torch.uint8, device="cpu"
        ).view(torch.float4_e2m1fn_x2)
      if numel == 0:
        if meta_val.dtype in fx_verifier.FLOAT8_TORCH_DTYPES:
          return torch.zeros(
              meta_val.shape, dtype=torch.uint8, device="cpu"
          ).view(meta_val.dtype)
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
    converted = [
        _materialize_fake_arg(
            elem,
            placeholder_env,
            concrete_tensor_ids=concrete_tensor_ids,
        )
        for elem in val
    ]
    if isinstance(val, tuple) and hasattr(val, "_fields"):
      return type(val)(*converted)
    return type(val)(converted)
  if isinstance(val, Mapping):
    return {
        k: _materialize_fake_arg(
            v,
            placeholder_env,
            concrete_tensor_ids=concrete_tensor_ids,
        )
        for k, v in val.items()
    }
  return val


def _signature(
    value: Any,
    *,
    concrete_tensor_ids: AbstractSet[int] | None = None,
) -> Any:
  """Builds a hashable invocation deduplication key for `value`.

  Concrete CPU integer and boolean tensor values are included in the key (by
  element tuple up to 4096 elements, or BLAKE2b digest beyond) because Pallas
  `PrefetchScalarGridSpec` and dynamic slice indices specialize bounds and
  schedules on runtime integer values, whereas symbolic FakeTensor-synthesized
  tensors (`id(value) not in concrete_tensor_ids`) key only by shape and dtype.
  """
  if isinstance(value, torch.Tensor):
    if concrete_tensor_ids is not None and id(value) not in concrete_tensor_ids:
      return ("symbolic_tensor", tuple(value.shape), value.dtype)
    if (
        not fx_verifier.is_fake_tensor(value)
        and value.device.type == "cpu"
        and value.dtype in fx_verifier.INT_OR_BOOL_TORCH_DTYPES
    ):
      numel = int(value.numel())
      if numel <= 4096:
        return (
            "tensor",
            tuple(value.shape),
            value.dtype,
            tuple(value.detach().reshape(-1).tolist()),
        )
      contig = value.detach().contiguous()
      raw = contig.view(torch.uint8).numpy().tobytes()
      return (
          "tensor",
          tuple(value.shape),
          value.dtype,
          hashlib.blake2b(raw, digest_size=16).digest(),
      )
    return ("tensor", tuple(value.shape), value.dtype)
  if isinstance(value, (tuple, list)):
    return tuple(
        _signature(v, concrete_tensor_ids=concrete_tensor_ids) for v in value
    )
  if isinstance(value, Mapping):
    return tuple(
        sorted(
            (str(k), _signature(v, concrete_tensor_ids=concrete_tensor_ids))
            for k, v in value.items()
        )
    )
  try:
    hash(value)
    return value
  except TypeError:
    return (type(value).__qualname__, repr(value))


# Pure, deterministic integer/boolean/shape operations safe to evaluate on small
# CPU concrete inputs during static FX graph inspection (`empty*` and random ops
# are intentionally excluded so uninitialized or nondeterministic values are
# never fed into kernel verification).
_PURE_CONCRETE_OP_NAMES: frozenset[str] = frozenset({
    "__lshift__",
    "__rshift__",
    "_to_copy",
    "_unsafe_index",
    "_unsafe_view",
    "abs",
    "add",
    "alias",
    "all",
    "amax",
    "amin",
    "any",
    "arange",
    "as_tensor",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "bool",
    "broadcast_to",
    "byte",
    "cat",
    "char",
    "chunk",
    "clamp",
    "clamp_max",
    "clamp_min",
    "clone",
    "concat",
    "constant_pad_nd",
    "contiguous",
    "copy",
    "cumprod",
    "cumsum",
    "detach",
    "dim",
    "div",
    "eq",
    "expand",
    "expand_as",
    "eye",
    "flatten",
    "floor_divide",
    "fmod",
    "full",
    "full_like",
    "gather",
    "ge",
    "gt",
    "hstack",
    "index",
    "index_select",
    "int",
    "item",
    "le",
    "lift_fresh",
    "lift_fresh_copy",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "long",
    "lt",
    "max",
    "maximum",
    "min",
    "minimum",
    "moveaxis",
    "movedim",
    "mul",
    "narrow",
    "ne",
    "neg",
    "new_full",
    "new_ones",
    "new_zeros",
    "numel",
    "ones",
    "ones_like",
    "pad",
    "permute",
    "pow",
    "prod",
    "ravel",
    "remainder",
    "repeat",
    "repeat_interleave",
    "reshape",
    "reshape_as",
    "roll",
    "rsub",
    "scalar_tensor",
    "select",
    "short",
    "sign",
    "size",
    "slice",
    "slice_scatter",
    "split",
    "squeeze",
    "stack",
    "sub",
    "sum",
    "swapaxes",
    "swapdims",
    "t",
    "take",
    "tensor",
    "tile",
    "to",
    "transpose",
    "true_divide",
    "type_as",
    "unbind",
    "unflatten",
    "unsqueeze",
    "view",
    "view_as",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
})

_PURE_VIEW_METHOD_NAMES: frozenset[str] = frozenset({
    "_unsafe_view",
    "alias",
    "as_strided",
    "broadcast_to",
    "chunk",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "flatten",
    "getitem",
    "moveaxis",
    "movedim",
    "narrow",
    "permute",
    "ravel",
    "reshape",
    "reshape_as",
    "select",
    "slice",
    "split",
    "split_with_sizes",
    "squeeze",
    "swapaxes",
    "swapdims",
    "t",
    "transpose",
    "unbind",
    "unflatten",
    "unfold",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "view",
    "view_as",
})


def _is_small_int_or_bool_value(val: Any) -> bool:
  """Returns whether `val` is a small integer/bool tensor, scalar, or tuple."""
  if isinstance(val, (int, bool)):
    return True
  if isinstance(val, torch.Tensor):
    return (
        val.dtype in fx_verifier.INT_OR_BOOL_TORCH_DTYPES
        and int(val.numel()) <= 4096
    )
  if isinstance(val, (tuple, list)):
    return len(val) <= 64 and all(_is_small_int_or_bool_value(x) for x in val)
  return False


def _fx_node_base_op_name(node: fx.Node) -> str:
  if node.op == "call_method":
    return str(node.target)
  raw_name = getattr(node.target, "__name__", str(node.target))
  return str(raw_name).split(".")[0]


def _mutated_fx_input_nodes(node: fx.Node) -> list[fx.Node]:
  """Returns input `fx.Node`s mutated or donated by `node`."""
  mutated: list[fx.Node] = []
  if node.op == "call_function" and (
      node.target is operator.setitem
      or node.target in fx_verifier.INPLACE_OPERATOR_TARGETS
  ):
    if node.args:
      mutated.extend(fx_verifier.iter_fx_nodes(node.args[0]))
    return mutated
  if node.op == "call_function":
    jc = fx_verifier.extract_jax_callable(node.target)
    if jc is not None:
      # pylint: disable=protected-access
      param_name_to_idx = getattr(jc, "_param_name_to_idx", {}) or {}
      total_params = max(
          len(node.args) + len(node.kwargs),
          len(getattr(jc, "_orig_params", ()) or ()),
      )
      donated = fx_verifier._get_donated_arg_indices(
          jc, total_params, include_io_aliases=True
      )
      # pylint: enable=protected-access
      for idx in donated:
        if 0 <= idx < len(node.args):
          mutated.extend(fx_verifier.iter_fx_nodes(node.args[idx]))
      for kw_name, kw_val in node.kwargs.items():
        kw_idx = param_name_to_idx.get(kw_name)
        if kw_idx is not None and kw_idx in donated:
          mutated.extend(fx_verifier.iter_fx_nodes(kw_val))
      return mutated
  out_kw = node.kwargs.get("out")
  if out_kw is not None:
    mutated.extend(fx_verifier.iter_fx_nodes(out_kw))
  base_name = _fx_node_base_op_name(node)
  if node.op == "call_method":
    if base_name.endswith("_") and not (
        base_name.startswith("__") and base_name.endswith("__")
    ):
      if node.args and isinstance(node.args[0], fx.Node):
        mutated.append(node.args[0])
    return mutated
  schema = getattr(node.target, "_schema", None)
  if schema is not None and getattr(schema, "is_mutable", False):
    for idx, s_arg in enumerate(schema.arguments):
      alias_info = getattr(s_arg, "alias_info", None)
      if not getattr(alias_info, "is_write", False):
        continue
      if idx < len(node.args):
        mutated.extend(fx_verifier.iter_fx_nodes(node.args[idx]))
      elif s_arg.name in node.kwargs:
        mutated.extend(fx_verifier.iter_fx_nodes(node.kwargs[s_arg.name]))
    return mutated
  if (
      base_name.endswith("_")
      and not (base_name.startswith("__") and base_name.endswith("__"))
      and node.args
      and isinstance(node.args[0], fx.Node)
  ):
    mutated.append(node.args[0])
  return mutated


def _is_allowed_pure_concrete_op(node: fx.Node) -> bool:
  """Returns whether `node` is an allowlisted pure deterministic int/bool op."""
  base_name = _fx_node_base_op_name(node)
  if node.op == "call_method":
    return base_name in _PURE_CONCRETE_OP_NAMES
  if node.op != "call_function" or not callable(node.target):
    return False
  mod_name = getattr(node.target, "__module__", "") or ""
  if mod_name in ("_operator", "operator"):
    return base_name in {
        "abs",
        "add",
        "and_",
        "eq",
        "floordiv",
        "ge",
        "getitem",
        "gt",
        "index",
        "invert",
        "le",
        "lshift",
        "lt",
        "mod",
        "mul",
        "ne",
        "neg",
        "not_",
        "or_",
        "pos",
        "rshift",
        "sub",
        "truth",
        "xor",
    }
  if mod_name == "builtins":
    return base_name in {"abs", "bool", "int", "len", "max", "min"}
  schema = getattr(node.target, "_schema", None)
  if schema is not None:
    ns = str(getattr(schema, "name", "")).split("::", 1)[0]
    if ns in ("aten", "prim"):
      return base_name in _PURE_CONCRETE_OP_NAMES
    return False
  if mod_name.startswith("torch"):
    return base_name in _PURE_CONCRETE_OP_NAMES
  return False


def _has_non_cpu_device_arg(
    item: Any, *, is_device_kwarg: bool = False
) -> bool:
  """Returns True if `item` specifies a non-CPU `torch.device`."""
  if isinstance(item, torch.device):
    return item.type != "cpu"
  if is_device_kwarg and isinstance(item, str):
    return item != "cpu" and not item.startswith("cpu:")
  if isinstance(item, (tuple, list)):
    return any(_has_non_cpu_device_arg(x) for x in item)
  if isinstance(item, Mapping):
    return any(
        _has_non_cpu_device_arg(v, is_device_kwarg=(k == "device"))
        for k, v in item.items()
    )
  return False


def _meta_val_all_cpu(meta_val: Any) -> bool:
  """Returns True if all tensor leaves in `meta_val` reside on CPU."""
  if isinstance(meta_val, torch.Tensor):
    return meta_val.device.type == "cpu"
  if isinstance(meta_val, (tuple, list)):
    return all(_meta_val_all_cpu(x) for x in meta_val)
  return True


def _try_eval_small_concrete_node(
    node: fx.Node,
    env: Mapping[fx.Node, Any],
) -> tuple[bool, Any]:
  """Evaluates a small pure int/bool CPU op when all inputs are known."""
  if not _is_allowed_pure_concrete_op(node):
    return False, None
  meta_val = node.meta.get("val")
  if meta_val is None or not _is_small_int_or_bool_value(meta_val):
    return False, None
  if not _meta_val_all_cpu(meta_val):
    return False, None
  if _has_non_cpu_device_arg(node.args) or _has_non_cpu_device_arg(node.kwargs):
    return False, None

  missing = False

  def _resolve(item: Any) -> Any:
    nonlocal missing
    if isinstance(item, fx.Node):
      if item not in env:
        missing = True
        return None
      val = env[item]
      if isinstance(val, torch.Tensor):
        if (
            fx_verifier.is_fake_tensor(val)
            or val.device.type != "cpu"
            or int(val.numel()) > 4096
        ):
          missing = True
          return None
        return val.detach().clone()
      return val
    if isinstance(item, (tuple, list)):
      resolved_items = [_resolve(x) for x in item]
      if isinstance(item, tuple) and hasattr(item, "_fields"):
        return type(item)(*resolved_items)
      return type(item)(resolved_items)
    if isinstance(item, Mapping):
      return {k: _resolve(v) for k, v in item.items()}
    if isinstance(item, slice):
      return slice(
          _resolve(item.start), _resolve(item.stop), _resolve(item.step)
      )
    return item

  eval_args = _resolve(node.args)
  eval_kwargs = _resolve(node.kwargs)
  if (
      missing
      or _has_non_cpu_device_arg(eval_args)
      or _has_non_cpu_device_arg(eval_kwargs)
  ):
    return False, None

  try:
    with torch.no_grad():
      if node.op == "call_function" and callable(node.target):
        out = node.target(*eval_args, **eval_kwargs)
      elif node.op == "call_method" and eval_args:
        out = getattr(eval_args[0], str(node.target))(
            *eval_args[1:], **eval_kwargs
        )
      else:
        return False, None
  except Exception:  # pylint: disable=broad-exception-caught
    return False, None

  if not _is_small_int_or_bool_value(out):
    return False, None
  if isinstance(out, torch.Tensor) and (
      fx_verifier.is_fake_tensor(out) or out.device.type != "cpu"
  ):
    return False, None
  return True, out


def _seed_node_roots(node: fx.Node, val: Any) -> set[Any]:
  """Returns initial provenance tokens for `node` including CPU storage ptr."""
  tokens: set[Any] = {node}
  if (
      isinstance(val, torch.Tensor)
      and not fx_verifier.is_fake_tensor(val)
      and val.device.type == "cpu"
      and int(val.untyped_storage().nbytes()) > 0
  ):
    tokens.add(f"ptr:{int(val.untyped_storage().data_ptr())}")
  return tokens


def _invalidate_mutated_env_entries(
    env_map: MutableMapping[fx.Node, Any],
    roots_map: Mapping[fx.Node, AbstractSet[Any]],
    targets: Sequence[fx.Node],
) -> None:
  """Drops `targets` and any storage-aliased entries from `env_map`."""
  bad_roots: set[Any] = set()
  bad_ptrs: set[int] = set()
  for t_node in targets:
    bad_roots.update(roots_map.get(t_node, {t_node}))
    val = env_map.get(t_node)
    if (
        isinstance(val, torch.Tensor)
        and not fx_verifier.is_fake_tensor(val)
        and val.device.type == "cpu"
        and int(val.untyped_storage().nbytes()) > 0
    ):
      bad_ptrs.add(int(val.untyped_storage().data_ptr()))
  to_drop = [
      cand
      for cand, val in env_map.items()
      if (roots_map.get(cand, {cand}) & bad_roots)
      or (
          isinstance(val, torch.Tensor)
          and not fx_verifier.is_fake_tensor(val)
          and val.device.type == "cpu"
          and int(val.untyped_storage().nbytes()) > 0
          and int(val.untyped_storage().data_ptr()) in bad_ptrs
      )
  ]
  for cand in to_drop:
    env_map.pop(cand, None)


def _subgraph_mutated_placeholder_indices(sub_gm: fx.GraphModule) -> set[int]:
  """Returns placeholder indices whose storage is mutated in `sub_gm`."""
  phs = [n for n in sub_gm.graph.nodes if n.op == "placeholder"]
  ph_to_idx = {ph: idx for idx, ph in enumerate(phs)}
  roots: dict[fx.Node, set[int]] = {ph: {idx} for idx, ph in enumerate(phs)}
  mutated_indices: set[int] = set()
  attr_sub_gms: dict[fx.Node, fx.GraphModule] = {}

  for sub_node in sub_gm.graph.nodes:
    if sub_node.op == "get_attr":
      sub_attr = fx_verifier.resolve_fx_attr(sub_gm, str(sub_node.target))
      if isinstance(sub_attr, fx.GraphModule):
        attr_sub_gms[sub_node] = sub_attr
      continue
    if sub_node.op not in ("call_function", "call_method"):
      continue
    nested_gms: list[fx.GraphModule] = []
    if sub_node.op == "call_function":
      for a in (*sub_node.args, *sub_node.kwargs.values()):
        if isinstance(a, fx.Node) and a in attr_sub_gms:
          nested_gms.append(attr_sub_gms[a])
    if nested_gms:
      hop_kind = fx_verifier.classify_higher_order_op(sub_node.target)
      if hop_kind is None:
        # Unrecognized higher-order op: conservatively treat all FX operands
        # passed to the op as potentially mutated.
        for op_node in fx_verifier.iter_fx_nodes(
            (sub_node.args, sub_node.kwargs)
        ):
          if op_node not in attr_sub_gms:
            mutated_indices.update(roots.get(op_node, set()))
        continue
      for ngm in nested_gms:
        nphs = [n for n in ngm.graph.nodes if n.op == "placeholder"]
        nops = fx_verifier.extract_hop_operands(
            sub_node, attr_sub_gms, len(nphs)
        )
        nested_mut = _subgraph_mutated_placeholder_indices(ngm)
        for n_idx in nested_mut:
          if n_idx < len(nops) and isinstance(nops[n_idx], fx.Node):
            mutated_indices.update(roots.get(nops[n_idx], set()))
      continue

    # pylint: disable=protected-access
    if fx_verifier._is_set_inplace_fx_node(sub_node):
      # pylint: enable=protected-access
      dst_node = (
          sub_node.args[0]
          if sub_node.args and isinstance(sub_node.args[0], fx.Node)
          else None
      )
      src_raw = (
          sub_node.args[1]
          if len(sub_node.args) >= 2
          else sub_node.kwargs.get("source")
      )
      if dst_node is not None:
        if dst_node in ph_to_idx:
          mutated_indices.add(ph_to_idx[dst_node])
        mutated_indices.update(roots.get(dst_node, set()))
      src_roots = (
          set(roots.get(src_raw, set()))
          if isinstance(src_raw, fx.Node)
          else set()
      )
      if dst_node is not None:
        roots[dst_node] = set(src_roots)
      roots[sub_node] = set(src_roots)
      continue

    for m_in in _mutated_fx_input_nodes(sub_node):
      if m_in in ph_to_idx:
        mutated_indices.add(ph_to_idx[m_in])
      mutated_indices.update(roots.get(m_in, set()))
    aliased = (
        fx_verifier.fx_op_aliases_inputs(
            sub_node.target, sub_node.args, sub_node.kwargs
        )
        if sub_node.op == "call_function"
        else []
    )
    if (
        not aliased
        and _fx_node_base_op_name(sub_node) in _PURE_VIEW_METHOD_NAMES
    ):
      if sub_node.args and isinstance(sub_node.args[0], fx.Node):
        aliased = [sub_node.args[0]]
    if aliased:
      merged: set[int] = set()
      for src in aliased:
        merged.update(roots.get(src, set()))
      if merged:
        roots[sub_node] = merged
  return mutated_indices


def _shapes_match(ph: fx.Node, val: Any) -> bool:
  ph_meta = ph.meta.get("val")
  if isinstance(ph_meta, torch.Tensor) and isinstance(val, torch.Tensor):
    return (
        tuple(ph_meta.shape) == tuple(val.shape) and ph_meta.dtype == val.dtype
    )
  return True


def _collect_graph_module_envs(
    root_gm: fx.GraphModule,
    root_env: Mapping[fx.Node, Any],
) -> list[tuple[fx.GraphModule, dict[fx.Node, Any]]]:
  """Collects `root_gm` and any HOP child `GraphModule`s with concrete envs."""
  initial_env = {
      n: v
      for n, v in root_env.items()
      if not (isinstance(v, torch.Tensor) and fx_verifier.is_fake_tensor(v))
  }
  initial_roots: dict[fx.Node, set[Any]] = {
      n: _seed_node_roots(n, v) for n, v in initial_env.items()
  }
  worklist: list[
      tuple[fx.GraphModule, dict[fx.Node, Any], dict[fx.Node, set[Any]]]
  ] = [(root_gm, initial_env, initial_roots)]
  visited_gm_ids: set[int] = {id(root_gm)}
  collected: list[tuple[fx.GraphModule, dict[fx.Node, Any]]] = []

  idx = 0
  while idx < len(worklist):
    parent_gm, parent_env, parent_roots = worklist[idx]
    idx += 1
    collected.append((parent_gm, parent_env))
    for n, val in list(parent_env.items()):
      parent_roots.setdefault(n, _seed_node_roots(n, val))
    attr_to_sub_gm: dict[fx.Node, fx.GraphModule] = {}
    for node in parent_gm.graph.nodes:
      if node.op == "get_attr":
        sub = fx_verifier.resolve_fx_attr(parent_gm, str(node.target))
        if isinstance(sub, fx.GraphModule):
          attr_to_sub_gm[node] = sub
        elif (
            isinstance(sub, torch.Tensor)
            and not fx_verifier.is_fake_tensor(sub)
            and sub.device.type == "cpu"
        ):
          parent_env[node] = sub
          parent_roots[node] = _seed_node_roots(node, sub)
      elif node.op in ("call_function", "call_method"):
        child_gms: list[fx.GraphModule] = []
        if node.op == "call_function":
          for a in (*node.args, *node.kwargs.values()):
            if isinstance(a, fx.Node) and a in attr_to_sub_gm:
              child_gms.append(attr_to_sub_gm[a])
        if child_gms:
          hop_kind = fx_verifier.classify_higher_order_op(node.target)
          if hop_kind is None:
            operand_nodes = [
                n
                for n in fx_verifier.iter_fx_nodes((node.args, node.kwargs))
                if n not in attr_to_sub_gm
            ]
            if operand_nodes:
              _invalidate_mutated_env_entries(
                  parent_env, parent_roots, operand_nodes
              )
            for child_gm in child_gms:
              visited_gm_ids.add(id(child_gm))
              worklist.append((child_gm, {}, {}))
            continue

          is_cond = hop_kind is fx_verifier.HopKind.COND
          is_while = hop_kind is fx_verifier.HopKind.WHILE_LOOP
          is_scan = hop_kind is fx_verifier.HopKind.SCAN
          known_cond_branch: int | None = None
          if is_cond and node.args:
            pred_arg = node.args[0]
            pred_val = (
                parent_env.get(pred_arg)
                if isinstance(pred_arg, fx.Node)
                else pred_arg
            )
            if isinstance(pred_val, bool):
              known_cond_branch = 0 if pred_val else 1
            elif (
                isinstance(pred_val, torch.Tensor)
                and not fx_verifier.is_fake_tensor(pred_val)
                and int(pred_val.numel()) == 1
            ):
              known_cond_branch = 0 if bool(pred_val.item()) else 1

          all_hop_mutated_nodes: list[fx.Node] = []
          for branch_idx, child_gm in enumerate(child_gms):
            child_phs = [
                n for n in child_gm.graph.nodes if n.op == "placeholder"
            ]
            operands = fx_verifier.extract_hop_operands(
                node, attr_to_sub_gm, len(child_phs)
            )
            mutated_ph_idxs = _subgraph_mutated_placeholder_indices(child_gm)
            for m_idx in mutated_ph_idxs:
              if m_idx < len(operands) and isinstance(operands[m_idx], fx.Node):
                all_hop_mutated_nodes.append(operands[m_idx])

            child_env: dict[fx.Node, Any] = {}
            child_roots: dict[fx.Node, set[Any]] = {}
            if len(child_phs) == len(operands):
              invariant_start = 0
              if is_while:
                invariant_start = (
                    len(node.args[2])
                    if len(node.args) > 2
                    and isinstance(node.args[2], (tuple, list))
                    else len(operands)
                )
              elif is_scan:
                n_carry = (
                    len(node.args[1])
                    if len(node.args) > 1
                    and isinstance(node.args[1], (tuple, list))
                    else len(operands)
                )
                n_xs = (
                    len(node.args[2])
                    if len(node.args) > 2
                    and isinstance(node.args[2], (tuple, list))
                    else 0
                )
                invariant_start = n_carry + n_xs

              for ph_idx, (ph, op_arg) in enumerate(zip(child_phs, operands)):
                if ph_idx in mutated_ph_idxs or ph_idx < invariant_start:
                  continue
                if not isinstance(op_arg, fx.Node) or op_arg not in parent_env:
                  continue
                val = parent_env[op_arg]
                if not _shapes_match(ph, val):
                  continue
                if is_cond and _is_small_int_or_bool_value(val):
                  # Only specialize int/bool operands into a `cond` branch when
                  # `pred` is known and selects this branch; untaken or unknown
                  # branches may rely on `pred` as a bounds guard.
                  if (
                      known_cond_branch is None
                      or branch_idx != known_cond_branch
                  ):
                    continue
                child_env[ph] = val
                child_roots[ph] = _seed_node_roots(ph, val)

            visited_gm_ids.add(id(child_gm))
            worklist.append((child_gm, child_env, child_roots))

          if all_hop_mutated_nodes:
            _invalidate_mutated_env_entries(
                parent_env, parent_roots, all_hop_mutated_nodes
            )
          continue

        # pylint: disable=protected-access
        if fx_verifier._is_set_inplace_fx_node(node):
          # pylint: enable=protected-access
          dst_node = (
              node.args[0]
              if node.args and isinstance(node.args[0], fx.Node)
              else None
          )
          src_raw = (
              node.args[1] if len(node.args) >= 2 else node.kwargs.get("source")
          )
          if dst_node is not None:
            _invalidate_mutated_env_entries(
                parent_env, parent_roots, [dst_node]
            )
          src_roots: set[Any] = (
              set(parent_roots.get(src_raw, {src_raw}))
              if isinstance(src_raw, fx.Node)
              else {node}
          )
          if dst_node is not None:
            parent_roots[dst_node] = set(src_roots)
          parent_roots[node] = set(src_roots)
          if isinstance(src_raw, fx.Node) and src_raw in parent_env:
            if dst_node is not None:
              parent_env[dst_node] = parent_env[src_raw]
            parent_env[node] = parent_env[src_raw]
          continue

        mutated_inputs = _mutated_fx_input_nodes(node)
        if mutated_inputs:
          _invalidate_mutated_env_entries(
              parent_env, parent_roots, mutated_inputs
          )
          continue
        if (
            node.op == "call_function"
            and fx_verifier.extract_jax_callable(node.target) is not None
        ):
          continue

        aliased_srcs = (
            fx_verifier.fx_op_aliases_inputs(
                node.target, node.args, node.kwargs
            )
            if node.op == "call_function"
            else []
        )
        if (
            not aliased_srcs
            and _fx_node_base_op_name(node) in _PURE_VIEW_METHOD_NAMES
        ):
          if node.args and isinstance(node.args[0], fx.Node):
            aliased_srcs = [node.args[0]]
        if aliased_srcs:
          merged_roots: set[Any] = set()
          for src_node in aliased_srcs:
            merged_roots.update(parent_roots.get(src_node, {src_node}))
          parent_roots[node] = merged_roots

        ok, concrete_val = _try_eval_small_concrete_node(node, parent_env)
        if ok:
          parent_env[node] = concrete_val
          parent_roots.setdefault(node, _seed_node_roots(node, concrete_val))

  for mod in root_gm.modules():
    if isinstance(mod, fx.GraphModule) and id(mod) not in visited_gm_ids:
      visited_gm_ids.add(id(mod))
      collected.append((mod, {}))
  return collected


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
          gm,
          args,
          kwargs,
          tpu_version=options.tpu_version,
          target_callable=target,
      )
  ]

  placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
  placeholder_env = fx_verifier.match_placeholders_to_inputs(
      placeholders, args, kwargs
  )

  seen_invocations: set[Any] = set()
  for sub_gm, sub_env in _collect_graph_module_envs(gm, placeholder_env):
    for node in sub_gm.graph.nodes:
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
      concrete_ids: set[int] = set()
      mat_args = tuple(
          _materialize_fake_arg(
              a,
              sub_env,
              concrete_tensor_ids=concrete_ids,
          )
          for a in node.args
      )
      mat_kwargs = {
          k: _materialize_fake_arg(
              v,
              sub_env,
              concrete_tensor_ids=concrete_ids,
          )
          for k, v in node.kwargs.items()
      }
      key = (
          id(op_handle),
          fx_verifier.get_op_display_name(op_handle),
          _signature(mat_args, concrete_tensor_ids=concrete_ids),
          _signature(mat_kwargs, concrete_tensor_ids=concrete_ids),
      )
      if key in seen_invocations:
        continue
      seen_invocations.add(key)
      reports.append(
          _verify_kernel(
              op_handle,
              mat_args,
              mat_kwargs,
              mode,
              options,
              concrete_arg_ids=frozenset(concrete_ids),
          )
      )
  return findings.merge_reports(*reports)
