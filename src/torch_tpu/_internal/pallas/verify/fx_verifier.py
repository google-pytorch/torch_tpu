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

"""Checks of `@jax_op` contracts, TPU lowering, FX graphs, and autograd."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
import functools
import inspect
import operator
import types
from typing import Any

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch._library import custom_ops
import torch.fx as fx
from torch.fx.experimental import proxy_tensor
from torch.utils import _python_dispatch
from torch_tpu._internal.pallas import pallas as pallas_lib
from torch_tpu._internal.pallas.verify import findings

_CONTRACTS = findings.Phase.CONTRACTS
_FX_GRAPH = findings.Phase.FX_GRAPH
_AUTOGRAD = findings.Phase.AUTOGRAD
_LOWERING = findings.Phase.LOWERING
# Longest lowering error message kept in a finding; Mosaic errors can embed
# whole MLIR modules.
_MAX_LOWERING_MESSAGE_CHARS = 1200
# JAX device kind that ops without a mesh are lowered for by default.
DEFAULT_TPU_DEVICE_KIND = "TPU v5 lite"

_VIEW_FALLBACK_NAMES = frozenset({
    "alias",
    "as_strided",
    "chunk",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "flatten",
    "movedim",
    "narrow",
    "permute",
    "reshape",
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
    "unsqueeze",
    "view",
    "view_as",
    "_unsafe_view",
})


def resolve_custom_op_def(op_or_fn: Any) -> custom_ops.CustomOpDef | None:
  """Resolves a PyTorch CustomOpDef from an op handle, overload, or qualname."""
  if isinstance(op_or_fn, custom_ops.CustomOpDef):
    return op_or_fn
  if isinstance(op_or_fn, str):
    return custom_ops.OPDEFS.get(op_or_fn)
  schema = getattr(op_or_fn, "_schema", None)
  if schema is not None and hasattr(schema, "name"):
    qualname = str(schema.name).split(".", 1)[0]
    return custom_ops.OPDEFS.get(qualname)
  qualname = getattr(op_or_fn, "_qualified_op_name", None)
  if isinstance(qualname, str):
    return custom_ops.OPDEFS.get(qualname)
  return None


def extract_jax_callable(op_or_fn: Any) -> pallas_lib.JaxCallable | None:
  """Extracts the underlying TorchTPU JaxCallable if `op_or_fn` wraps one."""
  if isinstance(op_or_fn, pallas_lib.JaxCallable):
    return op_or_fn
  op_def = resolve_custom_op_def(op_or_fn)
  if op_def is not None and isinstance(op_def._init_fn, pallas_lib.JaxCallable):
    return op_def._init_fn
  return None


def get_op_display_name(op_or_fn: Any) -> str:
  """Returns a stable human-readable identifier for an op or function."""
  op_def = resolve_custom_op_def(op_or_fn)
  if op_def is not None:
    return op_def._qualname
  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is not None:
    return jax_callable.name
  name = getattr(op_or_fn, "__name__", None)
  if isinstance(name, str):
    return name
  if isinstance(op_or_fn, functools.partial):
    return get_op_display_name(op_or_fn.func)
  return type(op_or_fn).__name__


def is_mock_tpu_mesh(mesh: jax.sharding.Mesh | None) -> bool:
  """Returns True if `mesh` uses non-CPU (TPU or compile-only mock) devices."""
  if mesh is None or mesh.devices.size == 0:
    return False
  first_dev = mesh.devices.flat[0]
  return (
      not isinstance(first_dev, jax.Device)
      or getattr(first_dev, "platform", None) != "cpu"
  )


def make_cpu_mesh_for(mesh: jax.sharding.Mesh) -> jax.sharding.Mesh | None:
  """Builds a CPU `jax.sharding.Mesh` matching `mesh` if enough CPU devices exist."""
  cpu_devs = jax.devices("cpu")
  if len(cpu_devs) < mesh.size:
    return None
  reshaped = np.array(cpu_devs[: mesh.size]).reshape(mesh.devices.shape)
  return jax.sharding.Mesh(reshaped, mesh.axis_names)


def _replace_mesh_if_mock(
    val: Any, mesh_replacement: jax.sharding.Mesh
) -> tuple[Any, bool]:
  if isinstance(val, jax.sharding.Mesh) and is_mock_tpu_mesh(val):
    return mesh_replacement, True
  return val, False


def replace_mock_meshes_in_pytree(
    tree: Any, mesh_replacement: jax.sharding.Mesh | None
) -> Any:
  """Replaces non-CPU `jax.sharding.Mesh` leaves in `tree` with `mesh_replacement`."""
  if mesh_replacement is None:
    return tree
  return jax.tree.map(
      lambda leaf: mesh_replacement
      if isinstance(leaf, jax.sharding.Mesh) and is_mock_tpu_mesh(leaf)
      else leaf,
      tree,
  )


def rebind_function_meshes(
    fn: Callable[..., Any],
    mesh_replacement: jax.sharding.Mesh,
) -> Callable[..., Any]:
  """Replaces closed-over, default, or partial-bound non-CPU meshes with `mesh_replacement`."""
  if isinstance(fn, functools.partial):
    new_func = rebind_function_meshes(fn.func, mesh_replacement)
    new_args = tuple(
        _replace_mesh_if_mock(a, mesh_replacement)[0] for a in fn.args
    )
    new_keywords = {
        k: _replace_mesh_if_mock(v, mesh_replacement)[0]
        for k, v in (fn.keywords or {}).items()
    }
    return functools.partial(new_func, *new_args, **new_keywords)

  if (
      type(fn).__name__ == "HashablePartial"
      and hasattr(fn, "f")
      and hasattr(fn, "args")
      and hasattr(fn, "kwargs")
  ):
    inner_fn = getattr(fn, "f")
    inner_args = getattr(fn, "args")
    inner_kwargs = getattr(fn, "kwargs") or {}
    new_func = rebind_function_meshes(inner_fn, mesh_replacement)
    new_args = tuple(
        _replace_mesh_if_mock(a, mesh_replacement)[0] for a in inner_args
    )
    new_keywords = {
        k: _replace_mesh_if_mock(v, mesh_replacement)[0]
        for k, v in inner_kwargs.items()
    }
    return functools.partial(new_func, *new_args, **new_keywords)

  unwrapped = inspect.unwrap(fn)
  if unwrapped is not fn:
    return rebind_function_meshes(unwrapped, mesh_replacement)

  if isinstance(unwrapped, types.FunctionType):
    changed = False
    new_cells = None
    if unwrapped.__closure__:
      cells_list = []
      for cell in unwrapped.__closure__:
        replaced, did_change = _replace_mesh_if_mock(
            cell.cell_contents, mesh_replacement
        )
        if did_change:
          cells_list.append(types.CellType(replaced))
          changed = True
        else:
          cells_list.append(cell)
      new_cells = tuple(cells_list)

    new_defaults = unwrapped.__defaults__
    if unwrapped.__defaults__:
      defs_list = []
      for d in unwrapped.__defaults__:
        replaced, did_change = _replace_mesh_if_mock(d, mesh_replacement)
        defs_list.append(replaced)
        changed = changed or did_change
      new_defaults = tuple(defs_list)

    new_kwdefaults = unwrapped.__kwdefaults__
    if unwrapped.__kwdefaults__:
      kwdefs_dict = {}
      for k, d in unwrapped.__kwdefaults__.items():
        replaced, did_change = _replace_mesh_if_mock(d, mesh_replacement)
        kwdefs_dict[k] = replaced
        changed = changed or did_change
      new_kwdefaults = kwdefs_dict

    if changed:
      rebound = types.FunctionType(
          unwrapped.__code__,
          unwrapped.__globals__,
          unwrapped.__name__,
          new_defaults,
          new_cells,
      )
      rebound.__kwdefaults__ = new_kwdefaults
      rebound.__annotations__ = dict(unwrapped.__annotations__)
      return rebound
  return unwrapped


def unwrap_raw_jax_fn(
    op_or_fn: Any,
    *,
    prefer_cpu_mesh: bool = False,
) -> tuple[Callable[..., Any], jax.sharding.Mesh | None]:
  """Returns `(raw_jax_fn, active_mesh)` unwrapped past `jax.jit`."""
  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is not None:
    raw_fn = inspect.unwrap(jax_callable.jit_fn)
    mesh = jax_callable.mesh
    if prefer_cpu_mesh and mesh is not None:
      cpu_mesh = make_cpu_mesh_for(mesh)
      if cpu_mesh is not None:
        raw_fn = rebind_function_meshes(raw_fn, cpu_mesh)
        return raw_fn, cpu_mesh
    return raw_fn, mesh

  raw_fn = inspect.unwrap(op_or_fn)
  return raw_fn, None


def _tensor_storage_span(tensor: torch.Tensor) -> tuple[int, int, int] | None:
  """Returns `(storage_ptr, low_byte, high_byte)` for eager storage overlap checks."""
  if tensor.device.type == "meta":
    return None
  storage = tensor.untyped_storage()
  base_ptr = storage.data_ptr()
  if base_ptr == 0 or tensor.numel() == 0:
    return None
  elem_size = tensor.element_size()
  min_elem = tensor.storage_offset()
  max_elem = tensor.storage_offset()
  for size, stride in zip(tensor.shape, tensor.stride()):
    if size <= 1:
      continue
    delta = (size - 1) * stride
    if delta >= 0:
      max_elem += delta
    else:
      min_elem += delta
  return (base_ptr, min_elem * elem_size, (max_elem + 1) * elem_size)


def _eager_tensors_alias(a: torch.Tensor, b: torch.Tensor) -> bool:
  """Returns True if two eager tensors share overlapping underlying storage."""
  if a is b:
    return True
  span_a = _tensor_storage_span(a)
  span_b = _tensor_storage_span(b)
  if span_a is None or span_b is None:
    return False
  ptr_a, lo_a, hi_a = span_a
  ptr_b, lo_b, hi_b = span_b
  if ptr_a != ptr_b:
    return False
  return lo_a < hi_b and lo_b < hi_a


def _iter_spec_axis_names(spec: Any) -> list[str]:
  """Extracts all mesh axis names referenced inside a JAX PartitionSpec."""
  names: list[str] = []
  if spec is None:
    return names
  for entry in spec:
    if entry is None:
      continue
    if isinstance(entry, str):
      names.append(entry)
    elif isinstance(entry, (tuple, list)):
      for sub in entry:
        if isinstance(sub, str):
          names.append(sub)
  return names


def _collect_jax_callable_tensor_args(
    jax_callable: pallas_lib.JaxCallable,
    norm_args: Sequence[Any],
    norm_kwargs: Mapping[str, Any],
) -> list[tuple[int, str, torch.Tensor]]:
  """Collects `(param_idx, param_name, tensor)` for non-static tensor inputs."""
  collected: list[tuple[int, str, torch.Tensor]] = []
  # pylint: disable=protected-access
  for i, arg in enumerate(norm_args):
    param = (
        jax_callable._orig_params[i]
        if i < len(jax_callable._orig_params)
        and jax_callable._orig_params[i].kind
        != inspect.Parameter.VAR_POSITIONAL
        else None
    )
    param_name = param.name if param is not None else f"arg_{i}"
    is_static = i in jax_callable.static_argnums or (
        param is not None
        and (
            param.name in jax_callable._static_param_names
            or not jax_callable._param_is_tensor.get(param.name, False)
        )
    )
    if is_static:
      continue
    if isinstance(arg, torch.Tensor):
      collected.append((i, param_name, arg))
    elif isinstance(arg, (list, tuple)):
      for sub_idx, elem in enumerate(arg):
        if isinstance(elem, torch.Tensor):
          collected.append((i, f"{param_name}[{sub_idx}]", elem))

  for k, arg in norm_kwargs.items():
    param = jax_callable._orig_params_by_name.get(k)
    is_static = k in jax_callable._static_param_names or (
        param is not None and not jax_callable._param_is_tensor.get(k, False)
    )
    if is_static:
      continue
    param_idx = jax_callable._param_name_to_idx.get(k, -1)
    if isinstance(arg, torch.Tensor):
      collected.append((param_idx, k, arg))
    elif isinstance(arg, (list, tuple)):
      for sub_idx, elem in enumerate(arg):
        if isinstance(elem, torch.Tensor):
          collected.append((param_idx, f"{k}[{sub_idx}]", elem))
  # pylint: enable=protected-access
  return collected


def _validate_partition_spec_entry(
    spec: Any,
    ndim: int,
    mesh: jax.sharding.Mesh,
    label: str,
    op_name: str,
    *,
    global_shape: Sequence[int] | None = None,
) -> list[findings.VerificationFinding]:
  """Checks rank, axis existence, uniqueness, and divisibility of one spec."""
  detected: list[findings.VerificationFinding] = []
  if spec is None:
    return detected
  spec_tuple = tuple(spec)
  if len(spec_tuple) > ndim:
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=(
                f"{label} PartitionSpec {spec!r} has length {len(spec_tuple)},"
                f" exceeding tensor rank {ndim}"
            ),
        )
    )
    return detected

  axis_sizes = dict(zip(mesh.axis_names, mesh.devices.shape))
  seen_axes: set[str] = set()
  for dim_idx, entry in enumerate(spec_tuple):
    if entry is None:
      continue
    axes = (entry,) if isinstance(entry, str) else tuple(entry)
    shard_factor = 1
    for ax in axes:
      if ax not in axis_sizes:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"{label} PartitionSpec {spec!r} references unknown mesh"
                    f" axis {ax!r}; available mesh axes are"
                    f" {tuple(mesh.axis_names)}"
                ),
            )
        )
      else:
        shard_factor *= int(axis_sizes[ax])
      if ax in seen_axes:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"{label} PartitionSpec {spec!r} uses mesh axis {ax!r}"
                    " multiple times across tensor dimensions"
                ),
            )
        )
      seen_axes.add(ax)

    if (
        global_shape is not None
        and dim_idx < len(global_shape)
        and shard_factor > 1
        and isinstance(global_shape[dim_idx], int)
        and global_shape[dim_idx] % shard_factor != 0
    ):
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"{label} global dimension {dim_idx} of size"
                  f" {global_shape[dim_idx]} is not divisible by mesh shard"
                  f" factor {shard_factor} for PartitionSpec {spec!r}"
              ),
          )
      )
  return detected


def verify_op_contracts(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
) -> findings.VerificationReport:
  """Checks static and eager contracts on a TorchTPU Pallas custom op."""
  kwargs = dict(example_kwargs or {})
  op_name = get_op_display_name(op_or_fn)
  detected: list[findings.VerificationFinding] = []

  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is None:
    return findings.VerificationReport()

  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      example_args, kwargs
  )
  donated_set = set(jax_callable._donate_argnums_set) | set(
      jax_callable.input_output_aliases.keys()
  )
  # pylint: enable=protected-access

  tensor_entries = _collect_jax_callable_tensor_args(
      jax_callable, norm_args, norm_kwargs
  )

  # 1. Check eager storage aliasing involving donated arguments.
  for idx_a in range(len(tensor_entries)):
    p_a, name_a, ten_a = tensor_entries[idx_a]
    for idx_b in range(idx_a + 1, len(tensor_entries)):
      p_b, name_b, ten_b = tensor_entries[idx_b]
      if (p_a in donated_set or p_b in donated_set) and _eager_tensors_alias(
          ten_a, ten_b
      ):
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.ALIASED_DONOR_INPUTS,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"inputs {name_a!r} (arg {p_a}) and {name_b!r} (arg {p_b})"
                    " alias the same storage while at least one is donated via"
                    f" donate_argnums={tuple(sorted(donated_set))}"
                ),
            )
        )

  # Donated buffers are reused or mutated by the kernel, which would corrupt
  # model parameters that later steps still read.
  for param_idx, name, tensor in tensor_entries:
    if param_idx in donated_set and isinstance(tensor, torch.nn.Parameter):
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.DONATED_PARAMETER,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"input {name!r} (arg {param_idx}) is an nn.Parameter but is"
                  " donated; the kernel may reuse its buffer and corrupt the"
                  " parameter, so pass a clone() or stop donating it"
              ),
          )
      )

  # 2. Check mesh device uniqueness & input/output PartitionSpec contracts.
  mesh = jax_callable.mesh
  in_specs = jax_callable.input_partition_specs
  out_specs = jax_callable.output_partition_specs

  if mesh is None and (in_specs is not None or out_specs is not None):
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_CONTRACTS,
            op_name=op_name,
            message=(
                "partition specs were provided without a `mesh` on the Pallas"
                " op"
            ),
        )
    )

  mesh_ok = True
  if mesh is not None:
    device_ids = [
        int(getattr(d, "id", idx)) for idx, d in enumerate(mesh.devices.flat)
    ]
    if len(set(device_ids)) != len(device_ids):
      mesh_ok = False
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_CONTRACTS,
              op_name=op_name,
              message=(
                  f"mesh contains duplicate device IDs {device_ids}; every"
                  " mesh entry must map to a distinct physical device"
              ),
          )
      )

    if in_specs is not None:
      non_static_top_args = [
          (i, a)
          for i, a in enumerate(norm_args)
          if i not in jax_callable.static_argnums
      ]
      if len(in_specs) != len(non_static_top_args):
        mesh_ok = False
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    f"input_partition_specs has length {len(in_specs)}, but op"
                    f" takes {len(non_static_top_args)} tensor argument(s)"
                ),
            )
        )
      else:
        for (_, arg_val), spec in zip(non_static_top_args, in_specs):
          if isinstance(arg_val, torch.Tensor):
            spec_errs = _validate_partition_spec_entry(
                spec, arg_val.ndim, mesh, "input", op_name
            )
            if spec_errs:
              mesh_ok = False
              detected.extend(spec_errs)

  # 3. Check zero-output silent skip & output PartitionSpec shapes.
  if mesh_ok:
    try:
      # Shape inference only traces, so it uses the op's own mesh and needs no
      # JAX devices.
      raw_fn, eval_mesh = unwrap_raw_jax_fn(jax_callable)
      dyn_phs: list[Any] = []
      pos_slots: list[tuple[bool, Any]] = []
      spec_idx = 0
      for idx, val in enumerate(norm_args):
        if idx in jax_callable.static_argnums or not isinstance(
            val, torch.Tensor
        ):
          pos_slots.append((True, val))
        else:
          spec = (
              in_specs[spec_idx]
              if in_specs is not None and spec_idx < len(in_specs)
              else None
          )
          spec_idx += 1
          pos_slots.append((False, len(dyn_phs)))
          dyn_phs.append(
              pallas_lib.jax_placeholder(
                  val, mesh=eval_mesh, partition_spec=spec
              )
          )
      kw_slots: dict[str, tuple[bool, Any]] = {}
      for k, val in norm_kwargs.items():
        if isinstance(val, torch.Tensor):
          kw_slots[k] = (False, len(dyn_phs))
          dyn_phs.append(pallas_lib.jax_placeholder(val, mesh=eval_mesh))
        else:
          kw_slots[k] = (True, val)

      def _shape_fn(*phs):
        c_args = [v if is_s else phs[v] for is_s, v in pos_slots]
        c_kwargs = {
            k: v if is_s else phs[v] for k, (is_s, v) in kw_slots.items()
        }
        return raw_fn(*c_args, **c_kwargs)

      out_struct = jax.eval_shape(_shape_fn, *dyn_phs)
      out_avals = [
          leaf
          for leaf in jax.tree.leaves(out_struct)
          if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
      ]
      if not out_avals:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.ZERO_OUTPUT_SILENT_SKIP,
                severity=findings.Severity.ERROR,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    "jax_op returns zero tensor outputs; PyCallCustomKernel in"
                    " TorchTPU silently skips execution when output_shapes is"
                    " empty, dropping any side effects or barriers"
                ),
            )
        )
      elif mesh is not None and out_specs is not None:
        # pylint: disable=protected-access
        norm_out_specs = pallas_lib._normalize_output_partition_specs(
            out_specs, len(out_avals)
        )
        # pylint: enable=protected-access
        if norm_out_specs is not None and len(norm_out_specs) != len(out_avals):
          detected.append(
              findings.VerificationFinding(
                  category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                  severity=findings.Severity.ERROR,
                  phase=_CONTRACTS,
                  op_name=op_name,
                  message=(
                      "output_partition_specs has length"
                      f" {len(norm_out_specs)}, expected {len(out_avals)}"
                  ),
              )
          )
        elif norm_out_specs is not None:
          for out_idx, (aval, spec) in enumerate(
              zip(out_avals, norm_out_specs)
          ):
            detected.extend(
                _validate_partition_spec_entry(
                    spec,
                    len(aval.shape),
                    mesh,
                    f"output[{out_idx}]",
                    op_name,
                    global_shape=aval.shape,
                )
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      reported_axis = False
      if mesh is not None and out_specs is not None:
        for out_idx, spec in enumerate(out_specs):
          for ax in _iter_spec_axis_names(spec):
            if ax not in mesh.axis_names:
              reported_axis = True
              detected.append(
                  findings.VerificationFinding(
                      category=findings.FindingCategory.PARTITION_SPEC_MISMATCH,
                      severity=findings.Severity.ERROR,
                      phase=_CONTRACTS,
                      op_name=op_name,
                      message=(
                          f"output[{out_idx}] PartitionSpec {spec!r} references"
                          f" unknown mesh axis {ax!r} ({exc})"
                      ),
                  )
              )
      if not reported_axis:
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
                severity=findings.Severity.INFO,
                phase=_CONTRACTS,
                op_name=op_name,
                message=(
                    "skipped the output checks because the output shapes could"
                    f" not be inferred ({type(exc).__name__}:"
                    f" {_truncate(str(exc), 300)})"
                ),
            )
        )

  return findings.VerificationReport.of(_CONTRACTS, detected)


def _truncate(text: str, limit: int) -> str:
  text = text.strip()
  return text if len(text) <= limit else text[:limit] + " ..."


def _abstract_tpu_mesh(device_kind: str) -> Any:
  """Returns a one-device abstract mesh that targets TPU `device_kind`."""
  device_cls = jax.sharding.AbstractDevice
  if "platform" in inspect.signature(device_cls).parameters:
    device = device_cls(device_kind, 1, "tpu")  # type: ignore[call-arg]
  else:
    # JAX releases up to 0.9 take no platform.
    device = device_cls(device_kind, 1)  # pyrefly: ignore[missing-argument]
  return jax.sharding.AbstractMesh(
      (1,),
      ("x",),
      axis_types=(jax.sharding.AxisType.Auto,),
      abstract_device=device,
  )


def verify_tpu_lowering(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    tpu_device_kind: str = DEFAULT_TPU_DEVICE_KIND,
) -> findings.VerificationReport:
  """Lowers a TorchTPU Pallas op for TPU on the host, without a TPU.

  This is the lowering TorchTPU performs the first time the op runs, so a
  failure here means the op fails on TPU too, for example because of
  unaligned block shapes or vector loads and stores on HBM refs.

  Args:
    op_or_fn: A `@jax_op` or `JaxCallable`.
    example_args: Example positional arguments.
    example_kwargs: Example keyword arguments.
    tpu_device_kind: JAX device kind of the TPU generation to lower for, such as
      "TPU v5 lite" or "TPU7x". Ops with a mesh are lowered for the device kind
      of their mesh devices instead.

  Returns:
    A report with a `LOWERING_FAILURE` error if lowering fails.
  """
  jax_callable = extract_jax_callable(op_or_fn)
  if jax_callable is None:
    return findings.VerificationReport()
  op_name = get_op_display_name(op_or_fn)
  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      example_args, dict(example_kwargs or {})
  )
  jax_args, jax_kwargs = jax_callable._get_jax_placeholders(
      norm_args, norm_kwargs
  )
  target = (
      jax.sharding.use_abstract_mesh(_abstract_tpu_mesh(tpu_device_kind))
      if jax_callable.mesh is None
      else contextlib.nullcontext()
  )
  try:
    # Matches the forward-compatibility mode TorchTPU lowers with.
    with target, jax._src.config.export_ignore_forward_compatibility(True):
      jax_callable.exported(*jax_args, **jax_kwargs)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    message = _truncate(f"{exc}", _MAX_LOWERING_MESSAGE_CHARS)
    if "Unsupported TPU device kind" in message:
      finding = findings.VerificationFinding(
          category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
          severity=findings.Severity.INFO,
          phase=_LOWERING,
          op_name=op_name,
          message=(
              "skipped TPU lowering because the TPU generation is unknown;"
              " build the mesh with torch.tpu.pallas.get_pallas_mesh or with"
              " devices whose device_kind names a TPU generation"
          ),
      )
      return findings.VerificationReport(findings=(finding,))
    finding = findings.VerificationFinding(
        category=findings.FindingCategory.LOWERING_FAILURE,
        severity=findings.Severity.ERROR,
        phase=_LOWERING,
        op_name=op_name,
        message=(
            "the kernel fails Pallas TPU lowering for"
            f" {tpu_device_kind if jax_callable.mesh is None else 'its mesh'},"
            f" so it cannot run on TPU ({type(exc).__name__}: {message})"
        ),
    )
    return findings.VerificationReport.of(_LOWERING, [finding])
  # pylint: enable=protected-access
  return findings.VerificationReport.of(_LOWERING)


def _fx_op_aliases_inputs(target: Any, args: Sequence[Any]) -> list[fx.Node]:
  """Returns input `fx.Node`s whose storage may be viewed/aliased by `target`."""
  schema = getattr(target, "_schema", None)
  if schema is not None:
    return_has_alias = any(
        getattr(ret, "alias_info", None) is not None for ret in schema.returns
    )
    if return_has_alias:
      aliased_nodes: list[fx.Node] = []
      for idx, schema_arg in enumerate(schema.arguments):
        if idx >= len(args):
          break
        if getattr(schema_arg, "alias_info", None) is not None:
          val = args[idx]
          if isinstance(val, fx.Node):
            aliased_nodes.append(val)
          elif isinstance(val, (list, tuple)):
            aliased_nodes.extend(x for x in val if isinstance(x, fx.Node))
      if aliased_nodes:
        return aliased_nodes

  target_name = getattr(target, "__name__", str(target)).split(".")[0]
  if target_name in _VIEW_FALLBACK_NAMES and args:
    first = args[0]
    if isinstance(first, fx.Node):
      return [first]
  return []


def _iter_fx_nodes(value: Any) -> list[fx.Node]:
  """Collects all `fx.Node` leaves inside nested tuples/lists/dicts."""
  result: list[fx.Node] = []
  if isinstance(value, fx.Node):
    result.append(value)
  elif isinstance(value, (tuple, list)):
    for item in value:
      result.extend(_iter_fx_nodes(item))
  elif isinstance(value, Mapping):
    for item in value.values():
      result.extend(_iter_fx_nodes(item))
  return result


def trace_to_fx_graph(
    fn_or_gm: Callable[..., Any],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
) -> fx.GraphModule:
  """Traces `fn_or_gm` into an `fx.GraphModule` on CPU using FakeTensor mode."""
  if isinstance(fn_or_gm, fx.GraphModule):
    return fn_or_gm
  kwargs = dict(example_kwargs or {})

  def _bound(*pos_args):
    return fn_or_gm(*pos_args, **kwargs)

  cloned_args = tuple(
      a.detach().clone() if isinstance(a, torch.Tensor) else a
      for a in example_args
  )
  # Module parameters and buffers are real tensors that the trace captures as
  # `get_attr` constants.
  return proxy_tensor.make_fx(
      _bound, tracing_mode="fake", _allow_non_fake_inputs=True
  )(*cloned_args)


def verify_fx_graph(
    fn_or_gm: Callable[..., Any],
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
) -> findings.VerificationReport:
  """Verifies linear buffer donation and alias invariants in a PyTorch FX graph."""
  gm = trace_to_fx_graph(fn_or_gm, example_args, example_kwargs)
  detected: list[findings.VerificationFinding] = []

  # Provenance map: fx.Node -> set of root storage identifiers.
  node_roots: dict[fx.Node, set[str]] = {}
  # Invalidated root map: root_id -> (donating_op_name, donating_fx_node_name).
  invalidated_roots: dict[str, tuple[str, str]] = {}
  # Nodes whose outputs are new buffers rather than views of their inputs.
  fresh_nodes: set[fx.Node] = set()

  for node in gm.graph.nodes:
    if node.op == "placeholder":
      node_roots[node] = {node.name}
      continue

    if node.op == "get_attr":
      node_roots[node] = {f"attr:{node.target}"}
      continue

    if node.op == "call_function" and node.target is operator.getitem:
      container = node.args[0]
      index = node.args[1] if len(node.args) > 1 else None
      container_val = (
          container.meta.get("val") if isinstance(container, fx.Node) else None
      )
      if (
          isinstance(container, fx.Node)
          and container in fresh_nodes
          and isinstance(index, int)
          and (
              container_val is None or isinstance(container_val, (tuple, list))
          )
      ):
        # Each output of an op that creates new buffers is its own buffer.
        node_roots[node] = {f"{r}[{index}]" for r in node_roots[container]}
        fresh_nodes.add(node)
      elif isinstance(container, fx.Node):
        # Indexing or slicing a tensor returns a view of it.
        node_roots[node] = set(node_roots.get(container, {node.name}))
      else:
        node_roots[node] = {node.name}
      continue

    # Check whether this node is a TorchTPU @pallas.jax_op custom op.
    jax_callable = (
        extract_jax_callable(node.target)
        if node.op == "call_function"
        else None
    )
    if jax_callable is not None:
      op_name = get_op_display_name(node.target)
      # pylint: disable=protected-access
      donated_set = set(jax_callable._donate_argnums_set) | set(
          jax_callable.input_output_aliases.keys()
      )
      # pylint: enable=protected-access

      # First check if any input to this jax_op was ALREADY invalidated earlier!
      for used_node in _iter_fx_nodes((node.args, node.kwargs)):
        stale = node_roots.get(used_node, set()) & invalidated_roots.keys()
        for root in sorted(stale):
          donor_op, donor_node = invalidated_roots[root]
          detected.append(
              findings.VerificationFinding(
                  category=findings.FindingCategory.USE_AFTER_DONATE,
                  severity=findings.Severity.ERROR,
                  phase=_FX_GRAPH,
                  op_name=op_name,
                  source_location=node.name,
                  message=(
                      f"FX node '{node.name}' ({op_name}) reads"
                      f" '{used_node.name}' derived from buffer '{root}', which"
                      f" was already donated and invalidated by '{donor_node}'"
                      f" ({donor_op})"
                  ),
              )
          )

      # Collect per-argument FX nodes to check aliased donor inputs & invalidate.
      arg_fx_nodes: list[tuple[int, fx.Node]] = []
      for arg_idx, arg_val in enumerate(node.args):
        if arg_idx in jax_callable.static_argnums:
          continue
        for sub_node in _iter_fx_nodes(arg_val):
          arg_fx_nodes.append((arg_idx, sub_node))

      for i_pos in range(len(arg_fx_nodes)):
        idx_a, n_a = arg_fx_nodes[i_pos]
        roots_a = node_roots.get(n_a, set())
        for j_pos in range(i_pos + 1, len(arg_fx_nodes)):
          idx_b, n_b = arg_fx_nodes[j_pos]
          roots_b = node_roots.get(n_b, set())
          shared = roots_a & roots_b
          if shared and (idx_a in donated_set or idx_b in donated_set):
            detected.append(
                findings.VerificationFinding(
                    category=findings.FindingCategory.ALIASED_DONOR_INPUTS,
                    severity=findings.Severity.ERROR,
                    phase=_FX_GRAPH,
                    op_name=op_name,
                    source_location=node.name,
                    message=(
                        f"FX node '{node.name}' ({op_name}) receives aliased"
                        f" arguments '{n_a.name}' (arg {idx_a}) and"
                        f" '{n_b.name}' (arg {idx_b}) sharing root storage"
                        f" {sorted(shared)} while donate_argnums="
                        f"{tuple(sorted(donated_set))}"
                    ),
                )
            )

      # Invalidate all provenance roots of donated arguments.
      for arg_idx, sub_node in arg_fx_nodes:
        if arg_idx not in donated_set:
          continue
        for root in sorted(node_roots.get(sub_node, set())):
          invalidated_roots[root] = (op_name, node.name)
          if root.startswith("attr:"):
            detected.append(
                findings.VerificationFinding(
                    category=findings.FindingCategory.DONATED_PARAMETER,
                    severity=findings.Severity.ERROR,
                    phase=_FX_GRAPH,
                    op_name=op_name,
                    source_location=node.name,
                    message=(
                        f"FX node '{node.name}' ({op_name}) donates arg"
                        f" {arg_idx} '{sub_node.name}', which is backed by"
                        f" module state or a captured tensor '{root[5:]}';"
                        " the kernel may reuse that buffer and corrupt it"
                        " for later calls, so donate a clone() instead"
                    ),
                )
            )

      # Output of `jax_op` is a newly materialized buffer (note `mutates_args=()`).
      node_roots[node] = {f"out:{node.name}"}
      fresh_nodes.add(node)
      continue

    # Standard FX operation or `output` return node: check all consumed nodes!
    consumer_desc = (
        "graph return (`output`)"
        if node.op == "output"
        else f"FX node '{node.name}' ({node.target})"
    )
    for used_node in _iter_fx_nodes((node.args, node.kwargs)):
      stale = node_roots.get(used_node, set()) & invalidated_roots.keys()
      for root in sorted(stale):
        donor_op, donor_node = invalidated_roots[root]
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.USE_AFTER_DONATE,
                severity=findings.Severity.ERROR,
                phase=_FX_GRAPH,
                op_name=donor_op,
                source_location=node.name,
                message=(
                    f"{consumer_desc} uses '{used_node.name}' (sharing storage"
                    f" with '{root}') after '{root}' was donated to"
                    f" '{donor_node}' ({donor_op})"
                ),
            )
        )

    view_source: fx.Node | None = None
    if node.op == "call_function":
      aliased_inputs = _fx_op_aliases_inputs(node.target, node.args)
      if aliased_inputs:
        merged_roots: set[str] = set()
        for src_node in aliased_inputs:
          merged_roots.update(node_roots.get(src_node, {src_node.name}))
        node_roots[node] = merged_roots
        continue
    elif node.op == "call_method" and str(node.target) in _VIEW_FALLBACK_NAMES:
      src = node.args[0] if node.args else None
      view_source = src if isinstance(src, fx.Node) else None
    if view_source is not None:
      node_roots[node] = set(node_roots.get(view_source, {view_source.name}))
    else:
      node_roots[node] = {f"val:{node.name}"}
      fresh_nodes.add(node)

  return findings.VerificationReport.of(_FX_GRAPH, detected)


def tensor_to_numpy(tensor: torch.Tensor) -> tuple[np.ndarray, Any]:
  """Returns `(host_numpy_array, target_jax_dtype)` for a `torch.Tensor`."""
  detached = tensor.detach().cpu()
  jax_dtype = pallas_lib.TORCH_TO_JAX_DTYPE_MAP.get(detached.dtype)
  if jax_dtype is None:
    raise NotImplementedError(
        f"unsupported dtype for interpret: {detached.dtype}"
    )
  if detached.dtype == torch.bfloat16:
    return detached.to(torch.float32).numpy(), jnp.bfloat16
  return detached.numpy(), jax_dtype


def build_sharded_cpu_array(
    local_tensor: torch.Tensor,
    cpu_mesh: jax.sharding.Mesh,
    spec: Any,
    *,
    distinguish_ranks: bool = False,
) -> jax.Array:
  """Builds a global sharded `jax.Array` on `cpu_mesh` from a local shard."""
  base_np, jax_dtype = tensor_to_numpy(local_tensor)
  sharding = jax.sharding.NamedSharding(
      cpu_mesh,
      spec if spec is not None else jax.sharding.PartitionSpec(),
  )
  global_shape = pallas_lib.get_global_shape(
      tuple(local_tensor.shape), cpu_mesh, spec
  )
  use_rank_offset = distinguish_ranks and global_shape != tuple(
      local_tensor.shape
  )

  per_device_arrays: list[jax.Array] = []
  for flat_rank, dev in enumerate(cpu_mesh.devices.flat):
    if use_rank_offset and np.issubdtype(base_np.dtype, np.floating):
      shard_np = base_np + np.asarray(flat_rank * 0.25, dtype=base_np.dtype)
    elif (
        use_rank_offset
        and np.issubdtype(base_np.dtype, np.integer)
        and base_np.dtype != np.bool_
    ):
      shard_np = base_np + np.asarray(flat_rank, dtype=base_np.dtype)
    else:
      shard_np = base_np
    per_device_arrays.append(
        jax.device_put(jnp.asarray(shard_np, dtype=jax_dtype), dev)
    )
  return jax.make_array_from_single_device_arrays(
      global_shape, sharding, per_device_arrays
  )


def _torch_to_jax_array(tensor: torch.Tensor, device: jax.Device) -> jax.Array:
  """Converts a CPU `torch.Tensor` into a `jax.Array` on `device`."""
  np_val, jax_dtype = tensor_to_numpy(tensor)
  return jax.device_put(jnp.asarray(np_val, dtype=jax_dtype), device)


def _jax_to_torch_tensor(
    arr: jax.Array, *, local_shard: bool = False
) -> torch.Tensor:
  """Converts a CPU `jax.Array` (or its rank-0 shard) into a `torch.Tensor`."""
  target = (
      arr.addressable_shards[0].data
      if local_shard and arr.addressable_shards
      else arr
  )
  np_arr = np.asarray(target)
  if target.dtype == jnp.bfloat16:
    return torch.from_numpy(np_arr.astype(np.float32)).to(torch.bfloat16)
  return torch.from_numpy(np_arr.copy())


def _run_jax_callable_on_cpu_interpret(
    jax_callable: pallas_lib.JaxCallable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
  """Executes a single-chip or multi-chip `JaxCallable` on CPU interpret."""
  raw_fn, cpu_mesh = unwrap_raw_jax_fn(jax_callable, prefer_cpu_mesh=True)
  if jax_callable.mesh is not None and is_mock_tpu_mesh(cpu_mesh):
    raise ValueError(
        f"CPU interpret requires at least {jax_callable.mesh.size} virtual CPU"
        f" devices, got {len(jax.devices('cpu'))}"
    )
  # pylint: disable=protected-access
  norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
      args, dict(kwargs)
  )
  # pylint: enable=protected-access
  norm_args = replace_mock_meshes_in_pytree(norm_args, cpu_mesh)
  norm_kwargs = replace_mock_meshes_in_pytree(norm_kwargs, cpu_mesh)
  cpu0 = jax.devices("cpu")[0]
  in_specs = jax_callable.input_partition_specs

  jax_args: list[Any] = []
  spec_idx = 0
  for i, a in enumerate(norm_args):
    if i in jax_callable.static_argnums:
      jax_args.append(a)
      continue
    spec = (
        in_specs[spec_idx]
        if in_specs is not None and spec_idx < len(in_specs)
        else None
    )
    spec_idx += 1
    if isinstance(a, torch.Tensor):
      if cpu_mesh is not None:
        jax_args.append(build_sharded_cpu_array(a, cpu_mesh, spec))
      else:
        jax_args.append(_torch_to_jax_array(a, cpu0))
    else:
      jax_args.append(a)

  jax_kwargs = {
      k: (
          (
              build_sharded_cpu_array(v, cpu_mesh, None)
              if cpu_mesh is not None
              else _torch_to_jax_array(v, cpu0)
          )
          if isinstance(v, torch.Tensor)
          else v
      )
      for k, v in norm_kwargs.items()
  }
  params = pltpu.InterpretParams(
      dma_execution_mode="on_wait",
      detect_races=False,
      uninitialized_memory="zero",
  )
  dev_ctx = (
      contextlib.nullcontext()
      if cpu_mesh is not None
      else jax.default_device(cpu0)
  )
  try:
    with dev_ctx, pltpu.force_tpu_interpret_mode(params):
      out = raw_fn(*jax_args, **jax_kwargs)
      jax.effects_barrier()
  finally:
    pltpu.reset_tpu_interpret_mode_state()

  return jax.tree.map(
      lambda leaf: _jax_to_torch_tensor(
          leaf, local_shard=(cpu_mesh is not None)
      )
      if isinstance(leaf, jax.Array)
      else leaf,
      out,
  )


class _CpuPallasInterpretDispatchMode(_python_dispatch.TorchDispatchMode):
  """Intercepts `@pallas.jax_op` calls below Autograd and runs them on CPU."""

  def __torch_dispatch__(self, func, types_unused, args=(), kwargs=None):
    del types_unused
    kwargs = kwargs or {}
    jax_callable = extract_jax_callable(func)
    if jax_callable is not None:
      return _run_jax_callable_on_cpu_interpret(jax_callable, args, kwargs)
    return func(*args, **kwargs)


def verify_autograd_parity(
    op_def: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> findings.VerificationReport:
  """Checks registered PyTorch custom autograd backward against `jax.vjp`."""
  resolved = resolve_custom_op_def(op_def)
  if resolved is None or resolved._backward_fn is None:  # pylint: disable=protected-access
    return findings.VerificationReport()

  jax_callable = extract_jax_callable(resolved)
  if jax_callable is None:
    return findings.VerificationReport()

  op_name = resolved._qualname  # pylint: disable=protected-access
  kwargs = dict(example_kwargs or {})

  diff_indices: list[int] = []
  pt_args: list[Any] = []
  for idx, arg in enumerate(example_args):
    if (
        isinstance(arg, torch.Tensor)
        and idx not in jax_callable.static_argnums
        and (arg.is_floating_point() or arg.is_complex())
    ):
      cloned = arg.detach().cpu().clone().requires_grad_(True)
      pt_args.append(cloned)
      diff_indices.append(idx)
    else:
      pt_args.append(arg)

  if not diff_indices:
    return findings.VerificationReport()

  detected: list[findings.VerificationFinding] = []
  try:
    with _CpuPallasInterpretDispatchMode():
      pt_out = resolved(*pt_args, **kwargs)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.VERIFICATION_INCOMPLETE,
            severity=findings.Severity.INFO,
            phase=_AUTOGRAD,
            op_name=op_name,
            message=f"skipped autograd parity check ({exc})",
        )
    )
    return findings.VerificationReport(findings=tuple(detected))

  pt_out_leaves, _ = jax.tree.flatten(pt_out)
  tensor_outs = [t for t in pt_out_leaves if isinstance(t, torch.Tensor)]
  # Integer outputs and outputs that do not depend on the inputs have no
  # gradient: they get zero cotangents and are left out of the backward pass.
  differentiable = [t.requires_grad for t in tensor_outs]
  if not any(differentiable):
    return findings.VerificationReport()

  rng = torch.Generator(device="cpu")
  rng.manual_seed(2026)
  raw_cotangents = [
      torch.randn(t.shape, dtype=torch.float32, generator=rng)
      if is_diff
      else torch.zeros(t.shape, dtype=torch.float32)
      for t, is_diff in zip(tensor_outs, differentiable)
  ]
  cot_norm = (
      float(sum(float((g * g).sum().item()) for g in raw_cotangents) ** 0.5)
      or 1.0
  )
  cotangents_pt = [
      (g / cot_norm).to(t.dtype) for g, t in zip(raw_cotangents, tensor_outs)
  ]
  cotangents_f32 = [c.to(torch.float32) for c in cotangents_pt]

  try:
    with _CpuPallasInterpretDispatchMode():
      pt_grads = torch.autograd.grad(
          outputs=[t for t, d in zip(tensor_outs, differentiable) if d],
          inputs=[pt_args[i] for i in diff_indices],
          grad_outputs=[c for c, d in zip(cotangents_pt, differentiable) if d],
          allow_unused=True,
      )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    detected.append(
        findings.VerificationFinding(
            category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
            severity=findings.Severity.ERROR,
            phase=_AUTOGRAD,
            op_name=op_name,
            message=(
                f"custom PyTorch backward raised {type(exc).__name__}: {exc}"
            ),
        )
    )
    return findings.VerificationReport.of(_AUTOGRAD, detected)

  for arg_pos, pt_grad in zip(diff_indices, pt_grads):
    if pt_grad is not None and pt_grad.shape != pt_args[arg_pos].shape:
      detected.append(
          findings.VerificationFinding(
              category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
              severity=findings.Severity.ERROR,
              phase=_AUTOGRAD,
              op_name=op_name,
              message=(
                  f"custom PyTorch backward for arg {arg_pos} returned shape"
                  f" {tuple(pt_grad.shape)}, expected"
                  f" {tuple(pt_args[arg_pos].shape)}"
              ),
          )
      )
  if detected:
    return findings.VerificationReport.of(_AUTOGRAD, detected)

  # First attempt analytical `jax.vjp` on single-device forward JAX functions
  # (succeeds when pure JAX ops or `@jax.custom_vjp` rules are present).
  raw_fwd_fn, cpu_mesh = unwrap_raw_jax_fn(jax_callable, prefer_cpu_mesh=True)
  ref_jax_grads: Sequence[Any] | None = None
  if cpu_mesh is None:
    # pylint: disable=protected-access
    norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
        pt_args, kwargs
    )
    # pylint: enable=protected-access
    norm_args = replace_mock_meshes_in_pytree(norm_args, cpu_mesh)
    norm_kwargs = replace_mock_meshes_in_pytree(norm_kwargs, cpu_mesh)
    cpu0 = jax.devices("cpu")[0]
    jax_primals = [_torch_to_jax_array(pt_args[i], cpu0) for i in diff_indices]

    def _pure_diff_fwd(*diff_jax_arrays):
      call_args = list(norm_args)
      for pos, arr in zip(diff_indices, diff_jax_arrays):
        call_args[pos] = arr
      for i, val in enumerate(call_args):
        if i not in diff_indices and isinstance(val, torch.Tensor):
          call_args[i] = _torch_to_jax_array(val, cpu0)
      call_kwargs = {
          k: _torch_to_jax_array(v, cpu0) if isinstance(v, torch.Tensor) else v
          for k, v in norm_kwargs.items()
      }
      return raw_fwd_fn(*call_args, **call_kwargs)

    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=False,
        uninitialized_memory="zero",
    )
    try:
      with jax.default_device(cpu0), pltpu.force_tpu_interpret_mode(params):
        jax_fwd_out, vjp_fn = jax.vjp(_pure_diff_fwd, *jax_primals)
        _, out_tree = jax.tree.flatten(jax_fwd_out)
        jax_cotangents = out_tree.unflatten(
            [_torch_to_jax_array(g, cpu0) for g in cotangents_pt]
        )
        ref_jax_grads = vjp_fn(jax_cotangents)
        jax.effects_barrier()
    except (ValueError, NotImplementedError, TypeError):
      ref_jax_grads = None
    finally:
      pltpu.reset_tpu_interpret_mode_state()

  if ref_jax_grads is not None:
    for arg_pos, pt_grad, ref_jax_grad in zip(
        diff_indices, pt_grads, ref_jax_grads
    ):
      ref_pt_grad = _jax_to_torch_tensor(ref_jax_grad)
      if pt_grad is None:
        if float(ref_pt_grad.abs().max().item()) > atol:
          detected.append(
              findings.VerificationFinding(
                  category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
                  severity=findings.Severity.ERROR,
                  phase=_AUTOGRAD,
                  op_name=op_name,
                  message=(
                      "custom backward returned None for differentiable input"
                      f" arg {arg_pos}, whereas jax.vjp reference has max"
                      " magnitude"
                      f" {float(ref_pt_grad.abs().max().item()):.4e}"
                  ),
              )
          )
        continue
      if not torch.allclose(
          pt_grad.to(torch.float32),
          ref_pt_grad.to(torch.float32),
          rtol=rtol,
          atol=atol,
      ):
        max_diff = float(
            (pt_grad.to(torch.float32) - ref_pt_grad.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        detected.append(
            findings.VerificationFinding(
                category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
                severity=findings.Severity.ERROR,
                phase=_AUTOGRAD,
                op_name=op_name,
                message=(
                    f"custom PyTorch backward for arg {arg_pos} diverges from"
                    " jax.vjp of forward Pallas kernel (max_abs_diff="
                    f"{max_diff:.4e}, rtol={rtol}, atol={atol})"
                ),
            )
        )
  else:
    # Bare `pl.pallas_call` and multi-chip `shard_map` Pallas primitives lack
    # default JAX JVP rules; verify the custom backward via Hutchinson
    # Rademacher central finite differences
    # `<g_pt, v> == ( <u, f(x + eps v)> - <u, f(x - eps v)> ) / (2 eps)` on
    # CPU TPU interpret mode.
    base_detached = [
        a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in pt_args
    ]
    for arg_pos, pt_grad in zip(diff_indices, pt_grads):
      x_base = base_detached[arg_pos]
      orig_dtype = x_base.dtype
      x_f32 = x_base.to(torch.float32)
      is_low_prec = orig_dtype in (torch.float16, torch.bfloat16)
      eps = 6.25e-2 if is_low_prec else (2.0**-10)
      fd_atol = max(atol, 5e-2 if is_low_prec else 2e-3)
      fd_rtol = max(rtol, 5e-2 if is_low_prec else 5e-3)

      probes: list[torch.Tensor] = []
      if pt_grad is not None:
        grad_f32 = pt_grad.detach().cpu().to(torch.float32)
        grad_max = float(grad_f32.abs().max().item())
        if grad_max > 1e-6:
          probes.append(grad_f32 / grad_max)
      for _ in range(4):
        bits = torch.randint(
            0, 2, x_base.shape, generator=rng, dtype=torch.int64
        )
        probes.append(bits.to(torch.float32) * 2.0 - 1.0)

      for probe_idx, v_f32 in enumerate(probes):
        x_pos = (x_f32 + eps * v_f32).to(orig_dtype)
        x_neg = (x_f32 - eps * v_f32).to(orig_dtype)
        actual_half_step = (
            x_pos.to(torch.float32) - x_neg.to(torch.float32)
        ) * 0.5
        args_pos = list(base_detached)
        args_neg = list(base_detached)
        args_pos[arg_pos] = x_pos
        args_neg[arg_pos] = x_neg

        out_pos = _run_jax_callable_on_cpu_interpret(
            jax_callable, args_pos, kwargs
        )
        out_neg = _run_jax_callable_on_cpu_interpret(
            jax_callable, args_neg, kwargs
        )
        pos_leaves, _ = jax.tree.flatten(out_pos)
        neg_leaves, _ = jax.tree.flatten(out_neg)
        pos_tensors = [t for t in pos_leaves if isinstance(t, torch.Tensor)]
        neg_tensors = [t for t in neg_leaves if isinstance(t, torch.Tensor)]

        phi_diff = sum(
            float(
                (u * (p.to(torch.float32) - n.to(torch.float32))).sum().item()
            )
            for u, p, n in zip(cotangents_f32, pos_tensors, neg_tensors)
        )
        fd_dir = phi_diff / (2.0 * eps)
        pt_dir = (
            0.0
            if pt_grad is None
            else float(
                (pt_grad.detach().cpu().to(torch.float32) * actual_half_step)
                .sum()
                .item()
            )
            / eps
        )
        abs_err = abs(pt_dir - fd_dir)
        if abs_err > fd_atol + fd_rtol * abs(fd_dir):
          detected.append(
              findings.VerificationFinding(
                  category=findings.FindingCategory.AUTOGRAD_PARITY_MISMATCH,
                  severity=findings.Severity.ERROR,
                  phase=_AUTOGRAD,
                  op_name=op_name,
                  message=(
                      f"custom PyTorch backward for arg {arg_pos} diverges"
                      " from central finite-difference directional JVP of"
                      f" forward Pallas kernel on probe #{probe_idx}"
                      f" (backward_dir={pt_dir:.4e}, fd_ref_dir={fd_dir:.4e},"
                      f" abs_err={abs_err:.4e})"
                  ),
              )
          )
          break

  return findings.VerificationReport.of(_AUTOGRAD, detected)
