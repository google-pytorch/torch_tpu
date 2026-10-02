# Copyright 2025 Google LLC
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

"""APIs for Pallas integration.

This module provides APIs to call Pallas kernels from PyTorch. Note that use
of these APIs requires that your environment also has JAX/Pallas installed.
"""

import collections.abc
import enum
import functools
import hashlib
import inspect
import itertools
import math
import pathlib
import types
import typing
from typing import Any, Callable, Sequence, Union, overload
from absl import logging
import frozendict
import numpy as np
import torch
from torch._library.custom_ops import CustomOpDef
import torch.distributed as dist
import torch.library
from torch_tpu._internal.device import _device_ops_backend
from torch_tpu._internal.distributed import device_mesh
from torch_tpu._internal.pallas import _compat
from torch_tpu._internal.pallas import tpu_torch_pallas
from torch_tpu._internal.pallas._mock_device import _PallasDevice
from torch_tpu._internal.utils import annotations

try:
  # TODO: TorchXLA hit issues when workin with both JAX and PT in the same
  # process, hitting deadlocks when each framework tried to access the PJRT
  # plugin. Unclear if this applies for trace-only, but we may need to better
  # protect against this, see @requires_jax in PT/XLA repo.
  # pylint: disable=g-import-not-at-top
  import jax
  import jax.export
  # pylint: enable=g-import-not-at-top
except ImportError as ex:
  raise ImportError(
      "JAX/Pallas is not available. Please install JAX to use this API."
  ) from ex


# Based on torch/_C/__init__.pyi
# Skip alias types since they are covered by other entries in the table. JAX
# also has similar alias types, but it relies on JAX config.py, which may not
# align to PT's configuration (jax.numpy.float_ => float32 | float64)
TORCH_TO_JAX_DTYPE_MAP = frozendict.frozendict({
    torch.float32: jax.numpy.float32.dtype,
    # torch.float: jax.numpy.float_.dtype,  # Avoid PT alias types
    torch.float64: jax.numpy.float64.dtype,
    torch.double: jax.numpy.double.dtype,
    torch.float16: jax.numpy.float16.dtype,
    torch.bfloat16: jax.numpy.bfloat16.dtype,
    torch.float8_e4m3fn: jax.numpy.float8_e4m3fn.dtype,
    torch.float8_e4m3fnuz: jax.numpy.float8_e4m3fnuz.dtype,
    torch.float8_e5m2: jax.numpy.float8_e5m2.dtype,
    torch.float8_e5m2fnuz: jax.numpy.float8_e5m2fnuz.dtype,
    torch.float8_e8m0fnu: jax.numpy.float8_e8m0fnu.dtype,
    torch.float4_e2m1fn_x2: jax.numpy.float4_e2m1fn.dtype,  # x2 is packed
    # torch.half: jax.numpy.float16.dtype,  # PT alias type
    torch.uint8: jax.numpy.uint8.dtype,
    torch.uint16: jax.numpy.uint16.dtype,
    torch.uint32: jax.numpy.uint32.dtype,
    torch.uint64: jax.numpy.uint64.dtype,
    torch.int8: jax.numpy.int8.dtype,
    torch.int16: jax.numpy.int16.dtype,
    # torch.short: jax.numpy.int16.dtype,  # PT alias type
    torch.int32: jax.numpy.int32.dtype,
    # torch.int: jax.numpy.int_.dtype,  # PT alias type
    torch.int64: jax.numpy.int64.dtype,
    # torch.long: jax.numpy.int64.dtype,  # PT alias type
    # torch.complex32: jax.numpy.complex32.dtype, # complex32 DNE in JAX
    torch.complex64: jax.numpy.complex64.dtype,
    # torch.chalf: jax.numpy.chalf.dtype, # PT alias type
    # torch.cfloat: jax.numpy.csingle.dtype,  # PT alias type
    torch.complex128: jax.numpy.complex128.dtype,
    torch.cdouble: jax.numpy.cdouble.dtype,
    # torch.quint8: jax.numpy.quint8.dtype,
    # torch.qint8: jax.numpy.qint8.dtype,
    # torch.qint32: jax.numpy.qint32.dtype,
    torch.bool: jax.numpy.bool_.dtype,
    # torch.quint4x2: jax.numpy.quint4x2.dtype,
    # torch.quint2x4: jax.numpy.quint2x4.dtype,
    # torch.bits1x8: jax.numpy.bits1x8.dtype,
    # torch.bits2x4: jax.numpy.bits2x4.dtype,
    # torch.bits4x2: jax.numpy.bits4x2.dtype,
    # torch.bits8: jax.numpy.bits8.dtype,
    # torch.bits16: jax.numpy.bits16.dtype,
})
# NOTE: If multiple Torch dtypes map to the same JAX dtype in the future,
# JAX_TO_TORCH_DTYPE_MAP will only preserve the last one. Handle duplicates
# explicitly if unpacked float4 types are added.
JAX_TO_TORCH_DTYPE_MAP = {v: k for k, v in TORCH_TO_JAX_DTYPE_MAP.items()}


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "custom_jax_kernel",
    "get_default_pallas_mesh",
    "get_pallas_mesh",
    "jax_op",
    # go/keep-sorted end
]


def _convert_shape(shape, mesh, partition_spec, op):
  """Converts a shape based on a mesh and partition specification."""
  if mesh is not None and partition_spec is not None:
    axis_to_index = {axis: index for index, axis in enumerate(mesh.axis_names)}
    ans = tuple(
        [
            op(s, mesh.devices.shape[axis_to_index[name]])
            if name is not None
            else s
            for s, name in zip(shape, partition_spec)
        ]
        + list(shape[len(partition_spec) :])
    )
  else:
    ans = shape
  return ans


def get_global_shape(local_shape, mesh, partition_spec):
  return _convert_shape(local_shape, mesh, partition_spec, lambda a, b: a * b)


def get_local_shape(global_shape, mesh, partition_spec):
  return _convert_shape(global_shape, mesh, partition_spec, lambda a, b: a // b)


def jax_placeholder(
    tensor: torch.Tensor | Sequence[torch.Tensor],
    mesh=None,
    partition_spec=None,
) -> Any:
  """Converts a torch tensor (or sequence of tensors) to a jax placeholder for tracing."""
  if isinstance(tensor, (tuple, list)):
    return type(tensor)(
        jax_placeholder(t, mesh=mesh, partition_spec=partition_spec)
        for t in tensor
    )
  if not isinstance(tensor, torch.Tensor):
    # Preserve POD constants / None types
    return tensor

  jax_dtype = TORCH_TO_JAX_DTYPE_MAP.get(tensor.dtype, None)
  logging.debug("[jax_placeholder] dtype: %s -> %s", tensor.dtype, jax_dtype)
  if jax_dtype is None:
    raise NotImplementedError(
        f"Unsupported dtype for pallas kernels: {tensor.dtype}"
    )

  jax_shape = tensor.shape
  if tensor.dtype == torch.float4_e2m1fn_x2 and len(jax_shape) > 0:
    if tensor.stride(-1) != 1:
      raise ValueError(
          "expected packed dimension of float4_e2m1fn_x2 tensor to be "
          f"contiguous (stride 1), got stride {tensor.stride(-1)} "
          f"(all strides: {tensor.stride()})"
      )
    jax_shape = jax_shape[:-1] + (jax_shape[-1] * 2,)

  if mesh is not None:
    spec = (
        partition_spec
        if partition_spec is not None
        else jax.sharding.PartitionSpec()
    )
    return jax.ShapeDtypeStruct(
        get_global_shape(jax_shape, mesh, spec),
        jax_dtype,
        sharding=jax.sharding.NamedSharding(mesh, spec),
    )
  else:
    return jax.ShapeDtypeStruct(jax_shape, jax_dtype)


def jax_placeholders(
    tensors: Sequence[Any] | torch.Tensor,
    mesh=None,
    partition_specs=None,
):
  """Converts a sequence of torch tensors to a sequence of jax placeholders."""
  if isinstance(tensors, torch.Tensor):
    return jax_placeholder(tensors, mesh, partition_specs)
  if mesh is not None:
    specs = (
        partition_specs
        if partition_specs is not None
        else [None] * len(tensors)
    )
    return [
        jax_placeholder(tensor, mesh, partition_spec)
        for tensor, partition_spec in zip(tensors, specs)
    ]
  else:
    return [jax_placeholder(tensor) for tensor in tensors]


def torch_placeholder(
    array: jax.core.ShapedArray | None,
    mesh: jax.sharding.Mesh | None = None,
    partition_spec: tuple[str | None, ...] | None = None,
    device: torch.device | str = "tpu",
) -> torch.Tensor | None:
  """Converts a jax output to a torch tensor for return type.

  JAX JIT functions only support JAX Array and None types, all other POD (Plain
  Old Data / scalar) types will error.

  Args:
    array: The JAX array to convert.
    mesh: Optional physical mesh.
    partition_spec: Optional partition spec for this output.
    device: The target torch device.

  Returns:
    The converted PyTorch tensor, or None.
  """
  if not isinstance(array, jax.core.ShapedArray):
    # Preserve POD constants / None types
    return array

  torch_dtype = JAX_TO_TORCH_DTYPE_MAP.get(array.dtype, None)
  logging.debug("[torch_placeholder] dtype: %s -> %s", array.dtype, torch_dtype)
  if torch_dtype is None:
    raise NotImplementedError(
        f"Unsupported dtype for pallas kernels: {array.dtype}"
    )
  spec = partition_spec
  if (
      spec is None
      and hasattr(array, "sharding")
      and hasattr(array.sharding, "spec")
  ):
    spec = array.sharding.spec
  torch_shape = get_local_shape(array.shape, mesh, spec)
  if torch_dtype == torch.float4_e2m1fn_x2 and len(torch_shape) > 0:
    if torch_shape[-1] % 2 != 0:
      raise ValueError(
          "expected last dimension of JAX local shape to be even to convert "
          f"to packed PyTorch dtype {torch_dtype}, got {torch_shape[-1]} "
          f"(local shape: {torch_shape}, global: {array.shape})"
      )
    torch_shape = torch_shape[:-1] + (torch_shape[-1] // 2,)

  return torch.empty(
      torch_shape,
      dtype=torch_dtype,
      device=device,
  )


def _normalize_output_partition_specs(
    output_specs: Any,
    num_outputs: int,
) -> Sequence[Any] | None:
  """Normalizes output partition specs across output leaves."""
  if output_specs is None:
    return None
  if (
      isinstance(output_specs, tuple)
      and len(output_specs) > 0
      and (isinstance(output_specs[0], str) or output_specs[0] is None)
  ):
    return [output_specs] * num_outputs
  return output_specs


def _get_kernel_invocation_key(
    trace_key: str,
    input_args: Sequence[torch.Tensor | Any],
    input_kwargs: dict[str, Any],
    static_argnums: tuple[int, ...] = (),
) -> str:
  """Construct a unique ID for the kernel invocation.

  Builds a string key that includes both the pallas compile args and
  the input shapes; if either of these changes, we need to recompile.
  For static args, includes value hash instead of shape.

  Args:
    trace_key: A string key used to identify the exported kernel.
    input_args: The input arguments to the kernel.
    input_kwargs: The input keyword arguments to the kernel.
    static_argnums: Tuple of argument positions that are compile-time constants.

  Returns:
    A string key that includes both the pallas compile args and
    the input shapes; if either of these changes, we need to recompile.
  """
  static_argnums_set = frozenset(static_argnums)

  def get_hash(i, item):
    if i in static_argnums_set:
      return f"static_{get_hash(-1, item)}"
    if isinstance(item, torch.Tensor):
      return f"{item.shape}x{item.dtype}"
    if isinstance(item, (tuple, list)):
      elem_hashes = ",".join(get_hash(-1, x) for x in item)
      return f"[{elem_hashes}]"
    return str(item)

  input_kwargs_str = ";".join(
      f"{key}:{get_hash(-1, value)}"
      for key, value in sorted(input_kwargs.items())
  )

  inputs_shapes = [get_hash(i, arg) for i, arg in enumerate(input_args)]
  inputs_shapes_str = ",".join(inputs_shapes)
  return ";".join([trace_key, inputs_shapes_str, input_kwargs_str])


def _unwrap_annotated(typ: Any) -> Any:
  """Unwraps typing.Annotated wrappers to get the underlying type."""
  while typing.get_origin(typ) is typing.Annotated:
    typ = typing.get_args(typ)[0]
  return typ


def _strip_optional(typ: Any) -> tuple[Any, bool]:
  """Splits an annotation into (non_optional_type, is_optional)."""
  typ = _unwrap_annotated(typ)
  if typing.get_origin(typ) not in (Union, types.UnionType):
    return typ, False
  union_types = [_unwrap_annotated(t) for t in typing.get_args(typ)]
  has_none = types.NoneType in union_types
  concrete_args = [t for t in union_types if t is not types.NoneType]
  if not concrete_args:
    return types.NoneType, True
  if len(concrete_args) == 1:
    return concrete_args[0], has_none
  reconstructed = concrete_args[0]
  for t in concrete_args[1:]:
    reconstructed = reconstructed | t
  return reconstructed, has_none


def _get_enum_type(typ: Any) -> type[enum.Enum] | None:
  """Returns the Enum subclass if typ is or contains an Enum in a Union."""
  underlying, _ = _strip_optional(typ)
  if isinstance(underlying, type) and issubclass(underlying, enum.Enum):
    return underlying
  if typing.get_origin(underlying) in (Union, types.UnionType):
    for arg in typing.get_args(underlying):
      arg_unwrapped = _unwrap_annotated(arg)
      if isinstance(arg_unwrapped, type) and issubclass(
          arg_unwrapped, enum.Enum
      ):
        return arg_unwrapped
  return None


def _get_sequence_element_type(typ: Any) -> Any | None:
  """If typ is a homogeneous Sequence/list/tuple, returns its element type."""
  origin = typing.get_origin(typ)
  if origin in (list, collections.abc.Sequence):
    args = typing.get_args(typ)
    if len(args) == 1:
      return args[0]
    return None
  if origin is tuple:
    args = typing.get_args(typ)
    if not args:
      return None
    if len(args) == 2 and args[1] is Ellipsis:
      return args[0]
    if all(a == args[0] for a in args):
      return args[0]
    return None
  return None


def _is_valid_base_type(annotation: Any) -> bool:
  annotation = _unwrap_annotated(annotation)
  if annotation in (
      bool,
      float,
      int,
      jax.Array,
      jax.Device,
      np.dtype,
      str,
      torch.Tensor,
      torch.device,
      torch.dtype,
      torch.types.Number,
  ):
    return True
  if isinstance(annotation, type):
    if issubclass(annotation, (np.dtype, np.generic, enum.Enum)):
      return True
  origin = typing.get_origin(annotation)
  if origin is typing.Literal:
    args = typing.get_args(annotation)
    return bool(args) and all(
        isinstance(a, (int, float, bool, str, enum.Enum)) for a in args
    )
  if origin in (Union, types.UnionType):
    args = typing.get_args(annotation)
    if set(args) <= {int, float, bool}:
      return True
    if all(isinstance(t, type) and issubclass(t, str) for t in args):
      return True
    if all(
        isinstance(t, type) and issubclass(t, int) and not issubclass(t, bool)
        for t in args
    ):
      return True
    if all(
        t in (torch.dtype, np.dtype)
        or (isinstance(t, type) and issubclass(t, (np.dtype, np.generic)))
        for t in args
    ):
      return True
    if all(t in (torch.device, jax.Device) for t in args):
      return True
  return False


def _is_valid_sequence_element_type(elem_type: Any) -> bool:
  underlying, is_optional = _strip_optional(elem_type)
  if underlying in (jax.Array, torch.Tensor):
    return True
  if is_optional:
    return False
  if underlying in (int, float, bool) or underlying == torch.types.Number:
    return True
  if (
      isinstance(underlying, type)
      and issubclass(underlying, int)
      and not issubclass(underlying, (bool, enum.Enum))
  ):
    return True
  if typing.get_origin(underlying) in (Union, types.UnionType):
    if set(typing.get_args(underlying)) <= {int, float, bool}:
      return True
  return False


def _is_valid_argument_type(annotation: Any) -> bool:
  """Returns True if the argument type is valid for a wrapped JAX function."""
  underlying, is_optional = _strip_optional(annotation)
  if _is_valid_base_type(underlying):
    return True
  elem_type = _get_sequence_element_type(underlying)
  if elem_type is not None and _is_valid_sequence_element_type(elem_type):
    return True
  return False


def _get_bound_names(fn: Any) -> set[str]:
  """Returns the set of parameter names bound by functools.partial."""
  bound_names = set()
  cur = fn
  while True:
    if isinstance(cur, functools.partial):
      if cur.keywords:
        bound_names.update(cur.keywords.keys())
      cur = cur.func
    elif hasattr(cur, "__wrapped__"):
      cur = cur.__wrapped__
    else:
      break
  return bound_names


def _is_valid_argument(
    param: inspect.Parameter,
    bound_names: collections.abc.Set[str] = frozenset(),
) -> bool:
  """Returns True if the argument is valid for a wrapped JAX function."""
  if param.name in bound_names:
    return True
  return _is_valid_argument_type(param.annotation)


def _verify_signature(
    signature: inspect.Signature,
    bound_names: collections.abc.Set[str] = frozenset(),
):
  """Verify that the signature contains only types supported by PyTorch custom_op.

  Args:
    signature: The signature of the function to verify.
    bound_names: Parameter names already bound (e.g. via functools.partial).

  Returns:
    True if the signature arguments are valid.
  """

  for param in signature.parameters.values():
    if param.annotation is inspect.Parameter.empty:
      raise ValueError(
          f"Missing argument type annotation for JAX function: {signature}."
      )

  def is_valid_return_base_type(typ: Any) -> bool:
    if typ in (jax.Array, torch.Tensor, int, float, bool, torch.types.Number):
      return True
    if (
        isinstance(typ, type)
        and issubclass(typ, int)
        and not issubclass(typ, (bool, enum.Enum))
    ):
      return True
    if typing.get_origin(typ) in (Union, types.UnionType):
      args = set(typing.get_args(typ))
      if args <= {int, float, bool}:
        return True
    return False

  def is_valid_result(result: Any):
    if result is None or result is types.NoneType:
      return True
    underlying, is_opt = _strip_optional(result)
    if is_opt:
      return is_valid_result(underlying)
    origin = typing.get_origin(result)
    if origin is None or origin in (Union, types.UnionType):
      return is_valid_return_base_type(result)
    args = typing.get_args(result)
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
      return is_valid_result(args[0])
    if origin in (
        tuple,
        list,
        collections.abc.Sequence,
    ):
      return all(is_valid_result(arg) for arg in args)
    return False

  invalid_arg_indices = [
      idx
      for idx, param in enumerate(signature.parameters.values())
      if not _is_valid_argument(param, bound_names=bound_names)
  ]
  result_valid = signature.return_annotation is None or is_valid_result(
      signature.return_annotation
  )

  if not invalid_arg_indices and result_valid:
    return

  error_messages: list[str] = []
  if invalid_arg_indices:
    error_messages.append(
        f"Arguments at indices {invalid_arg_indices} are invalid."
    )
  if not result_valid:
    error_messages.append("The return annotation is invalid.")

  raise ValueError(
      f"Invalid signature for JAX function: {signature}. "
      f"{' '.join(error_messages)} Only jax.Arrays and POD types are supported."
  )


def _is_tensor_annotation(annotation: Any) -> bool:
  """Returns True if the annotation represents a tensor or sequence of tensors."""
  underlying, _ = _strip_optional(annotation)
  if underlying in (jax.Array, torch.Tensor):
    return True
  elem_type = _get_sequence_element_type(underlying)
  if elem_type is not None:
    elem_underlying, _ = _strip_optional(elem_type)
    return elem_underlying in (jax.Array, torch.Tensor)
  return False


def _infer_static_argnums(signature: inspect.Signature) -> tuple[int, ...]:
  """Infers the static_argnums for a JAX function.

  All non-tensor arguments are considered static.

  Args:
    signature: The inspect.Signature of the JAX function.

  Returns:
    A tuple of argument indices that should be treated as static.
  """
  static_argnums = []
  for idx, (_, param) in enumerate(signature.parameters.items()):
    if param.kind is inspect.Parameter.KEYWORD_ONLY:
      continue
    if _is_tensor_annotation(param.annotation):
      continue
    static_argnums.append(idx)
  return tuple(static_argnums)


def _infer_static_argnames(signature: inspect.Signature) -> tuple[str, ...]:
  """Infers the static_argnames for keyword-only parameters of a JAX function."""
  static_argnames = []
  for name, param in signature.parameters.items():
    if param.kind is inspect.Parameter.KEYWORD_ONLY:
      if not _is_tensor_annotation(param.annotation):
        static_argnames.append(name)
  return tuple(static_argnames)


def _map_jax_base_to_torch(typ: Any) -> Any:
  if typ in ("int", int):
    return int
  if typ in ("float", float):
    return float
  if typ in ("bool", bool):
    return bool
  if typ in ("str", str):
    return str
  if typ in (
      "Tensor",
      "torch.Tensor",
      "Array",
      "jax.Array",
      jax.Array,
      torch.Tensor,
  ):
    return torch.Tensor
  if typ in (torch.dtype, np.dtype) or (
      isinstance(typ, type) and issubclass(typ, (np.dtype, np.generic))
  ):
    return torch.dtype
  if typ in (torch.device, jax.Device):
    return torch.device
  if typ == torch.types.Number:
    return torch.types.Number
  if isinstance(typ, type) and issubclass(typ, enum.Enum):
    if issubclass(typ, str):
      return str
    if issubclass(typ, int) and not issubclass(typ, bool):
      return int
    members = list(typ.__members__.values())
    if members:
      if all(isinstance(m.value, str) for m in members):
        return str
      if all(
          isinstance(m.value, int) and not isinstance(m.value, bool)
          for m in members
      ):
        return int
      if all(isinstance(m.value, float) for m in members):
        return float
      if all(isinstance(m.value, bool) for m in members):
        return bool
    return str
  origin = typing.get_origin(typ)
  if origin is typing.Literal:
    args = typing.get_args(typ)
    for candidate in (bool, int, float, str):
      if all(isinstance(a, candidate) for a in args):
        return candidate
    return str
  if origin in (Union, types.UnionType):
    args = typing.get_args(typ)
    if set(args) <= {int, float, bool}:
      return torch.types.Number
    if all(isinstance(t, type) and issubclass(t, str) for t in args):
      return str
    if all(
        isinstance(t, type) and issubclass(t, int) and not issubclass(t, bool)
        for t in args
    ):
      return int
    if all(
        t in (torch.dtype, np.dtype)
        or (isinstance(t, type) and issubclass(t, (np.dtype, np.generic)))
        for t in args
    ):
      return torch.dtype
    if all(t in (torch.device, jax.Device) for t in args):
      return torch.device
  return typ


def _map_jax_param_to_torch(typ: Any) -> Any:
  underlying, is_optional = _strip_optional(typ)
  elem_type = _get_sequence_element_type(underlying)
  if elem_type is not None:
    mapped_elem = _map_jax_param_to_torch(elem_type)
    if typing.get_origin(underlying) is list:
      seq_type = list[mapped_elem]  # pyrefly: ignore[not-a-type]
    else:
      seq_type = collections.abc.Sequence[mapped_elem]  # pyrefly: ignore[not-a-type]
    return (seq_type | types.NoneType) if is_optional else seq_type
  mapped = _map_jax_base_to_torch(underlying)
  return (mapped | types.NoneType) if is_optional else mapped


def _map_jax_return_to_torch(typ: Any) -> Any:
  if typ is None or typ is types.NoneType:
    return None
  if typ in (jax.Array, torch.Tensor):
    return torch.Tensor
  underlying, is_opt = _strip_optional(typ)
  if is_opt:
    mapped = _map_jax_return_to_torch(underlying)
    return mapped | types.NoneType
  origin = typing.get_origin(typ)
  if origin is tuple:
    args = typing.get_args(typ)
    if not args:
      return tuple[()]
    if len(args) == 2 and args[1] is Ellipsis:
      mapped_elem = _map_jax_return_to_torch(args[0])
      return tuple[mapped_elem, ...]  # pyrefly: ignore[not-a-type]
    mapped_args = tuple(_map_jax_return_to_torch(arg) for arg in args)
    return tuple[mapped_args]  # pyrefly: ignore[not-a-type]
  elem_type = _get_sequence_element_type(typ)
  if elem_type is not None:
    mapped_elem = _map_jax_return_to_torch(elem_type)
    if origin is list:
      return list[mapped_elem]  # pyrefly: ignore[not-a-type]
    return collections.abc.Sequence[mapped_elem]  # pyrefly: ignore[not-a-type]
  if (
      isinstance(typ, type)
      and issubclass(typ, int)
      and not issubclass(typ, (bool, enum.Enum))
  ):
    return int
  if typ == torch.types.Number or (
      origin in (Union, types.UnionType)
      and set(typing.get_args(typ)) <= {int, float, bool}
  ):
    return torch.types.Number
  return typ


def _map_jax_return_to_schema_str(typ: Any) -> str:
  """Converts a JAX return type annotation to a PyTorch C++ schema return string."""
  if typ is None or typ is types.NoneType:
    return "()"
  if typ in (jax.Array, torch.Tensor):
    return "Tensor"
  underlying, is_opt = _strip_optional(typ)
  if is_opt:
    base_str = _map_jax_return_to_schema_str(underlying)
    return f"{base_str}?"
  origin = typing.get_origin(typ)
  if origin is tuple:
    args = typing.get_args(typ)
    if not args:
      return "()"
    if len(args) == 2 and args[1] is Ellipsis:
      elem_str = _map_jax_return_to_schema_str(args[0])
      return f"{elem_str}[]"
    output_tys = [_map_jax_return_to_schema_str(arg) for arg in args]
    inner = ", ".join(output_tys)
    return f"({inner})"
  elem_type = _get_sequence_element_type(typ)
  if elem_type is not None:
    elem_str = _map_jax_return_to_schema_str(elem_type)
    return f"{elem_str}[]"
  if typ in (int, float, bool):
    return {"int": "SymInt", "float": "float", "bool": "bool"}[typ.__name__]
  if (
      isinstance(typ, type)
      and issubclass(typ, int)
      and not issubclass(typ, (bool, enum.Enum))
  ):
    return "SymInt"
  if typ == torch.types.Number or (
      origin in (Union, types.UnionType)
      and set(typing.get_args(typ)) <= {int, float, bool}
  ):
    return "Scalar"
  raise ValueError(f"Unsupported return type annotation: {typ}")


def _is_torch_supported_default(val: Any) -> bool:
  return val is None or isinstance(
      val, (int, float, bool, str, torch.device, torch.dtype)
  )


def _normalize_caller_arg(val: Any) -> Any:
  """Normalizes JAX values passed by caller into PyTorch custom_op types."""
  if val is None or isinstance(
      val, (torch.Tensor, int, float, bool, torch.dtype, torch.device)
  ):
    return val
  if isinstance(val, tuple):
    return tuple(_normalize_caller_arg(x) for x in val)
  if isinstance(val, list):
    return [_normalize_caller_arg(x) for x in val]
  if isinstance(val, enum.Enum):
    if isinstance(val.value, str):
      return str(val.value)
    if isinstance(val.value, int) and not isinstance(val.value, bool):
      return int(val.value)
    return val.value
  if isinstance(val, str):
    return val
  if val not in (int, float, bool, str, list, tuple, dict, set) and (
      isinstance(val, (np.dtype, np.generic))
      or hasattr(val, "dtype")
      or (isinstance(val, type) and issubclass(val, (np.generic, np.dtype)))
  ):
    try:
      jax_dt = np.dtype(val)
      if jax_dt in JAX_TO_TORCH_DTYPE_MAP:
        return JAX_TO_TORCH_DTYPE_MAP[jax_dt]
    except (TypeError, ValueError):
      pass
  if isinstance(val, jax.Device):
    return torch.device("tpu")
  return val


def _to_hashable_jax_static_arg(val: Any, orig_annotation: Any = None) -> Any:
  """Converts a PyTorch-dispatched static argument to a hashable JAX value."""
  if val is None:
    return None
  if orig_annotation is not None:
    enum_cls = _get_enum_type(orig_annotation)
    if enum_cls is not None and not isinstance(val, enum_cls):
      try:
        return enum_cls(val)
      except ValueError:
        if isinstance(val, str) and val in enum_cls.__members__:
          return enum_cls[val]
        raise ValueError(
            f"Value '{val}' is not a valid member of enum {enum_cls.__name__}"
        )
  if isinstance(val, (list, tuple)):
    elem_ann = (
        _get_sequence_element_type(_strip_optional(orig_annotation)[0])
        if orig_annotation is not None
        else None
    )
    return tuple(_to_hashable_jax_static_arg(x, elem_ann) for x in val)
  if isinstance(val, torch.dtype):
    if val not in TORCH_TO_JAX_DTYPE_MAP:
      raise NotImplementedError(f"Unsupported dtype for pallas kernels: {val}")
    return TORCH_TO_JAX_DTYPE_MAP[val]
  return val


def _get_torch_signature(
    signature: inspect.Signature,
    bound_names: collections.abc.Set[str] = frozenset(),
) -> inspect.Signature:
  """Converts a JAX signature to a Torch signature for custom_op schema."""
  pos_params = []
  kw_params = []

  for name, param in signature.parameters.items():
    if name in bound_names or not _is_valid_argument_type(param.annotation):
      continue

    mapped_annotation = _map_jax_param_to_torch(param.annotation)
    if param.default is not inspect.Parameter.empty:
      normalized_default = _normalize_caller_arg(param.default)
      if not _is_torch_supported_default(normalized_default):
        normalized_default = None
        _, is_opt = _strip_optional(mapped_annotation)
        if not is_opt:
          mapped_annotation = mapped_annotation | types.NoneType
      new_param = param.replace(
          annotation=mapped_annotation, default=normalized_default
      )
    else:
      new_param = param.replace(annotation=mapped_annotation)

    if new_param.kind is inspect.Parameter.KEYWORD_ONLY:
      kw_params.append(new_param)
    else:
      pos_params.append(new_param)

  new_parameters = pos_params + kw_params
  new_return_annotation = _map_jax_return_to_torch(signature.return_annotation)
  return inspect.Signature(
      new_parameters, return_annotation=new_return_annotation
  )


def _generate_torch_schema(
    torch_signature: inspect.Signature,
    ret_schema_str: str,
    mutates_args: tuple[str, ...] = (),
) -> str:
  """Generates an explicit PyTorch custom_op schema string."""

  def dummy_fn(*args, **kwargs):
    del args, kwargs

  dummy_params = list(torch_signature.parameters.values())
  dummy_sig = inspect.Signature(dummy_params, return_annotation=torch.Tensor)
  setattr(dummy_fn, "__signature__", dummy_sig)
  inferred = torch.library.infer_schema(dummy_fn, mutates_args=mutates_args)
  params_part = inferred.rsplit(" -> ", 1)[0]
  return f"{params_part} -> {ret_schema_str}"


_KernelResultT = Sequence[torch.Tensor] | torch.Tensor | None


class _JaxCustomOpDef(CustomOpDef):
  """CustomOpDef subclass that normalizes caller arguments for Torch schema."""

  def __call__(self, *args, **kwargs):
    norm_args = tuple(_normalize_caller_arg(a) for a in args)
    if kwargs:
      norm_kwargs = {k: _normalize_caller_arg(v) for k, v in kwargs.items()}
    else:
      norm_kwargs = {}
    return self._opoverload(*norm_args, **norm_kwargs)


def _normalize_mlir_module(mlir_module_text: str) -> str:
  """Returns the MLIR module re-printed without location debug info.

  Stripping locations lets differing caller stack frames produce the same
  deterministic fingerprint and reuse the compiled executable cache.

  Note that this only removes the outer module's debug information (#loc), not
  the inner debug info embedded inside the base64-encoded kernel
  (custom_call_config.body), which contains the Mosaic MLIR debug info that is
  more useful for debugging / profiling than the outer tracing call stack. The
  exact outer tracing call stack is deterministic but unstable, depending on
  which layer/callsite happens to trace first.

  Args:
    mlir_module_text: The textual MLIR module, e.g. from
      `jax.export.Exported.mlir_module()`.
  """
  with jax.interpreters.mlir.make_ir_context():
    ir_module = jax.interpreters.mlir.ir.Module.parse(mlir_module_text)
    return str(ir_module.operation.get_asm(enable_debug_info=False))


def _fingerprint_normalized_mlir(normalized_mlir: str) -> str:
  """Returns a content-addressable fingerprint of a normalized MLIR module."""
  # BLAKE2 is significantly faster than SHA-256, and 128 bits (16 bytes)
  # provides enough collision resistance before being fingerprinted down to
  # 64 bits in lower layers.
  return hashlib.blake2b(
      normalized_mlir.encode("utf-8"),
      digest_size=16,
  ).hexdigest()


class JaxCallable:
  """A torch callable function that wraps a JAX function.

  Supports static_argnums and static_argnames to allow passing non-tensor
  arguments directly.
  """

  name: str
  trace_key: str
  jit_fn: Any
  mesh: Any
  input_partition_specs: Any
  output_partition_specs: Any
  static_argnums: tuple[int, ...]
  static_argnames: tuple[str, ...]
  donate_argnums: list[int]
  input_output_aliases: dict[int, int]
  exported: Any
  __signature__: Any
  __globals__: Any

  def __init__(
      self,
      name: str,
      jit_fn: Any,
      trace_key: str,
      mesh: jax.sharding.Mesh | None = None,
      input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
      output_partition_specs: tuple[tuple[str, ...], ...] | None = None,
      static_argnums: tuple[int, ...] = (),
      donate_argnums: list[int] | None = None,
      input_output_aliases: dict[int, int] | None = None,
      static_argnames: tuple[str, ...] = (),
  ):
    """Initializes a JaxCallable, a cached callable for custom kernels.

    The callable is cached based on the (trace_key, input_shapes, input_kwargs),
    if a hash is not found the function is re-traced and re-registered. The JIT
    function passed in is a `jax.jit` wrapped pallas_call or function. This
    level of abstraction allows output types to be determined via tracing,
    instead of requiring they be passed in explicitly.

    Note that for JAX/python functions the trace_key should be the function's ID
    as well as a hash of any jit / pallas kwargs used to create the callable
    jit_fn.

    Args:
      name: The name of the JAX function, often including the function ID.
      jit_fn: The JAX function or wrapped pallas function output from `jax.jit`.
      trace_key: A string key of args / kwargs used to trace the exported
        function.
      mesh: If using a distributed kernel, provide the device mesh.
      input_partition_specs: If using a distributed kernel, provide the input
        partition specs for each input tensor.
      output_partition_specs: If using a distributed kernel, provide the output
        partition specs for each output tensor.
      static_argnums: Tuple of argument positions that are compile-time
        constants.
      donate_argnums: Indices of arguments to donate. See `jax.jit`.
      input_output_aliases: Indices of arguments to alias. This is deprecated
        and will be removed soon.
      static_argnames: Tuple of argument names that are compile-time constants.
    """

    self.name = name
    # TODO: Make trace_key stable across processes or remove it.
    self.trace_key = trace_key
    self.jit_fn = jit_fn
    self.output_shapes = {}  # cache output shapes per kernel specialization
    # Maps a process-local kernel key (representing the Python function,
    # argument shapes, and static arguments) to its content-addressable MLIR
    # module fingerprint.
    # Because `kernel_key` can vary across processes for the same Python
    # function, this mapping bridges process-dependent kernel keys to a
    # stable, deterministic fingerprint so that compilation caches can be
    # shared across processes.
    self.kernel_key_to_mlir_fingerprint = {}
    self.mesh = mesh
    self.input_partition_specs = input_partition_specs
    self.output_partition_specs = output_partition_specs
    self.static_argnums = static_argnums
    self.static_argnames = static_argnames
    # Ignore forward compatibility as we are directly using the exported
    # StableHLO. Without this flag we can be missing the latest optimizations.
    with jax._src.config.export_ignore_forward_compatibility(True):
      self.exported = jax.export.export(
          jit_fn,
          platforms=["tpu"],
          disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
      )
    if input_output_aliases is not None:
      _compat.warn_deprecation_with_skip(
          "input_output_aliases is deprecated and will be removed soon. Please"
          " use donate_argnums instead.",
          pathlib.Path(__file__).parent,
      )
      self.input_output_aliases = input_output_aliases
      # If donate_argnums is also provided, it must match.
      if donate_argnums is not None and self.input_output_aliases.keys() != set(
          donate_argnums
      ):
        raise ValueError(
            "donate_argnums must be None or the same as the keys of"
            " input_output_aliases."
        )
      self.donate_argnums = list(self.input_output_aliases.keys())
    else:
      self.donate_argnums = donate_argnums if donate_argnums is not None else []
      self.input_output_aliases = {}

    # Unwrap jax.jit and other wrappers to get the original function for its
    # __globals__.
    sig_source = inspect.unwrap(jit_fn)
    self._orig_signature = inspect.signature(sig_source, eval_str=True)
    self._orig_params = list(self._orig_signature.parameters.values())
    self._orig_params_by_name = self._orig_signature.parameters
    bound_names = _get_bound_names(sig_source)
    self.__signature__ = _get_torch_signature(
        self._orig_signature, bound_names=bound_names
    )
    self.__globals__ = None

    self._param_is_tensor = {
        name: (
            p.annotation is inspect.Parameter.empty
            or _is_tensor_annotation(p.annotation)
        )
        for name, p in self._orig_params_by_name.items()
    }
    self._static_param_names = set(self.static_argnames) | {
        self._orig_params[i].name
        for i in self.static_argnums
        if i < len(self._orig_params)
        and self._orig_params[i].kind != inspect.Parameter.VAR_POSITIONAL
    }
    self._param_name_to_idx = {
        p.name: idx for idx, p in enumerate(self._orig_params)
    }
    self._donate_argnums_set = set(self.donate_argnums)

    logging.debug("Creating JAX callable: %s", self)

  def __repr__(self):
    return (
        f"JaxCallable(name={self.name}, trace_key={self.trace_key},"
        f" static_argnums={self.static_argnums})"
    )

  def _validate_args(self, *args) -> None:
    """Validates that the arguments are supported by the kernel."""
    for i, arg in enumerate(args):
      if i in self.static_argnums:
        if isinstance(arg, torch.Tensor):
          raise ValueError(
              f"Argument at index {i} is marked as static but is a"
              " torch.Tensor. Tensors cannot be static arguments."
          )
        continue
      if arg is None or isinstance(arg, torch.Tensor):
        continue
      if isinstance(arg, (list, tuple)) and all(
          x is None or isinstance(x, torch.Tensor) for x in arg
      ):
        continue
      raise TypeError(
          "custom kernels only support torch.Tensor and None inputs, but"
          f" input {i} was of type {type(arg)}. Use static_argnums or"
          " functools.partial for non-tensor args."
      )

  def _normalize_invocation_args(
      self, args: Sequence[Any], kwargs: dict[str, Any]
  ) -> tuple[list[Any], dict[str, Any]]:
    """Binds arguments to signature and normalizes static arguments."""
    try:
      bound = self._orig_signature.bind(*args, **kwargs)
    except TypeError:
      bound = None

    if bound is not None:
      bound.apply_defaults()
      for name, val in bound.arguments.items():
        param = self._orig_params_by_name.get(name)
        if param is None or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
          continue
        is_tensor = self._param_is_tensor.get(name, False)
        is_static = name in self._static_param_names or not is_tensor

        # If schema defaulted non-optional container to None, restore original default.
        if val is None and param.default is not inspect.Parameter.empty:
          _, is_optional = _strip_optional(param.annotation)
          if not is_optional:
            val = param.default
            bound.arguments[name] = val

        if is_static:
          bound.arguments[name] = _to_hashable_jax_static_arg(
              val, param.annotation
          )

      norm_args = list(bound.args)
      for i in self.static_argnums:
        if i < len(norm_args) and (
            i >= len(self._orig_params)
            or self._orig_params[i].kind == inspect.Parameter.VAR_POSITIONAL
        ):
          norm_args[i] = _to_hashable_jax_static_arg(norm_args[i], None)
      return norm_args, dict(bound.kwargs)

    norm_args = []
    for i, arg in enumerate(args):
      param = self._orig_params[i] if i < len(self._orig_params) else None
      if i in self.static_argnums:
        ann = param.annotation if param is not None else None
        norm_args.append(_to_hashable_jax_static_arg(arg, ann))
      else:
        norm_args.append(arg)

    norm_kwargs = {}
    for k, v in kwargs.items():
      param = self._orig_params_by_name.get(k)
      ann = param.annotation if param is not None else None
      is_static = k in self._static_param_names or (
          param is not None and not self._param_is_tensor.get(k, False)
      )
      if is_static:
        norm_kwargs[k] = _to_hashable_jax_static_arg(v, ann)
      else:
        norm_kwargs[k] = v
    return norm_args, norm_kwargs

  def _get_jax_placeholders(
      self, norm_args: Sequence[Any], norm_kwargs: dict[str, Any] | None = None
  ) -> tuple[list[Any], dict[str, Any]]:
    """Converts normalized args and kwargs into JAX placeholders for tracing."""
    jax_args = jax_placeholders(
        norm_args, mesh=self.mesh, partition_specs=self.input_partition_specs
    )
    for i, ph in enumerate(jax_args):
      if isinstance(ph, list) and i not in self.static_argnums:
        param = self._orig_params[i] if i < len(self._orig_params) else None
        if param is not None:
          underlying, _ = _strip_optional(param.annotation)
          if typing.get_origin(underlying) is tuple:
            jax_args[i] = tuple(ph)

    jax_kwargs = {}
    if norm_kwargs:
      for k, v in norm_kwargs.items():
        param = self._orig_params_by_name.get(k)
        is_static = k in self._static_param_names or (
            param is not None and not self._param_is_tensor.get(k, False)
        )
        if is_static or not isinstance(v, (torch.Tensor, list, tuple)):
          jax_kwargs[k] = v
        else:
          ph = jax_placeholders(v, mesh=self.mesh)
          if isinstance(ph, list):
            if param is not None:
              underlying, _ = _strip_optional(param.annotation)
              if typing.get_origin(underlying) is tuple:
                ph = tuple(ph)
          jax_kwargs[k] = ph
    return jax_args, jax_kwargs

  def __call__(self, *args, **kwargs) -> _KernelResultT:
    self._validate_args(*args)
    norm_args, norm_kwargs = self._normalize_invocation_args(args, kwargs)

    # Check if we are running with wrapper tensors
    wrapper_arg = None
    for arg in itertools.chain(norm_args, norm_kwargs.values()):
      elems = arg if isinstance(arg, (list, tuple)) else (arg,)
      for x in elems:
        if hasattr(x, "_elem") and hasattr(x, "_env"):
          wrapper_arg = x
          break
      if wrapper_arg is not None:
        break

    if wrapper_arg is not None:
      env = wrapper_arg._env
      tensor_cls = type(wrapper_arg)

      def unwrap_elem(x):
        if hasattr(x, "_elem"):
          return x._elem
        if isinstance(x, (list, tuple)):
          return type(x)(unwrap_elem(e) for e in x)
        return x

      jax_args = [
          arg if i in self.static_argnums else unwrap_elem(arg)
          for i, arg in enumerate(norm_args)
      ]
      jax_kwargs = {k: unwrap_elem(v) for k, v in norm_kwargs.items()}

      jax_results = self.jit_fn(*jax_args, **jax_kwargs)

      def wrap(x):
        if isinstance(x, jax.Array):
          return tensor_cls(x, env)
        return x

      return jax.tree_util.tree_map(wrap, jax_results)

    kernel_key = _get_kernel_invocation_key(
        self.trace_key, norm_args, norm_kwargs, self.static_argnums
    )
    output_shapes, out_tree = self.output_shapes.get(kernel_key, (None, None))
    mlir_fingerprint = self.kernel_key_to_mlir_fingerprint.get(kernel_key)
    kernel_exists = (
        mlir_fingerprint is not None
        and tpu_torch_pallas.lookup_custom_kernel(self.name, mlir_fingerprint)
    )
    if not output_shapes or not kernel_exists:
      jax_args, jax_kwargs = self._get_jax_placeholders(norm_args, norm_kwargs)
      with jax._src.config.export_ignore_forward_compatibility(True):
        lowered = self.exported(*jax_args, **jax_kwargs)
      normalized_mlir = _normalize_mlir_module(lowered.mlir_module())
      mlir_fingerprint = _fingerprint_normalized_mlir(normalized_mlir)
      self.kernel_key_to_mlir_fingerprint[kernel_key] = mlir_fingerprint

      logging.debug(
          "Pallas kernel '%s' (invocation kernel_key=%s -> mlir_fingerprint=%s)"
          " MLIR module:\n%s",
          self.name,
          kernel_key,
          mlir_fingerprint,
          normalized_mlir,
      )

      tpu_torch_pallas.register_custom_kernel(
          self.name,
          mlir_fingerprint,
          serialized_mlir_module=lowered.mlir_module_serialized,
      )
      # Use out_tree to format the outputs shapes - i.e. pack in tuple, nested
      # tuple, etc.
      specs = _normalize_output_partition_specs(
          self.output_partition_specs, len(lowered.out_avals)
      ) or [None] * len(lowered.out_avals)
      output_shapes = [
          torch_placeholder(aval, mesh=self.mesh, partition_spec=spec)
          for aval, spec in zip(lowered.out_avals, specs)
      ]
      out_tree = lowered.out_tree
      self.output_shapes[kernel_key] = (output_shapes, out_tree)

    tensor_args: list[torch.Tensor] = []
    tensor_donate_argnums: list[int] = []
    tensor_io_aliases: dict[int, int] = {}

    def _add_tensor_arg(arg: Any, param_idx: int | None) -> None:
      if arg is None:
        return
      start_idx = len(tensor_args)
      if isinstance(arg, (list, tuple)):
        tensor_args.extend(t for t in arg if isinstance(t, torch.Tensor))
      elif isinstance(arg, torch.Tensor):
        tensor_args.append(arg)
      end_idx = len(tensor_args)
      if param_idx is not None and start_idx < end_idx:
        if param_idx in self._donate_argnums_set:
          tensor_donate_argnums.extend(range(start_idx, end_idx))
        if param_idx in self.input_output_aliases:
          tensor_io_aliases[start_idx] = self.input_output_aliases[param_idx]

    for i, arg in enumerate(norm_args):
      param = (
          self._orig_params[i]
          if i < len(self._orig_params)
          and self._orig_params[i].kind != inspect.Parameter.VAR_POSITIONAL
          else None
      )
      is_static = i in self.static_argnums or (
          param is not None
          and (
              param.name in self._static_param_names
              or not self._param_is_tensor.get(param.name, False)
          )
      )
      if not is_static:
        _add_tensor_arg(arg, i)

    for k, arg in norm_kwargs.items():
      param = self._orig_params_by_name.get(k)
      is_static = k in self._static_param_names or (
          param is not None and not self._param_is_tensor.get(k, False)
      )
      if not is_static:
        _add_tensor_arg(arg, self._param_name_to_idx.get(k))

    device_assignment: list[int] = []
    if self.mesh is not None:
      _validate_pallas_mesh_world_size(mesh=self.mesh)
      device_assignment = [int(d.id) for d in self.mesh.devices.flat]

    results = tpu_torch_pallas.call_custom_kernel(
        self.name,
        mlir_fingerprint,
        inputs=tensor_args,
        output_shapes=output_shapes,
        donate_argnums=tensor_donate_argnums,
        device_assignment=device_assignment,
    )

    for in_idx, out_idx in tensor_io_aliases.items():
      tensor_args[in_idx].copy_(results[out_idx])

    unflattened = out_tree.unflatten(results)
    return unflattened


def _validate_pallas_mesh_world_size(
    *,
    mesh: jax.sharding.Mesh | None = None,
    mesh_shape: Sequence[int] | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> None:
  """Validates that Pallas mesh, mesh_shape, and process_group match world size."""
  if not dist.is_initialized():
    return
  world_size = dist.get_world_size()
  if process_group is not None:
    pg_size = dist.get_world_size(process_group)
    if pg_size != world_size:
      raise ValueError(
          "torch_tpu pallas kernels currently require full world-size meshes"
          f" only. Got process_group with size {pg_size}, expected"
          f" {world_size}."
      )
  if mesh_shape is not None:
    mesh_size = math.prod(mesh_shape)
    if mesh_size != world_size:
      raise ValueError(
          f"Pallas mesh shape {mesh_shape} product ({mesh_size}) must match"
          f" world size ({world_size})."
      )
  if mesh is not None and mesh.size != world_size:
    raise ValueError(
        f"Pallas kernel mesh size ({mesh.size}) must match world size"
        f" ({world_size})."
    )


@overload
def custom_jax_kernel(
    jax_fn: None = None,
    name: str | None = None,
    static_argnums: tuple[int, ...] = (),
    donate_argnums: list[int] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    output_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    *,
    static_argnames: tuple[str, ...] = (),
    warn_deprecated: bool = True,
) -> Callable[[Callable[..., Any]], JaxCallable]:
  ...  # Overload stub


@overload
def custom_jax_kernel(
    jax_fn: Callable[..., Any],
    name: str | None = None,
    static_argnums: tuple[int, ...] = (),
    donate_argnums: list[int] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    output_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    *,
    static_argnames: tuple[str, ...] = (),
    warn_deprecated: bool = True,
) -> JaxCallable:
  ...  # Overload stub


def custom_jax_kernel(
    jax_fn=None,
    name=None,
    static_argnums=(),
    donate_argnums=None,
    input_output_aliases=None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs=None,
    output_partition_specs=None,
    *,
    static_argnames=(),
    warn_deprecated: bool = True,
):
  """Deprecated: Please use `jax_op` instead.

  A decorator that imports a JAX kernel into for use with torch_tpu.

  Often pallas kernel libraries are written with thin JAX usability layers, i.e.
  in tokamax. `custom_jax_kernel` allows for interop with these JAX kernel
  wrappers.

  ```py
  # Option 1: Using functools.partial (original approach)
  tokamax_ragged_dot = functools.partial(
      tokamax.ragged_dot,
      implementation="mosaic",
      preferred_element_type=jnp.bfloat16
  )
  ragged_dot = pallas.custom_jax_kernel(tokamax_ragged_dot)

  # Option 2: Using static_argnums
  ragged_dot = pallas.custom_jax_kernel(
      tokamax.ragged_dot,
      static_argnums=(2, 3),  # Mark positions 2 and 3 as static
  )
  # Call with static args directly:
  ragged_dot(lhs, rhs, "mosaic", jnp.bfloat16)
  ```

  Args:
    jax_fn: The JAX function to compile. If None then return a decorator that
      expects a jax_fn argument.
    name: Optional debug friendly kernel name, attempt to use the function's
      name if not provided.
    static_argnums: Tuple of argument positions that are compile-time constants.
      These args can be non-tensors and are used for caching by value.
    donate_argnums: The indexes of the arguments to donate. After an argument
      donated, it will be left in an invalid state and should not be used again.
    input_output_aliases: Indices of arguments to alias. This is deprecated and
      will be removed soon.
    mesh: If using a distributed kernel, provide the device mesh.
    input_partition_specs: If using a distributed kernel, provide the input
      partition specs for each input tensor.
    output_partition_specs: If using a distributed kernel, provide the output
      partition specs for each output tensor.
    static_argnames: Tuple of argument names that are compile-time constants.
    warn_deprecated: If True, issue a deprecation warning.

  Returns:
    A decorator that takes a JAX function and returns a callable that
    can be used to call the kernel with torch.Tensor inputs.
  """
  if warn_deprecated:
    _compat.warn_deprecation_with_skip(
        "the use of `custom_jax_kernel` is deprecated and will be removed soon."
        " Please use `jax_op` instead.",
        pathlib.Path(__file__).parent,
    )

  _validate_pallas_mesh_world_size(mesh=mesh)

  def decorator(jax_fn: Callable[..., Any]) -> JaxCallable:
    nonlocal name
    if name is None:
      name = getattr(jax_fn, "__name__", "kernel")
    name_key = f"{name}_{id(jax_fn)}"  # distinguish funcs with same name

    trace_key = _get_kernel_invocation_key(
        name_key,
        [],
        {
            "static_argnums": static_argnums,
            "static_argnames": static_argnames,
            "donate_argnums": donate_argnums,
        },
    )

    jit_kwargs: dict[str, Any] = {
        "static_argnums": static_argnums,
        "donate_argnums": donate_argnums,
        "keep_unused": True,
    }
    if static_argnames:
      jit_kwargs["static_argnames"] = static_argnames

    jit_fn = jax.jit(jax_fn, **jit_kwargs)
    return JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        input_output_aliases=input_output_aliases,
        static_argnames=static_argnames,
    )

  if jax_fn is None:
    return decorator
  return decorator(jax_fn)


@overload
def jax_op(
    name: str,
    fn: None = None,
    /,
    *,
    donate_argnums: Sequence[int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    output_partition_specs: tuple[tuple[str, ...], ...] | None = None,
) -> Callable[[Callable[..., Any]], CustomOpDef]:
  ...  # Overload stub


@overload
def jax_op(
    name: str,
    fn: Callable[..., Any],
    /,
    *,
    donate_argnums: Sequence[int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    output_partition_specs: tuple[tuple[str, ...], ...] | None = None,
) -> CustomOpDef:
  ...  # Overload stub


@annotations.experimental(
    "torch.tpu.pallas.jax_op is experimental and subject to change."
)
def jax_op(
    name,
    fn=None,
    /,
    *,
    donate_argnums=None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs=None,
    output_partition_specs=None,
):
  """Registers a JAX function as a custom PyTorch operation.

  This decorator allows a JAX function, typically a Pallas kernel wrapped
  with `custom_jax_kernel`, to be exposed as a `torch.library.custom_op`.
  It handles the necessary tracing and registration with the TPU backend.

  If arguments are donated those tensors will be left in an invalid state after
  the kernel is called and should not be used again. Consider using a wrapper
  function and copy_ or set_ to overwrite these invalid tensors to prevent
  accidental reuse.

  Args:
    name: The name of the custom operator.
    fn: The JAX function to wrap. If None, `pallas_op` acts as a decorator.
    donate_argnums: A sequence of argument indices to donate.
    mesh: If using a distributed kernel, provide the device mesh.
    input_partition_specs: If using a distributed kernel, provide the input
      partition specs for each input tensor.
    output_partition_specs: If using a distributed kernel, provide the output
      partition specs for each output tensor.

  Returns:
    A `CustomOpDef` instance if `fn` is provided, or a decorator that returns
    a `CustomOpDef` when applied to a function.
  """

  if "::" not in name:
    raise ValueError(f"Op name {name} does not contain a namespace.")

  if len(name.split("::")) != 2:
    raise ValueError(f"Op name {name} must have exactly one '::' separator.")

  def dec(fn: Callable[..., object]) -> CustomOpDef:
    nonlocal name

    bound_names = _get_bound_names(fn)
    # Unwrap any jax.jit or decorators before getting signature to preserve
    # globals.
    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    _verify_signature(signature, bound_names=bound_names)
    static_argnums = _infer_static_argnums(signature)
    static_argnames = _infer_static_argnames(signature)
    wrapped_fn = custom_jax_kernel(
        fn,
        name,
        donate_argnums=donate_argnums,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        warn_deprecated=False,
    )
    # Jax functions are inherently immutable, even donated args are immutable in
    # that they are in an invalid state rather than mutated state, users can
    # copy back if they require.
    # If we mark donated args as mutable then Dynamo will copy back the
    # original value of the donated arg which then defeats the purpose of
    # donating.
    mutates_args = ()

    # We require that the user pass us a function that is make_fx traceable,
    # so we can just register it as the Fake/meta kernel.
    def fake_fn(*args, **kwargs):
      # Symbolic dimensions can implicitly convert to int which then creates a
      # very confusing error message, so we check for them explicitly here.
      dev: torch.device | str = "meta"
      for arg in itertools.chain(args, kwargs.values()):
        tensors_to_check = arg if isinstance(arg, (list, tuple)) else (arg,)
        for t in tensors_to_check:
          if isinstance(t, torch.Tensor):
            dev = t.device
            if any(isinstance(d, torch.SymInt) for d in t.shape):
              raise RuntimeError(
                  "Symbolic dimensions are not supported in the default fake"
                  " kernel. Either remove symbolic dimensions or override this"
                  " default implementation."
              )
      # pylint: disable=protected-access
      norm_args, norm_kwargs = wrapped_fn._normalize_invocation_args(
          args, kwargs
      )
      jax_args, jax_kwargs = wrapped_fn._get_jax_placeholders(
          norm_args, norm_kwargs
      )
      with jax._src.config.export_ignore_forward_compatibility(True):
        lowered = wrapped_fn.exported(*jax_args, **jax_kwargs)
      specs = _normalize_output_partition_specs(
          output_partition_specs, len(lowered.out_avals)
      ) or [None] * len(lowered.out_avals)
      unflattened = lowered.out_tree.unflatten(
          torch_placeholder(aval, mesh=mesh, partition_spec=spec, device=dev)
          for aval, spec in zip(lowered.out_avals, specs)
      )
      # pylint: enable=protected-access
      return unflattened

    ret_schema_str = _map_jax_return_to_schema_str(signature.return_annotation)
    schema_str = _generate_torch_schema(
        wrapped_fn.__signature__, ret_schema_str, mutates_args=mutates_args
    )

    result = torch.library.custom_op(
        name,
        wrapped_fn,
        mutates_args=mutates_args,
        schema=schema_str,
    )
    result.__class__ = _JaxCustomOpDef

    result.register_fake(fake_fn)

    return result

  if fn is None:
    return dec
  else:
    return dec(fn)


def get_pallas_mesh(
    axis_names: Sequence[str] | str,
    mesh_shape: Sequence[int] | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> jax.sharding.Mesh:
  """Constructs a rank-aligned JAX Mesh using TPU runtime device metadata."""
  if not dist.is_initialized():
    raise RuntimeError(
        "torch.distributed is not initialized. dist.init_process_group()"
    )

  world_size = dist.get_world_size()
  _validate_pallas_mesh_world_size(
      mesh_shape=mesh_shape, process_group=process_group
  )
  if (
      mesh_shape is not None
      and not isinstance(axis_names, str)
      and len(axis_names) != len(mesh_shape)
  ):
    raise ValueError("does not match length")
  if (
      mesh_shape is None
      and not isinstance(axis_names, str)
      and len(axis_names) > 1
  ):
    raise ValueError("must be explicitly specified")

  attrs = _device_ops_backend._get_local_device_attributes()
  dev_name = attrs.get("device_kind")

  pg_devs = device_mesh.get_device_assignment(process_group)

  devices = [
      _PallasDevice(
          id=pg_devs[i] if (pg_devs and i < len(pg_devs)) else i,
          process_index=i,
          device_kind=dev_name,
      )
      for i in range(world_size)
  ]

  if mesh_shape is not None:
    mesh_devices = np.array(devices).reshape(mesh_shape)
  else:
    mesh_devices = np.array(devices)

  return jax.sharding.Mesh(mesh_devices, axis_names=axis_names)


def get_default_pallas_mesh(
    process_group: dist.ProcessGroup | None = None,
) -> jax.sharding.Mesh:
  """Constructs default 1D world rank-aligned JAX Mesh with axis ('rank',)."""
  return get_pallas_mesh(
      axis_names=("rank",), mesh_shape=None, process_group=process_group
  )
