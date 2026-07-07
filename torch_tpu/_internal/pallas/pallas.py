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

import inspect
import pathlib
import types
import typing
from typing import Any, Callable, Sequence, Union, overload
from absl import logging
import frozendict
import torch
from torch._library.custom_ops import CustomOpDef
import torch.library
from torch_tpu._internal.pallas import _compat
from torch_tpu._internal.pallas import tpu_torch_pallas

try:
  # TODO: TorchXLA hit issues when workin with both JAX and PT in the same
  # process, hitting deadlocks when each framework tried to access the PJRT
  # plugin. Unclear if this applies for trace-only, but we may need to better
  # protect against this, see @requires_jax in PT/XLA repo.
  import jax
  import jax.export
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
    # torch.float4_e2m1fn_x2: jax.numpy.float4_e2m1fn.dtype,  # x2 is packed
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
JAX_TO_TORCH_DTYPE_MAP = {v: k for k, v in TORCH_TO_JAX_DTYPE_MAP.items()}


def _convert_shape(shape, mesh, partition_spec, op):
  """Converts a shape based on a mesh and partition specification."""
  if mesh is not None and partition_spec is not None:
    ans = tuple(
        [
            op(s, mesh.devices.shape[mesh.axis_names.index(name)])
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
    tensor: torch.Tensor, mesh=None, partition_spec=None
) -> jax.ShapeDtypeStruct:
  """Converts a torch tensor to a jax placeholder for tracing."""
  if not isinstance(tensor, torch.Tensor):
    # Preserve POD constants / None types
    return tensor

  jax_dtype = TORCH_TO_JAX_DTYPE_MAP.get(tensor.dtype, None)
  logging.debug("[jax_placeholder] dtype: %s -> %s", tensor.dtype, jax_dtype)
  if jax_dtype is None:
    raise NotImplementedError(
        f"Unsupported dtype for pallas kernels: {tensor.dtype}"
    )
  if mesh is not None:
    return jax.ShapeDtypeStruct(
        get_global_shape(tensor.shape, mesh, partition_spec),
        jax_dtype,
        sharding=jax.sharding.NamedSharding(mesh, partition_spec),  # pyrefly: ignore[bad-argument-type]
    )
  else:
    return jax.ShapeDtypeStruct(tensor.shape, jax_dtype)


def jax_placeholders(
    tensors: Sequence[torch.Tensor] | torch.Tensor,
    mesh=None,
    partition_specs=None,
):
  """Converts a sequence of torch tensors to a sequence of jax placeholders."""
  if isinstance(tensors, torch.Tensor):
    return jax_placeholder(tensors, mesh, partition_specs)
  if mesh is not None:
    return [
        jax_placeholder(tensor, mesh, partition_spec)
        for tensor, partition_spec in zip(tensors, partition_specs)  # pyrefly: ignore[bad-argument-type]
    ]
  else:
    return [jax_placeholder(tensor) for tensor in tensors]


def torch_placeholder(
    tensor: jax.core.ShapedArray | None,
    mesh=None,
) -> torch.Tensor | None:
  """Converts a jax output to a torch tensor for return type.

  JAX JIT functions only support tensor and None types, all other POD types will
  error.
  """
  if not isinstance(tensor, jax.core.ShapedArray):
    # Preserve POD constants / None types
    return tensor

  torch_dtype = JAX_TO_TORCH_DTYPE_MAP.get(tensor.dtype, None)
  logging.debug(
      "[torch_placeholder] dtype: %s -> %s", tensor.dtype, torch_dtype
  )
  if torch_dtype is None:
    raise NotImplementedError(
        f"Unsupported dtype for pallas kernels: {tensor.dtype}"
    )
  return torch.empty(
      get_local_shape(tensor.shape, mesh, tensor.sharding.spec),
      dtype=torch_dtype,
      device="tpu",
  )


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
    return str(item)

  input_kwargs_str = ";".join(
      f"{get_hash(-1, key)}:{get_hash(-1, value)}"
      for key, value in input_kwargs.items()
  )

  inputs_shapes = [get_hash(i, arg) for i, arg in enumerate(input_args)]
  inputs_shapes_str = ",".join(inputs_shapes)
  return ";".join([trace_key, inputs_shapes_str, input_kwargs_str])


def _is_valid_base_type(annotation: type[Any]) -> bool:
  return annotation in (
      jax.Array,
      int,
      float,
      bool,
      str,
  )


def _get_underlying_type_from_optional(typ: type[Any]) -> type[Any] | None:
  """Returns the underlying type of an optional type, or None if not optional."""
  # Optional types are simply a union of the underlying type and None.
  # get_origin gets the base type of subscripted types
  # e.g. get_origin(Union[T, NoneType]) => Union
  if typing.get_origin(typ) not in (Union, types.UnionType):
    return None
  # get_args gets the subscripted type arguments
  # e.g. get_args(Union[T, NoneType]) => (T, NoneType)
  union_types = typing.get_args(typ)
  concrete_args = [t for t in union_types if t is not types.NoneType]

  # If there is more than one concrete type, then this is not a simple optional
  # type.
  if len(concrete_args) != 1:
    return None
  return concrete_args[0]


def _is_valid_argument_type(annotation: type[Any]) -> bool:
  """Returns True if the argument type is valid for a wrapped JAX function."""
  if (underling := _get_underlying_type_from_optional(annotation)) is not None:
    return _is_valid_base_type(underling)
  return _is_valid_base_type(annotation)


def _is_valid_argument(param: inspect.Parameter) -> bool:
  """Returns True if the argument is valid for a wrapped JAX function.

  Note this includes and defaulted kw only arguments such as those produces by
  `functools.partial`.

  Args:
    param: The function parameter to validate.
  """
  # Defaulted keyword only arguments are allowed as they will be simply bound to
  # the jax.jit call, we must support this as it is how functools.partial works.
  if (
      param.kind is inspect.Parameter.KEYWORD_ONLY
      and param.default is not inspect.Parameter.empty
  ):
    return True
  return _is_valid_argument_type(param.annotation)


def _verify_signature(signature: inspect.Signature):
  """verify that the signature contains only the allowed types.

  Allowed types are, where arguments can be optional:
    - jax.Array
    - int
    - float
    - bool
    - str

  Args:
    signature: The signature of the function to verify.

  Returns:
    True if the signature arguments are valid.
  """

  for param in signature.parameters.values():
    if param.annotation is inspect.Parameter.empty:
      raise ValueError(
          f"Missing argument type annotation for JAX function: {signature}."
      )

  def is_valid_result(result: Any):
    if result is None or result is types.NoneType:
      return True
    if result in (dict, list, tuple, set):
      return True
    origin = typing.get_origin(result)
    if origin is None:
      return _is_valid_base_type(result)
    args = typing.get_args(result)
    if origin in (tuple, list, set, dict, Union, types.UnionType):
      return all(is_valid_result(arg) for arg in args)
    return False

  invalid_arg_indices = [
      idx
      for idx, param in enumerate(signature.parameters.values())
      if not _is_valid_argument(param)
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


def _infer_static_argnums(signature: inspect.Signature):
  """Infers the static_argnums for a JAX function.

  All non-tensor arguments are considered static.

  Args:
    signature: The inspect.Signature of the JAX function.

  Returns:
    A tuple of argument indices that should be treated as static.
  """
  static_argnums = []
  for idx, (_, param) in enumerate(signature.parameters.items()):
    annotation = param.annotation
    if param.kind is inspect.Parameter.KEYWORD_ONLY:
      continue
    if annotation is jax.Array:
      continue
    if _get_underlying_type_from_optional(annotation) is jax.Array:
      continue
    static_argnums.append(idx)
  return tuple(static_argnums)


def _get_torch_signature(signature: inspect.Signature):
  """Converts a JAX signature to a Torch signature."""

  new_parameters = []

  def _map_jax_to_torch(typ):
    if typ is jax.Array:
      return torch.Tensor
    if (underling := _get_underlying_type_from_optional(typ)) is not None:
      return _map_jax_to_torch(underling) | types.NoneType
    origin = typing.get_origin(typ)
    if origin is not None:
      mapped_args = (_map_jax_to_torch(arg) for arg in typing.get_args(typ))
      return origin[*mapped_args]  # pyrefly: ignore[not-a-type]

    return typ

  for param in signature.parameters.values():
    # Skip invalid argument types, these should already have been verified
    # to have a default value.
    if not _is_valid_argument_type(param.annotation):
      continue
    new_parameters.append(
        param.replace(annotation=_map_jax_to_torch(param.annotation))
    )
  new_return_annotation = _map_jax_to_torch(signature.return_annotation)
  return inspect.Signature(
      new_parameters, return_annotation=new_return_annotation
  )


_KernelResultT = Sequence[torch.Tensor] | torch.Tensor | None


class JaxCallable:
  """A torch callable function that wraps a JAX function.

  Supports static_argnums to allow passing non-tensor arguments directly.
  """
  name: str
  trace_key: str
  jit_fn: Any
  mesh: Any
  input_partition_specs: Any
  static_argnums: tuple[int, ...]
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
      static_argnums: tuple[int, ...] = (),
      donate_argnums: list[int] | None = None,
      input_output_aliases: dict[int, int] | None = None,
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
      static_argnums: Tuple of argument positions that are compile-time
        constants.
      donate_argnums: Indices of arguments to donate. See `jax.jit`.
      input_output_aliases: Indices of arguments to alias. This is deprecated
        and will be removed soon.
    """

    self.name = name
    self.trace_key = trace_key
    self.jit_fn = jit_fn
    self.output_shapes = {}  # cache output shapes per kernel specialization
    self.mesh = mesh
    self.input_partition_specs = input_partition_specs
    self.static_argnums = static_argnums
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

    self.__signature__ = _get_torch_signature(inspect.signature(jit_fn))
    self.__globals__ = None

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
      if arg is not None and not isinstance(arg, torch.Tensor):
        raise TypeError(
            "custom kernels only support torch.Tensor and None inputs, but"
            f" input {i} was of type {type(arg)}. Use static_argnums or"
            " functools.partial for non-tensor args."
        )

  def __call__(self, *args, **kwargs) -> _KernelResultT:
    self._validate_args(*args)

    # Check if we are running with wrapper tensors
    wrapper_arg = None
    for arg in args:
      if hasattr(arg, "_elem") and hasattr(arg, "_env"):
        wrapper_arg = arg
        break
    if wrapper_arg is None:
      for arg in kwargs.values():
        if hasattr(arg, "_elem") and hasattr(arg, "_env"):
          wrapper_arg = arg
          break

    if wrapper_arg is not None:
      env = wrapper_arg._env
      tensor_cls = type(wrapper_arg)

      jax_args = []
      for i, arg in enumerate(args):
        if i in self.static_argnums:
          jax_args.append(arg)
        elif arg is None:
          jax_args.append(None)
        elif hasattr(arg, "_elem"):
          jax_args.append(arg._elem)
        else:
          jax_args.append(arg)

      jax_kwargs = {}
      for k, v in kwargs.items():
        if hasattr(v, "_elem"):
          jax_kwargs[k] = v._elem
        else:
          jax_kwargs[k] = v

      jax_results = self.jit_fn(*jax_args, **jax_kwargs)

      def wrap(x):
        if isinstance(x, jax.Array):
          return tensor_cls(x, env)
        return x

      return jax.tree_util.tree_map(wrap, jax_results)

    kernel_key = _get_kernel_invocation_key(
        self.trace_key, args, kwargs, self.static_argnums
    )
    output_shapes, out_tree = self.output_shapes.get(kernel_key, (None, None))
    kernel_exists = tpu_torch_pallas.lookup_custom_kernel(self.name, kernel_key)
    if not output_shapes or not kernel_exists:
      jax_args = jax_placeholders(
          args, mesh=self.mesh, partition_specs=self.input_partition_specs
      )
      with jax._src.config.export_ignore_forward_compatibility(True):
        lowered = self.exported(*jax_args, **kwargs)
      tpu_torch_pallas.register_custom_kernel(
          self.name,
          kernel_key,
          serialized_mlir_module=lowered.mlir_module_serialized,
      )
      # Use out_tree to format the outputs shapes - i.e. pack in tuple, nested
      # tuple, etc.
      output_shapes = [
          torch_placeholder(aval, mesh=self.mesh) for aval in lowered.out_avals
      ]
      out_tree = lowered.out_tree
      self.output_shapes[kernel_key] = (output_shapes, out_tree)

    tensor_args = [
        arg
        for i, arg in enumerate(args)
        if arg is not None and i not in self.static_argnums
    ]

    results = tpu_torch_pallas.call_custom_kernel(
        self.name,
        kernel_key,
        inputs=tensor_args,
        output_shapes=output_shapes,
        donate_argnums=self.donate_argnums,
    )

    for in_idx, out_idx in self.input_output_aliases.items():
      tensor_args[in_idx].copy_(results[out_idx])

    return out_tree.unflatten(results)


@overload
def custom_jax_kernel(
    jax_fn: None = None,
    name: str | None = None,
    static_argnums: tuple[int, ...] = (),
    donate_argnums: list[int] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    *,
    warn_deprecated: bool = True,
) -> Callable[[Callable[..., Any]], JaxCallable]:
  ...


@overload
def custom_jax_kernel(
    jax_fn: Callable[..., Any],
    name: str | None = None,
    static_argnums: tuple[int, ...] = (),
    donate_argnums: list[int] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
    *,
    warn_deprecated: bool = True,
) -> JaxCallable:
  ...


def custom_jax_kernel(
    jax_fn=None,
    name=None,
    static_argnums=(),
    donate_argnums=None,
    input_output_aliases=None,
    mesh=None,
    input_partition_specs=None,
    *,
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

  def decorator(jax_fn: Callable[..., Any]) -> JaxCallable:
    nonlocal name
    if name is None:
      name = getattr(jax_fn, "__name__", "kernel")
    name_key = f"{name}_{id(jax_fn)}"  # distinguish funcs with same name

    trace_key = _get_kernel_invocation_key(
        name_key,
        [],
        {"static_argnums": static_argnums, "donate_argnums": donate_argnums},
    )

    jit_fn = jax.jit(
        jax_fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )
    return JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        input_output_aliases=input_output_aliases,
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
) -> Callable[[Callable[..., Any]], CustomOpDef]:
  ...


@overload
def jax_op(
    name: str,
    fn: Callable[..., Any],
    /,
    *,
    donate_argnums: Sequence[int] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None,
) -> CustomOpDef:
  ...


def jax_op(
    name,
    fn=None,
    /,
    *,
    donate_argnums=None,
    mesh=None,
    input_partition_specs=None,
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
    mesh: jax.sharding.Mesh | None = None,
    input_partition_specs: tuple[tuple[str, ...], ...] | None = None

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

    signature = inspect.signature(fn, follow_wrapped=False)
    _verify_signature(signature)
    static_argnums = _infer_static_argnums(signature)
    wrapped_fn = custom_jax_kernel(
        fn,
        name,
        donate_argnums=donate_argnums,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        static_argnums=static_argnums,
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
      for arg in args:
        if isinstance(arg, torch.Tensor) and any(
            isinstance(d, torch.SymInt) for d in arg.shape
        ):
          raise RuntimeError(
              "Symbolic dimensions are not supported in the default fake"
              " kernel. Either remove symbolic dimensions or override this"
              " default implementation."
          )
      jax_args = jax_placeholders(
          args, mesh=mesh, partition_specs=input_partition_specs
      )
      with jax._src.config.export_ignore_forward_compatibility(True):
        lowered = wrapped_fn.exported(*jax_args, **kwargs)
      return lowered.out_tree.unflatten(
          torch_placeholder(aval, mesh=mesh) for aval in lowered.out_avals
      )

    result = torch.library.custom_op(
        name,
        wrapped_fn,
        mutates_args=mutates_args,
    )

    result.register_fake(fake_fn)

    return result

  if fn is None:
    return dec
  else:
    return dec(fn)
