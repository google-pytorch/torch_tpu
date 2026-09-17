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

"""Direct tracing and asynchronous compilation for TorchTPU.

This module provides an independent compilation API that bypasses PyTorch
Dynamo,
AOT Autograd, and FX graph transformations entirely. It executes callables
directly
within TorchTPU's C++ deferred execution engine (`INTERNAL_COMPILE_FX_GRAPH`),
extracts the resulting deferred IR, and submits XLA/PJRT compilation
asynchronously
to a background thread pool, returning a Future[CompiledArtifact].
"""

from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import functools
import threading
from typing import Any, Protocol, runtime_checkable

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
    CompiledArtifact,
    NoOpCompiledArtifact,
    TorchTpuCompiledExecutable,
)
from torch_tpu._internal.benchmarks import xprof_adapter

_TRACE_LOCK = threading.Lock()

_EXECUTOR: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="tpu_trace_compile"
    )
)


@dataclasses.dataclass(frozen=True)
class _TensorLayout:
  dtype: torch.dtype
  shape: torch.Size
  stride: tuple[int, ...]
  storage_offset: int


@dataclasses.dataclass(frozen=True)
class _TensorDedupeKey:
  """A key for deduplicating tensors based on their data and view layout."""

  data_ptr: int
  layout: _TensorLayout

  @classmethod
  def from_tensor(cls, tensor: torch.Tensor) -> "_TensorDedupeKey":
    return cls(
        data_ptr=tensor.data_ptr(),
        layout=_TensorLayout(
            dtype=tensor.dtype,
            shape=tensor.shape,
            stride=tensor.stride(),
            storage_offset=tensor.storage_offset(),
        ),
    )


@dataclasses.dataclass(frozen=True)
class _DedupedTensorOutput:
  tensor_idx: int


@dataclasses.dataclass(frozen=True)
class _NonTensorPassthrough:
  input_idx: int


def _find_extra_tensor_attributes(
    module: torch.nn.Module,
) -> list[tuple[torch.nn.Module, str, torch.Tensor]]:
  """Finds direct torch.Tensor attributes that are not parameters/buffers."""
  extra_tensors: list[tuple[torch.nn.Module, str, torch.Tensor]] = []
  for mod in module.modules():
    for attr_name, attr_val in list(mod.__dict__.items()):
      if (
          isinstance(attr_val, torch.Tensor)
          and not isinstance(attr_val, torch.nn.Parameter)
          and attr_name not in mod._buffers
          and attr_name not in mod._parameters
      ):
        extra_tensors.append((mod, attr_name, attr_val))
  return extra_tensors


def _get_tensor_storage_ptr(t: Any) -> int:
  """Returns untyped storage data pointer of a tensor, or 0 if empty."""
  if not isinstance(t, torch.Tensor) or t.numel() == 0:
    return 0
  try:
    return t.untyped_storage().data_ptr()
  except Exception:  # pylint: disable=broad-exception-caught
    return t.data_ptr()


def _assert_storage_disjoint(
    mutated_tensors: Sequence[Any],
    other_tensors: Sequence[Any] = (),
) -> None:
  """Validates that mutated tensors do not share storage with other tensors."""
  seen_storage_ptrs: set[int] = set()
  for t in mutated_tensors:
    s_ptr = _get_tensor_storage_ptr(t)
    if s_ptr != 0:
      if s_ptr in seen_storage_ptrs:
        raise NotImplementedError(
            "In-place mutation on aliased input tensors is not supported in"
            " trace_and_compile."
        )
      seen_storage_ptrs.add(s_ptr)

  for t in other_tensors:
    s_ptr = _get_tensor_storage_ptr(t)
    if s_ptr != 0 and s_ptr in seen_storage_ptrs:
      raise NotImplementedError(
          "In-place mutation on aliased input tensors is not supported in"
          " trace_and_compile."
      )


def _contains_tensors(obj: Any) -> bool:
  """Recursively checks if obj contains any torch.Tensor instances."""
  if isinstance(obj, torch.Tensor):
    return True
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return any(
        _contains_tensors(getattr(obj, f.name)) for f in dataclasses.fields(obj)
    )
  if isinstance(obj, (list, tuple, set, frozenset)):
    return any(_contains_tensors(item) for item in obj)
  if isinstance(obj, dict):
    return any(
        _contains_tensors(k) or _contains_tensors(v) for k, v in obj.items()
    )
  return False


@dataclasses.dataclass(frozen=True)
class _GeneratorContext:
  argument_generators: list[torch.Generator]
  export_mode_generators: dict[torch.Generator, Any]
  has_implicit_default_generator: bool


@dataclasses.dataclass(frozen=True)
class _TracingInputs:
  placeholder_flat_inputs: list[Any]
  argument_tensors: list[torch.Tensor]
  internal_layouts: list[Any] | None
  placeholder_params: dict[str, torch.Tensor]
  placeholder_buffers: dict[str, torch.Tensor]
  extra_tensors: list[torch.Tensor]
  placeholder_extra_tensors: list[torch.Tensor]


@dataclasses.dataclass(frozen=True)
class _MutationAnalysis:
  raw_outputs: Any
  flat_outputs: list[Any]
  out_spec: _pytree.TreeSpec
  mutated_input_map: dict[int, tuple[int, ...]]
  extra_mutated_inputs: list[int]


@dataclasses.dataclass(frozen=True)
class _DeduplicatedOutputs:
  deduped_outputs: list[torch.Tensor]
  reconstructor: tpu_torch_compile.FxOutputReconstructor
  unique_output_indices: list[int]
  constant_outputs: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class _TraceCompilationMetadata:
  in_spec: _pytree.TreeSpec
  is_simple_tensor_args: bool
  expected_tensor_specs: Sequence[
      tuple[int, torch.Size, torch.dtype, torch.device]
  ]
  expected_non_tensor_specs: Sequence[tuple[int, Any]]
  extra_tensors: Sequence[torch.Tensor]
  constant_outputs: tuple[Any, ...]
  mutations: _MutationAnalysis


class _TraceCompiledExecutableWrapper(CompiledArtifact):
  """Wrapper around TorchTpuCompiledExecutable.

  Supports arbitrary args/kwargs and nn.Modules.
  """

  def __init__(
      self,
      executable: CompiledArtifact,
      metadata: _TraceCompilationMetadata,
      module: torch.nn.Module | None = None,
      debug_container: TpuCompileDebug | None = None,
  ):
    super().__init__()
    self._executable = executable
    self._in_spec = metadata.in_spec
    self._is_simple_tensor_args = metadata.is_simple_tensor_args
    self.debug = debug_container
    self._fast_runner: Callable[..., Any] | None = getattr(
        executable, "_fast_runner", None
    )
    if self._fast_runner is not None:
      self._runner: Callable[..., Any] = self._fast_runner
    else:
      self._runner = self._call_executable_unpacked

    base_params: tuple[Any, ...] = ()
    if module is not None:
      base_params = (
          tuple(module.parameters())
          + tuple(module.buffers())
          + tuple(metadata.extra_tensors)
      )
    elif metadata.extra_tensors:
      base_params = tuple(metadata.extra_tensors)

    self._constant_outputs = tuple(metadata.constant_outputs)
    self._params_and_buffers = base_params + self._constant_outputs

    self._has_params = bool(self._params_and_buffers)
    self._expected_arg_count = len(metadata.expected_tensor_specs) + len(
        metadata.expected_non_tensor_specs
    )
    self._fast_specs = tuple(
        (spec[0], spec[1], spec[2], spec[3].type)
        for spec in metadata.expected_tensor_specs
    )
    self._expected_non_tensor_specs = tuple(metadata.expected_non_tensor_specs)
    self._mutated_input_map: dict[int, tuple[int, ...]] = dict(
        metadata.mutations.mutated_input_map
    )
    self._extra_mutated_inputs = tuple(metadata.mutations.extra_mutated_inputs)
    self._has_mutations = bool(
        self._mutated_input_map or self._extra_mutated_inputs
    )

  @property
  def underlying_executable(self) -> CompiledArtifact:
    return self._executable

  @property
  def mlir_text(self) -> str | None:
    return getattr(self._executable, "mlir_text", None)

  @mlir_text.setter
  def mlir_text(self, value: str) -> None:
    self._executable.mlir_text = value

  @property
  def mlir_texts(self) -> Sequence[str]:
    return getattr(self._executable, "mlir_texts", ())

  @property
  def graph_module_debug_str(self) -> str | None:
    return getattr(self._executable, "graph_module_debug_str", None)

  @graph_module_debug_str.setter
  def graph_module_debug_str(self, value: str) -> None:
    self._executable.graph_module_debug_str = value

  @property
  def graph_module_debug_strs(self) -> Sequence[str]:
    return getattr(self._executable, "graph_module_debug_strs", ())

  def updates_default_generator_state(self) -> bool:
    return getattr(
        self._executable, "updates_default_generator_state", lambda: False
    )()

  def fingerprint(self) -> str:
    return getattr(self._executable, "fingerprint", lambda: "")()

  def __getattr__(self, name: str) -> Any:
    return getattr(self._executable, name)

  def _call_executable_unpacked(self, inputs: Sequence[Any]) -> Any:
    return self._executable(*inputs)

  def _handle_mutations(self, flat_inputs: Sequence[Any], res: Any) -> Any:
    if not self._has_mutations:
      return res

    user_res = res
    if self._extra_mutated_inputs:
      user_res, extra_mutated_tensors = res
      for input_idx, mut_t in zip(
          self._extra_mutated_inputs, extra_mutated_tensors
      ):
        flat_inputs[input_idx].copy_(mut_t)

    if self._mutated_input_map:
      flat_outputs, out_spec = _pytree.tree_flatten(user_res)
      flat_outputs = list(flat_outputs)
      for input_idx, out_indices in self._mutated_input_map.items():
        flat_inputs[input_idx].copy_(flat_outputs[out_indices[0]])
        for out_idx in out_indices:
          flat_outputs[out_idx] = flat_inputs[input_idx]
      user_res = _pytree.tree_unflatten(flat_outputs, out_spec)

    return user_res

  def _validate_fast_tensor_specs(self, inputs: Sequence[Any]) -> None:
    for (
        idx,
        expected_shape,
        expected_dtype,
        expected_device_type,
    ) in self._fast_specs:
      actual = inputs[idx]
      if not isinstance(actual, torch.Tensor):
        raise TypeError(
            f"Expected tensor argument at index {idx}, but got"
            f" {type(actual).__name__}"
        )
      if actual.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for argument {idx}: expected {expected_shape}, got"
            f" {actual.shape}"
        )
      if actual.dtype != expected_dtype:
        raise TypeError(
            f"Dtype mismatch for argument {idx}: expected {expected_dtype}, got"
            f" {actual.dtype}"
        )
      if actual.device.type != expected_device_type:
        raise ValueError(
            f"Device mismatch for argument {idx}: expected"
            f" {expected_device_type}, got {actual.device}"
        )

  def _validate_runtime_mutation_aliasing(self, inputs: Sequence[Any]) -> None:
    mutated_indices = set(self._mutated_input_map.keys()).union(
        self._extra_mutated_inputs
    )
    mutated = [inputs[idx] for idx in mutated_indices]
    others = [
        inputs[idx] for idx in range(len(inputs)) if idx not in mutated_indices
    ]
    if self._has_params:
      others.extend(self._params_and_buffers)
    _assert_storage_disjoint(mutated, others)

  def _validate_tensor_specs(self, flat_inputs: Sequence[Any]) -> None:
    self._validate_fast_tensor_specs(flat_inputs)

    for idx, expected_val in self._expected_non_tensor_specs:
      actual = flat_inputs[idx]
      if type(actual) is not type(expected_val):
        raise TypeError(
            f"Type mismatch for non-tensor argument at index {idx}: expected"
            f" {type(expected_val).__name__}, got {type(actual).__name__}"
        )
      try:
        is_mismatch = bool(actual != expected_val)
      except Exception:  # pylint: disable=broad-exception-caught
        is_mismatch = True
      if is_mismatch:
        raise ValueError(
            f"Argument value mismatch for non-tensor argument at index {idx}:"
            f" expected {expected_val}, got {actual}. Non-tensor arguments are"
            " specialized at trace time."
        )

    if self._has_mutations:
      self._validate_runtime_mutation_aliasing(flat_inputs)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if not kwargs and self._is_simple_tensor_args:
      if len(args) != self._expected_arg_count:
        raise ValueError(
            f"Expected {self._expected_arg_count} arguments, but got"
            f" {len(args)}"
        )
      self._validate_fast_tensor_specs(args)
      if self._has_mutations:
        self._validate_runtime_mutation_aliasing(args)

      all_inputs = args + self._params_and_buffers if self._has_params else args
      if not self._has_mutations:
        return self._runner(all_inputs)

      res = self._runner(all_inputs)
      return self._handle_mutations(args, res)

    return self._call_general(args, kwargs)

  def _call_general(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    sorted_kwargs = dict(sorted(kwargs.items())) if kwargs else kwargs
    flat_inputs, in_spec = _pytree.tree_flatten((args, sorted_kwargs))
    if in_spec != self._in_spec:
      raise ValueError(
          f"Input structure mismatch: expected {self._in_spec}, got {in_spec}"
      )
    self._validate_tensor_specs(flat_inputs)
    all_inputs = (
        (*flat_inputs, *self._params_and_buffers)
        if self._has_params
        else flat_inputs
    )
    res = self._runner(all_inputs)
    return self._handle_mutations(flat_inputs, res)

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    raise TypeError(
        f"{self.__class__.__name__} cannot be pickled directly; pickle the"
        " underlying executable via .underlying_executable instead."
    )


def _setup_generators(flat_inputs: Sequence[Any]) -> _GeneratorContext:
  device = torch.accelerator.current_accelerator()
  device_module = getattr(torch, device.type)
  default_device_index = device.index or 0
  default_generator = device_module.default_generators[default_device_index]

  argument_generators: list[torch.Generator] = []
  for item in flat_inputs:
    if isinstance(item, torch.Generator) and item not in argument_generators:
      argument_generators.append(item)
  has_implicit_default_generator = default_generator not in argument_generators
  if has_implicit_default_generator:
    argument_generators.append(default_generator)

  export_mode_generators = {
      gen: tpu_torch_compile.make_export_mode_device_generator(gen)
      for gen in argument_generators
  }
  tpu_torch_compile.set_export_mode_default_device_generator(
      export_mode_generators[default_generator]
  )
  return _GeneratorContext(
      argument_generators=argument_generators,
      export_mode_generators=export_mode_generators,
      has_implicit_default_generator=has_implicit_default_generator,
  )


def _is_valid_layout(layout: Any, rank: int) -> bool:
  if isinstance(layout, tuple) and layout:
    minor_to_major = layout[0]
  elif isinstance(layout, list):
    minor_to_major = layout
  else:
    return False
  return len(minor_to_major) == rank and sorted(minor_to_major) == list(
      range(rank)
  )


def _extract_device_layout(val: Any) -> Any:
  """Extracts physical TPU layout if `val` has a materialized layout."""
  if (
      isinstance(val, torch.Tensor)
      and val.device.type == "tpu"
      and val.is_contiguous()
      and val.numel() > 0
      and not isinstance(val, FakeTensor)
  ):
    rank = val.dim()
    try:
      layout = tpu_torch_compile.get_device_layout_if_materialized(val)
      if layout is not None and _is_valid_layout(layout, rank):
        minor_to_major = layout[0] if isinstance(layout, tuple) else layout
        if minor_to_major == list(range(rank - 1, -1, -1)):
          return []
        return layout
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return []


def _extract_module(fn: Callable[..., Any]) -> torch.nn.Module | None:
  """Extracts underlying torch.nn.Module if fn is a module or bound method."""
  if isinstance(fn, torch.nn.Module):
    return fn
  self_obj = getattr(fn, "__self__", None)
  if isinstance(self_obj, torch.nn.Module):
    return self_obj
  if isinstance(fn, functools.partial):
    return _extract_module(fn.func)
  return None


@dataclasses.dataclass(frozen=True)
class _InputPlaceholders:
  placeholder_flat_inputs: list[Any]
  argument_tensors: list[torch.Tensor]
  layouts: list[Any]


def _build_input_placeholders(
    flat_inputs: Sequence[Any],
    gen_ctx: _GeneratorContext,
    argument_layouts: Sequence[Any] | None,
) -> _InputPlaceholders:
  """Builds placeholder inputs and extracts layouts for example inputs."""
  placeholder_flat_inputs: list[Any] = []
  argument_tensors: list[torch.Tensor] = []
  layouts: list[Any] = []
  tensor_input_idx = 0

  for item in flat_inputs:
    if isinstance(item, torch.Tensor):
      ph = tpu_torch_compile.placeholder_like(item)
      placeholder_flat_inputs.append(ph)
      argument_tensors.append(ph)
      if argument_layouts is not None and tensor_input_idx < len(
          argument_layouts
      ):
        layouts.append(argument_layouts[tensor_input_idx])
      else:
        layouts.append(_extract_device_layout(item))
      tensor_input_idx += 1
    elif isinstance(item, torch.Generator):
      placeholder_flat_inputs.append(gen_ctx.export_mode_generators[item])
    else:
      placeholder_flat_inputs.append(item)

  return _InputPlaceholders(
      placeholder_flat_inputs=placeholder_flat_inputs,
      argument_tensors=argument_tensors,
      layouts=layouts,
  )


@dataclasses.dataclass(frozen=True)
class _ModulePlaceholders:
  placeholder_params: dict[str, torch.Tensor]
  placeholder_buffers: dict[str, torch.Tensor]
  extra_tensors: list[torch.Tensor]
  placeholder_extra_tensors: list[torch.Tensor]
  argument_tensors: list[torch.Tensor]
  layouts: list[Any]


def _build_module_placeholders(
    target_mod: torch.nn.Module | None,
    saved_extra_tensor_attrs: list[tuple[torch.nn.Module, str, torch.Tensor]],
) -> _ModulePlaceholders:
  """Builds placeholders for module parameters, buffers, and extra attrs."""
  placeholder_params: dict[str, torch.Tensor] = {}
  placeholder_buffers: dict[str, torch.Tensor] = {}
  extra_tensors: list[torch.Tensor] = []
  placeholder_extra_tensors: list[torch.Tensor] = []
  argument_tensors: list[torch.Tensor] = []
  layouts: list[Any] = []

  if target_mod is None:
    return _ModulePlaceholders(
        placeholder_params=placeholder_params,
        placeholder_buffers=placeholder_buffers,
        extra_tensors=extra_tensors,
        placeholder_extra_tensors=placeholder_extra_tensors,
        argument_tensors=argument_tensors,
        layouts=layouts,
    )

  for name, p in target_mod.named_parameters():
    ph = tpu_torch_compile.placeholder_like(p)
    placeholder_params[name] = ph
    argument_tensors.append(ph)
    layouts.append(_extract_device_layout(p))

  for name, b in target_mod.named_buffers():
    ph = tpu_torch_compile.placeholder_like(b)
    placeholder_buffers[name] = ph
    argument_tensors.append(ph)
    layouts.append(_extract_device_layout(b))

  placeholder_by_tensor_id: dict[int, torch.Tensor] = {}
  for mod, attr_name, tensor in _find_extra_tensor_attributes(target_mod):
    if id(tensor) not in placeholder_by_tensor_id:
      ph = tpu_torch_compile.placeholder_like(tensor)
      placeholder_by_tensor_id[id(tensor)] = ph
      extra_tensors.append(tensor)
      placeholder_extra_tensors.append(ph)
      argument_tensors.append(ph)
      layouts.append(_extract_device_layout(tensor))
    else:
      ph = placeholder_by_tensor_id[id(tensor)]
    saved_extra_tensor_attrs.append((mod, attr_name, tensor))
    setattr(mod, attr_name, ph)

  return _ModulePlaceholders(
      placeholder_params=placeholder_params,
      placeholder_buffers=placeholder_buffers,
      extra_tensors=extra_tensors,
      placeholder_extra_tensors=placeholder_extra_tensors,
      argument_tensors=argument_tensors,
      layouts=layouts,
  )


def _build_generator_placeholders(
    gen_ctx: _GeneratorContext,
) -> tuple[list[torch.Tensor], list[Any]]:
  """Builds argument tensors and layouts for RNG generators."""
  argument_tensors: list[torch.Tensor] = []
  layouts: list[Any] = []
  for gen in gen_ctx.argument_generators:
    argument_tensors.append(
        tpu_torch_compile.get_device_state_tensor(
            gen_ctx.export_mode_generators[gen]
        )
    )
    layouts.append([])
  return argument_tensors, layouts


def _build_tracing_placeholders(
    target_mod: torch.nn.Module | None,
    flat_inputs: Sequence[Any],
    gen_ctx: _GeneratorContext,
    argument_layouts: Sequence[Any] | None,
    saved_extra_tensor_attrs: list[tuple[torch.nn.Module, str, torch.Tensor]],
) -> _TracingInputs:
  """Builds placeholder arguments, parameters, buffers, and extra attributes."""
  input_phs = _build_input_placeholders(flat_inputs, gen_ctx, argument_layouts)
  mod_phs = _build_module_placeholders(target_mod, saved_extra_tensor_attrs)
  gen_arg_tensors, gen_layouts = _build_generator_placeholders(gen_ctx)

  all_argument_tensors = (
      input_phs.argument_tensors + mod_phs.argument_tensors + gen_arg_tensors
  )
  all_layouts = input_phs.layouts + mod_phs.layouts + gen_layouts

  has_custom_layouts = argument_layouts is not None
  internal_layouts = (
      all_layouts if (has_custom_layouts or any(all_layouts)) else None
  )

  return _TracingInputs(
      placeholder_flat_inputs=input_phs.placeholder_flat_inputs,
      argument_tensors=all_argument_tensors,
      internal_layouts=internal_layouts,
      placeholder_params=mod_phs.placeholder_params,
      placeholder_buffers=mod_phs.placeholder_buffers,
      extra_tensors=mod_phs.extra_tensors,
      placeholder_extra_tensors=mod_phs.placeholder_extra_tensors,
  )


def _execute_trace(
    fn: Callable[..., Any],
    target_mod: torch.nn.Module | None,
    tracing_inputs: _TracingInputs,
    in_spec: _pytree.TreeSpec,
) -> tuple[Any, list[Any], list[int | None]]:
  """Executes `fn` in INTERNAL_COMPILE_FX_GRAPH mode on cloned placeholders."""
  previous_is_compiling = getattr(torch.compiler, "_is_compiling_flag", False)
  torch.compiler._is_compiling_flag = True
  try:
    trace_flat_inputs: list[Any] = []
    initial_versions: list[int | None] = []
    for ph in tracing_inputs.placeholder_flat_inputs:
      if isinstance(ph, torch.Tensor):
        cloned = ph.clone()
        trace_flat_inputs.append(cloned)
        initial_versions.append(cloned._version)
      else:
        trace_flat_inputs.append(ph)
        initial_versions.append(None)

    pos_args, kw_args = _pytree.tree_unflatten(trace_flat_inputs, in_spec)
    with xprof_adapter.TraceMe("trace_and_compile.trace"):
      if target_mod is not None:
        reparam_state = {
            **tracing_inputs.placeholder_params,
            **tracing_inputs.placeholder_buffers,
        }
        with torch.nn.utils.stateless._reparametrize_module(
            target_mod, reparam_state, tie_weights=True
        ):
          raw_outputs = fn(*pos_args, **kw_args)
      else:
        raw_outputs = fn(*pos_args, **kw_args)
    return raw_outputs, trace_flat_inputs, initial_versions
  finally:
    torch.compiler._is_compiling_flag = previous_is_compiling


def _assert_module_immutability(
    placeholder_params: dict[str, torch.Tensor],
    placeholder_buffers: dict[str, torch.Tensor],
    placeholder_extra_tensors: Sequence[torch.Tensor],
) -> None:
  """Ensures module parameters, buffers, and extra attributes are immutable."""
  for name, p in placeholder_params.items():
    if p._version != 0:
      raise NotImplementedError(
          f"In-place mutation of module parameter '{name}' is not supported in"
          " trace_and_compile."
      )
  for name, b in placeholder_buffers.items():
    if b._version != 0:
      raise NotImplementedError(
          f"In-place mutation of module buffer '{name}' is not supported in"
          " trace_and_compile."
      )
  for ph in placeholder_extra_tensors:
    if ph._version != 0:
      raise NotImplementedError(
          "In-place mutation of module extra tensor attribute is not supported"
          " in trace_and_compile."
      )


def _assert_no_storage_aliasing(
    flat_inputs: Sequence[Any],
    mutated_indices: Sequence[int],
    donated_inputs: Sequence[int] | None,
    module_params_and_buffers: Sequence[torch.Tensor] = (),
) -> None:
  """Validates mutated inputs do not share storage or conflict with donation."""
  if not mutated_indices:
    return

  mutated_set = set(mutated_indices)
  mutated = [flat_inputs[idx] for idx in mutated_indices]
  others = [
      flat_inputs[idx]
      for idx in range(len(flat_inputs))
      if idx not in mutated_set
  ]
  if module_params_and_buffers:
    others.extend(module_params_and_buffers)

  _assert_storage_disjoint(mutated, others)

  if donated_inputs:
    flat_to_arg_tensor_idx: dict[int, int] = {}
    tensor_count = 0
    for idx, item in enumerate(flat_inputs):
      if isinstance(item, torch.Tensor):
        flat_to_arg_tensor_idx[idx] = tensor_count
        tensor_count += 1

    mutated_arg_tensor_indices = {
        flat_to_arg_tensor_idx[idx]
        for idx in mutated_indices
        if idx in flat_to_arg_tensor_idx
    }
    conflicts = set(donated_inputs).intersection(mutated_arg_tensor_indices)
    if conflicts:
      raise ValueError(
          f"Cannot donate inputs {sorted(conflicts)} that are mutated in-place."
      )


def _resolve_mutation_mappings(
    mutated_input_indices: Sequence[int],
    trace_flat_inputs: Sequence[Any],
    flat_outputs: list[Any],
    raw_outputs: Any,
) -> tuple[
    dict[int, tuple[int, ...]],
    list[int],
    Any,
    list[Any],
    _pytree.TreeSpec,
]:
  """Maps mutated inputs to matching outputs or appends extra outputs."""
  mutated_input_map: dict[int, tuple[int, ...]] = {}
  extra_mutated_inputs: list[int] = []
  extra_mutation_tensors: list[torch.Tensor] = []

  for input_idx in mutated_input_indices:
    mutated_t = trace_flat_inputs[input_idx]
    matching_out_indices = tuple(
        out_idx
        for out_idx, out_t in enumerate(flat_outputs)
        if out_t is mutated_t
    )
    if matching_out_indices:
      mutated_input_map[input_idx] = matching_out_indices
    else:
      extra_mutated_inputs.append(input_idx)
      extra_mutation_tensors.append(mutated_t)

  if extra_mutation_tensors:
    raw_outputs = (raw_outputs, extra_mutation_tensors)
    flat_outputs, out_spec = _pytree.tree_flatten(raw_outputs)
  else:
    _, out_spec = _pytree.tree_flatten(raw_outputs)

  return (
      mutated_input_map,
      extra_mutated_inputs,
      raw_outputs,
      flat_outputs,
      out_spec,
  )


def _process_in_place_mutations(
    raw_outputs: Any,
    flat_inputs: Sequence[Any],
    trace_flat_inputs: Sequence[Any],
    initial_versions: Sequence[int | None],
    placeholder_params: dict[str, torch.Tensor],
    placeholder_buffers: dict[str, torch.Tensor],
    placeholder_extra_tensors: Sequence[torch.Tensor],
    donated_inputs: Sequence[int] | None,
    module_params_and_buffers: Sequence[torch.Tensor] = (),
) -> _MutationAnalysis:
  """Validates and analyzes in-place mutations occurring during tracing."""
  _assert_module_immutability(
      placeholder_params, placeholder_buffers, placeholder_extra_tensors
  )

  flat_outputs, _ = _pytree.tree_flatten(raw_outputs)
  flat_outputs = list(flat_outputs)

  mutated_input_indices = [
      idx
      for idx, (traced_ph, init_ver) in enumerate(
          zip(trace_flat_inputs, initial_versions)
      )
      if init_ver is not None and traced_ph._version != init_ver
  ]

  _assert_no_storage_aliasing(
      flat_inputs,
      mutated_input_indices,
      donated_inputs,
      module_params_and_buffers,
  )

  (
      mutated_input_map,
      extra_mutated_inputs,
      raw_outputs,
      flat_outputs,
      out_spec,
  ) = _resolve_mutation_mappings(
      mutated_input_indices, trace_flat_inputs, flat_outputs, raw_outputs
  )

  return _MutationAnalysis(
      raw_outputs=raw_outputs,
      flat_outputs=flat_outputs,
      out_spec=out_spec,
      mutated_input_map=mutated_input_map,
      extra_mutated_inputs=extra_mutated_inputs,
  )


def _deduplicate_outputs(
    flat_outputs: Sequence[Any],
    trace_flat_inputs: Sequence[Any],
    base_constant_offset: int,
    out_spec: _pytree.TreeSpec,
) -> _DeduplicatedOutputs:
  """Builds passthrough mappings and deduplicates output tensors."""
  input_passthrough_map: dict[int, int] = {}
  for input_idx, ph in enumerate(trace_flat_inputs):
    if not isinstance(ph, (torch.Tensor, int, float, bool, str, bytes)):
      input_passthrough_map[id(ph)] = input_idx

  deduped_outputs: list[torch.Tensor] = []
  expected_layouts: list[_TensorLayout] = []
  output_map: list[_DedupedTensorOutput | _NonTensorPassthrough | None] = []
  index_by_dedupe_key: dict[_TensorDedupeKey, int] = {}
  constant_outputs: list[Any] = []

  for item in flat_outputs:
    if item is None:
      output_map.append(None)
    elif isinstance(item, torch.Tensor):
      key = _TensorDedupeKey.from_tensor(item)
      if key not in index_by_dedupe_key:
        index_by_dedupe_key[key] = len(deduped_outputs)
        deduped_outputs.append(item)
        expected_layouts.append(key.layout)
      output_map.append(_DedupedTensorOutput(index_by_dedupe_key[key]))
    elif id(item) in input_passthrough_map:
      output_map.append(_NonTensorPassthrough(input_passthrough_map[id(item)]))
    else:
      const_idx = len(constant_outputs)
      constant_outputs.append(item)
      output_map.append(_NonTensorPassthrough(base_constant_offset + const_idx))

  unique_output_indices = [
      entry.tensor_idx
      for entry in output_map
      if isinstance(entry, _DedupedTensorOutput)
  ]

  reconstructor = tpu_torch_compile.FxOutputReconstructor(
      output_map,
      expected_layouts,
      out_spec,
      None,
  )

  return _DeduplicatedOutputs(
      deduped_outputs=deduped_outputs,
      reconstructor=reconstructor,
      unique_output_indices=unique_output_indices,
      constant_outputs=tuple(constant_outputs),
  )


def _resolve_generator_outputs(
    gen_ctx: _GeneratorContext,
    argument_tensors: list[torch.Tensor],
    deduped_outputs: list[torch.Tensor],
    internal_layouts: list[Any] | None,
) -> tuple[bool, bool]:
  """Appends RNG generator end states and checks whether this is a no-op graph.

  Returns:
    (updates_default_generator_state, is_noop_graph)
  """
  for gen in gen_ctx.argument_generators:
    end_state_tensor = tpu_torch_compile.get_device_state_tensor(
        gen_ctx.export_mode_generators[gen]
    )
    deduped_outputs.append(end_state_tensor)

  if gen_ctx.has_implicit_default_generator:
    begin_default_state_tensor = argument_tensors[-1]
    end_default_state_tensor = deduped_outputs[-1]
    updates_default_generator_state = (
        begin_default_state_tensor.data_ptr()
        != end_default_state_tensor.data_ptr()
    )
    if not updates_default_generator_state:
      if len(deduped_outputs) > 1:
        argument_tensors.pop()
        deduped_outputs.pop()
        if internal_layouts is not None:
          internal_layouts.pop()
      else:
        return False, True
  else:
    updates_default_generator_state = False

  return updates_default_generator_state, False


def _cleanup_tracing(
    saved_extra_tensor_attrs: Sequence[
        tuple[torch.nn.Module, str, torch.Tensor]
    ],
) -> None:
  try:
    for mod, attr_name, orig_tensor in saved_extra_tensor_attrs:
      setattr(mod, attr_name, orig_tensor)
  finally:
    tpu_torch_compile.clear_export_mode_default_device_generators()


@runtime_checkable
class CompiledCallable(Protocol):
  """A callable compiled artifact with optional debug information."""

  debug: TpuCompileDebug | None

  @property
  def underlying_executable(self) -> CompiledArtifact:
    ...

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    ...


def _make_noop_future(
    reconstructor: tpu_torch_compile.FxOutputReconstructor,
    metadata: _TraceCompilationMetadata,
    *,
    module: torch.nn.Module | None,
    debug_container: TpuCompileDebug | None,
) -> concurrent.futures.Future[CompiledCallable]:
  noop_artifact = _TraceCompiledExecutableWrapper(
      NoOpCompiledArtifact(reconstructor),
      metadata,
      module=module,
      debug_container=debug_container,
  )
  f: concurrent.futures.Future[CompiledCallable] = concurrent.futures.Future()
  f.set_result(noop_artifact)
  return f


def _package_executable_future(
    async_compile_result: tpu_torch_compile.AsyncCompileResult,
    reconstructor: tpu_torch_compile.FxOutputReconstructor,
    unique_output_indices: Sequence[int],
    updates_default_generator_state: bool,
    metadata: _TraceCompilationMetadata,
    *,
    module: torch.nn.Module | None,
    debug_container: TpuCompileDebug | None,
    enable_debug: bool,
) -> concurrent.futures.Future[CompiledCallable]:
  mlir_text: str | None = None
  if (
      enable_debug
      and debug_container is not None
      and async_compile_result.module is not None
  ):
    mlir_text = tpu_torch_compile.serialize_mlir_text(
        async_compile_result.module, enable_debug_info=True
    )
    debug_container.stablehlo_forward_text.append(mlir_text)

  def _build_artifact(
      loaded_executable: tpu_torch_compile.LoadedExecutableWithMetadata,
  ) -> CompiledCallable:
    executable = TorchTpuCompiledExecutable(
        executable=loaded_executable,
        reconstruct_fx_outputs_fn=reconstructor,
        updates_default_generator_state=updates_default_generator_state,
        dynamic_outputs=[False] * len(unique_output_indices),
        unique_output_indices=list(unique_output_indices),
    )
    if mlir_text is not None:
      executable.mlir_text = mlir_text
    if enable_debug and debug_container is not None:
      debug_container.compiled_executables.append(executable)

    return _TraceCompiledExecutableWrapper(
        executable,
        metadata,
        module=module,
        debug_container=debug_container,
    )

  if async_compile_result.is_ready():
    f: concurrent.futures.Future[CompiledCallable] = concurrent.futures.Future()
    try:
      loaded_exec = async_compile_result.get()
      f.set_result(_build_artifact(loaded_exec))
    except Exception as e:
      f.set_exception(e)
    return f

  def _wait_and_build() -> CompiledCallable:
    with xprof_adapter.TraceMe("trace_and_compile.wait_and_build"):
      loaded_executable = async_compile_result.get()
      return _build_artifact(loaded_executable)

  return _EXECUTOR.submit(_wait_and_build)


def trace_and_compile(
    fn: Callable[..., Any],
    example_args: Sequence[Any] | Any = (),
    example_kwargs: dict[str, Any] | None = None,
    *,
    argument_layouts: Sequence[Any] | None = None,
    donated_inputs: Sequence[int] | None = None,
    module_name: str | None = None,
    debug: bool = False,
) -> concurrent.futures.Future[CompiledCallable]:
  """Traces fn in INTERNAL_COMPILE_FX_GRAPH mode and asynchronously compiles.

  Accepts arbitrary positional arguments, keyword arguments, and non-tensor
  types.
  Tracing occurs synchronously on the calling thread under `_TRACE_LOCK`;
  lowering,
  cache query, and XLA compilation are dispatched asynchronously via TorchTPU's
  C++ compilation cache and resolve via the returned Future.

  Args:
    fn: Variadic callable taking arbitrary positional and keyword arguments.
    example_args: Positional argument(s) matching `fn` signature (tuple, list,
      or single argument).
    example_kwargs: Optional keyword argument dictionary for `fn`.
    argument_layouts: Optional custom memory layout per input tensor. If None,
      device layouts are automatically extracted from materialized TPU tensors.
    donated_inputs: Optional flat tensor input indices to donate to outputs.
    module_name: Optional identifier for XProf profiling and debug dumps.
    debug: Optional boolean indicating whether to collect compilation debug
      artifacts (accessible via the `.debug` attribute on the returned
      `CompiledCallable`).

  Returns:
    A concurrent.futures.Future resolving to a CompiledCallable.
  """
  pos_args = (
      tuple(example_args)
      if isinstance(example_args, (tuple, list))
      else (example_args,)
  )
  kw_args = (
      dict(sorted(example_kwargs.items())) if example_kwargs is not None else {}
  )
  is_simple_tensor_args = (not example_kwargs) and all(
      isinstance(x, torch.Tensor) for x in pos_args
  )

  flat_inputs, in_spec = _pytree.tree_flatten((pos_args, kw_args))

  for item in flat_inputs:
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
      if _contains_tensors(item):
        raise TypeError(
            f"Input dataclass '{type(item).__name__}' contains tensors but"
            " is not registered with PyTree. Register it using"
            " torch.utils._pytree.register_dataclass before compilation."
        )

  expected_tensor_specs = [
      (idx, item.shape, item.dtype, item.device)
      for idx, item in enumerate(flat_inputs)
      if isinstance(item, torch.Tensor)
  ]
  expected_non_tensor_specs = [
      (idx, item)
      for idx, item in enumerate(flat_inputs)
      if not isinstance(item, torch.Tensor)
  ]

  tensor_input_count = len(expected_tensor_specs)
  if donated_inputs:
    if len(set(donated_inputs)) != len(donated_inputs):
      raise ValueError(f"Duplicate indices in donated_inputs: {donated_inputs}")
    for d_idx in donated_inputs:
      if d_idx < 0 or d_idx >= tensor_input_count:
        raise ValueError(
            f"donated_inputs index {d_idx} is out of bounds for"
            f" {tensor_input_count} input tensor(s)."
        )

  debug_container = TpuCompileDebug() if debug else None
  target_mod = _extract_module(fn)

  saved_extra_tensor_attrs: list[tuple[torch.nn.Module, str, torch.Tensor]] = []
  with _TRACE_LOCK:
    try:
      with (
          execution_mode.set_eager_mode(
              execution_mode.EagerMode.INTERNAL_COMPILE_FX_GRAPH
          ),
          torch.no_grad(),
      ):
        gen_ctx = _setup_generators(flat_inputs)
        tracing_inputs = _build_tracing_placeholders(
            target_mod,
            flat_inputs,
            gen_ctx,
            argument_layouts,
            saved_extra_tensor_attrs,
        )
        raw_outputs, trace_flat_inputs, init_versions = _execute_trace(
            fn,
            target_mod,
            tracing_inputs,
            in_spec,
        )
        module_params_and_buffers: list[torch.Tensor] = []
        if target_mod is not None:
          module_params_and_buffers.extend(target_mod.parameters())
          module_params_and_buffers.extend(target_mod.buffers())
        if tracing_inputs.extra_tensors:
          module_params_and_buffers.extend(tracing_inputs.extra_tensors)

        mutations = _process_in_place_mutations(
            raw_outputs,
            flat_inputs,
            trace_flat_inputs,
            init_versions,
            tracing_inputs.placeholder_params,
            tracing_inputs.placeholder_buffers,
            tracing_inputs.placeholder_extra_tensors,
            donated_inputs,
            module_params_and_buffers,
        )
        base_constant_offset = (
            len(flat_inputs)
            + len(tracing_inputs.placeholder_params)
            + len(tracing_inputs.placeholder_buffers)
            + len(tracing_inputs.extra_tensors)
        )
        deduped = _deduplicate_outputs(
            mutations.flat_outputs,
            trace_flat_inputs,
            base_constant_offset,
            mutations.out_spec,
        )
        updates_gen_state, is_noop = _resolve_generator_outputs(
            gen_ctx,
            tracing_inputs.argument_tensors,
            deduped.deduped_outputs,
            tracing_inputs.internal_layouts,
        )

        metadata = _TraceCompilationMetadata(
            in_spec=in_spec,
            is_simple_tensor_args=is_simple_tensor_args,
            expected_tensor_specs=expected_tensor_specs,
            expected_non_tensor_specs=expected_non_tensor_specs,
            extra_tensors=tracing_inputs.extra_tensors,
            constant_outputs=deduped.constant_outputs,
            mutations=mutations,
        )

        if is_noop:
          return _make_noop_future(
              deduped.reconstructor,
              metadata,
              module=target_mod,
              debug_container=debug_container,
          )

        with xprof_adapter.TraceMe(
            "trace_and_compile.async_traverse_and_compile"
        ):
          async_compile_result = tpu_torch_compile.async_traverse_and_compile(
              result_tensors=deduped.deduped_outputs,
              argument_tensors=tracing_inputs.argument_tensors,
              build_mlir_module=bool(debug),
              argument_layouts=tracing_inputs.internal_layouts,
              donated_inputs=list(donated_inputs) if donated_inputs else [],
              module_name=module_name,
          )
    finally:
      _cleanup_tracing(saved_extra_tensor_attrs)

  return _package_executable_future(
      async_compile_result,
      deduped.reconstructor,
      deduped.unique_output_indices,
      updates_gen_state,
      metadata,
      module=target_mod,
      debug_container=debug_container,
      enable_debug=bool(debug),
  )
