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

"""TorchTPU support for torch.export to MLIR."""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import copy
import dataclasses
import re
import threading
from typing import Any, TypeAlias

from absl import logging
import torch
import torch.export
import torch.utils._pytree as pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.profiler import named_scope
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.benchmarks import xprof_adapter

EagerMode: TypeAlias = execution_mode.EagerMode
_fx_to_mlir_lock = threading.RLock()

__all__ = [
    "ExportedMlir",
    "exported_to_mlir",
    "fx_to_mlir",
]


def _extract_states_from_exported_program(exported_model):
  """Return a list of parameters that were lifted during export."""

  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = list(
      exported_model.graph_signature.parameters
  ) + list(exported_model.graph_signature.buffers)
  state_dict = copy.copy(exported_model.state_dict)
  if (constants := getattr(exported_model, "constants", None)) is not None:
    state_dict.update(constants)
  param_buffer_values = [state_dict[key] for key in param_and_buffer_keys]

  if hasattr(exported_model.graph_signature, "lifted_tensor_constants"):
    for name in exported_model.graph_signature.lifted_tensor_constants:
      param_buffer_values.append(exported_model.tensor_constants[name])

  return param_and_buffer_keys, param_buffer_values


def _extract_sample_arguments(exported: torch.export.ExportedProgram):
  """Return a list of sample arguments for the given exported program."""

  def _to_aval(arg_meta):
    """Convert from torch type to TPU meta buffer for tracing."""
    val = arg_meta["val"]
    is_scalar = not isinstance(val, torch.Tensor)
    if is_scalar:
      return tpu_torch_compile.placeholder([], type(arg_meta["val"]), False)

    tensor_meta = arg_meta["tensor_meta"]
    # tensor_meta is a torch.TensorMetadata, which is a NamedTuple and not
    # and actual tensor.
    # See torch/fx/passes/shape_prop.py
    return tpu_torch_compile.placeholder(
        list(tensor_meta.shape),
        tensor_meta.dtype,
        tensor_meta.requires_grad,
    )

  def _get_inputs(exported):
    """Return placeholders with input metadata."""
    placeholders = [p for p in exported.graph.nodes if p.op == "placeholder"]
    input_placeholders = [
        p
        for p, s in zip(placeholders, exported.graph_signature.input_specs)
        if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
    ]
    return input_placeholders

  args = _get_inputs(exported)
  return [_to_aval(arg.meta) for arg in args]


@dataclasses.dataclass(frozen=True)
class ExportedMlir:
  """The MLIR representation and compiled executable of an FX graph module.

  Attributes:
    module: The ContextedModule instance.
    executable: The compiled PJRT executable.
    mlir_result_tensors: A flattened list of tensors that the MLIR graph will
      actually produce as outputs.
    reconstruct_fx_outputs_fn: A function that takes the flattened MLIR outputs
      and reconstructs the original FX graph's output structure (e.g., restoring
      `None` values or nested tuples).
    updates_default_generator_state: Whether the MLIR module updates the default
      generator state. If so, one additional input tensor as the last argument
      is expected for the generator state tensor, and the one additional tensor
      as the last output is expected for the updated generator state tensor.
    is_noop: Whether the FX graph is a no-op (i.e. has no output tensors).
    dynamic_outputs: A sequence of booleans indicating whether each output
      tensor is dynamic.
    unique_output_indices: A sequence of original output indices corresponding
      to the deduplicated outputs.
  """

  module: tpu_torch_compile.ContextedModule | None
  executable: tpu_torch_compile.LoadedExecutableWithMetadata
  mlir_result_tensors: list[torch.Tensor]
  reconstruct_fx_outputs_fn: Callable[
      [Sequence[Any], Sequence[torch.Tensor]], Any
  ]
  updates_default_generator_state: bool
  is_noop: bool = False
  dynamic_outputs: Sequence[bool] | None = None
  unique_output_indices: Sequence[int] | None = None

  def serialize_text(self, enable_debug_info: bool = False) -> str:
    """Returns the MLIR representation of the graph as text."""
    return tpu_torch_compile.serialize_mlir_text(
        self.module, enable_debug_info=enable_debug_info
    )

  def serialize_bytecode(self) -> bytes:
    """Returns the MLIR representation of the graph as bytecode."""
    return tpu_torch_compile.serialize_mlir_bytecode(self.module)

  def serialize_portable_artifact(self) -> bytes:
    """Returns the MLIR representation of the graph as a versioned portable artifact."""
    return tpu_torch_compile.serialize_mlir_portable_artifact(self.module)


# Dynamo roots module FQNs at the frame local or global that holds the module,
# e.g. `L['self']` or `G['model']`. Un-inlined container indexing in Dynamo uses
# bracket syntax (e.g. `layers[0]` or `submodules['block']`).
_DYNAMO_SELF_ROOT_RE = re.compile(r"^[LG]\[['\"]self['\"]\]\.?")
_DYNAMO_ROOT_RE = re.compile(r"^[LG]\[['\"]([^'\"]+)['\"]\]")
_SUBSCRIPT_RE = re.compile(r"\[['\"]?([^'\"\]]+)['\"]?\]")


def _module_scope_entries(node: torch.fx.Node) -> list[tuple[str, str]]:
  """Returns the `nn_module_stack` keys and name scopes of a node's modules.

  Args:
    node: The FX node.

  Returns:
    A `(key, scope)` pair per enclosing module from outer to inner, where
    `scope` is formatted as described in `_extract_node_name_scopes`, and `key`
    identifies the module call (e.g. Dynamo suffixes the keys of repeated calls
    of a module with `@<n>`).
  """
  nn_stack = node.meta.get("nn_module_stack") or node.meta.get(
      "fwd_nn_module_stack"
  )
  if not isinstance(nn_stack, Mapping):
    return []

  entries: list[tuple[str, str]] = []
  prev_path = ""
  for key, value in nn_stack.items():
    if not isinstance(value, tuple) or len(value) < 2:
      continue
    fqn, mod_cls = value[0], value[1]
    # Make Dynamo FQNs relative to the compiled module like torch.export FQNs,
    # e.g. `L['self'].layers.0` to `layers.0` when compiling the module, and
    # `L['model'].layers.0` to `model.layers.0` when compiling a function that
    # takes the module as its `model` argument.
    path = _DYNAMO_SELF_ROOT_RE.sub("", str(fqn))
    path = _DYNAMO_ROOT_RE.sub(r"\1", path)
    path = _SUBSCRIPT_RE.sub(r".\1", path).lstrip(".")
    # Skip the root module (`""` in torch.export or `L['self']` in Dynamo) and
    # duplicate entries for the same module instance (e.g. `@1` stack keys).
    if not path or path == prev_path:
      continue
    rel_path = path.removeprefix(prev_path + ".") if prev_path else path
    prev_path = path

    cls_name = (
        mod_cls
        if isinstance(mod_cls, str)
        else getattr(mod_cls, "__name__", str(mod_cls))
    )
    cls_name = cls_name.rsplit(".", 1)[-1]
    clean_name = rel_path.replace("/", "_")
    entries.append((str(key), f"{clean_name}[{cls_name}]"))
  return entries


def _extract_node_name_scopes(node: torch.fx.Node) -> list[str]:
  """Extracts hierarchical module name scopes from FX node metadata.

  Args:
    node: The FX node, whose `nn_module_stack` (or `fwd_nn_module_stack` for
      backward nodes) metadata lists its enclosing modules.

  Returns:
    One scope per enclosing module from outer to inner, each formatted as the
    module path relative to its parent module followed by the module class
    name, e.g. `["layers.0[TransformerBlock]", "attn[Attention]"]`. The root
    module is omitted so that ops directly in it stay unscoped.
  """
  return [scope for _, scope in _module_scope_entries(node)]


class EagerLikeFxInterpreter(torch.fx.Interpreter):
  """Interpreter for TorchTPU.

  Interprets each node in the FX graph using the eager plugin with some
  additional setup so that torch.compile and torch.export produce similar MLIR
  to torch eager.
  """

  def __init__(
      self,
      module: torch.fx.GraphModule,
  ):
    super().__init__(module)
    self.state_dict = module.state_dict()
    self._name_scopes = named_scope.compute_graph_name_scopes(
        module.graph, _module_scope_entries
    )

  def _set_func_code_to_node_source_locations(self, func, node):
    """Update the function's code locations to the node's source locations.

    Overwrite the function's code locations to the node's source locations
    so that this node original source info appears on the python stack,
    making compiled/exported source locations resemble eager's source locations.
    """
    if not node.stack_trace:
      return

    # Note: This uses internal APIs and may break in the future.
    try:
      pt_trace = torch.fx.graph._parse_stack_trace(node.stack_trace)
      if pt_trace is None:
        # We will catch if something changes and this fails with unittests
        return

      # Overwrite function with node's source location info.
      # This is fairly unsafe, but it will set the python __code__ object that
      # is used in the captured stack trace to the node's source location, so it
      # will appear more like we were eagerly evaluating the function.
      func.__code__ = func.__code__.replace(
          co_filename=pt_trace.file,
          co_firstlineno=int(pt_trace.lineno),
          co_name=pt_trace.name,
      )
    except Exception as e:
      logging.warning("Failed to set func code to node source locations: %s", e)
    return

  @contextlib.contextmanager
  def _set_current_node(self, node: torch.fx.Node):
    """No-op override to prevent mutating process-global FX metadata.

    The base interpreter sets `torch.fx.traceback.current_meta`, which is a
    process-global dictionary. Overriding this prevents background compilation
    workers from corrupting autograd traceback state on the main thread.
    """
    yield

  def run_node(self, node) -> Any:
    """(Override) Run a single node in the FX graph, adding source locations."""

    def dispatch_fx_node(node):
      return super(EagerLikeFxInterpreter, self).run_node(node)

    self._set_func_code_to_node_source_locations(dispatch_fx_node, node)
    # Prefix the names of the node's ops with its module and named scopes.
    tpu_torch_compile.push_name_scopes(list(self._name_scopes.get(node, ())))
    try:
      return dispatch_fx_node(node)
    finally:
      tpu_torch_compile.pop_name_scopes()


def exported_to_mlir(
    exported: torch.export.ExportedProgram,
) -> ExportedMlir:
  """Converts a `torch.export.ExportedProgram` into its MLIR representation.

  This function serves as a high-level wrapper around `fx_to_mlir` for exported
  models. It extracts the FX graph, sample arguments, and lifted states
  (parameters, buffers, and constants) from the `ExportedProgram`, moves them
  to an available XLA device, and then generates the corresponding MLIR.

  An XLA device must be initialized before calling this function.

  Args:
    exported: The `torch.export.ExportedProgram` to convert.

  Returns:
    An `ExportedMlir` object containing the MLIR representation of the graph and
    FX output reconstruction information.
  """

  sample_args = _extract_sample_arguments(exported)
  _, state = _extract_states_from_exported_program(exported)

  args = state + sample_args
  args = [tpu_torch_compile.placeholder_like(arg) for arg in args]

  device = torch.accelerator.current_accelerator()
  to_xla = lambda x: x.to(device)
  args = pytree.tree_map_only(torch.Tensor, to_xla, args)
  module = exported.graph_module.to(device)

  return fx_to_mlir(module, args=args)


@dataclasses.dataclass(frozen=True)
class _DedupedTensorOutput:
  tensor_idx: int


@dataclasses.dataclass(frozen=True)
class _NonTensorPassthrough:
  input_idx: int


@dataclasses.dataclass(frozen=True)
class _TensorLayout:
  dtype: torch.dtype
  shape: torch.Size
  stride: tuple[int, ...]
  storage_offset: int


def _map_unused_fake_inputs_to_outputs(
    gm: torch.fx.GraphModule,
) -> Mapping[int, int]:
  """Maps indices of fake inputs that are directly passed through to outputs.

  This function inspects the given `GraphModule` to find "placeholder" nodes
  (inputs) that are also present in the output tuple. Specifically, it looks
  for `FakeScriptObject` instances that are passed from input to output without
  any intervening operations. This is useful for identifying outputs that don't
  correspond to actual computation results but are just forwarded inputs.

  Args:
    gm: The `torch.fx.GraphModule` to analyze.

  Returns:
    A dictionary mapping the index of an output in the flattened output tuple
    to the index of the corresponding input placeholder.
  """
  placeholders = list(enumerate(gm.graph.find_nodes(op="placeholder")))
  (graph_output_node,) = gm.graph.find_nodes(op="output")

  flattened_output_args, _ = pytree.tree_flatten(graph_output_node.args[0])
  output_nodes = list(enumerate(flattened_output_args))
  output_to_input_map = dict()

  for input_idx, input_node in placeholders:
    # TODO(aarfaian): Consider handling things beyond FakeScriptObject.
    if not isinstance(
        input_node.meta.get("val"),
        # pylint: disable-next=protected-access
        torch._library.fake_class_registry.FakeScriptObject,
    ):
      continue

    for output_idx, output_node in output_nodes:
      if input_node != output_node:
        continue

      output_to_input_map[output_idx] = input_idx

  return output_to_input_map


@dataclasses.dataclass(frozen=True)
class _TensorDedupeKey:
  """A key for deduplicating tensors based on their data and view layout.

  Two tensors are considered equivalent if and only if they are the same view of
  the same underlying buffer. The data_ptr encodes the buffer (and the storage
  offset, redundantly), and the layout encodes the view's interpretation of the
  buffer.
  """

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


def _process_fx_outputs(
    gm: torch.fx.GraphModule,
    outputs: Any,
    dynamic_outputs: Sequence[bool] | None = None,
) -> tuple[
    list[torch.Tensor],
    Callable[[Sequence[Any], Sequence[torch.Tensor]], Any],
    list[bool],
    list[int],
]:
  """Removes Nones, maps passthrough values, and dedups outputs.

  This is necessary because MLIR graphs do not support `None` values in the
  graph outputs, and we don't want to produce duplicate outputs for tensors that
  are the same.  Additionally, in some cases (e.g. DTensor) we will see some
  non-Tensor inputs that are passed through to the output tuple as well. This
  function filters `None` values, maps passthrough values, and reduces to unique
  tensors. It returns a tuple containing a list of result tensors and a function
  to reconstruct the original FX output structure.

  Args:
    gm: The original torch.fx.GraphModule instance.
    outputs: The outputs of the graph.
    dynamic_outputs: Optional sequence of booleans indicating dynamic outputs.

  Returns:
    A tuple containing a list of result tensors, a function to reconstruct
    the original FX output structure, and a list of booleans indicating dynamic
    outputs.

  Raises:
    TypeError: If an element in the flattened outputs is not a `torch.Tensor`
      or `None`.
  """
  output_to_input_map = _map_unused_fake_inputs_to_outputs(gm)
  flat_outputs, spec = pytree.tree_flatten(outputs)
  deduped_outputs: list[torch.Tensor] = []
  expected_layouts: list[_TensorLayout] = []
  deduped_dynamic_outputs: list[bool] = []
  unique_output_indices: list[int] = []
  output_map: list[_DedupedTensorOutput | _NonTensorPassthrough | None] = []
  index_by_dedupe_key: dict[_TensorDedupeKey, int] = {}

  for idx, item in enumerate(flat_outputs):
    if item is None:
      output_map.append(None)
    elif isinstance(item, torch.Tensor):
      key = _TensorDedupeKey.from_tensor(item)
      if key not in index_by_dedupe_key:
        index_by_dedupe_key[key] = len(deduped_outputs)
        deduped_outputs.append(item)
        expected_layouts.append(key.layout)
        unique_output_indices.append(idx)
        is_dyn = (
            dynamic_outputs[idx]
            if dynamic_outputs is not None and idx < len(dynamic_outputs)
            else False
        )
        deduped_dynamic_outputs.append(is_dyn)
      output_map.append(_DedupedTensorOutput(index_by_dedupe_key[key]))
    elif idx in output_to_input_map:
      output_map.append(_NonTensorPassthrough(output_to_input_map[idx]))
    else:
      raise TypeError(
          "Expected FX graph output to be a Tensor, None, or FakeScriptObject"
          f" but got {item}"
      )

  return (
      deduped_outputs,
      tpu_torch_compile.FxOutputReconstructor(
          output_map,
          expected_layouts,
          spec,
          deduped_dynamic_outputs,
      ),
      deduped_dynamic_outputs,
      unique_output_indices,
  )


@annotations.forbid_in_torch_compile(
    "A compiled graph already has MLIR debug locations for each op from the FX "
    "graph."
)
@contextlib.contextmanager
def enable_tracebacks(value: bool | None = True):
  """A context manager that overrides whether enabling MLIR tracebacks."""

  # Push the new state.
  tpu_torch_compile.push_enable_tracebacks(value)
  try:
    yield
  finally:
    # Guarantee the state is restored even if Python code raises an exception.
    tpu_torch_compile.pop_enable_tracebacks()


@contextlib.contextmanager
def _isolated_fx_name_scopes():
  """Isolates fx_to_mlir from ambient eager name scopes on the caller thread."""
  tpu_torch_compile.push_name_scopes([])
  try:
    yield
  finally:
    tpu_torch_compile.pop_name_scopes()


# TODO(cnchan): change build_mlir_module default to False after migrating all
# test usages with the new traverse_and_compile API.
def fx_to_mlir(
    module: torch.fx.GraphModule,
    args: list[torch.Tensor | Any],
    build_mlir_module: bool = True,
    use_stablehlo_bounds: bool = False,
    argument_layouts: list[list[int]] | None = None,
    dynamic_outputs: Sequence[bool] | None = None,
    donated_inputs: Sequence[int] | None = None,
    module_name: str | None = None,
) -> ExportedMlir:
  """Converts an FX graph module to MLIR using TorchTPU's FX-tracing mode.

  This function traces the given FX graph module by running it with an
  EagerLikeFxInterpreter in compilation mode.
  (`EagerMode._INTERNAL_COMPILE_FX_GRAPH`).
  The resulting deferred graph is then converted to MLIR bytes.

  Args:
    module: The `torch.fx.GraphModule` to be converted to MLIR.
    args: A list of input arguments to trace the module. These will be run
      through an FX graph interpreter to identify the graph's output tensors.
    build_mlir_module: Whether to build and return the MLIR module in the
      result.
    use_stablehlo_bounds: Whether to use StableHLO bounds.
    argument_layouts: A list of forced layouts for input arguments.
    dynamic_outputs: A list of booleans indicating whether the corresponding
      output is dynamic.
    donated_inputs: A list of argument indices to donate.
    module_name: Optional custom identifier for the compiled region.

  Returns:
    An `ExportedMlir` object containing the MLIR representation of the graph and
    FX output reconstruction information.
  """
  with (
      xprof_adapter.TraceMe("export.fx_to_mlir"),
      execution_mode.set_eager_mode(EagerMode._INTERNAL_COMPILE_FX_GRAPH),
  ):
    # Filter out non-tensor arguments.
    argument_tensors = [a for a in args if isinstance(a, torch.Tensor)]
    if argument_layouts is not None:
      assert len(argument_layouts) == len(argument_tensors), (
          f"argument_layouts size mismatch: expected {len(argument_tensors)},"
          f" got {len(argument_layouts)}"
      )
      internal_layouts = list(argument_layouts)
    else:
      internal_layouts = []

    with _fx_to_mlir_lock, _isolated_fx_name_scopes():
      # Run the module through the EagerLikeFxInterpreter with MLIR location
      # tracebacks enabled by default so that the MLIR we generate has file
      # location info.
      device = torch.accelerator.current_accelerator()
      device_module = getattr(torch, device.type)  # pyrefly: ignore[missing-attribute]

      # Add the default RNG generator for the device to the args if not already
      # present.
      default_device_index = device.index or 0  # pyrefly: ignore[missing-attribute]
      default_generator = device_module.default_generators[default_device_index]
      argument_generators = [
          arg for arg in args if isinstance(arg, torch.Generator)
      ]
      has_implicit_default_generator = (
          default_generator not in argument_generators
      )
      if has_implicit_default_generator:
        argument_generators.append(default_generator)

      export_mode_generators = {
          gen: tpu_torch_compile.make_export_mode_device_generator(gen)
          for gen in argument_generators
      }

      try:
        # Set export-mode override for the default generator.
        tpu_torch_compile.set_export_mode_default_device_generator(
            export_mode_generators[default_generator], default_device_index
        )

        # Include the state tensors from the export-mode generators in the
        # arguments for graph traversal, as these represent mutable RNG states
        # within the graph.
        for gen in argument_generators:
          argument_tensors.append(
              tpu_torch_compile.get_device_state_tensor(
                  export_mode_generators[gen]
              )
          )
          if internal_layouts:
            internal_layouts.append([])

        # We clone the args so that inplace updates do not overwrite the
        # placeholder args. These copies will be removed in the compiled code so
        # there is no performance impact.
        # Remove once b/491716758 is implemented.
        mapped_args = [
            export_mode_generators[x]
            if isinstance(x, torch.Generator)
            else (x.clone() if isinstance(x, torch.Tensor) else x)
            for x in args
        ]
        with xprof_adapter.TraceMe("EagerLikeFxInterpreter.run"):
          fx_outputs = EagerLikeFxInterpreter(module).run(*mapped_args)

        (
            result_tensors,
            reconstruct_fx_outputs_fn,
            deduped_dynamic_outputs,
            unique_output_indices,
        ) = _process_fx_outputs(
            module, fx_outputs, dynamic_outputs=dynamic_outputs
        )

        # Plumb the final state tensor as the last output of the graph so
        # that the updated RNG state can be returned from the executable and
        # persisted.
        result_tensors.extend(
            tpu_torch_compile.get_device_state_tensor(
                export_mode_generators[gen]
            )
            for gen in argument_generators
        )

        # Check if the module updates the default generator state.
        if has_implicit_default_generator:
          begin_default_state_tensor = argument_tensors[-1]
          end_default_state_tensor = result_tensors[-1]
          updates_default_generator_state = (
              begin_default_state_tensor.data_ptr()
              != end_default_state_tensor.data_ptr()
          )
          if not updates_default_generator_state:
            if len(result_tensors) > 1:
              # Remove the default generator state tensor from the arguments
              # and outputs since it is not used.
              argument_tensors.pop()
              result_tensors.pop()
              if internal_layouts:
                internal_layouts.pop()
            else:
              # No RNG update and nothing to compute: this graph is a no-op.
              return ExportedMlir(
                  module=None,
                  executable=None,
                  mlir_result_tensors=[],
                  reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
                  updates_default_generator_state=False,
                  is_noop=True,
                  dynamic_outputs=[],
                  unique_output_indices=[],
              )
        else:
          updates_default_generator_state = False
      finally:
        tpu_torch_compile.clear_export_mode_default_device_generators()

    compile_result = tpu_torch_compile.traverse_and_compile(
        result_tensors=result_tensors,
        argument_tensors=argument_tensors,
        build_mlir_module=build_mlir_module,
        use_stablehlo_bounds=use_stablehlo_bounds,
        argument_layouts=internal_layouts,
        donated_inputs=list(donated_inputs)
        if donated_inputs is not None
        else [],
        module_name=module_name,
    )

    return ExportedMlir(
        module=compile_result.module,
        executable=compile_result.executable,
        mlir_result_tensors=result_tensors,
        reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
        updates_default_generator_state=updates_default_generator_state,
        dynamic_outputs=deduped_dynamic_outputs,
        unique_output_indices=unique_output_indices,
    )
