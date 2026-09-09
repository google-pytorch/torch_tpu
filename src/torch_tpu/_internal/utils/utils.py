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

"""Tools and helper routines common to all examples and benchmarks."""

import collections
from collections.abc import MutableSequence
import contextlib
import functools
import os
import sys
import time
from typing import Any, Callable, Sequence, TextIO, TypeAlias

import torch
from torch.utils import _pytree
from torch_tpu._internal import execution_mode
from torch_tpu._internal import export as torch_tpu_export
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile

EagerMode: TypeAlias = execution_mode.EagerMode


def model_memory(
    model: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> float:
  """Estimate the physical memory requirements for a model.

  This function counts the total number of parameters plus
  the parameters that require gradients. It also collects
  all model buffers before finally multiplying this sum
  with the size of the model's data type.

  The function currently assumes that the whole model
  utilizes a single dtype.

  Args:
    model:        The PyTorch model to analyze.
    dtype:        The assumed dtype for the whole model.

  Returns:
    The estimated model memory requirements in GB.
  """
  total_params = 0
  for param in model.parameters():
    total_params += param.numel() * 2 if param.requires_grad else param.numel()
  total_buffers = sum(buf.numel() for buf in model.buffers())

  element_size = torch.tensor(0, dtype=dtype).element_size()
  total_memory_bytes = (total_params + total_buffers) * element_size

  return total_memory_bytes / (1024.0**3)


@contextlib.contextmanager
def open_output(
    filepath: str | None, mode: str = "w", default: TextIO = sys.stdout
):
  """Opens an output file if filepath is provided.

  Args:
    filepath: The path to the file to open. If None or empty, the default file
      is used.
    mode: The mode to open the file in.
    default: The default file to use if filepath is None.

  Only closes the file if it was opened (i.e., not the default).
  """
  if filepath:
    f = open(os.path.abspath(os.path.expanduser(filepath)), mode)
    try:
      yield f
    finally:
      f.close()
  else:
    yield default


def format_model(
    model: torch.nn.Module | Callable[..., Any],
    *input_tensors: torch.Tensor | tuple[torch.Tensor, ...],
    params: bool = False,
    pt: bool = False,
    aten: bool = False,
    shlo: bool = False,
) -> str:
  """Formats a model or tensor function in various ways.

  Use the various print method to provide details about a given model.

  Args:
    model:          The PyTorch model or tensor function to format.
    *input_tensors: The input tensor(s) to the model.
    params:         If True, include model parameter sizes.
    pt:             If True, include the output of PyTorch's print(model).
    aten:           If True, utilize OpTracer to collect and print the aten ops.
    shlo:           If True, dump generated StableHLO.

  Raises:
    NotImplementedError: For unrecognized dumper options.

  Returns:
    The formatted model as a string.
  """

  result = ""
  if not isinstance(model, torch.nn.Module):
    if isinstance(input_tensors, torch.Tensor):
      placeholders = (tpu_torch_compile.placeholder_like(input_tensors),)
    else:
      to_placeholder = tpu_torch_compile.placeholder_like
      placeholders = _pytree.tree_map_only(
          torch.Tensor, to_placeholder, input_tensors
      )

    # Trace the model in INTERNAL_COMPILE_FX_GRAPH mode so that we can capture
    # the full graph.
    with execution_mode.set_eager_mode(EagerMode.INTERNAL_COMPILE_FX_GRAPH):
      model = torch.fx.experimental.proxy_tensor.make_fx(model)(*placeholders)

  if params:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result += f"Parameters: {total}, Trainable: {train}\n"
    result += (
        f"Estimated Memory: {model_memory(model):.3f} "
        "GB (based on # of parameters and dtype)\n"
    )

  if pt:
    # If printing GraphModule, use print_readable to include type information.
    module_str = (
        model.print_readable(print_output=False)
        if isinstance(model, torch.fx.GraphModule)
        else str(model)
    )
    result += module_str + "\n"

  if aten:
    with OpTracer() as tracer:
      model(*input_tensors)
    result += tracer._pformat() + "\n"  # pylint: disable=protected-access

  if shlo:
    if isinstance(model, torch.fx.GraphModule):
      exported = torch.export.export(model, args=(*input_tensors,))
      mlir_text = torch_tpu_export.exported_to_mlir(exported).serialize_text()
      result += mlir_text + "\n"
    else:
      with execution_mode.set_eager_mode(EagerMode.INTERNAL_COMPILE_FX_GRAPH):
        results = model(*input_tensors)

      shlo = sync.computation_mlir(results)  # pyrefly: ignore[bad-assignment]
      result += shlo + "\n"  # pyrefly: ignore[unsupported-operation]

  return result


def format_tensor(
    tensor: torch.Tensor, desc: str = "", data: bool = False
) -> str:
  """Formats the most common properties of a PyTorch tensor.

  Args:
      tensor: The tensor to inspect.
      desc:   An optional descriptive string.
      data:   If True, also include the tensor's contents. If the tensor is on a
        device, it is first copied to CPU.

  Returns:
      A string representation of the tensor's properties.
  """

  if not isinstance(tensor, torch.Tensor):
    return f"Error: Expected a torch.Tensor, but got {type(tensor)}"

  lines = []
  if desc:
    lines.append(desc)
  lines.append(f"Shape (Size):     {tensor.shape}")
  lines.append(f"Number of Dims:   {tensor.ndim}")
  lines.append(f"Data Type (dtype):{tensor.dtype}")
  lines.append(f"Device:           {tensor.device}")
  lines.append(f"Min:              {tensor.cpu().min().item():.2f}")
  lines.append(f"Max:              {tensor.cpu().max().item():.2f}")
  lines.append(f"Mean:             {tensor.cpu().mean().item():.2f}")
  lines.append(f"Var:              {tensor.cpu().var().item():.2f}")
  lines.append(f"Requires Grad:    {tensor.requires_grad}")
  lines.append(f"Memory Layout:    {tensor.layout}")
  lines.append(f"Is a Leaf:        {tensor.is_leaf}")
  lines.append(f"Memory Usage (B): {tensor.nbytes}")
  lines.append(f"Strides:          {tensor.stride()}")
  if tensor.grad_fn is not None:
    lines.append(f"Gradient Function: {tensor.grad_fn}")
  else:
    lines.append("Gradient Function: None (or is a leaf tensor)")

  if data:
    if str(tensor.device) != "cpu":
      lines.append(tensor.to("cpu"))
    else:
      lines.append(str(tensor))

  return "\n".join(lines)


def get_tensor_summary(tensor: torch.Tensor, data: bool = True) -> str:
  """Returns a one-line, compact summary of a PyTorch tensor.

  The summary includes shape, dtype, device, min, max, mean, variance, and
  requires grad.

  Example output:
  (2, 3) Min:-0.85 Max:1.69 Mean:0.31 Var:0.77 Grad:Y float32 cpu

  Note: variance is calculated with correction=0, since we want the variance of
  the actual values, not the estimated variance a sample of a population.

  Args:
    tensor (torch.Tensor): The tensor to summarize.
    data (bool): If True, also include details on tensors data (Min, Max, etc)
      else only returns shape and dtype info.

  Returns:
    A compact string representation of the tensor with summary statistics.

  Raises:
    TypeError: If the tensor is not a torch.Tensor.
  """
  if not isinstance(tensor, torch.Tensor):
    raise TypeError(f"Expected a torch.Tensor, but got {type(tensor)}")

  shape = tuple(tensor.shape)
  grad = "Y" if tensor.requires_grad else "N"
  dtype = str(tensor.dtype).split("torch.")[-1]
  device = tensor.device

  if not data:
    template = "torch.tensor(...{shape}, dtype={dtype}, device={device}{grad})"
    grad = ", requires_grad=True" if tensor.requires_grad else ""
    return template.format(
        shape=shape, dtype=tensor.dtype, device=f"'{device}'", grad=grad
    )

  template = (
      "{shape}  Min:{min_val: .2f}  Max:{max_val: .2f}  Mean:{mean_val: .2f}  "
      "Var:{var_val: .2f}  Grad:{grad}  {dtype}  {device}"
  )

  if tensor.numel() > 0:
    t_cpu = tensor.to("cpu")
    min_val = t_cpu.min().item()
    max_val = t_cpu.max().item()
    if tensor.dtype.is_floating_point:
      mean_val = t_cpu.mean().item()
      var_val = t_cpu.var(correction=0).item()
    else:
      mean_val = t_cpu.float().mean().item()
      var_val = t_cpu.float().var(correction=0).item()
    return template.format(
        shape=shape,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        var_val=var_val,
        grad=grad,
        dtype=dtype,
        device=device,
    )
  else:
    return (
        f"{shape}  Min:n/a  Max:n/a  Mean:n/a  Var:n/a  Grad:{grad}  {dtype}  "
        f"{device}"
    )


class _InputArgumentMetadata:
  """A class to log a single argument's metadata.

  Note the stringification is pushed into the `__str__` method which makes
  this class more beneficial for python logging.
  """

  def __init__(self, arg: Any):
    self._arg = arg

  def __str__(self):
    if isinstance(self._arg, torch.Tensor):
      return get_tensor_summary(self._arg, data=False)
    return str(self._arg)

  def __repr__(self):
    return self.__str__()


class InputMetadata:
  """A class to log call inputs without requiring access to their data.

  This avoids the need to send any part of a tensor back to the host when
  printing. This also defers the evaluation of the text until `__str__` eval
  time allowing for lazy evaluation in logging.
  """

  def __init__(self, inputs: Sequence[Any]):
    self._inputs = _pytree.tree_map_only(
        torch.Tensor, _InputArgumentMetadata, inputs
    )
    if not isinstance(self._inputs, Sequence):
      self._inputs = [self._inputs]

  def __str__(self):
    return str(self._inputs)


class OpTracer(
    torch.utils._python_dispatch.TorchDispatchMode,  # pylint: disable=protected-access
    contextlib.ContextDecorator,
):
  """This context manager intercepts and stores PyTorch ops.

  It can also be used a decorator.

  Example usage as context manager:
    x = torch.tensor(2.0)
    with OpTracer() as tracer:
      _ = x ** 0.5
    print(tracer_utils.pformat_op_tracer(tracer))

  Example usage as a wrapper:
    def f(x):
      return x ** 0.5

    tracer = OpTracer()
    wrapper = tracer(f)
    _ = wrapper(torch.tensor(2.0))
    print(tracer_utils.pformat_op_tracer(tracer))

  Example usage as a decorator:
    tracer = OpTracer()
    @tracer
    def f(x):
      return x ** 0.5
    _ = f(torch.tensor(2.0))
    print(tracer_utils.pformat_op_tracer(tracer))

  Attributes:
    ops_log (dict[str, set[str]]): A dict mapping stringified namespaces to a
      set of stringified ops.
  """

  def __enter__(self):
    super().__enter__()  # pytype: disable=attribute-error
    self.ops_log = collections.defaultdict(dict)
    return self

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    aten = str(func)
    mapping = self.ops_log[func.namespace]
    mapping[aten] = mapping.get(aten, 0) + 1
    return func(*args, **kwargs)

  def num_atens(self) -> int:
    return sum(self.ops_log["aten"].values())

  ### TODO(yho): This really belongs in tracer_utils.py, but format_model
  ### uses it, so we'll leave it here for now.
  def _pformat(self) -> str:
    """Pretty-formats the collected data from OpTracer.

    Returns:
      A string representation of the collected data.
    """
    lines = []

    for key in sorted(self.ops_log):
      num_ops = len(self.ops_log[key])
      lines.append(f"Ops for namespace: '{key}' ({num_ops} ops)")
      for op in self.ops_log[key]:
        num_calls = self.ops_log[key][op]
        lines.append(f"  {op}: {num_calls}")
    return "\n".join(lines)


Event: TypeAlias = dict[str, Any]


class ActivationTracer(contextlib.ContextDecorator):
  """Traces activations in a model's forward and backward pass.

  This uses PyTorch's hooks API to intercept nn.Modules and
  log the input and output tensors of each module.

  Example usage:
    model = MyModel()
    with ActivationTracer(model) as collector:
      # Forward pass within this block will trigger hooks
      output = model(input_tensor)
    forward_log = collector.forward_log
    forward_pre_log = collector.forward_pre_log
    print(collector.pformat())

  Attributes:
    forward_pre_log: Events logged before the forward pass of each module.
    forward_log: Events logged after the forward pass of each module.
    backward_pre_log: Events logged before the backward pass of each module.
    backward_log: Events logged after the backward pass of each module.
  """

  forward_pre_log: MutableSequence[Event]
  forward_log: MutableSequence[Event]
  backward_pre_log: MutableSequence[Event]
  backward_log: MutableSequence[Event]

  def __init__(self, model: torch.nn.Module):
    self._model = model
    self._hook_handles: MutableSequence[torch.utils.hooks.RemovableHandle] = []
    self.forward_pre_log: MutableSequence[Event] = []
    self.forward_log: MutableSequence[Event] = []
    self.backward_pre_log: MutableSequence[Event] = []
    self.backward_log: MutableSequence[Event] = []

  # Based on signature at
  # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
  def _forward_pre_hook(
      self,
      module: torch.nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      depth: int,
  ) -> None:
    """Logs the forward pass of a module. Called before a forward pass.

    Args:
      module: The module that was called.
      args: The positional arguments of the module.
      kwargs: The keyword arguments of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.forward_pre_log.append({
        "module": module,
        "args": args,
        "kwargs": kwargs,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  # Based on signature at
  # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
  def _forward_hook(
      self,
      module: torch.nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      output: Any,
      depth: int,
  ) -> None:
    """Logs the forward pass of a module. Called after a forward pass.

    Args:
      module: The module that was called.
      args: The positional arguments of the module.
      kwargs: The keyword arguments of the module.
      output: The output tensor(s) of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.forward_log.append({
        "module": module,
        "args": args,
        "kwargs": kwargs,
        "output": output,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  # Based on signature at
  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
  def _backward_pre_hook(
      self,
      module: torch.nn.Module,
      grad_output: tuple[torch.Tensor, ...],
      depth: int,
  ) -> None:
    """Logs the backward pass of a module. Called before a backward pass.

    Args:
      module: The module that was called.
      grad_output: The gradient of the output tensor(s) of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.backward_pre_log.append({
        "module": module,
        "grad_output": grad_output,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  # Based on signature at
  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
  def _backward_hook(
      self,
      module: torch.nn.Module,
      grad_input: tuple[torch.Tensor, ...],
      grad_output: tuple[torch.Tensor, ...],
      depth: int,
  ) -> None:
    """Logs the backward pass of a module. Called after a backward pass.

    Args:
      module: The module that was called.
      grad_input: The gradient of the input tensor(s) to the module.
      grad_output: The gradient of the output tensor(s) of the module.
      depth: The depth of the module in the model hierarchy.
    """

    self.backward_log.append({
        "module": module,
        "grad_input": grad_input,
        "grad_output": grad_output,
        "depth": depth,
        "time": time.perf_counter(),
        "idx": self._idx,
    })
    self._idx += 1

  def _register_hooks_recursive(self, module: torch.nn.Module, depth: int):
    """Registers hooks on the current module and recursively on its children."""
    self._hook_handles.append(
        module.register_forward_pre_hook(
            functools.partial(self._forward_pre_hook, depth=depth),
            with_kwargs=True,
        )
    )
    self._hook_handles.append(
        module.register_forward_hook(
            functools.partial(self._forward_hook, depth=depth), with_kwargs=True
        )
    )
    self._hook_handles.append(
        module.register_full_backward_pre_hook(
            functools.partial(self._backward_pre_hook, depth=depth)
        )
    )
    self._hook_handles.append(
        module.register_full_backward_hook(
            functools.partial(self._backward_hook, depth=depth)
        )
    )

    # Only increase depth for non-container modules.
    child_depth = (
        depth
        if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict))
        else depth + 1
    )
    for _, child in module.named_children():
      self._register_hooks_recursive(child, child_depth)

  def __enter__(self):
    self.forward_log.clear()
    self.forward_pre_log.clear()
    self.backward_log.clear()
    self.backward_pre_log.clear()
    self._idx = 0

    self._register_hooks_recursive(self._model, 0)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Unregister and remove the hooks at the end of the context manager
    # in case the use wants to call the model outside the context manager.
    for handle in self._hook_handles:
      handle.remove()
    self._hook_handles.clear()
    return False


def parse_version(version: str) -> tuple[int, ...]:
  """Parses the leading numeric components of a version string into a tuple.

  Stops at the first non-numeric component so pre-release / nightly suffixes
  (e.g. "0.0.41.dev20260101") compare by their numeric prefix.

  Args:
    version: A version string such as "0.0.41".

  Returns:
    The leading numeric components, e.g. (0, 0, 41).
  """
  components = []
  for part in version.split("."):
    digits = ""
    for ch in part:
      if not ch.isdigit():
        break
      digits += ch
    if not digits:
      break
    components.append(int(digits))
  return tuple(components)


def libtpu_version() -> tuple[int, ...] | None:
  """Returns the parsed version of the installed libtpu, or None if unavailable."""
  try:
    import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  except ImportError:
    return None
  version = getattr(libtpu, "__version__", None)
  if not version:
    return None
  return parse_version(version)


def libtpu_at_least(min_version: tuple[int, ...]) -> bool:
  """Returns True if libtpu is built from source / in-tree (None) or >= min_version."""
  ver = libtpu_version()
  if ver is None:
    return True  # In-tree / source builds link PJRT directly without a wheel
  return ver >= min_version


def get_primary_tensor_output(out: Any) -> torch.Tensor | None:
  """Recursively extracts a primary output torch.Tensor from model output objects.

  Hugging Face and PyTorch models return output tensors wrapped in heterogeneous
  structures (e.g., HF ModelOutput dataclasses, tuples, lists, or dicts).
  This function resolves the primary numerical tensor using the following
  priority resolution strategy:
    1. Direct Tensor: Returns the input if it is already a torch.Tensor.
    2. Priority Attributes & Keys: Checks attributes and dictionary keys in
       precedence order to isolate the primary prediction head and avoid
       auxiliary tensors (e.g., loss, KV caches, image features):
       - `.logits` / "logits"
       - `.last_hidden_state` / "last_hidden_state"
       - `.sample` / "sample"
       - `.reconstruction` / "reconstruction"
       - `.waveform` / "waveform"
       - `.audio` / "audio"
       - `.sequences` / "sequences"
       - `.prediction_logits` / "prediction_logits"
       - `.output` / "output"
    3. Sequences (tuple/list): Recursively unwraps the first element (out[0]).
    4. Dictionaries (dict):
       - Fallback: Scans out.values() recursively to extract the first
         available tensor (for custom model dictionaries).

  Args:
    out: The raw model output object returned by forward execution.

  Returns:
    The extracted primary torch.Tensor, or None if no tensor could be found.
  """
  if isinstance(out, torch.Tensor):
    return out

  priority_keys = (
      "logits",
      "last_hidden_state",
      "sample",
      "reconstruction",
      "waveform",
      "audio",
      "sequences",
      "prediction_logits",
      "output",
  )
  for k in priority_keys:
    if hasattr(out, k) and (val := getattr(out, k)) is not None:
      return get_primary_tensor_output(val)
    if isinstance(out, dict) and k in out and out[k] is not None:
      return get_primary_tensor_output(out[k])

  if isinstance(out, (tuple, list)) and len(out) > 0:
    return get_primary_tensor_output(out[0])

  if isinstance(out, dict):
    for v in out.values():
      if (res := get_primary_tensor_output(v)) is not None:
        return res

  return None
