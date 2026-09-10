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

"""Simple trainer library for compiling and running PyTorch training steps on TPU."""

import functools
import inspect
import typing
from typing import Any, Callable, Sequence

import torch
import torch.func as func
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree
from examples.tracing import optimizers

# Re-export optimizer classes for simple_trainer
Optimizer = optimizers.Optimizer
AdamW = optimizers.AdamW
SGD = optimizers.SGD
ReferenceAdamw = optimizers.ReferenceAdamw
TorchAdamw = optimizers.TorchAdamw
ReferenceSgd = optimizers.ReferenceSgd
TorchSgd = optimizers.TorchSgd
Muon = optimizers.Muon
ReferenceMuon = optimizers.ReferenceMuon
TorchMuon = optimizers.TorchMuon


def _compute_loss(
    model: torch.nn.Module,
    key_map: dict[str, str],
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    inputs: Any,
    targets: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Computes loss for functional execution."""
  p_orig = {key_map[k]: v for k, v in params.items()}
  b_orig = {key_map[k]: v for k, v in buffers.items()}

  if isinstance(inputs, dict):
    args, kwargs = (), inputs
  elif isinstance(inputs, (tuple, list)):
    args, kwargs = tuple(inputs), {}
  else:
    args, kwargs = (inputs,), {}

  out = func.functional_call(model, (p_orig, b_orig), args, kwargs)

  loss = None
  if targets is not None:
    loss = torch.nn.functional.mse_loss(out, targets)
  elif hasattr(out, "loss") and out.loss is not None:
    loss = out.loss
  elif isinstance(out, dict) and "loss" in out:
    loss = out["loss"]
  elif hasattr(out, "logits") and out.logits is not None:
    loss = torch.mean(out.logits)
  elif hasattr(out, "sample") and out.sample is not None:
    loss = torch.mean(out.sample)
  else:
    flat_out, _ = _pytree.tree_flatten(out)
    for item in flat_out:
      if torch.is_tensor(item):
        loss = torch.mean(item)
        break

  if loss is None:
    raise TypeError(f"Cannot extract loss from output of type {type(out)}")

  return loss, buffers


def _invoke_compile_fn(
    compile_fn: Callable[..., Any],
    graph_module: torch.fx.GraphModule,
    flat_inputs: Sequence[Any],
) -> Callable[..., Any]:
  """Invokes compile_fn with either (graph_module, flat_inputs) or (graph_module)."""
  try:
    sig = inspect.signature(compile_fn)
    params = list(sig.parameters.values())
    pos_params = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    has_var_args = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
    )
    if len(pos_params) == 1 and not has_var_args:
      return compile_fn(graph_module)
    return compile_fn(graph_module, flat_inputs)
  except (ValueError, TypeError):
    try:
      return compile_fn(graph_module, flat_inputs)
    except TypeError:
      return compile_fn(graph_module)


class SingleTraceTrainer:
  """Facilitates compiled train steps with a single offload containing the optimizer step.

  This trainer captures the entire training loop (forward pass, loss
  computation,
  gradient calculation, and optimizer step) into a single functional graph using
  `make_fx`. This graph is then compiled and executed as a single unit
  (offload),
  reducing interpreter overhead and enabling better fusion opportunities by the
  compiler.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      optimizer: Any,
      compile_fn: Callable[..., Any] | None = None,
  ):
    """Initializes the SingleTraceTrainer."""
    self.model = model
    self.optimizer = optimizer
    self.compile_fn = compile_fn

    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
      tie_weights()

    raw_params = dict(model.named_parameters())
    raw_buffers = dict(model.named_buffers())

    self._key_map = {}
    initial_params = {}
    for name, param in raw_params.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      initial_params[safe_name] = param.detach()

    self.buffers = {}
    for name, buf in raw_buffers.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      self.buffers[safe_name] = buf.detach()

    self.param_group = self.optimizer.init_param_group(initial_params)

  # Update state in this container and in user model.
  # Not technically required for params; all of the optimizers internalize an
  # # inplace update, and that part of the trace is not functionalized.
  # Buffers on the other hand may be mutated by the model inside stuff traced
  # by func.grad_and_value. This is functionalized so we will need an update
  # for buffers.
  def _update(self, new_pg, updated_bufs):
    self.buffers = updated_bufs
    self.param_group = new_pg
    self.model.load_state_dict(self.bufs, strict=False)

  @property
  def params(self) -> dict[str, torch.Tensor]:
    """Returns parameters with their original names."""
    return {self._key_map[k]: v for k, v in self.param_group.params.items()}

  @property
  def bufs(self) -> dict[str, torch.Tensor]:
    """Returns buffers with their original names."""
    return {self._key_map[k]: v for k, v in self.buffers.items()}

  def make_compiled_train_step(
      self,
      example_inputs: Any,
      example_targets: Any = None,
      compile_fn: Callable[..., Any] | None = None,
  ) -> Callable[..., torch.Tensor]:
    """Returns a callable for the training step with the model compiled."""
    if compile_fn is None:
      compile_fn = self.compile_fn
    if compile_fn is None:
      raise ValueError(
          "A compilation callback (compile_fn) must be provided either to"
          " SingleTraceTrainer or to make_compiled_train_step."
      )

    # We don't actually want to trace through example_targets if it is None.
    # Static compiler doesn't support optional input args.
    flat_inputs, in_spec = _pytree.tree_flatten(
        (self.param_group, self.buffers, example_inputs)
        + ((example_targets,) if example_targets is not None else ())
    )

    # Pure tensor-level functional step captured by make_fx and compiled.
    def flattened_stateless_train_step(*flat_args):
      structured_args = _pytree.tree_unflatten(flat_args, in_spec)
      p_group, bufs, inps = structured_args[:3]
      tgts = structured_args[-1] if example_targets is not None else None

      bound_loss = functools.partial(
          _compute_loss,
          self.model,
          self._key_map,
          inputs=inps,
          targets=tgts,
      )

      grads, (loss, updated_bufs) = func.grad_and_value(
          bound_loss, has_aux=True, argnums=0
      )(p_group.params, bufs)
      new_p_group = self.optimizer(p_group, grads)

      flat_outputs, _ = _pytree.tree_flatten((loss, new_p_group, updated_bufs))
      return tuple(flat_outputs)

    unified_graph = make_fx(
        flattened_stateless_train_step, tracing_mode="fake"
    )(*flat_inputs)

    compiled_step = _invoke_compile_fn(
        compile_fn,
        typing.cast(torch.fx.GraphModule, unified_graph),
        flat_inputs,
    )

    # Build a template to capture output structure (`TreeSpec`) for
    # restoring backend flat results.
    dummy_loss = torch.tensor(0.0)
    _, out_spec = _pytree.tree_flatten(
        (dummy_loss, self.param_group, self.buffers)
    )

    # Stateful wrapper returned to user; handles flattening, running compiled
    # graph, and state updates.
    def train_step(inputs: Any, targets: Any = None) -> torch.Tensor:
      # Targets allowed here iff they were provided as examples.
      assert (targets is None) == (
          example_targets is None
      ), "Targets passed to step but not as examples, or vice versa."

      flat_inputs, _ = _pytree.tree_flatten(
          (self.param_group, self.buffers, inputs)
          + ((targets,) if targets is not None else ())
      )

      result = compiled_step(*flat_inputs)

      assert out_spec is not None
      loss, new_param_group, updated_bufs = _pytree.tree_unflatten(
          result, out_spec
      )

      self._update(new_param_group, updated_bufs)

      return loss

    return train_step
