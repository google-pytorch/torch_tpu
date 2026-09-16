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

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class _SparseGradPlaceholder:
  indices: torch.Tensor
  values: torch.Tensor
  size: tuple[int, ...]


def _sparse_grad_placeholder_unflatten(children: Any, size: Any):
  items = tuple(children)
  return _SparseGradPlaceholder(items[0], items[1], tuple(size))


_pytree.register_pytree_node(
    _SparseGradPlaceholder,
    lambda x: ([x.indices, x.values], x.size),
    _sparse_grad_placeholder_unflatten,
)


def _decomp_embedding_sparse_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
  if scale_grad_by_freq:
    raise RuntimeError(
        "embedding_sparse_backward(): scale_grad_by_freq not supported with"
        " sparse gradients"
    )
  indices_1d = indices.reshape(1, -1).to(torch.long)
  values = grad.reshape(-1, grad.shape[-1])
  if padding_idx >= 0:
    mask = (indices_1d != padding_idx).reshape(-1, 1)
    values = torch.where(mask, values, torch.zeros_like(values))
  return torch.ops.aten._sparse_coo_tensor_unsafe.default(
      indices_1d, values, (num_weights, grad.shape[-1])
  )


def _decomp_embedding_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> torch.Tensor:
  if sparse:
    return _decomp_embedding_sparse_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq
    )
  return torch.ops.aten.embedding_dense_backward.default(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq
  )


def _decomp_embedding_bag_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    max_indices: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    sparse: bool,
    per_sample_weights: torch.Tensor | None,
    padding_idx: int = -1,
) -> torch.Tensor:
  if not sparse:
    return NotImplemented
  index_grad = grad.index_select(0, offset2bag)
  if mode == 1:
    bag_size_f = bag_size.to(grad.dtype)
    safe_bag_size = torch.where(
        bag_size_f == 0, torch.ones_like(bag_size_f), bag_size_f
    )
    inv_bag_size = (
        safe_bag_size.reciprocal().unsqueeze(1).index_select(0, offset2bag)
    )
    index_grad = index_grad * inv_bag_size
  if per_sample_weights is not None:
    index_grad = index_grad * per_sample_weights.unsqueeze(1)
  return _decomp_embedding_sparse_backward(
      index_grad, indices, num_weights, padding_idx, scale_grad_by_freq
  )


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
      optimizer: Any = None,
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

    if self.optimizer is not None:
      self.param_group = self.optimizer.init_param_group(initial_params)
    else:
      # Without an optimizer, store raw parameter tensors directly in the
      # param_group container for pure forward + backward execution.
      self.param_group = initial_params

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
    p = (
        self.param_group.params
        if hasattr(self.param_group, "params")
        else self.param_group
    )
    return {self._key_map[k]: v for k, v in p.items()}

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

    traced_out_spec = None

    # Pure tensor-level functional step captured by make_fx and compiled.
    def flattened_stateless_train_step(*flat_args):
      nonlocal traced_out_spec
      structured_args = _pytree.tree_unflatten(flat_args, in_spec)
      p_group, bufs, inps = structured_args[:3]
      tgts = structured_args[-1] if example_targets is not None else None

      params_to_diff = p_group.params if hasattr(p_group, "params") else p_group
      diff_target = params_to_diff if params_to_diff else inps

      def bound_loss(target, bufs_arg):
        p = target if params_to_diff else {}
        inp = inps if params_to_diff else target
        return _compute_loss(
            self.model,
            self._key_map,
            params=p,
            buffers=bufs_arg,
            inputs=inp,
            targets=tgts,
        )

      grads, (loss, updated_bufs) = func.grad_and_value(
          bound_loss, has_aux=True, argnums=0
      )(diff_target, bufs)

      if self.optimizer is not None:
        new_p_group = self.optimizer(p_group, grads)
        flat_outputs, traced_out_spec = _pytree.tree_flatten(
            (loss, new_p_group, updated_bufs)
        )

      else:
        # Without an optimizer, surface gradients as explicit graph outputs.
        # This prevents compiler dead-code elimination (DCE) from optimizing
        # away the backward pass and ensures gradients can be synchronized.
        def _wrap_sparse(x):
          if isinstance(x, torch.Tensor) and x.is_sparse:
            # pylint: disable=protected-access
            return _SparseGradPlaceholder(
                x._indices(), x._values(), tuple(x.shape)
            )
          return x

        grads = _pytree.tree_map(_wrap_sparse, grads)
        flat_outputs, traced_out_spec = _pytree.tree_flatten(
            (loss, grads, updated_bufs)
        )
      return tuple(flat_outputs)

    decomp_table = {
        torch.ops.aten.embedding_backward.default: _decomp_embedding_backward,
        torch.ops.aten.embedding_sparse_backward.default: (
            _decomp_embedding_sparse_backward
        ),
        torch.ops.aten._embedding_bag_backward.default: (
            _decomp_embedding_bag_backward
        ),
    }
    unified_graph = make_fx(
        flattened_stateless_train_step,
        decomposition_table=decomp_table,
        tracing_mode="fake",
    )(*flat_inputs)

    for node in list(unified_graph.graph.nodes):
      if (
          node.op == "call_function"
          and node.target == torch.ops.aten._indices.default
          and node.args[0].op == "call_function"
          and node.args[0].target
          == torch.ops.aten._sparse_coo_tensor_unsafe.default
      ):
        node.replace_all_uses_with(node.args[0].args[0])
      elif (
          node.op == "call_function"
          and node.target == torch.ops.aten._values.default
          and node.args[0].op == "call_function"
          and node.args[0].target
          == torch.ops.aten._sparse_coo_tensor_unsafe.default
      ):
        node.replace_all_uses_with(node.args[0].args[1])
    unified_graph.graph.eliminate_dead_code()
    unified_graph.recompile()

    compiled_step = _invoke_compile_fn(
        compile_fn,
        typing.cast(torch.fx.GraphModule, unified_graph),
        flat_inputs,
    )

    out_spec = traced_out_spec
    if out_spec is None:
      # Build a template to capture output structure (`TreeSpec`) for
      # restoring backend flat results.
      dummy_loss = torch.tensor(0.0)
      if self.optimizer is not None:
        _, out_spec = _pytree.tree_flatten(
            (dummy_loss, self.param_group, self.buffers)
        )
      else:
        # Construct dummy gradients matching the output structure: input tree
        # for parameterless modules, or parameter dictionary for parameterized ones.
        if not self.param_group:
          dummy_grads = _pytree.tree_map(
              lambda _: torch.tensor(0.0), example_inputs
          )
        else:
          dummy_grads = {k: torch.tensor(0.0) for k in self.param_group.keys()}
        _, out_spec = _pytree.tree_flatten(
            (dummy_loss, dummy_grads, self.buffers)
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
      if self.optimizer is not None:
        loss, new_param_group, updated_bufs = _pytree.tree_unflatten(
            result, out_spec
        )
        self._update(new_param_group, updated_bufs)
      else:
        loss, grads, updated_bufs = _pytree.tree_unflatten(result, out_spec)
        self.buffers = updated_bufs
        self.model.load_state_dict(self.bufs, strict=False)

        def _unwrap_sparse(x):
          if isinstance(x, _SparseGradPlaceholder):
            return torch.ops.aten._sparse_coo_tensor_unsafe(
                x.indices, x.values, x.size
            )
          return x

        grads = _pytree.tree_map(
            _unwrap_sparse,
            grads,
            is_leaf=lambda x: isinstance(x, _SparseGradPlaceholder),
        )
        # Populate param.grad so callers (e.g. benchmark synchronization harness)
        # can eagerly synchronize device execution on parameter gradients.
        if isinstance(grads, dict):
          for name, param in self.model.named_parameters():
            safe_name = name.replace(".", "_")
            if safe_name in grads:
              param.grad = grads[safe_name]

      return loss

    return train_step
