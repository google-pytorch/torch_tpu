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

from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import inspect
import sys
import typing
from typing import Any

import torch
import torch.func as func
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils import _python_dispatch
from torch.utils import _pytree
from torch_tpu._internal.optimizers import adamw
from torch_tpu._internal.optimizers import muon
from torch_tpu._internal.optimizers import optim
from torch_tpu._internal.optimizers import sgd
from torch_tpu._internal.utils import annotations as _annotations

__all__ = [
    "SingleTraceTrainer",
]

# Re-export optimizer classes for simple_trainer
Optimizer = optim.Optimizer
ReferenceAdamW = adamw.ReferenceAdamW
AdamW = adamw.AdamW
ReferenceSgd = sgd.ReferenceSgd
Sgd = sgd.Sgd
ReferenceMuon = muon.ReferenceMuon
Muon = muon.Muon


@contextlib.contextmanager
def _declare_tracing_to_transformers():
  """Makes Hugging Face mask creation take its trace-safe path under make_fx.

  `transformers.masking_utils.is_tracing` does not recognize make_fx inside
  torch.func transforms (tensors are functorch wrappers rather than
  FakeTensors), so HF evaluates data-dependent checks such as
  `fast_all(padding_mask)` and `find_packed_sequence_indices` and tracing fails.
  While active, HF instead builds attention masks in-graph. Masks that do not
  depend on graph inputs are folded back to `is_causal=True` by the
  `detect_causal_mask` compiler pass.

  Only `masking_utils` is patched: flipping the global torch.compiler flag also
  changes HF MoE code paths that are incompatible with torch.func.
  """
  masking_utils = sys.modules.get("transformers.masking_utils")
  if masking_utils is None or not hasattr(masking_utils, "is_tracing"):
    yield
    return
  original = getattr(masking_utils, "is_tracing")
  setattr(masking_utils, "is_tracing", lambda tensor=None: True)
  try:
    yield
  finally:
    setattr(masking_utils, "is_tracing", original)


def _compute_loss(
    model: torch.nn.Module,
    key_map: Mapping[str, str],
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    inputs: Any,
    targets: Any,
    compute_loss_fn: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
  """Computes loss for functional execution.

  If `compute_loss_fn` is set, the loss is `compute_loss_fn(model, args,
  kwargs)`. Otherwise, it is derived from `targets` or the model output.
  """
  p_orig = {key_map[k]: v for k, v in params.items()}
  b_orig = {key_map[k]: v for k, v in buffers.items()}

  if isinstance(inputs, dict):
    args, kwargs = (), inputs
  elif isinstance(inputs, (tuple, list)):
    args, kwargs = tuple(inputs), {}
  else:
    args, kwargs = (inputs,), {}

  with torch.nn.attention.sdpa_kernel([
      torch.nn.attention.SDPBackend.OVERRIDEABLE,
      torch.nn.attention.SDPBackend.MATH,
  ]):
    if compute_loss_fn is not None:
      # Swaps in `params` and `buffers` like `func.functional_call`, but lets
      # `compute_loss_fn` call the model. `params` lists tied weights (such as
      # an LM head sharing the embedding weight) only once, and
      # `tie_weights=True` swaps them in for every alias.
      with stateless._reparametrize_module(  # pylint: disable=protected-access
          model, {**p_orig, **b_orig}, tie_weights=True
      ):
        return compute_loss_fn(model, args, kwargs), buffers
    out = func.functional_call(model, (p_orig, b_orig), args, kwargs)

  loss = None
  if targets is not None:
    loss = torch.nn.functional.mse_loss(out, targets)
  elif hasattr(out, "loss") and out.loss is not None:
    loss = out.loss
  elif isinstance(out, dict) and out.get("loss") is not None:
    loss = out["loss"]
  elif hasattr(out, "logits") and out.logits is not None:
    loss = torch.mean(out.logits)
  elif hasattr(out, "sample") and out.sample is not None:
    loss = torch.mean(out.sample)
  elif isinstance(out, dict) and out.get("sample") is not None:
    loss = torch.mean(out["sample"])
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


@dataclasses.dataclass(frozen=True)
class _SubclassMeta:
  """Rebuilds a tensor subclass (e.g. DTensor) from its plain local tensor.

  Supports traceable wrapper subclasses with exactly one plain inner tensor.
  Other inner attributes, such as DTensor's `device_mesh`, are not tensors and
  are kept as constants.
  """

  cls: type[torch.Tensor]
  inner_attr: str
  inner_constants: dict[str, Any]
  ctx: Any
  outer_size: torch.Size
  outer_stride: tuple[int, ...]

  def wrap(self, inner: torch.Tensor) -> torch.Tensor:
    return self.cls.__tensor_unflatten__(  # pytype: disable=attribute-error
        {self.inner_attr: inner, **self.inner_constants},
        self.ctx,
        self.outer_size,
        self.outer_stride,
    )

  def is_sharded(self, inner: torch.Tensor) -> bool:
    """Whether `inner` holds only part of the logical (outer) tensor."""
    return tuple(inner.shape) != tuple(self.outer_size)


def _unwrap_subclass(
    name: str, t: torch.Tensor
) -> tuple[torch.Tensor, _SubclassMeta | None]:
  """Returns `t`'s plain inner tensor and how to rewrap it.

  Plain tensors are returned unchanged with no metadata.

  Args:
    name: Parameter or buffer name, for error messages.
    t: The tensor to unwrap.

  Raises:
    NotImplementedError: If `t` is a subclass that does not wrap exactly one
      plain tensor.
  """
  if not _python_dispatch.is_traceable_wrapper_subclass(t):
    return t, None
  attrs, ctx = t.__tensor_flatten__()  # pytype: disable=attribute-error
  inner = {attr: getattr(t, attr) for attr in attrs}
  tensor_attrs = [a for a, v in inner.items() if isinstance(v, torch.Tensor)]
  if len(tensor_attrs) != 1 or _python_dispatch.is_traceable_wrapper_subclass(
      inner[tensor_attrs[0]]
  ):
    raise NotImplementedError(
        "SingleTraceTrainer only supports tensor subclasses that wrap one plain"
        f" tensor (such as DTensor); {name} is a {type(t).__name__} with inner"
        f" tensors {tensor_attrs}."
    )
  inner_attr = tensor_attrs[0]
  return inner[inner_attr], _SubclassMeta(
      cls=type(t),
      inner_attr=inner_attr,
      inner_constants={a: v for a, v in inner.items() if a != inner_attr},
      ctx=ctx,
      outer_size=t.size(),
      outer_stride=t.stride(),
  )


def _to_plain_tensor(name: str, t: torch.Tensor) -> torch.Tensor:
  """Returns `t`'s local tensor if it is a subclass, else `t`."""
  return _unwrap_subclass(name, t)[0]


@_annotations.experimental(
    "SingleTraceTrainer is experimental and may change or be removed without"
    " notice."
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
      compute_loss_fn: Callable[..., torch.Tensor] | None = None,
  ):
    """Initializes the SingleTraceTrainer.

    Args:
      model: The model to train.
      optimizer: The optimizer applied inside the compiled step. If `None`, the
        step only computes gradients and stores them in `param.grad`.
      compile_fn: The default compilation callback for
        `make_compiled_train_step`.
      compute_loss_fn: Optional `compute_loss_fn(model, args, kwargs) -> loss`
        that replaces the built-in loss. It is called while the traced
        parameters and buffers are swapped into `model`, and it does not receive
        the targets passed to the train step.
    """
    self._model = model
    self._optimizer = optimizer
    self._compile_fn = compile_fn
    self._compute_loss_fn = compute_loss_fn

    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
      tie_weights()

    raw_params = dict(model.named_parameters())
    raw_buffers = dict(model.named_buffers())

    # Tensor subclass params and buffers (e.g. SimpleFSDP's DTensor shards) are
    # stored and traced as their plain local tensors, so the optimizer only
    # sees, and keeps state for, the local shard. They are rewrapped inside the
    # traced step, where DTensor dispatch records the collectives.
    self._subclass_metas: dict[str, _SubclassMeta] = {}
    self._key_map = {}
    initial_params = {}
    for name, param in raw_params.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      local, meta = _unwrap_subclass(name, param.detach())
      initial_params[safe_name] = local
      if meta is not None:
        self._subclass_metas[safe_name] = meta

    self._buffers = {}
    for name, buf in raw_buffers.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      local, meta = _unwrap_subclass(name, buf)
      self._buffers[safe_name] = local.detach()
      if meta is not None:
        self._subclass_metas[safe_name] = meta

    if self._optimizer is not None:
      if not getattr(self._optimizer, "_SUPPORTS_SHARDED_PARAMS", True):
        sharded = [
            self._key_map[k]
            for k, meta in self._subclass_metas.items()
            if k in initial_params and meta.is_sharded(initial_params[k])
        ]
        if sharded:
          raise NotImplementedError(
              f"{type(self._optimizer).__name__} needs full parameter"
              " matrices and does not support sharded parameters, but"
              f" {len(sharded)} parameters are sharded (e.g. {sharded[0]})."
          )
      self._param_group = self._optimizer.init_param_group(initial_params)
    else:
      # Without an optimizer, store raw parameter tensors directly in the
      # param_group container for pure forward + backward execution.
      self._param_group = initial_params

  def _wrap_subclasses(
      self, tensors: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    """Rewraps plain local tensors of subclass params/buffers (by safe name)."""
    if not self._subclass_metas:
      return tensors
    return {
        k: self._subclass_metas[k].wrap(v) if k in self._subclass_metas else v
        for k, v in tensors.items()
    }

  # Update state in this container and in user model.
  # Not technically required for params; all of the optimizers internalize an
  # # inplace update, and that part of the trace is not functionalized.
  # Buffers on the other hand may be mutated by the model inside stuff traced
  # by func.grad_and_value. This is functionalized so we will need an update
  # for buffers.
  def _update(self, new_pg, updated_bufs):
    self._buffers = updated_bufs
    self._param_group = new_pg
    self._model.load_state_dict(self._bufs, strict=False)

  @property
  @_annotations.experimental(
      "param_group is experimental and may change or be removed without notice."
  )
  def param_group(self) -> Any:
    """Returns the parameters and optimizer state that the train step updates.

    This is the `ParamGroup` returned by the optimizer's `init_param_group`, or
    a dict of parameters if the trainer has no optimizer.
    """
    return self._param_group

  @property
  @_annotations.experimental(
      "params is experimental and may change or be removed without notice."
  )
  def params(self) -> Mapping[str, torch.Tensor]:
    """Returns parameters with their original names.

    Tensor subclass parameters (e.g. DTensor) are returned rewrapped around
    the trainer's current local tensors.
    """
    p = (
        self._param_group.params
        if hasattr(self._param_group, "params")
        else self._param_group
    )
    return {self._key_map[k]: v for k, v in self._wrap_subclasses(p).items()}

  @property
  def _bufs(self) -> Mapping[str, torch.Tensor]:
    """Returns buffers with their original names."""
    return {
        self._key_map[k]: v
        for k, v in self._wrap_subclasses(self._buffers).items()
    }

  def _subclass_grad_and_value(
      self,
      params: Mapping[str, torch.Tensor],
      bufs: Mapping[str, torch.Tensor],
      inputs: Any,
      targets: Any,
  ) -> tuple[
      Mapping[str, torch.Tensor], torch.Tensor, Mapping[str, torch.Tensor]
  ]:
    """Computes local grads, loss and buffers for models with subclass state.

    Runs inside the traced step. `params` and `bufs` hold plain local tensors;
    subclass ones (e.g. DTensor shards) are rewrapped first, so tensor
    subclass dispatch records its local ops and collectives (for SimpleFSDP:
    the all-gather in forward and the reduce-scatter in backward). Gradients
    come from `torch.autograd.grad`, as in torchtitan's graph trainer, because
    `torch.func` transforms do not compose with DTensor.

    Args:
      params: Plain local parameter tensors, by safe name.
      bufs: Plain local buffer tensors, by safe name.
      inputs: Train step inputs.
      targets: Train step targets, or None.

    Returns:
      `(grads, loss, updated_bufs)`. Gradients are local tensors laid out like
      the local parameter shards, and buffers are local tensors.
    """
    if not params:
      raise NotImplementedError(
          "SingleTraceTrainer does not support tensor subclass buffers on a"
          " model without parameters."
      )
    wrapped_params = {}
    for k, v in self._wrap_subclasses(params).items():
      # Rewrapped subclasses are fresh leaves; plain proxies need grads too.
      wrapped_params[k] = v if v.requires_grad else v.requires_grad_()
    wrapped_bufs = self._wrap_subclasses(bufs)
    # The traced tensors stay swapped into the model during backward too, so
    # activation checkpointing recomputes the forward with them rather than
    # with the model's real parameters. The autograd engine must run backward
    # on this thread, where make_fx's tracing modes are active.
    with (
        stateless._reparametrize_module(  # pylint: disable=protected-access
            self._model,
            {
                **{self._key_map[k]: v for k, v in wrapped_params.items()},
                **{self._key_map[k]: v for k, v in wrapped_bufs.items()},
            },
            tie_weights=True,
        ),
        torch.enable_grad(),
        torch.autograd.set_multithreading_enabled(False),
    ):
      loss, updated_bufs = _compute_loss(
          self._model,
          self._key_map,
          params=wrapped_params,
          buffers=wrapped_bufs,
          inputs=inputs,
          targets=targets,
          compute_loss_fn=self._compute_loss_fn,
      )
      names = list(wrapped_params)
      raw_grads = torch.autograd.grad(
          loss, [wrapped_params[k] for k in names], allow_unused=True
      )

    grads = {}
    for k, grad in zip(names, raw_grads):
      if grad is None:
        grads[k] = torch.zeros_like(params[k])
        continue
      param = wrapped_params[k]
      placements = getattr(param, "placements", None)
      if placements is not None and tuple(grad.placements) != tuple(placements):
        # E.g. a Replicate param whose grad is still Partial.
        grad = grad.redistribute(param.device_mesh, placements)
      grads[k] = _to_plain_tensor(self._key_map[k], grad)

    if hasattr(loss, "full_tensor"):  # A DTensor loss.
      loss = loss.full_tensor()
    loss = _to_plain_tensor("loss", loss)
    updated_bufs = {
        k: _to_plain_tensor(self._key_map[k], v)
        for k, v in updated_bufs.items()
    }
    return grads, loss.detach(), updated_bufs

  @_annotations.experimental(
      "make_compiled_train_step() is experimental and may change or be removed"
      " without notice."
  )
  def make_compiled_train_step(
      self,
      example_inputs: Any,
      example_targets: Any = None,
      compile_fn: Callable[..., Any] | None = None,
  ) -> Callable[..., torch.Tensor]:
    """Returns a callable for the training step with the model compiled."""
    if compile_fn is None:
      compile_fn = self._compile_fn
    if compile_fn is None:
      raise ValueError(
          "A compilation callback (compile_fn) must be provided either to"
          " SingleTraceTrainer or to make_compiled_train_step."
      )

    # We don't actually want to trace through example_targets if it is None.
    # Static compiler doesn't support optional input args.
    flat_inputs, in_spec = _pytree.tree_flatten(
        (self._param_group, self._buffers, example_inputs)
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
            self._model,
            self._key_map,
            params=p,
            buffers=bufs_arg,
            inputs=inp,
            targets=tgts,
            compute_loss_fn=self._compute_loss_fn,
        )

      if self._subclass_metas:
        grads, loss, updated_bufs = self._subclass_grad_and_value(
            params_to_diff, bufs, inps, tgts
        )
      else:
        grads, (loss, updated_bufs) = func.grad_and_value(
            bound_loss, has_aux=True, argnums=0
        )(diff_target, bufs)

      if self._optimizer is not None:
        new_p_group = self._optimizer(p_group, grads)
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
    with _declare_tracing_to_transformers():
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
      if self._optimizer is not None:
        _, out_spec = _pytree.tree_flatten(
            (dummy_loss, self._param_group, self._buffers)
        )
      else:
        # Construct dummy gradients matching the output structure: input tree
        # for parameterless modules, or parameter dictionary for parameterized ones.
        if not self._param_group:
          dummy_grads = _pytree.tree_map(
              lambda _: torch.tensor(0.0), example_inputs
          )
        else:
          dummy_grads = {k: torch.tensor(0.0) for k in self._param_group.keys()}
        _, out_spec = _pytree.tree_flatten(
            (dummy_loss, dummy_grads, self._buffers)
        )

    # Stateful wrapper returned to user; handles flattening, running compiled
    # graph, and state updates.
    def train_step(inputs: Any, targets: Any = None) -> torch.Tensor:
      # Targets allowed here iff they were provided as examples.
      assert (targets is None) == (
          example_targets is None
      ), "Targets passed to step but not as examples, or vice versa."

      flat_inputs, _ = _pytree.tree_flatten(
          (self._param_group, self._buffers, inputs)
          + ((targets,) if targets is not None else ())
      )

      result = compiled_step(*flat_inputs)

      assert out_spec is not None
      if self._optimizer is not None:
        loss, new_param_group, updated_bufs = _pytree.tree_unflatten(
            result, out_spec
        )
        self._update(new_param_group, updated_bufs)
      else:
        loss, grads, updated_bufs = _pytree.tree_unflatten(result, out_spec)
        self._buffers = updated_bufs
        self._model.load_state_dict(self._bufs, strict=False)

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
          if self._subclass_metas:
            grads = {
                k: v.detach() for k, v in self._wrap_subclasses(grads).items()
            }
          for name, param in self._model.named_parameters():
            safe_name = name.replace(".", "_")
            if safe_name in grads:
              param.grad = grads[safe_name]

      return loss

    return train_step
