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

  updated_buffers = dict(buffers)
  if buffers:
    orig_forward = model.forward

    def _capturing_forward(*a, **kw):
      res = orig_forward(*a, **kw)
      for k, orig_name in key_map.items():
        if k in buffers:
          updated_buffers[k] = model.get_buffer(orig_name)
      return res

    try:
      model.forward = _capturing_forward
      out = func.functional_call(model, (p_orig, b_orig), args, kwargs)
    finally:
      model.forward = orig_forward
  else:
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

  return loss, updated_buffers


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

    # Map sanitized buffer names (dots replaced by underscores for FX proxy
    # compatibility) directly to the owning submodule's `_buffers` dict and
    # local attribute name. This allows `_update` and fast-path training loops
    # to write updated buffers back into `self.model` in O(1) time per buffer
    # without calling `self.model.load_state_dict(..., strict=False)` on every
    # step (which incurs heavy Python recursion and dictionary allocation).
    self._buffer_target_map: dict[str, tuple[dict[str, Any], str]] = {}
    self.buffers = {}
    for name, buf in raw_buffers.items():
      safe_name = name.replace(".", "_")
      self._key_map[safe_name] = name
      self.buffers[safe_name] = buf.detach()
      if "." in name:
        mod_path, attr_name = name.rsplit(".", 1)
        target_mod = self.model.get_submodule(mod_path)
      else:
        target_mod, attr_name = self.model, name
      self._buffer_target_map[safe_name] = (target_mod._buffers, attr_name)

    # Monotonically increasing version counter incremented whenever parameters
    # or buffers are updated. Compiled evaluation steps compare `self._version`
    # against their cached version to detect whether their flattened prefix
    # tuple needs to be rebuilt.
    self._version: int = 0
    if self.optimizer is not None:
      self.param_group = self.optimizer.init_param_group(initial_params)
    else:
      # Without an optimizer, store raw parameter tensors directly in the
      # param_group container for pure forward + backward execution.
      self.param_group = initial_params

  # Update state in this container and in user model.
  # Not technically required for params; all of the optimizers internalize an
  # inplace update, and that part of the trace is not functionalized.
  # Buffers on the other hand may be mutated by the model inside stuff traced
  # by func.grad_and_value. This is functionalized so we will need an update
  # for buffers.
  def _update(self, new_pg, updated_bufs):
    self._version += 1
    self.buffers = updated_bufs
    self.param_group = new_pg
    if updated_bufs:
      # Write updated buffer tensors directly into each owning submodule's
      # `_buffers` dictionary using the precomputed `_buffer_target_map`.
      for k, v in updated_bufs.items():
        if k in self._buffer_target_map:
          mod_bufs, attr = self._buffer_target_map[k]
          mod_bufs[attr] = v
        else:
          self.model.load_state_dict(self.bufs, strict=False)
          break

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

    # Extract the direct C++ `_fast_runner` callable if supported by the
    # compiled executable (e.g., single-partition `_SplitCompiledExecutable` or
    # `TorchTpuCompiledExecutable`). Calling `fast_runner(flat_args)` bypasses
    # FX `GraphModule.forward` and AOTAutograd boxing overhead.
    fast_runner = getattr(compiled_step, "_fast_runner", None)
    static_prefix: tuple[Any, ...] | None = None
    fast_grad_pairs: tuple[tuple[torch.nn.Parameter, int], ...] | None = None
    fast_buf_updates: (
        tuple[tuple[int, int, str, dict[str, Any], str], ...] | None
    ) = None
    fast_loss_idx: int = 0
    prefix_list: list[Any] | None = None

    # Compile-time output layout analysis for optimizer-free training steps
    # (pure forward + backward passes, as used in layer benchmarks).
    # Instead of calling `_pytree.tree_unflatten(result, out_spec)` on every
    # training step to reconstruct nested dicts of `(loss, grads_dict, bufs_dict)`,
    # we unflatten a synthetic tuple of integer indices `0..num_leaves-1` once
    # here at compile time. This reveals the exact flat output index for:
    #   1. `loss_idx`: The scalar loss tensor.
    #   2. `grads_idx[param_name]`: The gradient tensor for each parameter.
    #   3. `bufs_idx[buf_name]`: The updated tensor for each mutated buffer.
    if self.optimizer is None and out_spec is not None:
      flat_prefix, _ = _pytree.tree_flatten((self.param_group, self.buffers))
      try:
        idx_tuple = tuple(range(out_spec.num_leaves))
        loss_idx, grads_idx, bufs_idx = _pytree.tree_unflatten(
            idx_tuple, out_spec
        )
        # Ensure all leaves mapped cleanly to integer positions (no nested
        # `_SparseGradPlaceholder` or non-tensor leaves in the gradients/buffers).
        if (
            isinstance(loss_idx, int)
            and isinstance(grads_idx, dict)
            and all(isinstance(v, int) for v in grads_idx.values())
            and isinstance(bufs_idx, dict)
            and all(isinstance(v, int) for v in bufs_idx.values())
        ):
          fast_loss_idx = loss_idx
          # Pre-bind each `nn.Parameter` object directly to its corresponding
          # flat output tuple index so runtime gradient assignment is a simple
          # `param.grad = res[grad_idx]` loop without string lookups.
          fast_grad_pairs = tuple(
              (param, grads_idx[name.replace(".", "_")])
              for name, param in self.model.named_parameters()
              if name.replace(".", "_") in grads_idx
          )
          if not bufs_idx:
            # When the module has no mutated buffers, the flattened prefix of
            # `(params, buffers)` is completely invariant across steps and can
            # be stored as an immutable `static_prefix` tuple.
            static_prefix = tuple(flat_prefix)
          elif all(k in self._buffer_target_map for k in bufs_idx):
            # When the module updates stateful buffers (e.g. `BatchNorm` running
            # mean/variance), store the prefix as a mutable `prefix_list`.
            # Precompute `fast_buf_updates` tuples containing:
            #   - `p_idx`: Index in `prefix_list` to overwrite for the next step.
            #   - `r_idx`: Index in `fast_runner` output tuple containing the new buffer.
            #   - `k`: Sanitized buffer key in `self.buffers`.
            #   - `mod_bufs`, `attr`: Direct reference to the submodule's
            #     `_buffers` dict and attribute name for O(1) in-place update.
            prefix_list = list(flat_prefix)
            num_params = len(self.param_group)
            fast_buf_updates = tuple(
                (
                    num_params + i,
                    bufs_idx[k],
                    k,
                    self._buffer_target_map[k][0],
                    self._buffer_target_map[k][1],
                )
                for i, k in enumerate(self.buffers.keys())
                if k in bufs_idx
            )
      except Exception:  # pylint: disable=broad-except
        fast_grad_pairs = None
        fast_buf_updates = None

    # Specialize the returned training step closure based on:
    #   1. Input structure (tuple of tensors vs. single tensor vs. with targets).
    #   2. Whether stateful buffers are updated (`static_prefix` vs. `prefix_list`).
    #   3. Number of parameters (unrolling the 2-parameter case eliminates loop
    #      overhead for standard weight+bias layers like Linear/LayerNorm).
    if (
        fast_runner is not None
        and fast_grad_pairs is not None
        and static_prefix is not None
        and example_targets is None
    ):
      if isinstance(example_inputs, tuple) and all(
          isinstance(x, torch.Tensor) for x in example_inputs
      ):
        # Fast path 1a: Tuple of tensor inputs, no buffers, exactly 2 parameters
        # (unrolled gradient assignment).
        if len(fast_grad_pairs) == 2:
          p0, idx0 = fast_grad_pairs[0]
          p1, idx1 = fast_grad_pairs[1]

          def train_step_fast_tuple_2(
              inputs: Any, targets: Any = None
          ) -> torch.Tensor:
            assert targets is None
            res = fast_runner(static_prefix + inputs)
            p0.grad = res[idx0]
            p1.grad = res[idx1]
            return res[fast_loss_idx]

          setattr(train_step_fast_tuple_2, "is_pre_compiled", True)
          return train_step_fast_tuple_2

        # Fast path 1b: Tuple of tensor inputs, no buffers, N parameters.
        def train_step_fast_tuple(
            inputs: Any, targets: Any = None
        ) -> torch.Tensor:
          assert targets is None
          res = fast_runner(static_prefix + inputs)
          for param, grad_idx in fast_grad_pairs:
            param.grad = res[grad_idx]
          return res[fast_loss_idx]

        setattr(train_step_fast_tuple, "is_pre_compiled", True)
        return train_step_fast_tuple

      elif isinstance(example_inputs, torch.Tensor):
        # Fast path 2a: Single tensor input, no buffers, exactly 2 parameters
        # (unrolled gradient assignment).
        if len(fast_grad_pairs) == 2:
          p0, idx0 = fast_grad_pairs[0]
          p1, idx1 = fast_grad_pairs[1]

          def train_step_fast_tensor_2(
              inputs: Any, targets: Any = None
          ) -> torch.Tensor:
            assert targets is None
            res = fast_runner((*static_prefix, inputs))
            p0.grad = res[idx0]
            p1.grad = res[idx1]
            return res[fast_loss_idx]

          setattr(train_step_fast_tensor_2, "is_pre_compiled", True)
          return train_step_fast_tensor_2

        # Fast path 2b: Single tensor input, no buffers, N parameters.
        def train_step_fast_tensor(
            inputs: Any, targets: Any = None
        ) -> torch.Tensor:
          assert targets is None
          res = fast_runner((*static_prefix, inputs))
          for param, grad_idx in fast_grad_pairs:
            param.grad = res[grad_idx]
          return res[fast_loss_idx]

        setattr(train_step_fast_tensor, "is_pre_compiled", True)
        return train_step_fast_tensor

    elif (
        fast_runner is not None
        and fast_grad_pairs is not None
        and static_prefix is not None
        and isinstance(example_inputs, torch.Tensor)
        and isinstance(example_targets, torch.Tensor)
    ):
      # Fast path 3: Single tensor input + single tensor target, no buffers.
      def train_step_fast_tensor_targets(
          inputs: Any, targets: Any = None
      ) -> torch.Tensor:
        assert targets is not None
        res = fast_runner((*static_prefix, inputs, targets))
        for param, grad_idx in fast_grad_pairs:
          param.grad = res[grad_idx]
        return res[fast_loss_idx]

      setattr(train_step_fast_tensor_targets, "is_pre_compiled", True)
      return train_step_fast_tensor_targets

    elif (
        fast_runner is not None
        and fast_grad_pairs is not None
        and fast_buf_updates is not None
        and prefix_list is not None
        and example_targets is None
    ):
      # Fast path 4: Modules with stateful buffers (e.g. BatchNorm). After each
      # step, `prefix_list[p_idx]` is updated in place so subsequent steps
      # receive the updated running statistics without re-flattening pytrees.
      if isinstance(example_inputs, torch.Tensor):

        def train_step_fast_tensor_with_bufs(
            inputs: Any, targets: Any = None
        ) -> torch.Tensor:
          assert targets is None
          self._version += 1
          res = fast_runner((*prefix_list, inputs))
          for param, grad_idx in fast_grad_pairs:
            param.grad = res[grad_idx]
          for p_idx, r_idx, k, mod_bufs, attr in fast_buf_updates:
            new_b = res[r_idx]
            prefix_list[p_idx] = new_b
            self.buffers[k] = new_b
            mod_bufs[attr] = new_b
          return res[fast_loss_idx]

        setattr(train_step_fast_tensor_with_bufs, "is_pre_compiled", True)
        return train_step_fast_tensor_with_bufs

      elif isinstance(example_inputs, tuple) and all(
          isinstance(x, torch.Tensor) for x in example_inputs
      ):

        def train_step_fast_tuple_with_bufs(
            inputs: Any, targets: Any = None
        ) -> torch.Tensor:
          assert targets is None
          self._version += 1
          res = fast_runner((*prefix_list, *inputs))
          for param, grad_idx in fast_grad_pairs:
            param.grad = res[grad_idx]
          for p_idx, r_idx, k, mod_bufs, attr in fast_buf_updates:
            new_b = res[r_idx]
            prefix_list[p_idx] = new_b
            self.buffers[k] = new_b
            mod_bufs[attr] = new_b
          return res[fast_loss_idx]

        setattr(train_step_fast_tuple_with_bufs, "is_pre_compiled", True)
        return train_step_fast_tuple_with_bufs

    # Fallback general stateful wrapper returned to user when an optimizer is
    # present, sparse gradients are used, or inputs/outputs are arbitrary pytrees.
    def train_step(inputs: Any, targets: Any = None) -> torch.Tensor:
      # Targets allowed here iff they were provided as examples.
      assert (targets is None) == (
          example_targets is None
      ), "Targets passed to step but not as examples, or vice versa."

      # Avoid full `_pytree.tree_flatten` overhead when `static_prefix` is known
      # and inputs are simple tensors or flat tuples/lists of tensors.
      if static_prefix is not None:
        if targets is None:
          if isinstance(inputs, torch.Tensor):
            flat_inputs = (*static_prefix, inputs)
          elif isinstance(inputs, (tuple, list)) and all(
              isinstance(x, torch.Tensor) for x in inputs
          ):
            flat_inputs = (*static_prefix, *inputs)
          else:
            flat_inputs = (*static_prefix, *_pytree.tree_leaves(inputs))
        elif isinstance(inputs, torch.Tensor) and isinstance(
            targets, torch.Tensor
        ):
          flat_inputs = (*static_prefix, inputs, targets)
        else:
          flat_inputs = (
              *static_prefix,
              *_pytree.tree_leaves((inputs, targets)),
          )
      else:
        flat_inputs, _ = _pytree.tree_flatten(
            (self.param_group, self.buffers, inputs)
            + ((targets,) if targets is not None else ())
        )

      if fast_runner is not None:
        result = fast_runner(flat_inputs)
      else:
        result = compiled_step(*flat_inputs)

      if fast_grad_pairs is not None and static_prefix is not None:
        for param, grad_idx in fast_grad_pairs:
          param.grad = result[grad_idx]
        return result[fast_loss_idx]

      assert out_spec is not None
      if self.optimizer is not None:
        loss, new_param_group, updated_bufs = _pytree.tree_unflatten(
            result, out_spec
        )
        self._update(new_param_group, updated_bufs)
      else:
        loss, grads, updated_bufs = _pytree.tree_unflatten(result, out_spec)
        if updated_bufs:
          self._version += 1
          self.buffers = updated_bufs
          for k, v in updated_bufs.items():
            if k in self._buffer_target_map:
              mod_bufs, attr = self._buffer_target_map[k]
              mod_bufs[attr] = v
            else:
              self.model.load_state_dict(self.bufs, strict=False)
              break

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

    setattr(train_step, "is_pre_compiled", True)
    return train_step

  def make_compiled_eval_step(
      self,
      example_inputs: Any,
      compile_fn: Callable[..., Any] | None = None,
  ) -> Callable[..., Any]:
    """Returns a callable for the evaluation step with the model compiled.

    Similar to `make_compiled_train_step`, this traces the forward evaluation
    pass into a single functional FX graph (`make_fx` under `torch.no_grad()`)
    and compiles it once, returning a specialized low-overhead callable that
    bypasses Dynamo per-step guard evaluations.
    """
    if compile_fn is None:
      compile_fn = self.compile_fn
    if compile_fn is None:
      raise ValueError(
          "A compilation callback (compile_fn) must be provided either to"
          " SingleTraceTrainer or to make_compiled_eval_step."
      )

    # Flatten model parameters (extracting `.params` if `self.param_group` is an
    # optimizer `ParamGroup` dataclass) and buffers once to form the static
    # prefix of arguments passed to the functional graph.
    params_dict = (
        self.param_group.params
        if hasattr(self.param_group, "params")
        else self.param_group
    )
    flat_params_bufs, pb_spec = _pytree.tree_flatten(
        (params_dict, self.buffers)
    )
    num_static = len(flat_params_bufs)
    static_prefix = tuple(flat_params_bufs)
    flat_inp, inp_spec = _pytree.tree_flatten(example_inputs)
    flat_inputs = [*static_prefix, *flat_inp]
    out_spec_holder: list[Any] = [None]

    def flattened_stateless_eval_step(*flat_args):
      p_dict, bufs = _pytree.tree_unflatten(flat_args[:num_static], pb_spec)
      inp = _pytree.tree_unflatten(flat_args[num_static:], inp_spec)
      p_orig = {self._key_map[k]: v for k, v in p_dict.items()}
      b_orig = {self._key_map[k]: v for k, v in bufs.items()}
      if isinstance(inp, dict):
        args, kwargs = (), inp
      elif isinstance(inp, (tuple, list)):
        args, kwargs = tuple(inp), {}
      else:
        args, kwargs = (inp,), {}
      out = func.functional_call(self.model, (p_orig, b_orig), args, kwargs)
      flat_out, out_spec = _pytree.tree_flatten(out)
      out_spec_holder[0] = out_spec
      return tuple(flat_out)

    # Trace the stateless forward pass with FakeTensors and eliminate dead code
    # before invoking the backend compiler.
    with torch.no_grad():
      unified_graph = make_fx(
          flattened_stateless_eval_step,
          tracing_mode="fake",
      )(*flat_inputs)
      unified_graph.graph.eliminate_dead_code()
      unified_graph.recompile()

      compiled_step = _invoke_compile_fn(
          compile_fn,
          typing.cast(torch.fx.GraphModule, unified_graph),
          flat_inputs,
      )

    fast_runner = getattr(compiled_step, "_fast_runner", None)
    out_spec = out_spec_holder[0]
    # Check if the model returns a single raw Tensor (not wrapped in a tuple/dict)
    # so we can return `res[0]` directly without calling `_pytree.tree_unflatten`.
    is_single_tensor_out = (
        out_spec is not None
        and out_spec.num_leaves == 1
        and out_spec.type is None
    )
    # When no optimizer is attached and the model has no buffers, the parameter
    # prefix is guaranteed not to change between evaluation steps.
    is_invariant_prefix = self.optimizer is None and not self.buffers
    cached_version = self._version
    cached_prefix = static_prefix

    def _get_prefix() -> tuple[Any, ...]:
      """Returns the flattened parameter/buffer prefix, refreshing on updates."""
      nonlocal cached_version, cached_prefix
      if self._version != cached_version:
        curr_params = (
            self.param_group.params
            if hasattr(self.param_group, "params")
            else self.param_group
        )
        cached_prefix = tuple(_pytree.tree_leaves((curr_params, self.buffers)))
        cached_version = self._version
      return cached_prefix

    # Pure tensor metadata view/alias operations that do not perform device
    # computation. When an evaluation FX graph contains only placeholders,
    # outputs, and these view targets (e.g. `nn.Dropout` or `nn.Flatten` in eval
    # mode), compiling it to XLA forces PJRT to launch a standalone device copy
    # kernel (~100us overhead). Bypassing `_fast_runner` and executing the view
    # directly on host metadata preserves zero-copy sharing of the underlying TPU
    # DeviceBuffer.
    _VIEW_OR_NOOP_TARGETS = {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.t.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.slice.Tensor,
    }
    has_compute_ops = any(
        n.op not in ("placeholder", "output")
        and not (n.op == "call_function" and n.target in _VIEW_OR_NOOP_TARGETS)
        for n in unified_graph.graph.nodes
    )

    # Zero-compute fast path: if the traced graph performs only view/alias
    # operations or is an identity no-op, execute the FX GraphModule directly
    # on host tensor metadata instead of dispatching a PJRT XLA copy kernel.
    if not has_compute_ops:
      # Exact identity graph (`return x`, e.g. `nn.Dropout(p).eval()`).
      if (
          num_static == 0
          and is_single_tensor_out
          and isinstance(example_inputs, torch.Tensor)
          and len(unified_graph.graph.nodes) == 2
      ):

        def eval_step_noop_tensor(inputs: Any) -> Any:
          return inputs

        setattr(eval_step_noop_tensor, "is_pre_compiled", True)
        return eval_step_noop_tensor

      # Single-tensor input with pure view operations (e.g. `nn.Flatten().eval()`).
      if is_single_tensor_out and isinstance(example_inputs, torch.Tensor):
        if num_static == 0:

          def eval_step_view_tensor_0(inputs: Any) -> Any:
            return unified_graph(inputs)[0]

          setattr(eval_step_view_tensor_0, "is_pre_compiled", True)
          return eval_step_view_tensor_0

        def eval_step_view_tensor(inputs: Any) -> Any:
          return unified_graph(
              *(static_prefix if is_invariant_prefix else _get_prefix()),
              inputs,
          )[0]

        setattr(eval_step_view_tensor, "is_pre_compiled", True)
        return eval_step_view_tensor

      # Tuple of tensor inputs with pure view operations.
      if isinstance(example_inputs, tuple) and all(
          isinstance(x, torch.Tensor) for x in example_inputs
      ):

        def eval_step_view_tuple(inputs: Any) -> Any:
          res = unified_graph(
              *(static_prefix if is_invariant_prefix else _get_prefix()),
              *inputs,
          )
          return (
              res[0]
              if is_single_tensor_out
              else _pytree.tree_unflatten(res, out_spec)
          )

        setattr(eval_step_view_tuple, "is_pre_compiled", True)
        return eval_step_view_tuple

    # Standard compiled execution fast paths: dispatch directly to C++
    # `fast_runner`, specialized on input type (tuple vs. single tensor),
    # output structure (single tensor vs. multi-leaf pytree), and prefix
    # invariance (`static_prefix` vs. `_get_prefix()`).
    if fast_runner is not None:
      if isinstance(example_inputs, tuple) and all(
          isinstance(x, torch.Tensor) for x in example_inputs
      ):
        if is_single_tensor_out:
          if is_invariant_prefix:

            def eval_step_fast_tuple_single(inputs: Any) -> Any:
              return fast_runner(static_prefix + inputs)[0]

            setattr(eval_step_fast_tuple_single, "is_pre_compiled", True)
            return eval_step_fast_tuple_single
          else:

            def eval_step_fast_tuple_single_dyn(inputs: Any) -> Any:
              return fast_runner(_get_prefix() + inputs)[0]

            setattr(eval_step_fast_tuple_single_dyn, "is_pre_compiled", True)
            return eval_step_fast_tuple_single_dyn
        else:

          def eval_step_fast_tuple_multi(inputs: Any) -> Any:
            return _pytree.tree_unflatten(
                fast_runner(
                    (static_prefix if is_invariant_prefix else _get_prefix())
                    + inputs
                ),
                out_spec,
            )

          setattr(eval_step_fast_tuple_multi, "is_pre_compiled", True)
          return eval_step_fast_tuple_multi
      elif isinstance(example_inputs, torch.Tensor):
        if is_single_tensor_out:
          if is_invariant_prefix:

            def eval_step_fast_tensor_single(inputs: Any) -> Any:
              return fast_runner((*static_prefix, inputs))[0]

            setattr(eval_step_fast_tensor_single, "is_pre_compiled", True)
            return eval_step_fast_tensor_single
          else:

            def eval_step_fast_tensor_single_dyn(inputs: Any) -> Any:
              return fast_runner((*_get_prefix(), inputs))[0]

            setattr(eval_step_fast_tensor_single_dyn, "is_pre_compiled", True)
            return eval_step_fast_tensor_single_dyn
        else:

          def eval_step_fast_tensor_multi(inputs: Any) -> Any:
            return _pytree.tree_unflatten(
                fast_runner((
                    *(static_prefix if is_invariant_prefix else _get_prefix()),
                    inputs,
                )),
                out_spec,
            )

          setattr(eval_step_fast_tensor_multi, "is_pre_compiled", True)
          return eval_step_fast_tensor_multi

    # General fallback for arbitrary pytree inputs or backends without `_fast_runner`.
    def eval_step(inputs: Any) -> Any:
      f_inp, _ = _pytree.tree_flatten(inputs)
      pref = static_prefix if is_invariant_prefix else _get_prefix()
      res = compiled_step(*pref, *f_inp)
      if is_single_tensor_out:
        return res[0]
      return _pytree.tree_unflatten(res, out_spec)

    setattr(eval_step, "is_pre_compiled", True)
    return eval_step
