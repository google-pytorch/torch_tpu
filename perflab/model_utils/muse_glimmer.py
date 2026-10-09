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

"""Muse Glimmer (text decoder) for perflab, built from the glimmer prototype.

The model and its training setup follow the team's Muse Glimmer TPU benchmark,
`torch_tpu/google/prototypes/glimmer/glimmer_tpu_test.py`, which runs a vendored
torchtitan Muse Glimmer (`torch_tpu.google.prototypes.glimmer`):

* flex attention with per-document, per-window block masks. On TPU, torch_tpu
  redirects flex attention to SDPA and, given `_tpu_doc_len`, runs it per
  document.
* rotate-half (`CosSinRoPE`) RoPE rather than the model's complex RoPE, which
  lowers poorly on TPU.
* parameters built directly in the benchmark dtype.
* with more than one rank: the `spmd_types` backend, SimpleFSDP `fully_shard`
  with `param_dtype` compute and fp32 gradient reduction, and (for the 30B
  flavor) full activation checkpointing.
* AdamW (lr 1e-4, eps 1e-3, weight decay 0.01) after clipping gradients to a
  global norm of 5.

Each rank trains on `num_docs` packed documents of `doc_len` tokens: the token
stream is `num_docs * doc_len` long, RoPE positions restart per document and
attention never crosses documents, so `num_docs` is the per-rank batch size.

The prototype vendors torchtitan and claims the `torchtitan` module namespace:
it aliases `sys.modules["torchtitan"]` to itself and resolves every
`torchtitan.*` import, including ones it makes at run time, to its own modules.
It therefore cannot share a process with code that uses the real torchtitan
(e.g. the Qwen3.8 benchmarks). Importing this module does not import the
prototype; `glimmer()` does, after evicting any real torchtitan modules, and
the switch lasts for the rest of the process. perflab runs every benchmark case
in its own process (`distributed_runner` spawns new ranks per target and
`torch_benchmark_test` forks a child per case unless `--norun_in_subprocess`),
so only the benchmark that calls `glimmer()` sees the prototype.
"""

from collections.abc import Callable
import contextlib
import dataclasses
import functools
import importlib
import sys
import types
from typing import Any

import torch
from torch import nn
from torch.distributed import tensor as dtensor
from torch.nn.attention import flex_attention as flex_attention_lib

_GLIMMER_PACKAGE = "torch_tpu.google.prototypes.glimmer"

# Training hyperparameters of the prototype's 30B runs.
_LEARNING_RATE = 1e-4
_ADAM_EPS = 1e-3
_WEIGHT_DECAY = 0.01
_GRADIENT_CLIP_NORM = 5.0
# Seeds the parameter initialization and the synthetic batch, as in the
# prototype.
_INIT_SEED = 42

# Flavors without activation checkpointing, like the prototype's debug model.
_NO_AC_FLAVORS = frozenset({"debugmodel"})

# Enters the SPMD mesh the forward pass runs under; set by a multi-rank
# `build_model`.
_forward_mesh: Callable[[], contextlib.AbstractContextManager[None]] | None = (
    None
)


def glimmer() -> types.ModuleType:
  """Imports the glimmer prototype and gives it the `torchtitan` namespace.

  Real torchtitan modules already imported in this process are evicted from
  `sys.modules` first, so the prototype's `torchtitan.*` imports cannot resolve
  to them. Modules that imported real torchtitan keep their references, but
  this process must not import more of it afterwards. See the module docstring.

  Returns:
    The `torch_tpu.google.prototypes.glimmer` package.
  """
  if _GLIMMER_PACKAGE in sys.modules:
    return sys.modules[_GLIMMER_PACKAGE]
  for name in list(sys.modules):
    if name == "torchtitan" or name.startswith("torchtitan."):
      del sys.modules[name]
  return importlib.import_module(_GLIMMER_PACKAGE)


def _submodule(name: str) -> types.ModuleType:
  glimmer()
  return importlib.import_module(f"{_GLIMMER_PACKAGE}.{name}")


@dataclasses.dataclass(frozen=True)
class ModelBundle:
  """A built Muse Glimmer model and its parallelism setup.

  Attributes:
    model: The model, SimpleFSDP-sharded if `parallel_dims` is set.
    parallel_dims: The prototype's `ParallelDims`, or None on one rank.
    parallel_config: The prototype's `ParallelismConfig`.
    vocab_size: The model's vocabulary size.
  """

  model: nn.Module
  parallel_dims: Any
  parallel_config: Any
  vocab_size: int


def text_config(flavor: str, num_layers: int, doc_len: int) -> Any:
  """Returns the prototype's text-only Muse Glimmer config, with CosSinRoPE.

  Args:
    flavor: "30B" or "debugmodel".
    num_layers: Number of decoder layers to keep, counted from the last layer.
      Muse Glimmer counts its iRoPE / sliding-window cycle from the end, so the
      last 4 layers are one full cycle of 3 sliding-window RoPE layers and a
      global NoPE layer. Values <= 0 keep all layers.
    doc_len: Document length. Positions restart per document, so this is the
      model's context length (RoPE cache size).

  Returns:
    A `MuseGlimmerModel.Config` with flex attention.
  """
  rope_lib = _submodule("models.common.rope")
  config = (
      glimmer()
      .model_registry(flavor, seq_len=doc_len, attn_backend="flex")
      .model
  )
  if num_layers > 0:
    config.layers = config.layers[-num_layers:]
  # The prototype's default `--rope=cos_sin`: the same rotation as ComplexRoPE
  # up to a fixed permutation of the q/k head dims, without the stride-2 lane
  # slices `view_as_complex` lowers to on TPU.
  for layer in config.layers:
    rope_config = layer.attention.rope
    if rope_config is not None:
      layer.attention.rope = rope_lib.CosSinRoPE.Config(
          **dataclasses.asdict(rope_config)
      )
  return config


def _apply_spmd_safe(model: nn.Module) -> None:
  """Wraps the transformer block class's forward in `spmd_safe` once.

  Like the prototype, the class rather than each block is wrapped, so that
  torch.compile does not recompile per block.

  Args:
    model: The (unsharded) model.
  """
  layer_cls = type(next(iter(model.layers.values())))
  if getattr(layer_cls.forward, "_perflab_spmd_safe", False):
    return
  forward = torch.tpu.distributed.spmd_safe(layer_cls.forward)
  forward._perflab_spmd_safe = True  # pylint: disable=protected-access
  layer_cls.forward = forward


def build_model(
    flavor: str,
    num_layers: int,
    doc_len: int,
    device: str,
    dtype: torch.dtype,
    *,
    world_size: int = 1,
    spmd_safe: bool = False,
) -> ModelBundle:
  """Builds Muse Glimmer like the prototype's `_build_model`.

  With `world_size > 1` (an initialized process group is required), the model
  is fully sharded with SimpleFSDP over all ranks using the `spmd_types`
  backend, with `dtype` compute and fp32 gradient reduction, and every block is
  activation-checkpointed (except for the debug model).

  Args:
    flavor: "30B" or "debugmodel".
    num_layers: See `text_config`.
    doc_len: See `text_config`.
    device: Device string (e.g. "tpu", "cuda" or "cpu").
    dtype: Parameter dtype.
    world_size: Number of ranks to shard the model across.
    spmd_safe: Wrap each block's forward in `torch.tpu.distributed.spmd_safe` so
      its collectives fuse with the surrounding ops (TPU only).

  Returns:
    The model bundle, in train mode.
  """
  g = glimmer()
  configs = _submodule("config.configs")
  titan_utils = _submodule("tools.utils")

  parallel_config = configs.ParallelismConfig(
      data_parallel_shard_degree=world_size, spmd_backend="spmd_types"
  )
  parallel_dims = None
  if world_size > 1:
    parallel_dims_lib = _submodule("distributed.parallel_dims")
    spmd_types_lib = _submodule("distributed.spmd_types")
    parallel_dims = parallel_dims_lib.ParallelDims.from_config(
        parallel_config, world_size=world_size
    )
    parallel_dims.build_mesh()
    spmd_types_lib.set_spmd_meshes(
        dense_mesh=parallel_dims.spmd_dense_mesh(),
        sparse_mesh=parallel_dims.spmd_sparse_mesh(),
    )
    global _forward_mesh  # pylint: disable=global-statement
    _forward_mesh = functools.partial(
        spmd_types_lib.set_current_spmd_mesh, parallel_dims.spmd_dense_mesh()
    )

  config = text_config(flavor, num_layers, doc_len)
  config.update_from_config(
      config=types.SimpleNamespace(parallelism=parallel_config)
  )
  with torch.device("meta"), titan_utils.set_default_dtype(dtype):
    model = config.build()

  if spmd_safe:
    _apply_spmd_safe(model)

  if parallel_dims is not None:
    model.parallelize(parallel_dims)
    fsdp_mesh = parallel_dims.get_optional_mesh(
        ["dp_shard", "cp"], include_singleton_axes=True
    )._flatten("fsdp")
    model = g.data_parallel(
        model,
        fsdp_mesh,
        mode="fully_shard",
        mp_policy=g.MixedPrecisionPolicy(
            param_dtype=dtype, reduce_dtype=torch.float32
        ),
        non_dp_mesh=parallel_dims.get_optional_mesh("tp"),
    )
    if flavor not in _NO_AC_FLAVORS:
      activation_checkpoint = _submodule("distributed.activation_checkpoint")
      activation_checkpoint.FullAC.Config(preserve_rng_state=False).build(
          dump_folder="/tmp"
      ).apply(model)

  model.to_empty(device=device)
  torch.manual_seed(_INIT_SEED)
  with torch.no_grad(), g.disable_active_parametrization():
    model.init_states(buffer_device=torch.device(device))
  model.train()
  return ModelBundle(
      model=model,
      parallel_dims=parallel_dims,
      parallel_config=parallel_config,
      vocab_size=config.vocab_size,
  )


def make_batch(
    bundle: ModelBundle,
    num_docs: int,
    doc_len: int,
    rank: int,
    world_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
  """Returns this rank's packed tokens and model kwargs (incl. `labels`).

  Mirrors the prototype's `_make_batch`: synthetic tokens and labels for all
  ranks from a fixed seed, sliced per rank, with per-document positions and
  the document / sliding-window attention masks.

  Args:
    bundle: The model bundle.
    num_docs: Documents per rank.
    doc_len: Tokens per document.
    rank: This rank.
    world_size: Number of ranks.
    device: Device to place the tensors on.

  Returns:
    `(tokens, kwargs)`: packed `(num_docs * doc_len,)` tokens, and the
    `labels`, `positions` and `attention_masks` keyword arguments.
  """
  seq_len = num_docs * doc_len
  global_tokens = seq_len * world_size
  generator = torch.Generator(device="cpu").manual_seed(_INIT_SEED)
  tokens = torch.randint(
      0, bundle.vocab_size, (global_tokens,), generator=generator
  )
  labels = torch.randint(
      0, bundle.vocab_size, (global_tokens,), generator=generator
  )
  positions = torch.arange(doc_len).repeat(num_docs * world_size)

  rank_slice = slice(rank * seq_len, (rank + 1) * seq_len)
  raw_batch = {
      "input": tokens[rank_slice].to(device),
      "positions": positions[rank_slice].to(device),
      "labels": labels[rank_slice].to(device),
  }
  if bundle.parallel_dims is not None:
    inputs, labels, kwargs = bundle.model.preprocess_inputs(
        raw_batch,
        parallel_dims=bundle.parallel_dims,
        parallelism=bundle.parallel_config,
    )
  else:
    inputs = raw_batch["input"]
    labels = raw_batch["labels"]
    kwargs = {
        "positions": raw_batch["positions"],
        "attention_masks": bundle.model.get_attention_masks(
            positions=raw_batch["positions"]
        ),
    }
  # Documents never attend across each other, so the TPU flex attention
  # redirect can run attention per document instead of over the packed
  # sequence.
  for mask in kwargs["attention_masks"].values():
    mask._tpu_doc_len = doc_len  # pylint: disable=protected-access
  kwargs["labels"] = labels
  return inputs, kwargs


def _mesh_context() -> contextlib.AbstractContextManager[None]:
  """The prototype's forward mesh context, if `build_model` registered one.

  It only carries SPMD type metadata. Under torch.compile it is skipped, as in
  the prototype, whose compiled regions (blocks) run inside it rather than
  tracing it.
  """
  if _forward_mesh is None or torch.compiler.is_compiling():
    return contextlib.nullcontext()
  return _forward_mesh()


def cross_entropy_loss(
    model: nn.Module,
    input_args: tuple[Any, ...],
    input_kwargs: dict[str, Any],
) -> torch.Tensor:
  """The prototype's training loss: fp32 cross entropy over global tokens.

  The per-rank sum is divided by the number of tokens across all ranks, as
  SimpleFSDP sums gradients across ranks.

  Args:
    model: The model (returns fp32 logits).
    input_args: `(tokens,)`.
    input_kwargs: Model kwargs plus `labels`.

  Returns:
    The scalar loss.
  """
  input_kwargs = dict(input_kwargs)
  labels = input_kwargs.pop("labels")
  world_size = (
      torch.distributed.get_world_size()
      if torch.distributed.is_initialized()
      else 1
  )
  with _mesh_context():
    logits = model(*input_args, **input_kwargs)
    loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
  return loss / float(labels.numel() * world_size)


def _clip_grad_norm_(
    parameters: list[nn.Parameter], max_norm: float
) -> torch.Tensor:
  """The prototype's `_clip_grad_norm`, without the host sync.

  Args:
    parameters: Parameters whose gradients to clip in place.
    max_norm: Max global gradient norm.

  Returns:
    The global gradient norm before clipping.
  """
  gradients = [
      # isinstance rather than the prototype's hasattr(to_local), which
      # torch.compile does not resolve on tensor subclasses.
      p.grad.to_local() if isinstance(p.grad, dtensor.DTensor) else p.grad
      for p in parameters
      if p.grad is not None
  ]
  sum_squared_norms = (
      torch.stack(torch._foreach_norm([g.float() for g in gradients]))  # pylint: disable=protected-access
      .pow(2)
      .sum()
  )
  if torch.distributed.is_initialized():
    torch.distributed.all_reduce(sum_squared_norms)
  total_norm = sum_squared_norms.sqrt()
  clip_coeff = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
  torch._foreach_mul_(gradients, clip_coeff)  # pylint: disable=protected-access
  return total_norm


class ClippedAdamW(torch.optim.AdamW):
  """AdamW that first clips the gradients to a global norm, as the prototype."""

  def __init__(self, params, *, max_grad_norm: float, **kwargs):
    super().__init__(params, **kwargs)
    self._max_grad_norm = max_grad_norm

  @torch.no_grad()
  def step(self, closure=None):
    _clip_grad_norm_(
        [p for group in self.param_groups for p in group["params"]],
        self._max_grad_norm,
    )
    return super().step(closure)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
  """Returns the prototype's optimizer for `model`.

  AdamW with the prototype's hyperparameters and `foreach` implementation,
  plus (off CPU) `capturable=True`, which keeps the step counter on device so a
  compiled training step has no host sync.

  Args:
    model: The model.
  """
  capturable = next(model.parameters()).device.type != "cpu"
  with glimmer().disable_active_parametrization():
    return ClippedAdamW(
        model.parameters(),
        max_grad_norm=_GRADIENT_CLIP_NORM,
        lr=_LEARNING_RATE,
        eps=_ADAM_EPS,
        weight_decay=_WEIGHT_DECAY,
        foreach=True,
        capturable=capturable,
    )


def allow_block_mask_cache_under_checkpoint() -> None:
  """Lets torch.compile trace the TPU flex redirect inside checkpointed blocks.

  The redirect caches the attention bias on the BlockMask; that write happens
  inside the activation-checkpointed blocks and need not be replayed when the
  backward recomputes the forward (the prototype sets the same flag).
  """
  torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True  # pylint: disable=protected-access


def use_sdpa_flex_attention() -> None:
  """Runs the model's flex attention as dense masked SDPA, for CPU tests.

  Eager flex attention has no backward on CPU. Like torch_tpu's TPU redirect
  (without its per-document folding), this materializes each block mask and
  calls SDPA.
  """
  attention = _submodule("models.common.attention")

  def sdpa_flex_attention(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      *,
      score_mod: Any = None,
      block_mask: Any = None,
      scale: float | None = None,
      enable_gqa: bool = False,
      **unused_kwargs: Any,
  ) -> tuple[torch.Tensor, Any]:
    if score_mod is not None:
      raise NotImplementedError("score_mod is not supported.")
    mask = None
    if block_mask is not None:
      batch, heads = block_mask.kv_num_blocks.shape[:2]
      mask = flex_attention_lib.create_mask(
          block_mask.mask_mod,
          batch,
          heads,
          query.shape[-2],
          key.shape[-2],
          device=query.device,
      )
    out = nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=enable_gqa
    )
    return out, flex_attention_lib.AuxOutput()

  attention.FlexAttention._compiled_flex_attn = sdpa_flex_attention  # pylint: disable=protected-access
