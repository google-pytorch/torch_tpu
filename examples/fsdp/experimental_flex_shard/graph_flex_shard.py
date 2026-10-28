# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the accompanying FLEX_SHARD_LICENSE.
"""Experimental graph executor for FlexShard storage, not the CUDA runtime.

Only 1D fully-sharded or replicated data parallelism is supported. Collectives
and their adjoints are functional; no streams, parameter-swapping hooks, or
storage resizing enter the training graph. This first prototype uses one
parameter per communication bucket so accesses remain pure inside checkpoints.
Bucket participation must be rank-uniform. HSDP, outer TP/EP, and coalesced
multi-parameter buckets are intentionally rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from flex_shard import BucketSpec, MixedPrecisionPolicy
from flex_shard.flex_shard.bucket_runtime import ParamOwnerRef
from flex_shard.flex_shard.bucket_storage import (
  ShardedBucketStorage,
  _assign_params_to_buckets,
)
from flex_shard.flex_shard.flex_shard import (
  _check_not_already_flex_sharded,
  _validate_param_dtype_overrides,
)
from flex_shard.flex_shard.placement_contract import (
  GlobalLayout,
  Placement,
  get_global_layout,
  get_partial_grad_group,
)
from flex_shard.flex_shard.utils import (
  _get_managed_named_params,
  _get_shared_param_names,
  _validate_bucket_uniform_dtype_and_placement,
  _validate_placements,
)
from torch import nn
from torch.distributed import _functional_collectives as fc
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class GraphShard(Placement):
  """Storage placement consumed only by this graph executor.

  dim=None replicates. Integer dimensions use ceil-sized, padded chunks.
  This is intentionally not accepted by FlexShard's CUDA eager executor.
  """

  dim: int | None = 0

  def bucket_compatibility_key(self) -> object:
    return (type(self), self.dim is None)

  def compute_local_shape(self, global_shape, rank, world_size):
    shape = list(global_shape)
    if self.dim is not None:
      chunk = math.ceil(shape[self.dim] / world_size)
      shape[self.dim] = max(0, min(chunk, shape[self.dim] - rank * chunk))
    return torch.Size(shape)

  def extract_local_shard(self, param, rank, world_size):
    if self.dim is None:
      return param
    chunk = math.ceil(param.shape[self.dim] / world_size)
    start = min(rank * chunk, param.shape[self.dim])
    length = self.compute_local_shape(param.shape, rank, world_size)[self.dim]
    return param.narrow(self.dim, start, length)

  def global_layout(self, info, rank, world_size):
    offset = [0] * len(info.global_shape)
    if self.dim is not None:
      chunk = math.ceil(info.global_shape[self.dim] / world_size)
      offset[self.dim] = min(rank * chunk, info.global_shape[self.dim])
    present = info.local_numel > 0
    return GlobalLayout(
      tuple(info.global_shape),
      (tuple(offset),) if present else (),
      ((0,) * len(offset),) if present else (),
      (tuple(info.local_shape),) if present else (),
    )


def _pad_chunk(tensor, shape, dim, chunk):
  missing = chunk - tensor.shape[dim]
  if missing == 0:
    return tensor.reshape(-1)
  padding_shape = list(shape)
  padding_shape[dim] = missing
  return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=dim).reshape(
    -1
  )


class _GraphUnshard(torch.autograd.Function):
  @staticmethod
  def forward(bucket, *shards):
    send = torch.cat([
      shard.to(info.unsharded_dtype).reshape(-1)
      if bucket.replicated
      else _pad_chunk(
        shard.to(info.unsharded_dtype),
        info.global_shape,
        info.placement.dim,
        chunk,
      )
      for shard, info, chunk in zip(shards, bucket.infos, bucket.chunks)
    ])
    if bucket.replicated or bucket.world_size == 1:
      gathered = send
    else:
      # Functional wrappers wait while tracing, or on first eager use.
      gathered = fc.all_gather_single(send, 0, bucket.group)
    rows = gathered.reshape(1 if bucket.replicated else bucket.world_size, -1)
    outputs = []
    offset = 0
    for info, chunk, numel in zip(bucket.infos, bucket.chunks, bucket.numels):
      if bucket.replicated:
        full = rows[0, offset : offset + numel].reshape(info.global_shape)
      else:
        shape = list(info.global_shape)
        shape[info.placement.dim] = chunk
        full = torch.cat(
          [
            rows[r, offset : offset + numel].reshape(shape)
            for r in range(bucket.world_size)
          ],
          dim=info.placement.dim,
        )
        full = full.narrow(
          info.placement.dim, 0, info.global_shape[info.placement.dim]
        )
      outputs.append(full)
      offset += numel
    return tuple(outputs)

  @staticmethod
  def setup_context(ctx, inputs, output):
    bucket, *shards = inputs
    ctx.bucket = bucket
    ctx.trainable = tuple(shard.requires_grad for shard in shards)
    ctx.set_materialize_grads(False)
    frozen = tuple(
      out for out, trainable in zip(output, ctx.trainable) if not trainable
    )
    if frozen:
      ctx.mark_non_differentiable(*frozen)

  @staticmethod
  def backward(ctx, *grads):
    bucket = ctx.bucket
    if not torch.compiler.is_compiling() and any(
      grad is not None and grad.layout != torch.strided for grad in grads
    ):
      raise NotImplementedError(
        "Graph executor requires dense strided gradients"
      )
    reference = next((g for g in grads if g is not None), None)
    if reference is None:
      return (None,) * (1 + len(grads))
    dtype = bucket.infos[0].grad_reduce_dtype
    full_grads = [
      reference.new_zeros(info.global_shape, dtype=dtype)
      if grad is None
      else grad.to(dtype)
      for grad, info in zip(grads, bucket.infos)
    ]
    if bucket.replicated:
      send = torch.cat([g.reshape(-1) for g in full_grads])
      reduced = (
        send
        if bucket.world_size == 1
        else fc.all_reduce(send, "sum", bucket.group)
      )
    else:
      rows = []
      for rank in range(bucket.world_size):
        rows.append(
          torch.cat([
            _pad_chunk(
              info.placement.extract_local_shard(g, rank, bucket.world_size),
              info.global_shape,
              info.placement.dim,
              chunk,
            )
            for g, info, chunk in zip(full_grads, bucket.infos, bucket.chunks)
          ])
        )
      send = torch.cat(rows)
      reduced = (
        send
        if bucket.world_size == 1
        else fc.reduce_scatter_single(send, "sum", 0, bucket.group)
      )
    if bucket.storage.gradient_reduce_op == dist.ReduceOp.AVG:
      factor = bucket.storage.gradient_divide_factor or bucket.world_size
      reduced = reduced / factor
    results = []
    offset = 0
    for info, chunk, numel, trainable in zip(
      bucket.infos, bucket.chunks, bucket.numels, ctx.trainable
    ):
      shape = list(info.global_shape)
      if not bucket.replicated:
        shape[info.placement.dim] = chunk
      grad = reduced[offset : offset + numel].reshape(shape)
      if not bucket.replicated:
        grad = grad.narrow(
          info.placement.dim, 0, info.local_shape[info.placement.dim]
        )
      results.append(
        grad if trainable and grads[len(results)] is not None else None
      )
      offset += numel
    return (None, *results)


class _GraphBucket:
  def __init__(self, storage):
    self.storage = storage
    self.infos = tuple(storage.param_infos.values())
    self.world_size = storage._mesh.size()
    self.group = storage._mesh.get_group()
    self.replicated = self.infos[0].placement.dim is None
    self.chunks = tuple(
      0
      if self.replicated
      else math.ceil(info.global_shape[info.placement.dim] / self.world_size)
      for info in self.infos
    )
    self.numels = tuple(
      info.global_numel
      if self.replicated
      else chunk
      * math.prod(
        size
        for dim, size in enumerate(info.global_shape)
        if dim != info.placement.dim
      )
      for info, chunk in zip(self.infos, self.chunks)
    )
    self.owners = tuple(
      tuple(
        ParamOwnerRef.resolve(storage._module, name)
        for name in (info.fqn, *info.shared_fqns)
      )
      for info in self.infos
    )

  def materialize(self):
    shards = tuple(
      owners[0].module._parameters[owners[0].param_name]
      for owners in self.owners
    )
    return _GraphUnshard.apply(self, *shards)


_WRAP_CLASSES = {}


def _read_graph_param(module, name):
  return module._graph_param_getters[name].materialize()[0]


def _install_param_properties(runtimes):
  modules = {}
  for runtime in runtimes:
    for owner in runtime.owners[0]:
      modules.setdefault(owner.module, {})[owner.param_name] = runtime
  for module, getters in modules.items():
    names = tuple(getters)
    key = (type(module), names)
    cls = _WRAP_CLASSES.get(key)
    if cls is None:
      cls_name = f"GraphFlexShard{type(module).__name__}_{len(_WRAP_CLASSES)}"
      cls = type(
        cls_name,
        (type(module),),
        {
          "__module__": __name__,
          **{
            name: property(partial(_read_graph_param, name=name))
            for name in names
          },
        },
      )
      globals()[cls_name] = cls
      _WRAP_CLASSES[key] = cls
    module._graph_param_getters = getters
    module.__class__ = cls


def graph_flex_shard(model: nn.Module, buckets: list[BucketSpec]) -> nn.Module:
  """Use real FlexShard storage/bucketing with a portable graph executor."""
  _check_not_already_flex_sharded(model)
  reserved = (
    "_graph_param_getters",
    "_graph_flex_shard_buckets",
    "_sharded_bucket_storages",
  )
  for module in model.modules():
    if any(hasattr(module, name) for name in reserved):
      raise ValueError("Module uses a reserved graph-executor attribute")
    if isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and (
      module.sparse or module.max_norm is not None
    ):
      raise NotImplementedError(
        "Sparse or max_norm embeddings are unsupported; forward must read weights only"
      )
  if not buckets or any(bucket.mesh.ndim != 1 for bucket in buckets):
    raise ValueError("Graph executor requires nonempty buckets with 1D meshes")
  named = _get_managed_named_params(model)
  if not named or any(
    isinstance(p, DTensor) or p.device.type == "meta" for _, p in named
  ):
    raise ValueError("Expected initialized ordinary local parameters")
  if len({p.device for _, p in named}) != 1:
    raise ValueError("All parameters must be on one device")
  if any(spec.mesh.device_type != named[0][1].device.type for spec in buckets):
    raise ValueError("Parameter device and every bucket mesh must match")
  if any(
    get_global_layout(p) is not None or get_partial_grad_group(p) is not None
    for _, p in named
  ):
    raise NotImplementedError(
      "Outer layouts and partial-gradient groups are unsupported"
    )
  shared = _get_shared_param_names(model)
  assigned = _assign_params_to_buckets(
    [name for name, _ in named], buckets, shared
  )
  original = dict(named)
  placements = {}
  for spec, names in zip(buckets, assigned):
    selected = [(name, original[name]) for name in names]
    if not selected:
      continue
    if (
      spec.offload_policy
      or spec.pre_backward_hook
      or spec.post_reduce_hook
      or spec.defer_post_backward
      or spec.fsdp2_compatible
    ):
      raise NotImplementedError("Eager-only bucket settings are not supported")
    if spec.gradient_reduce_op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
      raise ValueError("Only SUM and AVG gradient reduction are supported")
    if len(selected) != 1:
      raise NotImplementedError(
        "Checkpoint-safe prototype requires one parameter per bucket"
      )
    resolved = spec.placement_fn(selected, spec.mesh)
    _validate_placements(resolved, selected)
    for name, parameter in selected:
      (placement,) = resolved[name]
      if type(placement) is not GraphShard:
        raise TypeError(
          "Graph executor currently supports GraphShard placements only"
        )
      if placement.dim is not None:
        if (
          not parameter.ndim
          or not -parameter.ndim <= placement.dim < parameter.ndim
        ):
          raise ValueError(f"Invalid GraphShard dimension for {name}")
        resolved[name] = (GraphShard(placement.dim % parameter.ndim),)
    placements.update(resolved)
  _validate_param_dtype_overrides(named, assigned, buckets)
  _validate_bucket_uniform_dtype_and_placement(
    assigned, placements, buckets, named
  )
  runtimes = []
  for spec, names in zip(buckets, assigned):
    if not names:
      continue
    selected = [(name, original[name]) for name in names]
    infos, total_bytes = ShardedBucketStorage.create_param_infos(
      selected,
      spec.mesh,
      {name: placements[name] for name in names},
      spec.mp_policy,
    )
    for name, info in infos.items():
      info.shared_fqns = tuple(shared.get(name, []))
    storage = ShardedBucketStorage(
      torch.empty(total_bytes, dtype=torch.uint8, device=named[0][1].device),
      infos,
      spec.mesh,
      total_bytes,
      model,
      gradient_reduce_op=spec.gradient_reduce_op,
      gradient_divide_factor=spec.gradient_divide_factor,
    )
    storage.copy_params_from(selected)
    runtimes.append(_GraphBucket(storage))
  # Every bucket has validated and allocated before any original param changes.
  for runtime in runtimes:
    runtime.storage.install_sharded_params(named[0][1].device)
  _install_param_properties(runtimes)
  model._graph_flex_shard_buckets = runtimes
  model._sharded_bucket_storages = [runtime.storage for runtime in runtimes]
  return model


def data_parallel(
  model: nn.Module,
  device_mesh: DeviceMesh,
  mode: str = "replicate",
  mp_policy: MixedPrecisionPolicy | None = None,
  shard_dim: int | dict[str, int] = 0,
) -> nn.Module:
  """SimpleFSDP-shaped entry point; FSDP/DDP only, with SUM semantics."""
  if mode not in ("fully_shard", "replicate"):
    raise NotImplementedError(
      "HSDP and outer TP/EP layouts are not implemented"
    )
  specs = []
  aliases = _get_shared_param_names(model)
  for name, param in model.named_parameters():
    dim = shard_dim[name] if isinstance(shard_dim, dict) else shard_dim
    if param.ndim and not -param.ndim <= dim < param.ndim:
      raise ValueError(f"Invalid shard dimension for {name}")
    dim = dim % param.ndim if param.ndim else 0
    placement = GraphShard(
      None
      if mode == "replicate"
      or param.ndim == 0
      or param.shape[dim] < device_mesh.size()
      else dim
    )
    specs.append(
      BucketSpec(
        patterns=[name, *aliases.get(name, [])],
        mesh=device_mesh,
        mp_policy=mp_policy,
        placement_fn=partial(_fixed_placement, placement=placement),
        gradient_reduce_op=dist.ReduceOp.SUM,
      )
    )
  return graph_flex_shard(model, specs)


def _fixed_placement(named, mesh, placement):
  return {name: (placement,) for name, _ in named}
