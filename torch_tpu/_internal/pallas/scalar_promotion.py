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

"""Module for scalar-to-tensor promotion in JAX Pallas TPU calls."""

import functools
import inspect
import jax
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

# Save original
orig_pallas_call = pl.pallas_call

# Get no_block_spec from signature to avoid strict deps issues with JAX _src
try:
  _sig = inspect.signature(orig_pallas_call)
  no_block_spec = _sig.parameters["in_specs"].default
except (ValueError, KeyError):
  # Fallback
  no_block_spec = object()


class PromotedRef:
  """Wrapper for Ref to support scalar promotion."""

  def __init__(self, val):
    self.val = val

  def __getitem__(self, idx):
    # y_ref[...] -> idx is Ellipsis. self.val[...] returns 1D, we squeeze to 0D.
    # y_ref[()] -> idx is (). self.val[0] returns 0D.
    if isinstance(idx, tuple) and not idx:
      return self.val[0]
    res = self.val[idx]
    # We expect res to be 1D if idx was Ellipsis or slice.
    # If it is already 0D (e.g. if they indexed with int), we don't squeeze.
    if hasattr(res, "ndim") and res.ndim == 1:
      return res.squeeze(0)
    return res

  def __setitem__(self, idx, val):
    # Support write if needed, though we assume read-only for now.
    # val is 0D, we expand to 1D to write to 1D Ref.
    if isinstance(idx, tuple) and not idx:
      self.val[0] = val
    else:
      if not hasattr(val, "ndim"):
        val = jnp.asarray(val)
      self.val[idx] = val[None]


def wrap_index_map(orig_index_map):
  """Wraps index_map to append 0 for promoted 1D dimension."""

  @functools.wraps(orig_index_map)
  def new_index_map(*args):
    res = orig_index_map(*args)
    if not isinstance(res, tuple):
      res = (res,)
    return res + (0,)

  return new_index_map


def patched_pallas_call(kernel, out_shape, **kwargs):
  """Patched pallas_call to support scalar promotion."""
  in_specs = kwargs.get("in_specs", no_block_spec)
  out_specs = kwargs.get("out_specs", no_block_spec)

  def call_wrapper(*args, **wrapped_kwargs):
    # Flatten inputs
    flat_args, in_tree = jax.tree_util.tree_flatten(args)

    spec_tree = None
    if in_specs is no_block_spec:
      flat_in_specs = [None] * len(flat_args)
    else:
      try:
        flat_in_specs, spec_tree = jax.tree_util.tree_flatten(in_specs)
      except TypeError:
        flat_in_specs = [in_specs] * len(flat_args)
        spec_tree = None

    promoted_indices = []
    new_flat_args = []
    new_flat_in_specs = []

    for i, (arg, spec) in enumerate(zip(flat_args, flat_in_specs)):
      if hasattr(arg, "ndim") and arg.ndim == 0:
        mem_space = getattr(spec, "memory_space", None)
        is_smem_or_sem = False
        if mem_space is not None:
          mem_space_str = str(mem_space).lower()
          if "smem" in mem_space_str or "semaphore" in mem_space_str:
            is_smem_or_sem = True

        if not is_smem_or_sem:
          promoted_indices.append(i)
          new_flat_args.append(arg[None])
          if isinstance(spec, pl.BlockSpec):
            new_block_shape = (1,)
            new_index_map = (
                wrap_index_map(spec.index_map)
                if spec.index_map is not None
                else None
            )
            new_spec = spec.replace(
                block_shape=new_block_shape, index_map=new_index_map
            )
            new_flat_in_specs.append(new_spec)
          else:
            new_flat_in_specs.append(None)
          continue

      new_flat_args.append(arg)
      new_flat_in_specs.append(spec)

    # Flatten outputs
    flat_out_shapes, out_tree = jax.tree_util.tree_flatten(out_shape)

    out_spec_tree = None
    if out_specs is no_block_spec:
      flat_out_specs = [None] * len(flat_out_shapes)
    else:
      try:
        flat_out_specs, out_spec_tree = jax.tree_util.tree_flatten(out_specs)
      except TypeError:
        flat_out_specs = [out_specs] * len(flat_out_shapes)
        out_spec_tree = None

    promoted_out_indices = []
    new_flat_out_shapes = []
    new_flat_out_specs = []

    for i, (shape_struct, spec) in enumerate(
        zip(flat_out_shapes, flat_out_specs)
    ):
      if hasattr(shape_struct, "shape") and not shape_struct.shape:
        mem_space = getattr(spec, "memory_space", None)
        is_smem_or_sem = False
        if mem_space is not None:
          mem_space_str = str(mem_space).lower()
          if "smem" in mem_space_str or "semaphore" in mem_space_str:
            is_smem_or_sem = True

        if not is_smem_or_sem:
          promoted_out_indices.append(i)
          new_shape_struct = jax.ShapeDtypeStruct((1,), shape_struct.dtype)
          new_flat_out_shapes.append(new_shape_struct)
          if isinstance(spec, pl.BlockSpec):
            new_block_shape = (1,)
            new_index_map = (
                wrap_index_map(spec.index_map)
                if spec.index_map is not None
                else None
            )
            new_spec = spec.replace(
                block_shape=new_block_shape, index_map=new_index_map
            )
            new_flat_out_specs.append(new_spec)
          else:
            new_flat_out_specs.append(None)
          continue

      new_flat_out_shapes.append(shape_struct)
      new_flat_out_specs.append(spec)

    if not promoted_indices and not promoted_out_indices:
      return orig_pallas_call(kernel, out_shape, **kwargs)(
          *args, **wrapped_kwargs
      )

    # Reconstruct inputs
    new_args = jax.tree_util.tree_unflatten(in_tree, new_flat_args)
    if in_specs is no_block_spec:
      new_in_specs = no_block_spec
    elif spec_tree is not None:
      new_in_specs = jax.tree_util.tree_unflatten(spec_tree, new_flat_in_specs)
    else:
      new_in_specs = new_flat_in_specs[0]

    # Reconstruct outputs
    new_out_shape = jax.tree_util.tree_unflatten(out_tree, new_flat_out_shapes)
    if out_specs is no_block_spec:
      new_out_specs = no_block_spec
    elif out_spec_tree is not None:
      new_out_specs = jax.tree_util.tree_unflatten(
          out_spec_tree, new_flat_out_specs
      )
    else:
      new_out_specs = new_flat_out_specs[0]

    num_inputs = len(flat_args)

    # Wrap kernel
    def wrapped_kernel(*kernel_args):
      new_kernel_args = list(kernel_args)
      for idx in promoted_indices:
        new_kernel_args[idx] = PromotedRef(kernel_args[idx])
      for idx in promoted_out_indices:
        kernel_arg_idx = num_inputs + idx
        new_kernel_args[kernel_arg_idx] = PromotedRef(
            kernel_args[kernel_arg_idx]
        )
      return kernel(*new_kernel_args)

    new_kwargs = dict(kwargs)
    new_kwargs["in_specs"] = new_in_specs
    new_kwargs["out_specs"] = new_out_specs

    new_pallas_fn = orig_pallas_call(
        wrapped_kernel, new_out_shape, **new_kwargs
    )

    # We need to wrap the returned function to squeeze the outputs back
    @functools.wraps(new_pallas_fn)
    def pallas_call_wrapper(*args, **wrapped_kwargs):
      res = new_pallas_fn(*args, **wrapped_kwargs)
      flat_res, res_tree = jax.tree_util.tree_flatten(res)
      for idx in promoted_out_indices:
        flat_res[idx] = flat_res[idx].squeeze(0)
      return jax.tree_util.tree_unflatten(res_tree, flat_res)

    return pallas_call_wrapper(*new_args, **wrapped_kwargs)

  return call_wrapper


def patch():
  pl.pallas_call = patched_pallas_call
  # Also patch in jax.experimental.pallas.tpu if it was imported separately
  # Actually they point to the same function.
  if hasattr(pltpu, "pallas_call"):
    pltpu.pallas_call = patched_pallas_call


def unpatch():
  pl.pallas_call = orig_pallas_call
  if hasattr(pltpu, "pallas_call"):
    pltpu.pallas_call = orig_pallas_call
