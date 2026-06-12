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

"""Lazy bucketed compilation artifact for framework-level bucketing."""

from collections.abc import Callable, Sequence
import os
from typing import Any
from absl import logging
import torch
from torch._guards import TracingContext
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile.dynamic import sym_shape_manager
from torch_tpu._internal.compile.dynamic.graph_transformations import (
    LowerCatToIndexCopyPass,
)
from torch_tpu._internal.compile.dynamic.state_staged_tensor import (
    StateStagedTensor,
)


def _get_pad_value(tensor: torch.Tensor, node_name: str) -> Any:
  """Determines the correct padding value based on tensor dtype and name."""
  if "mask" in node_name or "bias" in node_name:
    if tensor.dtype.is_floating_point:
      # Use -10000.0 as a safe float mask value (compatible with
      # bfloat16/float16).
      return -10000.0
    elif tensor.dtype == torch.bool:
      return False
    else:
      return 0
  return 0  # Default padding for data/activations is 0


def _find_dynamic_dim_idx(tensor) -> int | None:
  if hasattr(tensor, "shape"):
    for idx, s in enumerate(tensor.shape):
      if isinstance(s, torch.SymInt):
        return idx
  return None


class LazyBucketedCompiledExecutable(compiler.CompiledArtifact):
  """A wrapper that lazy-compiles static PJRT executables for buckets."""

  def __init__(
      self,
      graph_module: torch.fx.GraphModule,
      is_fwd: bool,
      static_compiler: compiler.Compiler,
      buckets: list[int] | None,
      example_inputs: Sequence[Any],
  ):
    self._graph_module = graph_module
    self._is_fwd = is_fwd

    tracing_context = TracingContext.try_get()
    self._mutated_inp_indices = set()
    self._num_mutated_inps = 0
    if tracing_context is not None and hasattr(tracing_context, "fw_metadata"):
      fw_metadata = tracing_context.fw_metadata
      if fw_metadata is not None:
        self._mutated_inp_indices = set(
            getattr(fw_metadata, "mutated_inp_runtime_indices", [])
        )
        self._num_mutated_inps = len(self._mutated_inp_indices)
        logging.info(
            "[BucketedCompiler] fw_metadata.mutated_inp_runtime_indices: %s",
            list(self._mutated_inp_indices),
        )
    self._static_compiler = static_compiler
    # Sort manual buckets if provided, otherwise keep as None to trigger
    # auto-bucketing
    self._buckets = sorted(buckets) if buckets is not None else None
    self._executables: dict[int, compiler.CompiledArtifact] = {}
    self._backend_device = None
    self._input_plan = None
    self._output_plan = None
    self._disable_optimizations = (
        os.environ.get("TORCH_TPU_DISABLE_BUCKET_OPTS", "0") == "1"
    )

    # Track placeholder nodes to identify input names (for mask detection)
    self._placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]

    # Create SymShapeManager to analyze dynamic dimensions
    self._sym_shape_manager = sym_shape_manager.SymShapeManager(
        graph_module, example_inputs
    )

    # Check if we have cat nodes that we can lower
    has_cat = False
    for node in graph_module.graph.nodes:
      if (
          node.op == "call_function"
          and node.target == torch.ops.aten.cat.default
      ):
        tensors = node.args[0]
        if isinstance(tensors, (list, tuple)) and len(tensors) == 2:
          has_cat = True
          break

    self._cache_output_indices = []
    self._cache_input_indices = []
    self._x_input_indices = []
    self._cache_len_arg_idx = None
    self._logical_cache_len = None
    self._cache_len_scalar_indices = set()
    self._has_index_copy = False

    if has_cat:
      logging.info("[BucketedCompiler] Found cat nodes, lowering to index_copy")
      last_ph = None
      for node in graph_module.graph.nodes:
        if node.op == "placeholder":
          last_ph = node

      with graph_module.graph.inserting_after(last_ph):
        self._index_ph = graph_module.graph.placeholder("kv_cache_index")

      if self._placeholders:
        ref_ph = self._placeholders[0]
        if "val" in ref_ph.meta and hasattr(ref_ph.meta["val"], "fake_mode"):
          fake_mode = ref_ph.meta["val"].fake_mode
          device = ref_ph.meta["val"].device
          real_tensor = torch.tensor([0], dtype=torch.int64, device=device)
          index_fake = fake_mode.from_tensor(real_tensor)
          self._index_ph.meta["val"] = index_fake

      self._static_input_indices = set()
      if tracing_context is not None and hasattr(
          tracing_context, "fw_metadata"
      ):
        fw_metadata = tracing_context.fw_metadata
        if fw_metadata is not None:
          self._static_input_indices = set(
              getattr(fw_metadata, "static_input_indices", [])
          )
      self._parameter_phs = {
          self._placeholders[idx]
          for idx in self._static_input_indices
          if idx < len(self._placeholders)
      }

      self._index_copy_pass = LowerCatToIndexCopyPass(
          self._index_ph, self._parameter_phs
      )
      self._index_copy_pass(graph_module)
      graph_module.recompile()
      self._has_index_copy = len(self._index_copy_pass.replaced_nodes) > 0

      if not self._has_index_copy:
        graph_module.graph.erase_node(self._index_ph)
        graph_module.recompile()

    if self._has_index_copy:
      # Resolve input indices
      self._cache_input_indices = sorted([
          self._placeholders.index(ph)
          for ph in self._index_copy_pass.cache_phs
          if ph in self._placeholders
      ])
      self._cache_to_len_scalar_idx = {}
      for cache_ph, len_ph in self._index_copy_pass.cache_to_len_ph.items():
        if len_ph in self._placeholders and cache_ph in self._placeholders:
          cache_idx = self._placeholders.index(cache_ph)
          len_idx = self._placeholders.index(len_ph)
          self._cache_to_len_scalar_idx[cache_idx] = len_idx
      self._cache_len_scalar_indices = set(
          self._cache_to_len_scalar_idx.values()
      )

      # Group placeholders that share the same SymInt symbol
      shared_symbols = set()
      for idx in list(self._cache_len_scalar_indices):
        val = example_inputs[idx]
        if isinstance(val, torch.SymInt):
          shared_symbols.add(str(val))

      if shared_symbols:
        for idx in self._sym_shape_manager.dynamic_scalar_indices:
          val = example_inputs[idx]
          if isinstance(val, torch.SymInt) and str(val) in shared_symbols:
            if idx not in self._cache_len_scalar_indices:
              self._cache_len_scalar_indices.add(idx)

      self._x_input_indices = sorted([
          self._placeholders.index(ph)
          for ph in self._index_copy_pass.x_phs
          if ph in self._placeholders
      ])

      # Resolve dynamic dimension indices for cache and x inputs
      self._cache_dim_indices = {}
      for idx in self._cache_input_indices:
        if hasattr(example_inputs[idx], "ndim"):
          dim_idx = _find_dynamic_dim_idx(example_inputs[idx])
          if dim_idx is None:
            # Fallback based on ndim. StateStagedTensor base is 5D [1, B, H,
            # bucket_len, D]
            if example_inputs[idx].ndim == 5:
              dim_idx = 3
            elif example_inputs[idx].ndim == 4:
              dim_idx = 2
            else:
              dim_idx = 1
          self._cache_dim_indices[idx] = dim_idx
        else:
          self._cache_dim_indices[idx] = None

      self._x_dim_indices = {}
      for idx in self._x_input_indices:
        if hasattr(example_inputs[idx], "ndim"):
          dim_idx = _find_dynamic_dim_idx(example_inputs[idx])
          if dim_idx is None:
            dim_idx = 1 if example_inputs[idx].ndim in (2, 3) else 2
          self._x_dim_indices[idx] = dim_idx
        else:
          self._x_dim_indices[idx] = None

      for idx, ph in enumerate(self._placeholders):
        val = ph.meta.get("val", None)
        shape = list(val.shape) if hasattr(val, "shape") else None
        logging.info(
            "[BucketedCompiler] Placeholder %d: name=%s, shape=%s",
            idx,
            ph.name,
            shape,
        )
      logging.info("[BucketedCompiler] x_phs: %s", self._index_copy_pass.x_phs)

      logging.info(
          "[BucketedCompiler] Resolved cache input indices: %s, x input"
          " indices: %s",
          self._cache_input_indices,
          self._x_input_indices,
      )
      logging.info(
          "[BucketedCompiler] Cache dim indices: %s, x dim indices: %s",
          self._cache_dim_indices,
          self._x_dim_indices,
      )

      # Find the SymInt representing cache seq len
      self._cache_len_symints = set()
      for idx in self._cache_input_indices:
        dim = self._cache_dim_indices[idx]
        if dim is not None:
          shape = example_inputs[idx].shape
          if dim < len(shape):
            s = shape[dim]
            if isinstance(s, (torch.SymInt, int)):
              self._cache_len_symints.add(str(s))

      for idx in self._sym_shape_manager.dynamic_scalar_indices:
        val = example_inputs[idx]
        if isinstance(val, torch.SymInt):
          if str(val) in self._cache_len_symints:
            self._cache_len_scalar_indices.add(idx)
      logging.info(
          "[BucketedCompiler] Cache len scalar indices: %s",
          self._cache_len_scalar_indices,
      )

      # Identify which output indices correspond to the replaced nodes
      output_node = [n for n in graph_module.graph.nodes if n.op == "output"][0]
      logging.info("[BucketedCompiler] Graph outputs: %s", output_node.args[0])
      for i, arg in enumerate(output_node.args[0]):
        if isinstance(arg, torch.fx.Node):
          logging.info(
              "[BucketedCompiler]   Output %d: %s, val=%s",
              i,
              arg,
              arg.meta.get("val", None),
          )
      for i, arg in enumerate(output_node.args[0]):
        if arg in self._index_copy_pass.replaced_nodes:
          cache_input = arg.args[0]
          if cache_input in self._index_copy_pass.subclass_cache_phs:
            self._cache_output_indices.append(i)
      logging.info(
          "[BucketedCompiler] Cache output indices: %s",
          self._cache_output_indices,
      )

    # Locate the first dynamic dimension index to read seq_len at runtime
    self._dynamic_tensor_idx = None
    self._dynamic_dim_idx = None
    self._dynamic_scalar_idx = None

    for idx, meta in self._sym_shape_manager.input_tensors_metadata.items():
      if idx in self._mutated_inp_indices:
        continue  # Skip mutated inputs (caches)
      if idx in self._cache_input_indices:
        continue  # Skip identified cache inputs
      if meta.dynamic_dims:
        self._dynamic_tensor_idx = idx
        self._dynamic_dim_idx = meta.dynamic_dims[0]
        break

    if (
        self._dynamic_tensor_idx is None
        and self._sym_shape_manager.dynamic_scalar_indices
    ):
      self._dynamic_scalar_idx = list(
          self._sym_shape_manager.dynamic_scalar_indices
      )[0]

  def _get_backend_device(self) -> torch.device:
    if self._backend_device is None:
      self._backend_device = torch.accelerator.current_accelerator()
    return self._backend_device

  def _get_auto_bucket(self, seq_len: int) -> int:
    """Computes the next power of 2 for sequence length to limit recompiles."""
    if seq_len <= 0:
      return 5
    # Find next power of 2 (e.g., 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    bucket_len = 1 << (seq_len - 1).bit_length()
    return max(bucket_len, 4) + 1

  def _resolve_bucket(self, seq_len: int) -> int:
    """Finds the smallest bucket size >= seq_len (manual or auto)."""
    if self._buckets is not None:
      # Manual bucketing: find the smallest defined bucket that fits the shape
      for b in self._buckets:
        if b >= seq_len:
          return b + 1
      # Fallback to seq_len itself if it exceeds the largest manual bucket
      logging.warning(
          "[BucketedCompiler] Sequence length %d exceeds the largest bucket %d."
          " Falling back to static compile.",
          seq_len,
          self._buckets[-1],
      )
      return seq_len + 1
    else:
      # Automatic bucketing: pad to the next power of 2
      return self._get_auto_bucket(seq_len)

  def _fast_slice(
      self, tensor: torch.Tensor, dim: int, size: int
  ) -> torch.Tensor:
    """Slices a tensor along specified dimension to size using fast paths."""
    if tensor.ndim == 4 and dim == 2:
      return tensor[:, :, :size, :]
    elif tensor.ndim == 4 and dim == 1:
      return tensor[:, :size, :, :]
    elif tensor.ndim == 3 and dim == 1:
      return tensor[:, :size, :]
    # Fallback
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, size)
    return tensor[tuple(slices)]

  def _pad_tensor(
      self, tensor: torch.Tensor, dim: int, target_len: int, pad_value: Any
  ) -> torch.Tensor:
    """Pads a tensor along the specified dimension to target_len."""
    curr_len = tensor.shape[dim]
    if curr_len >= target_len:
      return tensor

    # Fast path for 2D tensor padding along dim 1 (common for masks)
    if tensor.ndim == 2 and dim == 1:
      return torch.nn.functional.pad(
          tensor, (0, target_len - curr_len), value=pad_value
      )

    # Pad configuration format: (pad_last_dim_left, pad_last_dim_right,
    # pad_prev_dim_left, ...)
    # PyTorch pads starting from the last dimension backwards.
    pad_config = [0] * (2 * tensor.ndim)
    # Map dimension index to PyTorch's padding index order
    py_dim = tensor.ndim - 1 - dim
    pad_config[2 * py_dim + 1] = target_len - curr_len

    return torch.nn.functional.pad(tensor, pad_config, value=pad_value)

  def _slice_tensor(
      self, tensor: torch.Tensor, dim: int, target_len: int
  ) -> torch.Tensor:
    """Slices a tensor along the specified dimension to target_len."""
    logging.info(
        "[LazyBucketed] _slice_tensor: input shape=%s, dim=%d, target_len=%d",
        list(tensor.shape),
        dim,
        target_len,
    )
    if tensor.shape[dim] <= target_len:
      return tensor
    res = self._fast_slice(tensor, dim, target_len).contiguous()

    # Dynamo specializes on contiguous strides. If we slice a padded tensor,
    # it retains the strides of the larger padded tensor, triggering
    # recompilations. We force the layout to be contiguous to match its logical
    # shape.
    # For shape[0] == 1, contiguous() might not force stride[0] == shape[1].
    # We use as_strided to force standard strides.
    expected_stride = []
    current_stride = 1
    for size in reversed(res.shape):
      expected_stride.append(current_stride)
      current_stride *= size
    expected_stride = tuple(reversed(expected_stride))

    if res.stride() != expected_stride:
      logging.info(
          "[LazyBucketed] Forcing stride from %s to %s",
          res.stride(),
          expected_stride,
      )
      res = res.as_strided(res.shape, expected_stride)

    logging.info(
        "[LazyBucketed] _slice_tensor: output shape=%s, stride=%s",
        list(res.shape),
        res.stride(),
    )
    return res

  def __call__(self, *args: Any) -> Any:
    # Handle single list/tuple argument wrapping (standard Dynamo/AOT pattern)
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
      args = args[0]

    logging.info("[LazyBucketed] __call__ args: %s", [type(a) for a in args])

    # 1. Compute start and update_len if we have index_copy
    start = 0
    update_len = 1
    if self._has_index_copy:
      if self._x_input_indices:
        x_idx = self._x_input_indices[0]
        dim_idx = self._x_dim_indices[x_idx]
        if dim_idx is not None:
          update_len = args[x_idx].shape[dim_idx]
        else:
          val = args[x_idx]
          update_len = int(val) if isinstance(val, (int, torch.SymInt)) else val
      if self._cache_input_indices:
        cache_idx = self._cache_input_indices[0]
        if cache_idx in getattr(self, "_cache_to_len_scalar_idx", {}):
          len_idx = self._cache_to_len_scalar_idx[cache_idx]
          val = args[len_idx]
          start = int(val) if isinstance(val, (int, torch.SymInt)) else val
        else:
          dim_idx = self._cache_dim_indices[cache_idx]
          if dim_idx is not None:
            start = args[cache_idx].shape[dim_idx]
          else:
            val = args[cache_idx]
            start = int(val) if isinstance(val, (int, torch.SymInt)) else val

    # 2. Determine the runtime sequence length (S) for bucketing
    if self._has_index_copy:
      seq_len = start + update_len
    elif self._dynamic_tensor_idx is not None:
      seq_len = args[self._dynamic_tensor_idx].shape[self._dynamic_dim_idx]
    elif self._dynamic_scalar_idx is not None:
      val = args[self._dynamic_scalar_idx]
      seq_len = int(val) if isinstance(val, (int, torch.SymInt)) else val
    else:
      logging.warning(
          "[BucketedCompiler] No dynamic sequence dimension identified."
          " Executing directly."
      )
      return self._static_compiler(self._graph_module, args)(args)

    # 3. Resolve bucket size (B)
    bucket_len = self._resolve_bucket(seq_len)
    logging.info(
        "[LazyBucketed] seq_len=%s (start=%s, update_len=%s),"
        " resolved bucket_len=%s",
        seq_len,
        start,
        update_len,
        bucket_len,
    )

    # Build input plan on first run
    if self._input_plan is None or self._disable_optimizations:
      self._input_plan = []
      for idx, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
          meta = self._sym_shape_manager.input_tensors_metadata.get(idx)
          is_cache = idx in self._cache_input_indices
          if (meta and meta.dynamic_dims) or is_cache:
            node_name = self._placeholders[idx].name
            pad_value = _get_pad_value(arg, node_name)
            if is_cache:
              dim_idx = self._cache_dim_indices.get(idx)
              self._input_plan.append(("CACHE", idx, dim_idx, pad_value))
            else:
              assert meta is not None
              self._input_plan.append(
                  ("PAD", idx, meta.dynamic_dims, pad_value)
              )
          else:
            self._input_plan.append(("PASS", idx))
        elif (
            isinstance(arg, (int, torch.SymInt))
            and idx in self._sym_shape_manager.dynamic_scalar_indices
        ):
          is_cache_len = idx in self._cache_len_scalar_indices
          self._input_plan.append(("SCALAR", idx, is_cache_len))
        else:
          self._input_plan.append(("PASS", idx))

    # Execute input plan to construct padded_args and runtime_args
    padded_args = []
    runtime_args = []
    for plan in self._input_plan:
      action = plan[0]
      if action == "PASS":
        val = args[plan[1]]
        padded_args.append(val)
        runtime_args.append(val)
      elif action == "SCALAR":
        val = args[plan[1]]
        is_cache_len = plan[2]
        if is_cache_len:
          padded_args.append(bucket_len - update_len)
        else:
          padded_args.append(bucket_len)
        runtime_args.append(val)
      elif action == "CACHE":
        idx = plan[1]
        dim_idx = plan[2]
        pad_value = plan[3]
        arg = args[idx]

        base = getattr(arg, "_base_tensor", None)
        if not self._disable_optimizations and base is not None:
          # Subclass input path (5D)
          padded_tensor = base
          base_dim_idx = dim_idx + 1 if dim_idx is not None else None
          if base_dim_idx is not None:
            if padded_tensor.shape[base_dim_idx] > bucket_len:
              padded_tensor = self._fast_slice(
                  padded_tensor, base_dim_idx, bucket_len
              )
            elif padded_tensor.shape[base_dim_idx] < bucket_len:
              padded_tensor = self._pad_tensor(
                  padded_tensor, base_dim_idx, bucket_len, pad_value
              )
        else:
          # Standard tensor input path (4D)
          padded_tensor = arg
          if dim_idx is not None:
            if padded_tensor.shape[dim_idx] > bucket_len:
              padded_tensor = self._fast_slice(
                  padded_tensor, dim_idx, bucket_len
              )
            elif padded_tensor.shape[dim_idx] < bucket_len:
              padded_tensor = self._pad_tensor(
                  padded_tensor, dim_idx, bucket_len, pad_value
              )

        padded_args.append(padded_tensor)
        runtime_args.append(padded_tensor)
      elif action == "PAD":
        idx = plan[1]
        dynamic_dims = plan[2]
        pad_value = plan[3]
        arg = args[idx]

        padded_tensor = arg
        base = getattr(arg, "_base_tensor", getattr(arg, "_base", None))
        reused_base = False
        if base is not None and base.ndim == arg.ndim:
          match = True
          for d, sz in enumerate(arg.shape):
            if d in dynamic_dims:
              if base.shape[d] < bucket_len:
                match = False
                break
            else:
              if base.shape[d] != sz:
                match = False
                break
          if match:
            padded_tensor = base
            for dim_idx in dynamic_dims:
              if padded_tensor.shape[dim_idx] > bucket_len:
                padded_tensor = self._fast_slice(
                    padded_tensor, dim_idx, bucket_len
                )
            reused_base = True

        if not reused_base:
          for dim_idx in dynamic_dims:
            padded_tensor = self._pad_tensor(
                padded_tensor, dim_idx, bucket_len, pad_value
            )

        padded_args.append(padded_tensor)
        runtime_args.append(padded_tensor)

    if self._has_index_copy:
      device = self._get_backend_device()
      runtime_index = torch.arange(
          start, start + update_len, dtype=torch.int64, device=device
      )
      padded_index = runtime_index

      padded_args.append(padded_index)
      runtime_args.append(runtime_index)

    # 4. Get or compile the static executable for bucket size (B)
    if bucket_len not in self._executables:
      logging.info(
          "[BucketedCompiler] Cache MISS. Compiling static graph for bucket"
          " size: %d",
          bucket_len,
      )
      self._executables[bucket_len] = self._static_compiler(
          self._graph_module, padded_args, is_fwd=self._is_fwd
      )

    executable = self._executables[bucket_len]

    # 5. Execute on TPU
    outputs = executable(runtime_args)

    # 6. Slice dynamic outputs back to original sequence length (S)
    flat_outputs, spec = _pytree.tree_flatten(outputs)

    # Build output plan on first run
    if self._output_plan is None or self._disable_optimizations:
      self._output_plan = []
      for i, out in enumerate(flat_outputs):
        if i < len(self._sym_shape_manager.outputs_sym_shape):
          if isinstance(out, torch.Tensor):
            if (
                not self._disable_optimizations
                and i in self._cache_output_indices
            ):
              self._output_plan.append(("CACHE", i))
            else:
              sym_shape_info = self._sym_shape_manager.outputs_sym_shape[i]
              self._output_plan.append(("SLICE_TENSOR", i, sym_shape_info))
          else:
            sym_shape_info = self._sym_shape_manager.outputs_sym_shape[i]
            self._output_plan.append(("SCALAR", i, sym_shape_info))
        else:
          self._output_plan.append(("PASS", i))

    # Execute output plan
    sliced_flat_outputs = []
    for plan in self._output_plan:
      action = plan[0]
      i = plan[1]
      out = flat_outputs[i]

      if action == "PASS":
        sliced_flat_outputs.append(out)
      elif action == "CACHE":
        if self._has_index_copy:
          # AOTAutograd subclass path is active, it will wrap the output.
          # We must return the raw 5D base tensor.
          sliced_flat_outputs.append(out)
        else:
          # Fallback (e.g. eager or standard compiled path)
          logical_view = out[0, :, :, :seq_len, :]
          sliced_out = StateStagedTensor(logical_view, out, seq_len, bucket_len)
          sliced_flat_outputs.append(sliced_out)
      elif action == "SLICE_TENSOR":
        sym_shape_info = plan[2]
        runtime_shape = sym_shape_info.get_output_runtime_shape(args)
        sliced_out = out
        for dim_idx, target_len in enumerate(runtime_shape):
          if out.shape[dim_idx] > target_len:
            if not self._disable_optimizations and out.ndim == 4:
              logical_view = self._fast_slice(sliced_out, dim_idx, target_len)
              sliced_out = StateStagedTensor(
                  logical_view, out.unsqueeze(0), target_len, bucket_len
              )
            else:
              sliced_out = self._slice_tensor(sliced_out, dim_idx, target_len)
        sliced_flat_outputs.append(sliced_out)
      elif action == "SCALAR":
        sym_shape_info = plan[2]
        runtime_shape = sym_shape_info.get_output_runtime_shape(args)
        if runtime_shape:
          sliced_flat_outputs.append(runtime_shape[0])
        else:
          sliced_flat_outputs.append(out)

    return _pytree.tree_unflatten(sliced_flat_outputs, spec)

  def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    raise NotImplementedError(
        "Serialization support for LazyBucketedCompiledExecutable not yet"
        " implemented"
    )
