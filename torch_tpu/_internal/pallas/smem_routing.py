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

"""Module for automatic SMEM routing of scalar tensors in JAX Pallas TPU calls."""

from collections.abc import Callable
import contextlib
import inspect
from typing import Any, cast
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_DEFAULT_BACKEND: str | None = None


def _get_default_backend() -> str:
  global _DEFAULT_BACKEND
  if _DEFAULT_BACKEND is None:
    try:
      _DEFAULT_BACKEND = jax.default_backend()
    except RuntimeError:
      _DEFAULT_BACKEND = "cpu"
  return _DEFAULT_BACKEND


@contextlib.contextmanager
def mock_tpu_compilation_environment(chip_version: str = "TPU v5e") -> Any:
  """Context manager to pretend we are compiling for a TPU device.

  This allows JAX Pallas to resolve hardware specs (like VMEM size)
  during offline compilation on CPU-only build machines (e.g. during wheel
  generation or CPU testing). This has zero effect on end-user workloads
  running on actual TPUs.

  Any valid TPU version works here; we just need to satisfy JAX's compiler
  checks with a representative TPU profile while running in CPU mode.

  Args:
    chip_version: The TPU chip version string to mock (defaults to "TPU v5e").
  """
  backend = _get_default_backend()

  if backend == "tpu":
    yield
    return

  try:
    # Create an abstract TPU device spec (e.g. TPU v5e, 1 core)
    sig = inspect.signature(jax.sharding.AbstractDevice)
    device_kwargs = {"device_kind": chip_version, "num_cores": 1}
    if "platform" in sig.parameters:
      device_kwargs["platform"] = "tpu"
    device = jax.sharding.AbstractDevice(**device_kwargs)
    # Create an empty abstract mesh tied to this device
    mock_mesh = jax.sharding.AbstractMesh(
        axis_sizes=(), axis_names=(), abstract_device=device
    )
    # Enter the JAX abstract mesh context
    with jax.sharding.use_abstract_mesh(mock_mesh):
      yield
  except (AttributeError, ValueError):
    # Fallback to normal environment if JAX sharding APIs are unavailable
    yield


# Get no_block_spec from signature to avoid strict deps issues with JAX _src
try:
  _sig = inspect.signature(pl.pallas_call)
  no_block_spec = _sig.parameters["in_specs"].default
except (ValueError, KeyError):
  no_block_spec = object()


def replace_grid_spec(grid_spec: Any, **changes: Any) -> Any:
  """Helper to copy GridSpec with modified fields.

  Bypasses __init__ limitations.

  Args:
    grid_spec: The GridSpec object to copy.
    **changes: The fields to modify and their new values.

  Returns:
    The new GridSpec object.
  """
  if hasattr(grid_spec, "replace"):
    return grid_spec.replace(**changes)

  cls = type(grid_spec)

  if grid_spec.grid_names is not None:
    grid = tuple(zip(grid_spec.grid_names, grid_spec.grid))
  else:
    grid = grid_spec.grid

  kwargs = {
      "grid": grid,
      "in_specs": grid_spec.in_specs,
      "out_specs": grid_spec.out_specs,
      "scratch_shapes": grid_spec.scratch_shapes,
  }

  if hasattr(grid_spec, "num_scalar_prefetch"):
    kwargs["num_scalar_prefetch"] = grid_spec.num_scalar_prefetch

  kwargs.update(changes)

  sig = inspect.signature(cls.__init__)
  supported_args = sig.parameters.keys()
  filtered_kwargs = {
      k: v for k, v in kwargs.items() if k in supported_args and k != "self"
  }

  return cls(**filtered_kwargs)


def _route_0d_to_smem(
    flat_items: list[Any],
    flat_specs: list[Any],
    is_output: bool = False,
) -> list[Any]:
  """Ensures 0D items have SMEM memory space and scalar index map in specs."""
  new_flat_specs = []
  for item, spec in zip(flat_items, flat_specs):
    if is_output:
      is_0d = hasattr(item, "shape") and not item.shape
    else:
      is_0d = hasattr(item, "ndim") and item.ndim == 0

    if is_0d:
      index_map = (
          spec.index_map
          if (isinstance(spec, pl.BlockSpec) and spec.index_map is not None)
          else lambda *args: ()
      )
      if isinstance(spec, pl.BlockSpec):
        new_spec = spec.replace(memory_space=pltpu.SMEM, index_map=index_map)
      else:
        new_spec = pl.BlockSpec(memory_space=pltpu.SMEM, index_map=index_map)
      new_flat_specs.append(new_spec)
    else:
      new_flat_specs.append(spec)
  return new_flat_specs


def pallas_call(
    kernel: Callable[..., Any], out_shape: Any, **kwargs: Any
) -> Callable[..., Any]:
  """Wrapper around pl.pallas_call with automatic scalar routing to SMEM."""
  grid_spec = kwargs.get("grid_spec", None)
  in_specs = kwargs.get("in_specs", no_block_spec)
  out_specs = kwargs.get("out_specs", no_block_spec)

  if grid_spec is not None:
    in_specs = grid_spec.in_specs
    out_specs = grid_spec.out_specs

  def call_wrapper(*args: Any, **wrapped_kwargs: Any) -> Any:
    flat_args, _ = jax.tree_util.tree_flatten(args)
    flat_out_shapes, out_tree = jax.tree_util.tree_flatten(out_shape)

    any_0d_input = any(hasattr(x, "ndim") and x.ndim == 0 for x in flat_args)
    any_0d_output = any(
        hasattr(x, "shape") and not x.shape for x in flat_out_shapes
    )

    if not any_0d_input and not any_0d_output:
      with mock_tpu_compilation_environment("TPU v5e"):
        return pl.pallas_call(kernel, out_shape, **kwargs)(
            *args, **wrapped_kwargs
        )

    num_scalar_prefetch = 0
    if grid_spec is not None and hasattr(grid_spec, "num_scalar_prefetch"):
      num_scalar_prefetch = grid_spec.num_scalar_prefetch

    pref_args = args[:num_scalar_prefetch]
    flat_pref_args, _ = jax.tree_util.tree_flatten(pref_args)
    num_actual_prefetch_leaves = len(flat_pref_args)

    flat_in_specs: list[Any]
    spec_tree = None
    prefetch_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    if in_specs is no_block_spec:
      flat_in_specs = [prefetch_spec] * num_actual_prefetch_leaves + [
          pl.BlockSpec()
      ] * (len(flat_args) - num_actual_prefetch_leaves)
    else:
      flat_in_specs_provided, spec_tree = jax.tree_util.tree_flatten(in_specs)
      flat_in_specs = [
          prefetch_spec
      ] * num_actual_prefetch_leaves + flat_in_specs_provided

    flat_in_specs = cast(list[Any], flat_in_specs)
    new_flat_in_specs = _route_0d_to_smem(
        flat_args, flat_in_specs, is_output=False
    )

    out_spec_tree = None
    if out_specs is no_block_spec:
      flat_out_specs = [pl.BlockSpec()] * len(flat_out_shapes)
    else:
      flat_out_specs, out_spec_tree = jax.tree_util.tree_flatten(out_specs)

    new_flat_out_specs = _route_0d_to_smem(
        flat_out_shapes, flat_out_specs, is_output=True
    )

    non_pref_args = args[num_scalar_prefetch:]
    _, non_pref_in_tree = jax.tree_util.tree_flatten(non_pref_args)

    if in_specs is no_block_spec:
      new_in_specs = jax.tree_util.tree_unflatten(
          non_pref_in_tree, new_flat_in_specs[num_actual_prefetch_leaves:]
      )
    else:
      new_in_specs = jax.tree_util.tree_unflatten(
          spec_tree, new_flat_in_specs[num_actual_prefetch_leaves:]
      )

    if out_specs is no_block_spec:
      new_out_specs = jax.tree_util.tree_unflatten(out_tree, new_flat_out_specs)
    else:
      new_out_specs = jax.tree_util.tree_unflatten(
          out_spec_tree, new_flat_out_specs
      )

    new_kwargs = dict(kwargs)
    if grid_spec is not None:
      new_grid_spec = replace_grid_spec(
          grid_spec, in_specs=new_in_specs, out_specs=new_out_specs
      )
      new_kwargs["grid_spec"] = new_grid_spec
    else:
      new_kwargs["in_specs"] = new_in_specs
      new_kwargs["out_specs"] = new_out_specs

    with mock_tpu_compilation_environment("TPU v5e"):
      return pl.pallas_call(kernel, out_shape, **new_kwargs)(
          *args, **wrapped_kwargs
      )

  return call_wrapper
