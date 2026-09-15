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

"""Generate blockwise scaled mm Pallas kernel MLIR module embedded C++ files."""

from collections.abc import Callable, Sequence
from typing import Any

from absl import app
from absl import flags
import jax.numpy as jnp
from torch_tpu._internal.pallas import pallas_kernel_generate_utils as kernel_utils
from torch_tpu.ops.scaled_mm.kernels import blockwise128_scaled_mm_kernel as blockwise128_kernels
from torch_tpu.ops.scaled_mm.kernels import blockwise_scaled_mm_kernel as kernels

_HEADER = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "header",
    None,
    "Header file to include in the generated kernel string.",
)

_IMPLEMENTATION = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "implementation",
    None,
    "Implementation file to include in the generated kernel string.",
)

_KERNEL_NAME = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "kernel_name",
    None,
    "Name of the kernel to generate. Used as prefix for C++ symbols.",
)

_RECIPE = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "recipe",
    "mxfp8",
    "Block scaling recipe to generate (e.g. 'mxfp8'). Add new recipes to"
    " _RECIPE_GENERATORS.",
)

_IN_DTYPE = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "in_dtype",
    "float8_e4m3fn",
    "Input dtype of the kernel (float8_e4m3fn or float8_e5m2).",
)

_OUT_DTYPE = flags.DEFINE_string(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "out_dtype",
    "float32",
    "Output dtype of the kernel (float32 or bfloat16).",
)

_BLOCK_SIZE = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "block_size",
    32,
    "Block size for scaling factors.",
)

_TILE_M = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "tile_m",
    128,
    "Tile size along M dimension.",
)

_TILE_N = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "tile_n",
    128,
    "Tile size along N dimension.",
)

_TILE_K = flags.DEFINE_integer(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "tile_k",
    256,
    "Tile size along K dimension.",
)

_HAS_BIAS = flags.DEFINE_boolean(  # PYTHON_FLAG_OK=Stand-alone build-time kernel generator tool.
    "has_bias",
    False,
    "Whether the kernel fuses bias addition.",
)


def _get_in_dtype(name: str) -> Any:
  if name == "float8_e4m3fn":
    return jnp.float8_e4m3fn
  elif name == "float8_e5m2":
    return jnp.float8_e5m2
  else:
    raise ValueError(f"Unsupported in_dtype: {name}")


def _get_out_dtype(name: str) -> Any:
  if name == "float32":
    return jnp.float32
  elif name == "bfloat16":
    return jnp.bfloat16
  else:
    raise ValueError(f"Unsupported out_dtype: {name}")


# ==============================================================================
# Recipe-Based Kernel Generator Registry
# ==============================================================================
#
# Extension Point:
# To add a new block-scaling format (e.g., 2D block scaling, 128 block-scaling,
# or formats with fused bias):
#   1. Implement a generator function with the signature:
#        def _generate_<recipe_name>(*, in_dtype, out_dtype, block_size,
#                                    tile_m, tile_n, tile_k, has_bias=False,
#                                    **kwargs) -> str
#      that returns the serialized StableHLO MLIR module string.
#      Optional features like `has_bias` or custom tile dimensions can be
#      handled natively by recipe generators that support them.
#   2. Register it in `_RECIPE_GENERATORS` under its recipe key.
#   3. Expose the recipe in `defs.bzl` via `define_scaled_mm_kernel(recipe=..., has_bias=...)`.
#
# This unifies kernel generation across block-scaling formats without creating
# separate conflicting build-time generation scripts.
# ==============================================================================


def _generate_mxfp8(
    *,
    in_dtype: Any,
    out_dtype: Any,
    block_size: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    has_bias: bool = False,
    **kwargs: Any,
) -> str:
  """Generates MXFP8 (1x32 block scaling) StableHLO MLIR module string."""
  if has_bias:
    raise NotImplementedError(
        "Bias fusion is not currently supported for recipe 'mxfp8'."
    )
  del kwargs
  return kernels.export_mlir_module(
      block_size=block_size,
      out_dtype=out_dtype,
      in_dtype=in_dtype,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
  )


def _generate_blockwise_128(
    *,
    in_dtype: Any,
    out_dtype: Any,
    block_size: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    has_bias: bool = False,
    **kwargs: Any,
) -> str:
  """Generates BlockWise1x128 x BlockWise128x128 StableHLO MLIR module string."""
  del kwargs
  return blockwise128_kernels.export_mlir_module(
      block_size=block_size,
      out_dtype=out_dtype,
      in_dtype=in_dtype,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      has_bias=has_bias,
  )


_RECIPE_GENERATORS: dict[str, Callable[..., str]] = {
    "mxfp8": _generate_mxfp8,
    "blockwise_128": _generate_blockwise_128,
    # [Extension Point]: Register new block-scaling recipes here.
}


def register_recipe(name: str, generator: Callable[..., str]) -> None:
  """Registers a kernel generator for a block-scaling recipe.

  Args:
    name: Recipe identifier (e.g. 'mxfp8').
    generator: Callable taking (in_dtype, out_dtype, block_size, tile_m, tile_n,
      tile_k, **kwargs) and returning the serialized StableHLO MLIR module
      string.
  """
  _RECIPE_GENERATORS[name] = generator


def get_supported_recipes() -> list[str]:
  """Returns the sorted list of registered recipe names."""
  return sorted(_RECIPE_GENERATORS.keys())


def generate_kernel_module(
    recipe: str,
    in_dtype_name: str,
    out_dtype_name: str,
    block_size: int = 32,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 256,
    has_bias: bool = False,
    **kwargs: Any,
) -> str:
  """Generates a StableHLO MLIR module string for the specified recipe.

  Args:
    recipe: Recipe identifier (e.g. 'mxfp8').
    in_dtype_name: Input dtype name ('float8_e4m3fn' or 'float8_e5m2').
    out_dtype_name: Output dtype name ('float32' or 'bfloat16').
    block_size: Block size along K for scale factors (default: 32).
    tile_m: Tile size along M (default: 128).
    tile_n: Tile size along N (default: 128).
    tile_k: Tile size along K (default: 256).
    has_bias: Whether the kernel fuses bias addition (default: False).
    **kwargs: Additional recipe-specific kwargs forwarded to generator.

  Returns:
    Serialized StableHLO MLIR module string.
  """
  if recipe not in _RECIPE_GENERATORS:
    supported_recipes = ", ".join(get_supported_recipes())
    raise ValueError(
        f"Unsupported recipe '{recipe}'. Supported recipes: {supported_recipes}"
    )

  in_dtype = _get_in_dtype(in_dtype_name)
  out_dtype = _get_out_dtype(out_dtype_name)

  generator = _RECIPE_GENERATORS[recipe]
  return generator(
      in_dtype=in_dtype,
      out_dtype=out_dtype,
      block_size=block_size,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      has_bias=has_bias,
      **kwargs,
  )


def main(argv: Sequence[str]) -> None:
  del argv
  if not _HEADER.value or not _IMPLEMENTATION.value:
    raise ValueError("--header and --implementation must be specified.")
  if not _KERNEL_NAME.value:
    raise ValueError("--kernel_name must be specified.")

  mlir_str = generate_kernel_module(
      recipe=_RECIPE.value,
      in_dtype_name=_IN_DTYPE.value,
      out_dtype_name=_OUT_DTYPE.value,
      block_size=_BLOCK_SIZE.value,
      tile_m=_TILE_M.value,
      tile_n=_TILE_N.value,
      tile_k=_TILE_K.value,
      has_bias=_HAS_BIAS.value,
  )

  kernel_utils.generate_embedded_file(
      _HEADER.value,
      _IMPLEMENTATION.value,
      [(_KERNEL_NAME.value, mlir_str.encode("utf-8"))],
  )


if __name__ == "__main__":
  app.run(main)
