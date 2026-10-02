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

"""Build rules for scaled mm kernel variants."""

load("//build_files:build_defs.bzl", "if_oss", "torch_tpu_cc_library")

def define_scaled_mm_kernel(
        name,
        in_dtype,
        recipe = "mxfp8",
        out_dtype = "float32",
        block_size = 32,
        tile_m = 128,
        tile_n = 128,
        tile_k = 256,
        has_bias = False,
        kernel_name = None):
    """Defines a scaled mm kernel target and compiles an embedded C++ library.

    This macro invokes the hermetic build-time kernel generator tool
    (`scaled_mm_generate`) to produce StableHLO MLIR artifacts embedded as C++
    data arrays.

    Extension Point:
      To instantiate kernels for alternative block-scaling formats or parameter
      sweeps, specify the `recipe` argument (e.g. `recipe = "custom_format"`)
      registered in `scaled_mm_generate.py`. Optional post-GEMM fusions such as
      `has_bias = True` can also be passed for recipes that support them.

    Args:
      name: The name of the target key (e.g. "blockwise_scaled_mm_e4m3").
      in_dtype: String, input dtype ("float8_e4m3fn" or "float8_e5m2").
      recipe: String, scaling recipe identifier (default: "mxfp8").
      out_dtype: String, output dtype ("float32").
      block_size: Integer, scaling block size (default: 32).
      tile_m: Integer, tile size along M (default: 128).
      tile_n: Integer, tile size along N (default: 128).
      tile_k: Integer, tile size along K (default: 256).
      has_bias: Boolean, whether the kernel fuses bias addition (default: False).
      kernel_name: Optional string, C++ variable name prefix (default: name).
    """
    if not kernel_name:
        kernel_name = name

    genrule_name = name
    header_name = name + "_embed.h"
    source_name = name + "_embed.cc"
    lib_name = name + "_embed"

    cmd_discovery = """
            # Discover PyTorch and CUDA libraries to support local torch wheel build mode.
            # We search specifically within the generator's runfiles to maintain hermeticity
            gen_tool="$(location //src/torch_tpu/_internal/ops/scaled_mm/kernels:scaled_mm_generate_bin)"
            if [[ -d "$${gen_tool}.runfiles" ]]; then
              lib_dirs=$$(find -L "$${gen_tool}.runfiles" \\( -name "libtorch.so" -o -name "libcudart.so*" -o -name "libnvrtc.so*" \\) -exec dirname {} + 2>/dev/null | sort -u)
              if [[ -n "$$lib_dirs" ]]; then
                export LD_LIBRARY_PATH="$$(echo "$$lib_dirs" | tr '\\n' ':')$${LD_LIBRARY_PATH:-}"
              fi
            fi
"""

    bias_flag = " \\\n                --has_bias" if has_bias else ""

    cmd_body = """
            read -ra out_files <<< "$(OUTS)"
            h_path=$${out_files[0]}
            cc_path=$${out_files[1]}
            # Set JAX_PLATFORMS=cpu to cross-compile TPU kernels without needing
            # access to physical TPU hardware or libtpu lockfiles during the build.
            JAX_PLATFORMS=cpu $(location //src/torch_tpu/_internal/ops/scaled_mm/kernels:scaled_mm_generate_bin) \
                --recipe=%s \
                --in_dtype=%s \
                --out_dtype=%s \
                --block_size=%d \
                --tile_m=%d \
                --tile_n=%d \
                --tile_k=%d \
                --kernel_name=%s%s \
                --header $$h_path \
                --implementation $$cc_path
""" % (recipe, in_dtype, out_dtype, block_size, tile_m, tile_n, tile_k, kernel_name, bias_flag)

    native.genrule(
        name = genrule_name,
        outs = [header_name, source_name],
        cmd = if_oss(
            select({
                "//shims/torch:use_cuda_torch": "set -e\n" + cmd_discovery + cmd_body,
                "//shims/torch:use_local_torch": "set -e\n" + cmd_discovery + cmd_body,
                "//conditions:default": "set -e\n" + cmd_body,
            }),
            "set -e\n" + cmd_body,
        ),
        tools = ["//src/torch_tpu/_internal/ops/scaled_mm/kernels:scaled_mm_generate_bin"],
    )

    torch_tpu_cc_library(
        name = lib_name,
        srcs = [source_name],
        hdrs = [header_name],
    )
