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

"""Build rules for SDPA kernels variants."""

load("//build_files:build_defs.bzl", "if_oss", "torch_tpu_cc_library")

def define_sdpa_kernel(name, forward, dtype, is_causal):
    """Defines a SDPA kernel target.

    Args:
      name: The name of the target key (e.g. "sdpa_fwd_f32_all_mlir").
      forward: Boolean, true for forward, false for backward.
      dtype: String, "float32" or "bfloat16".
      is_causal: Boolean, true for causal, false for non-causal.
    """
    direction_flag = "--forward" if forward else "--backward"
    is_causal_str = "True" if is_causal else "False"

    genrule_name = name
    header_name = name + "_embed.h"
    source_name = name + "_embed.cc"
    lib_name = name + "_embed"

    cmd_discovery = """
            # Discover PyTorch and CUDA libraries to support local torch wheel build mode.
            # We search specifically within the generator's runfiles to maintain hermeticity
            gen_tool="$(location //torch_tpu/ops/scaled_dot_product_attention/kernels:scaled_dot_product_attention_generate)"
            if [[ -d "$${gen_tool}.runfiles" ]]; then
              lib_dirs=$$(find -L "$${gen_tool}.runfiles" \\( -name "libtorch.so" -o -name "libcudart.so*" -o -name "libnvrtc.so*" \\) -exec dirname {} + 2>/dev/null | sort -u)
              if [[ -n "$$lib_dirs" ]]; then
                export LD_LIBRARY_PATH="$$(echo "$$lib_dirs" | tr '\\n' ':')$${LD_LIBRARY_PATH:-}"
              fi
            fi
"""

    cmd_body = """
            read -ra out_files <<< "$(OUTS)"
            h_path=$${out_files[0]}
            cc_path=$${out_files[1]}
            # Set JAX_PLATFORMS=cpu to cross-compile TPU kernels without needing
            # access to physical TPU hardware or libtpu lockfiles during the build.
            JAX_PLATFORMS=cpu $(location //torch_tpu/ops/scaled_dot_product_attention/kernels:scaled_dot_product_attention_generate) %s \
                --dtype=%s \
                --is_causal=%s \
                --kernel_name=%s \
                --header $$h_path \
                --implementation $$cc_path
""" % (direction_flag, dtype, is_causal_str, name)

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
        tools = ["//torch_tpu/ops/scaled_dot_product_attention/kernels:scaled_dot_product_attention_generate"],
    )

    torch_tpu_cc_library(
        name = lib_name,
        srcs = [source_name],
        hdrs = [header_name],
    )
