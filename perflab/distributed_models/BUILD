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

load(
    "//build_files:build_defs.bzl",
    "internal_target",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "qwen",
    srcs = ["qwen.py"],
    deps = [
        "//perflab:compile",
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/model_utils:ragged_moe",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "@pypi//fairscale",
        "@pypi//transformers",
    ],
)

# The Qwen3.8 benchmark depends on torchtitan, which is not available in OSS,
# so it is a separate non-OSS target that benchmark binaries add with
# `if_oss([], [...])`. Discovery imports whichever model modules are present.
internal_target(
    torch_tpu_py_library,
    name = "qwen3_8",
    srcs = ["qwen3_8.py"],
    deps = [
        "//perflab:compile",
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/model_utils:torchtitan_qwen3_5",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//shims/torchtitan/experiments/graph_trainer:simple_fsdp",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "llama",
    srcs = ["llama.py"],
    deps = [
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//shims/torch:pytorch",
        "@pypi//fairscale",
        "@pypi//llama_models",
    ],
)

# Depends on torchtitan's SimpleFSDP, so like `qwen3_8` it is a separate non-OSS
# target that benchmark binaries add with `if_oss([], [...])`.
internal_target(
    torch_tpu_py_library,
    name = "synthetic",
    srcs = ["synthetic.py"],
    deps = [
        "//perflab:compile",
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//shims/torch:pytorch",
        "//shims/torchtitan/experiments/graph_trainer:simple_fsdp",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "distributed_models",
    srcs = ["__init__.py"],
    deps = [
        ":llama",  # buildcleaner: keep
        ":qwen",  # buildcleaner: keep
    ],
)

# Muse Glimmer from the glimmer prototype, which is not available in OSS.
internal_target(
    torch_tpu_py_library,
    name = "muse_glimmer",
    srcs = ["muse_glimmer.py"],
    deps = [
        "//perflab:compile",
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/model_utils:muse_glimmer",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
