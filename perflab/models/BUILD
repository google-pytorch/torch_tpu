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
    name = "models",
    srcs = [
        "__init__.py",
        "gemma.py",
        "gen_benchmarks.py",
        "gpt.py",
        "llama.py",
        "qwen.py",
    ],
    allow_multiple_srcs = True,
    deps = [
        "//perflab:context",
        "//perflab:model_helpers",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:target",
        "//perflab:torch_device_ops",
        "//perflab/gen_utils",
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
    ],
)

# Qwen3.8 benchmarks depend on torchtitan, which is not available in OSS, so
# they are a separate non-OSS target that benchmark binaries add with
# `if_oss([], [...])`. Discovery imports whichever model modules are present.
internal_target(
    torch_tpu_py_library,
    name = "qwen3_8",
    srcs = ["qwen3_8.py"],
    deps = [
        "//perflab:context",
        "//perflab:registry",
        "//perflab:step_lib",
        "//perflab:torch_device_ops",
        "//perflab/model_utils:torchtitan_qwen3_5",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_package_end()
