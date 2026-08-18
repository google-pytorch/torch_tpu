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

# Description:
#   Qwen-Image model prototypes, sharding, and pipelines for TorchTPU.

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "config",
    srcs = ["config.py"],
    imports = ["."],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "utils",
    srcs = ["utils.py"],
    imports = ["."],
    deps = [
        "//shims/safetensors",
        "//shims/torch:pytorch",
        "@pypi//datasets",
        "@pypi//pandas",
        "@pypi//pillow",
    ],
)

torch_tpu_py_library(
    name = "tpu_sharding",
    srcs = ["tpu_sharding.py"],
    imports = ["."],
    deps = [
        ":config",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//accelerate",
        "@pypi//numpy",
    ],
)

torch_tpu_py_library(
    name = "qwen_image_tpu",
    srcs = ["qwen_image_tpu.py"],
    imports = ["."],
    deps = [
        ":config",
        ":tpu_sharding",
        "@pypi//pillow",
        # Keep the top-level diffusers target (instead of :diffusers_lite)
        # so it maps cleanly to the external PyPI dependency in OSS.
        "@pypi//diffusers",  # buildcleaner: keep
        "@pypi//peft",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "infer_qwen20b_tpu",
    srcs = ["infer_qwen20b_tpu.py"],
    imports = ["."],
    deps = [
        ":config",
        ":qwen_image_tpu",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "train_torch_tpu",
    srcs = ["train_torch_tpu.py"],
    imports = ["."],
    deps = [
        ":config",
        ":utils",
        "@pypi//datasets",
        # Keep the top-level diffusers target (instead of :diffusers_lite)
        # so it maps cleanly to the external PyPI dependency in OSS.
        "@pypi//diffusers",  # buildcleaner: keep
        "@pypi//numpy",
        "//shims/safetensors",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//tqdm",
    ],
)

torch_tpu_py_test(
    name = "qwen_image_test",
    srcs = ["qwen_image_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":config",
        ":infer_qwen20b_tpu",
        ":tpu_sharding",
        ":train_torch_tpu",
        ":utils",
        "@pypi//pillow",
        "//shims/absl_py/testing:absltest",
        # Keep the top-level diffusers target (instead of :diffusers_lite)
        # so it maps cleanly to the external PyPI dependency in OSS.
        "@pypi//diffusers",  # buildcleaner: keep
        "//shims/safetensors",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//transformers",
    ],
)

torch_tpu_package_end()
