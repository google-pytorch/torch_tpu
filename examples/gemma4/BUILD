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
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "splash_attention",
    srcs = ["splash_attention.py"],
    deps = [
        "//shims/jax:pallas_tpu_ops",
        "//shims/torch:pytorch",
        "//shims/torchax:ops",
        "//torch_tpu/_internal/pallas",
        "@pypi//jax",
        "@pypi//torchax",
    ],
)

torch_tpu_py_library(
    name = "tpu_attention",
    srcs = ["tpu_attention.py"],
    deps = [
        ":splash_attention",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":splash_attention",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    tags = [
        "requires-tpu-v5p",
    ],
    deps = [
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "correctness_test",
    srcs = ["correctness_test.py"],
    deps = [
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "@pypi//transformers",  # build_cleaner: keep
    ],
)

torch_tpu_py_test(
    name = "correctness_jax_test",
    srcs = ["correctness_jax_test.py"],
    deps = [
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "@pypi//gemma",
        "@pypi//jax",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "correctness_tpu_test",
    srcs = ["correctness_tpu_test.py"],
    tags = [
        "requires-tpu-v5p",
    ],
    deps = [
        ":model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
    ],
)

# This must be the last line in the BUILD file.
torch_tpu_package_end()
