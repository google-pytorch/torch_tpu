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
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/ops:splash_attention",
    ],
)

torch_tpu_py_library(
    name = "model_baseline",
    srcs = ["model_baseline.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_test(
    name = "model_baseline_test",
    srcs = ["model_baseline_test.py"],
    deps = [
        ":model_baseline",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
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
        "//tests:seed_test_utils",
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
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "@pypi//transformers",
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
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/ops:splash_attention",
    ],
)

torch_tpu_py_test(
    name = "gemma4_hf_example",
    srcs = ["gemma4_hf_example.py"],
    shuffle_tests = False,  # This script doesn't understand --test_randomize_ordering_seed.
    tags = [
        "noasan",  # OOM
        "nomsan",  # OOM
        "notsan",  # OOM
        "requires-tpu-v5p",
    ],
    deps = [
        "//examples:paths",
        "//examples/huggingface_transformers:model_configs",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:test_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_package_end()
