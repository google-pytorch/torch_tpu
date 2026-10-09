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
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "ragged_moe",
    srcs = ["ragged_moe.py"],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

# torchtitan is not available in OSS, so this and its test are not built in OSS.
internal_target(
    torch_tpu_py_library,
    name = "torchtitan_qwen3_5",
    srcs = ["torchtitan_qwen3_5.py"],
    deps = [
        "//shims/torch:pytorch",
        # google3's copybara rewrite of torchtitan's
        # `distributed/activation_checkpoint.py` (imported by
        # `torchtitan.models`) imports `torchtitan.experiments.tpu.utils`, which
        # the root torchtitan library cannot depend on. Every torchtitan
        # consumer adds it (e.g. torch_tpu/google/examples/benchmarks/e2e).
        "//shims/torchtitan/experiments/tpu:utils",  # buildcleaner: keep
        "//shims/torchtitan/models",
    ],
)

torch_tpu_py_library(
    name = "model_utils",
    srcs = ["__init__.py"],
)

torch_tpu_py_test(
    name = "ragged_moe_test",
    srcs = ["ragged_moe_test.py"],
    deps = [
        ":ragged_moe",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//transformers",
    ],
)

internal_target(
    torch_tpu_py_test,
    name = "torchtitan_qwen3_5_test",
    srcs = ["torchtitan_qwen3_5_test.py"],
    deps = [
        ":torchtitan_qwen3_5",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//transformers",
    ],
)

# Muse Glimmer from the glimmer prototype, which is not available in OSS.
internal_target(
    torch_tpu_py_library,
    name = "muse_glimmer",
    srcs = ["muse_glimmer.py"],
    deps = [
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        # Imported lazily; see the module docstring.
        "//src/torch_tpu/google/prototypes/glimmer",  # buildcleaner: keep
    ],
)

internal_target(
    torch_tpu_py_test,
    name = "muse_glimmer_test",
    srcs = ["muse_glimmer_test.py"],
    deps = [
        ":muse_glimmer",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
