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

# We use importlib.resources to load models.csv directly.

torch_tpu_py_library(
    name = "gen_utils",
    srcs = ["gen_utils.py"],
    data = ["models.csv"],
    deps = [
        ":load_model",
        "//examples/benchmarks/e2e/harness:context",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "load_model",
    srcs = ["load_model.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:torch_device_ops",
        "//shims/torch:pytorch",
        "//tests:module_registry",
    ],
)

torch_tpu_py_test(
    name = "gen_utils_test",
    srcs = ["gen_utils_test.py"],
    deps = [
        ":gen_utils",
        "//shims/absl_py/testing:absltest",
        "//tests:seed_test_utils",
        "//torch_tpu",
    ],
)

torch_tpu_package_end()
