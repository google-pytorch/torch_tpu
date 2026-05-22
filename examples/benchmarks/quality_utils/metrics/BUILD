# Copyright 2025 Google LLC
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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "init",
    srcs = ["__init__.py"],
)

torch_tpu_py_library(
    name = "perplexity_metric",
    srcs = ["perplexity_metric.py"],
    deps = [
        ":init",  # build_cleaner: keep
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "data_loader",
    srcs = ["data_loader.py"],
    deps = [
        "//shims/absl_py/logging",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//grain",
        "@pypi//tensorflow_datasets",
    ],
)

torch_tpu_py_test(
    name = "perplexity_metric_test",
    srcs = ["perplexity_metric_test.py"],
    notest_oss = "Disabled during initial build",
    deps = [
        ":perplexity_metric",
        "//examples:init",  # build_cleaner: keep
        "//examples/benchmarks:init",  # build_cleaner: keep
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_package_end()
