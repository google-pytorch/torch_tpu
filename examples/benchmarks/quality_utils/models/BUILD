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
    name = "configs",
    srcs = ["configs.py"],
    deps = [
        ":init",  # build_cleaner: keep
        "//examples:paths",
        "@pypi//llama_models",
    ],
)

torch_tpu_py_library(
    name = "meta_llama3_quality_benchmark",
    srcs = ["meta_llama3_quality_benchmark.py"],
    deps = [
        ":configs",
        "//examples/benchmarks/e2e:device_utils",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/torch:pytorch",
        "@pypi//fairscale",
        "@pypi//llama_models",
    ],
)

torch_tpu_py_library(
    name = "llama3_2_1b_quality_benchmark",
    srcs = ["llama3_2_1b_quality_benchmark.py"],
    deps = [
        "//examples:paths",
        "//examples/benchmarks/e2e:device_utils",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

torch_tpu_py_library(
    name = "qwen3_1_7b_quality_benchmark",
    srcs = ["qwen3_1_7b_quality_benchmark.py"],
    deps = [
        "//examples:paths",
        "//examples/benchmarks/e2e:device_utils",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/safetensors",  # build_cleaner: keep
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "llama3_2_1b_quality_benchmark_test",
    srcs = ["llama3_2_1b_quality_benchmark_test.py"],
    notest_oss = "Disabled during initial build",
    deps = [
        ":llama3_2_1b_quality_benchmark",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "qwen3_1_7b_quality_benchmark_test",
    srcs = ["qwen3_1_7b_quality_benchmark_test.py"],
    notest_oss = "Disabled during initial build",
    deps = [
        ":qwen3_1_7b_quality_benchmark",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//transformers",
    ],
)

torch_tpu_package_end()
