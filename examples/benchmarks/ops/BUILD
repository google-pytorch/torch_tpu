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
    name = "op_capture",
    srcs = ["op_capture.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "op_input_loader",
    srcs = ["op_input_loader.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_test(
    name = "op_input_loader_test",
    srcs = ["op_input_loader_test.py"],
    deps = [
        ":op_input_loader",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "aten_op_benchmarks",
    srcs = ["aten_op_benchmarks.py"],
    data = glob(["captured_ops/*.json"]),
    shard_count = 10,
    tags = [
        "nomsan",  # b/516945438
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_input_loader",
        "//examples/benchmarks/e2e:benchmark_utils",
        "//examples/benchmarks/e2e:common",
        "//examples/benchmarks/e2e:device_utils",
        "//examples/benchmarks/e2e:mlcompass_utils",
        "//examples/benchmarks/e2e/harness:metrics",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//numpy",
    ],
)

torch_tpu_package_end()
