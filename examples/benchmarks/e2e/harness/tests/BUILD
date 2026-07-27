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
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "measure_test",
    srcs = ["measure_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:measure",
        "//examples/benchmarks/e2e/harness:metrics",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "mode_test",
    srcs = ["mode_test.py"],
    deps = [
        "//examples/benchmarks/e2e:common",
        "//examples/benchmarks/e2e/harness:mode",
        "//examples/benchmarks/e2e/harness:target",
        "//shims/absl_py/testing:absltest",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:compile",
        "//examples/benchmarks/e2e/harness:registry",
        "//examples/benchmarks/e2e/harness:step_lib",
        "//shims/absl_py/testing:absltest",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "step_test",
    srcs = ["step_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//examples/benchmarks/e2e/harness:discovery",
        "//examples/benchmarks/e2e/harness:measure",
        "//examples/benchmarks/e2e/harness:step_lib",
        "//examples/benchmarks/e2e/harness:target",
        "//examples/benchmarks/e2e/harness:torch_device_ops",
        "//examples/benchmarks/e2e/harness/steps",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "target_test",
    srcs = ["target_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "torch_device_ops_test",
    srcs = ["torch_device_ops_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//examples/benchmarks/e2e/harness:target",
        "//examples/benchmarks/e2e/harness:torch_device_ops",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "compile_test",
    srcs = ["compile_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:compile",
        "//examples/benchmarks/e2e/harness:target",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "context_test",
    srcs = ["context_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:target",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "discovery_test",
    srcs = ["discovery_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:discovery",
        "//examples/benchmarks/e2e/harness:registry",
        "//shims/absl_py/testing:absltest",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "loss_test",
    srcs = ["loss_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:loss",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_package_end()
