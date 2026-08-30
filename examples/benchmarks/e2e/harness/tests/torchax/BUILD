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
    name = "torchax_device_ops_test",
    srcs = ["torchax_device_ops_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:target",
        "//examples/benchmarks/e2e/harness/torchax:torchax_device_ops",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "torchax_step_test",
    srcs = ["torchax_step_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness:compile",
        "//examples/benchmarks/e2e/harness:context",
        "//examples/benchmarks/e2e/harness:registry",
        "//examples/benchmarks/e2e/harness:step_lib",
        "//examples/benchmarks/e2e/harness:target",
        "//examples/benchmarks/e2e/harness/torchax:torchax_step",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//flax",  # buildcleaner: keep
        "@pypi//torchax",
    ],
)

torch_tpu_py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    deps = [
        "//examples/benchmarks/e2e/harness/torchax:optimizer",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//optax",
    ],
)

torch_tpu_package_end()
