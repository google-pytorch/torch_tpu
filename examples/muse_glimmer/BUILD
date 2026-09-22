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
    ],
)

torch_tpu_py_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    deps = [
        ":model",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "@pypi//tensorboardx",
    ],
)

torch_tpu_py_test(
    name = "benchmark_test",
    size = "large",
    srcs = ["benchmark_test.py"],
    args = [
        "--alsologtostderr",
        # The full 30B configuration only fits on an accelerator; the test
        # target runs the small configuration as a smoke test.
        "--smoke_test=true",
    ],
    nopresubmit = "slow benchmark example",
    tags = [
        "nofastbuild",  # Measures performance.
    ],
    deps = [
        ":benchmark",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_package_end()
