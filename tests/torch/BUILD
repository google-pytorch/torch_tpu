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

load("@rules_python//python:py_binary.bzl", "py_binary")
load(
    "//build_files:build_defs.bzl",
    "torch_tpu_oss_torch_test",
    "torch_tpu_package_end",
)

package(default_applicable_licenses = ["//:license"])

py_binary(
    name = "torch_test_runner",
    testonly = True,
    srcs = ["torch_test_runner.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
    ],
)

# TODO(b/562086911): auto-generate all test targets defined in
# https://github.com/pytorch/pytorch/tree/main/test.

torch_tpu_oss_torch_test(
    name = "tensor_iterator_test",
    torch_test_module = "test_tensor_iterator",
)

torch_tpu_package_end()
