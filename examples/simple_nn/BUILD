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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
)

torch_tpu_py_test(
    name = "simple_ops_nn_example",
    srcs = ["simple_ops_nn_example.py"],
    nopresubmit = "slow example",
    shuffle_tests = False,
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
    ],
)

torch_tpu_package_end()
