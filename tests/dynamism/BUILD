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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "compile_test",
    srcs = ["compile_test.py"],
    tags = [
        # TODO: b/503386468 - Re-enable once the test is fixed.
        # Skip this test on tpu v6 and v7 runners: Fatal crash in XLA XPrecisionCombineEmitter due
        # to SublaneBytes alignment assertion (32 vs 512) for dynamic 64-bit shapes (int64/float64).
        "fails-on-tpu-v6",
        "fails-on-tpu-v7",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "sym_shape_manager_test",
    srcs = ["sym_shape_manager_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:sym_shape_manager",
    ],
)

torch_tpu_py_test(
    name = "graph_transformations_test",
    srcs = ["graph_transformations_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:graph_transformations",
        "//torch_tpu/_internal/compile/dynamic:sym_shape_manager",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_package_end()
