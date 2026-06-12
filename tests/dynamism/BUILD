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
    notest_oss = "OSS needs to be updated with the latest libtpu release that contains cl/904008746.",
    tags = [
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
    name = "precompilation_test",
    srcs = ["precompilation_test.py"],
    tags = [
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

torch_tpu_py_test(
    name = "dynamic_adapters_test",
    srcs = ["dynamic_adapters_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:dynamic_adapters",
    ],
)

torch_tpu_py_test(
    name = "bucket_compile_test",
    srcs = ["bucket_compile_test.py"],
    tags = [
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

torch_tpu_package_end()
