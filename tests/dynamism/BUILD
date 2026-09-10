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
    "tpu_gen",
)

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
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "sym_constraint_test",
    srcs = ["sym_constraint_test.py"],
    notest_oss = "OSS needs to be updated with the latest libtpu release that contains cl/904008746.",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "generative_ops_test",
    srcs = ["generative_ops_test.py"],
    notest_oss = "OSS needs to be updated with the latest libtpu release for bounded dynamism.",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "dynamic_view_test",
    srcs = ["dynamic_view_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster dynamic shape compilation on TPU v7",
    ),
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "sym_expr_arg_test",
    srcs = ["sym_expr_arg_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster symbolic expression evaluation on TPU v7",
    ),
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "symbol_bounds_test",
    srcs = ["symbol_bounds_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:symbol_bounds",
    ],
)

torch_tpu_py_test(
    name = "sym_shape_manager_test",
    srcs = ["sym_shape_manager_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:sym_shape_manager",
        "//torch_tpu/_internal/compile/dynamic:sym_utils",
    ],
)

torch_tpu_py_test(
    name = "graph_transformations_test",
    srcs = ["graph_transformations_test.py"],
    deps = [
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:graph_transformations",
        "//torch_tpu/_internal/compile/dynamic:sym_shape_manager",
        "//torch_tpu/_internal/compile/dynamic:view_ops_passes",
    ],
)

torch_tpu_py_test(
    name = "jagged_dynamism_test",
    srcs = ["jagged_dynamism_test.py"],
    notest_oss = "OSS needs to be updated with the latest libtpu release for bounded dynamism.",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "view_decomposition_test",
    srcs = ["view_decomposition_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:seed_test_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile/dynamic:view_decomposition",
        "//torch_tpu/_internal/utils:test_utils",
        "@pypi//sympy",
    ],
)

torch_tpu_package_end()
