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

# Domain experts for this package:
# mkhanna15
# salkan0

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
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_backend",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    tags = [
        "nomsan",  # TODO(b/507536079): 3p 'py/torch' error: use-of-uninitialized-value in reduced_float_type_copy
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "multi_bucket_compile_test",
    srcs = ["multi_bucket_compile_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:torch_tpu_compiled_executable",
        "//src/torch_tpu/_internal/compile:tpu_backend",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "sym_constraint_test",
    srcs = ["sym_constraint_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "generative_ops_test",
    srcs = ["generative_ops_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "dynamic_view_test",
    srcs = ["dynamic_view_test.py"],
    # Run on single-chip TPU v7 for faster dynamic shape compilation and
    # to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster dynamic shape compilation on single-chip TPU v7",
    ),
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/dynamic:view_decomposition",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "sym_expr_arg_test",
    srcs = ["sym_expr_arg_test.py"],
    # Run on single-chip TPU v7 for faster symbolic expression evaluation and compilation,
    # utilizing idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster symbolic expression evaluation on single-chip TPU v7",
    ),
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "symbol_bounds_test",
    srcs = ["symbol_bounds_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/dynamic:output_sym_shape",
        "//src/torch_tpu/_internal/compile/dynamic:sym_bounds_manager",
        "//src/torch_tpu/_internal/compile/dynamic:symbol_bounds",
        "//tests:seed_test_utils",
        "@pypi//sympy",
    ],
)

torch_tpu_py_test(
    name = "graph_transform_context_test",
    srcs = ["graph_transform_context_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/dynamic:graph_transform_context",
        "//src/torch_tpu/_internal/compile/dynamic:output_sym_shape",
        "//src/torch_tpu/_internal/compile/dynamic:sym_bounds_manager",
        "//src/torch_tpu/_internal/compile/dynamic:sym_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "graph_transformations_test",
    srcs = ["graph_transformations_test.py"],
    deps = [
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/dynamic:graph_transform_context",
        "//src/torch_tpu/_internal/compile/dynamic:graph_transformations",
        "//src/torch_tpu/_internal/compile/dynamic:sym_bounds_manager",
        "//src/torch_tpu/_internal/compile/dynamic:view_ops_passes",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "jagged_dynamism_test",
    srcs = ["jagged_dynamism_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_backend",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "view_decomposition_test",
    srcs = ["view_decomposition_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/dynamic:view_decomposition",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//sympy",
    ],
)

torch_tpu_py_test(
    name = "dynamic_slice_test",
    srcs = ["dynamic_slice_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster dynamic shape compilation on single-chip TPU v7",
    ),
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/dynamic:graph_transform_context",
        "//src/torch_tpu/_internal/compile/dynamic:graph_transformations",
        "//src/torch_tpu/_internal/compile/dynamic:sym_bounds_manager",
        "//src/torch_tpu/_internal/compile/dynamic:view_ops_passes",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "sdpa_vlog_test",
    srcs = ["sdpa_vlog_test.py"],
    # TT_KERNEL logs its arguments at VLOG(1).
    args = ["--vmodule=scaled_dot_product_attention_aten_kernels=1"],
    notest_oss = True,  # NOTEST_OSS_OK=The C++ --vmodule flag is only bridged into absl.flags in internal test binaries.
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
