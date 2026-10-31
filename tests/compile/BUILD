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

# Domain experts for this package:
# zhanyong-wan
# basioli-k
# arfaian
# chizhang529
# cbasile-g
# ghpvnist
# mkhanna15

load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
    "tpu_gen",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files(["data/goldfish.jpg"])

torch_tpu_py_test(
    name = "flex_attention_test",
    srcs = ["flex_attention_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:flex_attention",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "detect_causal_mask_test",
    srcs = ["detect_causal_mask_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:detect_causal_mask",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "mark_embedded_constants_test",
    srcs = ["mark_embedded_constants_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/compile/fx_passes:mark_embedded_constants",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "backend_serialization_test",
    srcs = ["backend_serialization_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
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
    name = "fx_output_reconstructor_test",
    srcs = ["fx_output_reconstructor_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",  # buildcleaner: keep
        "//src/torch_tpu/_internal/export",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "small_graph_test",
    srcs = ["small_graph_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "nomsan",  # b/516945438
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:nested_compile_region",
        "//src/torch_tpu/_internal/compile:torch_tpu_compiled_executable",
        "//src/torch_tpu/_internal/compile:tpu_backend",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "split_compile_test",
    srcs = ["split_compile_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-mem:28g",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:split_compiler",
        "//src/torch_tpu/_internal/distributed:collective_ops",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "heterogeneous_device_test",
    srcs = ["heterogeneous_device_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "activation_checkpointing_test",
    timeout = "long",
    srcs = ["activation_checkpointing_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:debug",
        "//src/torch_tpu/_internal/compile/fx_passes:mark_activation_checkpoints",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "upstream_activation_checkpointing_test",
    timeout = "long",
    srcs = ["upstream_activation_checkpointing_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "activation_checkpointing_mem_test",
    timeout = "long",
    srcs = ["activation_checkpointing_mem_test.py"],
    nopresubmit_oss = "timeout",
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    use_tcmalloc = True,
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:mark_activation_checkpoints",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "dtensor_test",
    srcs = ["dtensor_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "random_ops_test",
    srcs = ["random_ops_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:debug",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "ensure_no_dynamic_test",
    srcs = ["ensure_no_dynamic_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//sympy",
    ],
)

torch_tpu_py_test(
    name = "compile_with_distributed_test",
    size = "large",
    timeout = "long",
    srcs = ["compile_with_distributed_test.py"],
    shard_count = 2,
    tags = if_oss(
        # Run test shards exclusively to prevent device contention on /dev/vfio/* when acquiring the
        # 8-chip TPU slice concurrently.
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/555762239): fix the test in asan mode.
        "requires-accel-hbm",
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests/distributed:continuous_test_utils",
    ],
)

torch_tpu_py_test(
    name = "compile_api_test",
    srcs = ["compile_api_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:debug",
        "//src/torch_tpu/_internal/compile:tpu_backend",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/device_utils:annotations",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "stablehlo_compile_testing",
    testonly = True,
    srcs = ["stablehlo_compile_testing.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/minipyspy:minipyspy_enable",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/device:_device_ops_backend",  # buildcleaner: keep
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "stablehlo_compile_test",
    timeout = "long",
    srcs = ["stablehlo_compile_test.py"],
    exec_properties = {
        "cpp_link.mem": "20g",  # Needed for asan.
    },
    linking_mode = "static",
    main = "stablehlo_compile_test.py",
    nopresubmit = "Slow test.",
    tags = [
        "cpu:1",  # Ensure reproducible performance on Forge.
        "nofastbuild",  # Measures performance.
        "requires-tpu-v6e",
    ],
    deps = [
        ":stablehlo_compile_testing",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/minipyspy:minipyspy_enable",
        "//shims/torch:pytorch",  # buildcleaner: keep
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "output_modify_test",
    srcs = ["output_modify_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "compile_with_inference_mode_test",
    srcs = ["compile_with_inference_mode_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "context_manager_graph_break_test",
    srcs = ["context_manager_graph_break_test.py"],
    env = {
        "TORCH_LOGS": "graph_breaks",  # For human-facing diagnostics only.
        "TORCH_LOGS_FORMAT": "%(message)s",
    },
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/precision",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "context_manager_errors_test",
    srcs = ["context_manager_errors_test.py"],
    tags = [
        "nosan",  # *san builds are low-value for error tests.
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compiler_options",
        "//src/torch_tpu/_internal/device_utils:annotations",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/export",
        "//src/torch_tpu/_internal/precision",
        "//src/torch_tpu/_internal/profiler",
        "//src/torch_tpu/_internal/profiler:named_scope",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "trace_autograd_ops_test",
    srcs = ["trace_autograd_ops_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    strict = "true",
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in 3 in reduced_float_type_copy
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests:seed_test_utils",
        "//tests:subprocess_test_utils",
    ],
)

torch_tpu_py_test(
    name = "aot_autograd_cache_test",
    srcs = ["aot_autograd_cache_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "//tests:subprocess_test_utils",
    ],
)

torch_tpu_py_test(
    name = "reorder_symints_test",
    srcs = ["reorder_symints_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:reorder_symints",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "propagate_symints_test",
    srcs = ["propagate_symints_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:propagate_symints",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "clone_mutated_returned_placeholders_test",
    srcs = ["clone_mutated_returned_placeholders_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:clone_mutated_returned_placeholders",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "sink_get_attr_constants_test",
    srcs = ["sink_get_attr_constants_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:sink_get_attr_constants",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "force_collectives_output_test",
    srcs = ["force_collectives_output_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:force_collectives_output",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "async_compile_test",
    srcs = ["async_compile_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "async_compile_benchmark_test",
    timeout = "eternal",
    srcs = ["async_compile_benchmark_test.py"],
    notap = "Slow benchmark test.",
    tags = [
        "cpu:8",
        "nofastbuild",  # Measures performance.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//tests:module_registry",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "async_compiled_artifact_test",
    srcs = ["async_compiled_artifact_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:torch_tpu_compiled_executable",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "jagged_compile_test",
    srcs = ["jagged_compile_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
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
    name = "fold_gqa_test",
    srcs = ["fold_gqa_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile/fx_passes:fold_gqa",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "debug_test",
    srcs = ["debug_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "compile_teardown_test",
    srcs = ["compile_teardown_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "nested_compile_region_benchmark_test",
    timeout = "long",
    srcs = ["nested_compile_region_benchmark_test.py"],
    # Run on single-chip TPU v7 for faster TorchDynamo/Inductor compilation
    # and to utilize idle single-chip v7 runner capacity in OSS presubmit.
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Faster TorchDynamo/Inductor compilation on single-chip TPU v7",
    ),
    tags = [
        "nofastbuild",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/compile:nested_compile_region",
        "//src/torch_tpu/_internal/compile:tpu_backend",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
