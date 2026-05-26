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

exports_files(["data/goldfish.jpg"])

torch_tpu_py_test(
    name = "backend_serialization_test",
    srcs = ["backend_serialization_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix ASAN failure
    tags = [
        "requires-net:external",
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
    name = "small_graph_test",
    srcs = ["small_graph_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/compile:compiler",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "activation_checkpointing_test",
    srcs = ["activation_checkpointing_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
    ],
)

torch_tpu_py_test(
    name = "activation_checkpointing_mem_test",
    srcs = ["activation_checkpointing_mem_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "dtensor_test",
    srcs = ["dtensor_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "random_ops_test",
    srcs = ["random_ops_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "ensure_no_dynamic_test",
    srcs = ["ensure_no_dynamic_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:compiler",
        "//torch_tpu/_internal/utils",
        "@pypi//sympy",
    ],
)

torch_tpu_py_test(
    name = "compile_with_distributed_test",
    size = "large",
    timeout = "long",
    srcs = ["compile_with_distributed_test.py"],
    notest_oss = "Disabled in OSS due to infra limitations on distributed tests",
    shard_count = 1,
    tags = [
        "requires-accel-hbm",
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/device:_device_module",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "compile_api_test",
    srcs = ["compile_api_test.py"],
    tags = [
        "noasan",  # TODO(b/497852581): fix the test in asan mode.
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_library(
    name = "stablehlo_compile_testing",
    srcs = ["stablehlo_compile_testing.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/minipyspy:minipyspy_enable",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/device:_device_ops_backend",  # buildcleaner: keep
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
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "output_modify_test",
    srcs = ["output_modify_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:compiler",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "compile_with_inference_mode_test",
    srcs = ["compile_with_inference_mode_test.py"],
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "model_coverage_hf_test",
    timeout = "long",
    srcs = ["model_coverage_hf_test.py"],
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 6,
    tags = [
        "cpu:4",
        "nomsan",  # 3p 'rust/tempfile' error: use-of-uninitialized-value in tempfile::file::imp::platform::reopen
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/utils",
        "@pypi//accelerate",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "model_coverage_hf_guitar_test",
    timeout = "long",
    srcs = ["model_coverage_hf_test.py"],
    args = [
        "--model_sizes_to_run=small",
        "--model_sizes_to_run=medium",
        "--model_sizes_to_run=large",
    ],
    linking_mode = "static",
    main = "model_coverage_hf_test.py",
    nobuild_oss = "Not intended to run in OSS",
    notap = "For Guitar only.",
    shard_count = 12,
    tags = [
        "cpu:4",
        "guitar",
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/utils",
        "@pypi//accelerate",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "model_coverage_timm_test",
    timeout = "long",
    srcs = ["model_coverage_timm_test.py"],
    data = ["data/goldfish.jpg"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 6,
    tags = [
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v5lite",  #
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/utils",
        "@pypi//pillow",
    ],
)

torch_tpu_py_test(
    name = "trace_autograd_ops_test",
    srcs = ["trace_autograd_ops_test.py"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = "true",
    tags = [
        "nomsan",  # 3p 'py/torch' error: use-of-uninitialized-value in 3 in reduced_float_type_copy
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    notest_oss = "subprocess spawning to set env vars timeout in oss",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_package_end()
