load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

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
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

exports_files(["data/goldfish.jpg"])

torch_tpu_py_test(
    name = "small_graph_test",
    srcs = ["small_graph_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "dtensor_test",
    srcs = ["dtensor_test.py"],
    notest_oss = "b/493050035 -- revisit once distributed tests are fixed in OSS",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
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
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "ensure_no_dynamic_test",
    srcs = ["ensure_no_dynamic_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "compile_with_distributed_test",
    srcs = ["compile_with_distributed_test.py"],
    env = {
        "ACCELERATOR_TYPE": "v5litepod-8",  # Needed for OSS.
    },
    shard_count = 1,
    tags = [
        # In OSS, we have to run the test exclusively to avoid the
        # "TPU initialization failed ... Device or resource busy" error.
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO: fix the test in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",  # buildcleaner: keep
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
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/api",
    ],
)

pytype_strict_library(
    name = "stablehlo_compile_testing",
    srcs = ["stablehlo_compile_testing.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/minipyspy:minipyspy_enable",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/api",
        "//torch_tpu/api:_device_ops_backend",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "stablehlo_compile_test",
    size = "enormous",
    srcs = ["stablehlo_compile_test.py"],
    args = [
        "--enable_tensorboard_logging=False",
    ],
    exec_properties = {
        "cpp_link.mem": "20g",  # Needed for asan.
    },
    linking_mode = "static",
    main = "stablehlo_compile_test.py",
    notap = "Disable on TAP until we can get faster performance or reduce number of runs. Currently runs ~1h.",
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
    ],
)

torch_tpu_py_test(
    name = "output_modify_test",
    srcs = ["output_modify_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
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
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "model_coverage_hf_test",
    timeout = "long",
    srcs = ["model_coverage_hf_test.py"],
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    shard_count = 6,
    tags = [
        "cpu:4",
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
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
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
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "@pypi//accelerate",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "model_coverage_timm_test",
    timeout = "long",
    srcs = ["model_coverage_timm_test.py"],
    data = ["data/goldfish.jpg"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 6,
    tags = [
        "noasan",  # TODO: fix the test in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-net:external",
        "requires-tpu-v5lite",  #
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "@pypi//pillow",
    ],
)

torch_tpu_py_test(
    name = "trace_autograd_ops_test",
    srcs = ["trace_autograd_ops_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    notest_oss = "Disabled during switch from opt-in to opt-out",
    strict = "true",
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_package_end()
