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
load("//shims/mpm:genmpm.bzl", "genmpm")
load("//shims/pkg:pkg.bzl", "pkg_binary_with_runfiles")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "benchmark_utils",
    srcs = ["benchmark_utils.py"],
    deps = [
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/xprof:xprof_analysis_client",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/utils:device_utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "mlcompass_utils",
    srcs = ["mlcompass_utils.py"],
    deps = [
        ":benchmark_utils",
        "//shims/absl_py/logging",
        "//torch_tpu/_internal/shims/mlcompass:benchmark_data",
        "//torch_tpu/_internal/shims/mlcompass:export_lib_borg",
        "//torch_tpu/_internal/utils:log_utils",
    ],
)

pytype_strict_library(
    name = "performance_utils",
    testonly = 1,
    srcs = ["performance_utils.py"],
    deps = [
        ":benchmark_function_db",
        ":benchmark_utils",
        ":mlcompass_utils",
        ":model_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/torch",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:xprof_analysis_client",
        "//torch_tpu/_internal/utils:device_utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//tensorboardx",
    ],
)

pytype_strict_library(
    name = "quality_utils",
    testonly = 1,
    srcs = ["quality_utils.py"],
    deps = [
        ":benchmark_utils",
        ":mlcompass_utils",
        "//examples/benchmarks/quality_utils:quality_benchmark_model",
        "//examples/benchmarks/quality_utils/metrics:data_loader",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:gpu_env",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        "//shims/absl_py/testing:parameterized",
    ],
)

torch_tpu_py_test(
    name = "model_performance_benchmarks",
    timeout = "long",
    srcs = ["model_performance_benchmarks.py"],
    # This allows us to use a single .par file for remote cluster execution.
    linking_mode = "static",
    main = "model_performance_benchmarks.py",
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    notap = "Tested on guitar workflow.",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        "nofastbuild",  # Benchmark.
        # Needed for getting xprof links
        "requires-net:external",
        "exclusive",  # copybara:comment(oss-only)
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "model_quality_benchmarks",
    timeout = "long",
    srcs = ["model_quality_benchmarks.py"],
    # This allows us to use a single .par file for remote cluster execution.
    linking_mode = "static",
    main = "model_quality_benchmarks.py",
    notap = "Tested on guitar workflow.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 4,
    strict = True,
    tags = [
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":quality_utils",
        "//examples/benchmarks/quality_utils/metrics:data_loader",
        "//examples/benchmarks/quality_utils/metrics:perplexity_metric",
        "//examples/benchmarks/quality_utils/models:meta_llama3_quality_benchmark",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

pkg_binary_with_runfiles(
    name = "model_performance_benchmarks_with_runfiles",
    testonly = 1,
    binary = ":model_performance_benchmarks",
)

pkg_binary_with_runfiles(
    name = "model_quality_benchmarks_with_runfiles",
    testonly = 1,
    binary = ":model_quality_benchmarks",
)

# Using a temporal MPM instead of a durable MPM as these are only used for testing. It also allows
# testing changes from a local workspace.
genmpm(
    name = "model_performance_benchmarks_mpm",
    testonly = 1,
    temporal = True,
    deps = [":model_performance_benchmarks_with_runfiles"],
)

genmpm(
    name = "model_quality_benchmarks_mpm",
    testonly = 1,
    temporal = True,
    deps = [":model_quality_benchmarks_with_runfiles"],
)

pytype_strict_library(
    name = "layer_configs",
    testonly = True,
    srcs = ["layer_configs.py"],
    deps = ["//shims/torch:pytorch"],
)

torch_tpu_py_test(
    name = "layer_performance_benchmarks",
    timeout = "long",
    srcs = ["layer_performance_benchmarks.py"],
    main = "layer_performance_benchmarks.py",
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    notap = "Tested on guitar workflow.",
    shard_count = 8,
    strict = True,
    tags = [
        "cpu:4",
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":layer_configs",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

pkg_binary_with_runfiles(
    name = "layer_performance_benchmarks_with_runfiles",
    testonly = 1,
    binary = ":layer_performance_benchmarks",
)

# Using a temporal MPM instead of a durable MPM as these are only used for testing. It also allows
# testing changes from a local workspace.
genmpm(
    name = "layer_performance_benchmarks_mpm",
    testonly = 1,
    temporal = True,
    deps = [":layer_performance_benchmarks_with_runfiles"],
)

torch_tpu_py_test(
    name = "layer_performance_benchmarks_cuda",
    timeout = "long",
    srcs = ["layer_performance_benchmarks.py"],
    args = ["--platform=b200_1"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {
        "cpp_link.mem": "16g",
    },
    main = "layer_performance_benchmarks.py",
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    notap = "Tested on guitar workflow.",
    shard_count = 2,
    strict = True,
    tags = [
        "cpu:4",
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-gpu-sm100-full-only",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":layer_configs",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "distributed_benchmark_tests",
    timeout = "long",
    srcs = ["model_performance_benchmarks.py"],
    args = [
        "--platform=gfc_2x2x1",
    ],
    # This allows us to use a single .par file for remote cluster execution.
    linking_mode = "static",
    main = "model_performance_benchmarks.py",
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    notap = "Tested on guitar workflow.",
    tags = [
        "cpu:4",
        # This makes sure that we have enough host RAM to load weights from host to device HBM.
        "requires-accel-hbm",
        "requires-tpu-v7x:4",
        # Needed for getting xprof links
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "model_benchmark_cuda_tests",
    timeout = "long",
    srcs = ["model_performance_benchmarks.py"],
    args = ["--platform=b200_1"],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {
        "cpp_link.mem": "16g",
    },
    main = "model_performance_benchmarks.py",
    nobuild_oss = "Disabled during switch from opt-in to opt-out",
    notap = "Frequently causes Forge OOM.",
    shard_count = 2,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "nofastbuild",  # Benchmark.
        "requires-accel-hbm",
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        ":benchmark_utils",
        ":performance_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

pytype_strict_library(
    name = "benchmark_function_db",
    testonly = 1,
    srcs = ["benchmark_function_db.py"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils:device_utils",
    ],
)

pytype_strict_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    deps = [
        "//examples/deepseek",
        "//shims/torch:pytorch",
        "//tests:module_registry",
        "//torch_tpu/_internal/utils:device_utils",
        "@pypi//fairscale",
        "@pypi//llama_models",
        "@pypi//transformers",
    ],
)

torch_tpu_package_end()
