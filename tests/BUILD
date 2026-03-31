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
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
)

pytype_strict_library(
    name = "module_registry",
    srcs = ["module_registry.py"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/etils/epath",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "@pypi//timm",
        "@pypi//torchvision",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "module_registry_test",
    timeout = "long",
    srcs = ["module_registry_test.py"],
    data = ["//tests/compile:data/goldfish.jpg"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 6,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-net:external",
    ],
    deps = [
        ":module_registry",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "@pypi//pillow",
    ],
)

pytype_strict_library(
    name = "op_testing",
    testonly = True,
    srcs = ["op_testing.py"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/safetensors",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/compiler_options",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

pytype_strict_library(
    name = "dynamism_test_utils",
    testonly = True,
    srcs = ["dynamism_test_utils.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/utils",
    ],
)

# We have multiple targets for ops_test.py:
# 1. ops_test_tpu: TPU vs CPU.
# 2. ops_test_xla_cuda: XLA:CUDA vs CPU.
# 3. ops_test_xla_cpu: XLA:CPU vs CPU.
# 4. ops_test_tpu_vs_gpu: TPU vs GPU.
# 5. ops_test_gen_gpu_golden: generates the golden file for the GPU backend
# 6. ops_test_grad: gradient on TPU vs CPU.
# 7. ops_test_perf: TPU only, for measuring compilation/execution time.
# 8. ops_test_dynamic: TPU only, bounds a dimension and compares TPU vs CPU.
# 9. ops_test_tpu_vs_gpu_compiled: TPU only, torch.compile TpuBackend vs GPU inductor backend.

#
# TorchTPU can run on "tpu" or "xla_cuda" or "xla_cpu" devices using the --torch_tpu_device flag.
#

# (1) and (2) are identical except for the compilation mode. We chose to
# manually maintain the duplicate tests instead of using a for-loop to
# generate the targets because:
#
# - The for-loop approach is incompatible with build_cleaner.
# - The for-loop approach breaks the Cider-torch_tpu_py_test integration (in particular,
#   it breaks running a single test case directly in the Cider UI).

torch_tpu_py_test(
    name = "ops_test_tpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 50,
    tags = [
        # Will re-enable this slow test until we have a proper way to speed it up.
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_grad",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--compute_grad=True",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_test.py",
    notest_oss = "Will re-enable this slow test until we have a proper way to speed it up.",
    shard_count = 50,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_tpu_vs_gpu_compiled",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--use_compiled=True",
        "--test_mode=torch_tpu_vs_gpu",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden_compiled0.gz",
        "ops_test_gpu_golden_compiled1.gz",
        "ops_test_gpu_golden_compiled2.gz",
    ],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 40,
    tags = [
        "noasan",  # TODO: fix the timeout of test_grid_sampler in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the timeout of test_grid_sampler in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_xla_cuda",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=xla_cuda",
        "--max_samples_per_op_dtype=3",  # Reduce test samples as XLA:CUDA is not a priority.
    ],
    main = "ops_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 20,
    tags = [
        "requires-gpu-sm80-only",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "ops_test_xla_cpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=xla_cpu",
        "--max_samples_per_op_dtype=3",  # Reduce test samples as XLA:CPU is not a priority.
    ],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_test.py",
    # This test takes almost 30 minutes to run, since it is only using one
    # machine per build for now and it is not essential for the OSS release.
    notest_oss = "Test is too slow to run in OSS CI",
    shard_count = 25,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_tpu_vs_gpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=torch_tpu_vs_gpu",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_test.py",
    notest_oss = "Will re-enable this slow test until we have a proper way to speed it up.",
    shard_count = 50,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

# TODO: Document how to update the perf data in OSS.
torch_tpu_py_test(
    name = "ops_test_perf",
    timeout = "long",  # 15 minutes. The test finishes in ~5 minutes, which is the default limit.
    srcs = ["ops_test.py"],
    args = [
        # go/keep-sorted start
        "--alsologtostderr",
        "--max_samples_per_op_dtype=10",
        "--print_op_inputs=2",
        "--test_mode=perf",
        # Needed to fix flag 'vmodule' parsing error when using C++ flag in Python with Bazel.
        "--undefok=vmodule",
        "--vmodule=compilation_cache=3",  # Log SHLO strings.
        # go/keep-sorted end
    ],
    main = "ops_test.py",
    nolocal = "slow test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 50,
    tags = [
        "cpu:8",  # Specify the number of CPUs for reproducible results.
        "nofastbuild",  # Measures performance.
        "requires-net:external",  # For writing remote files.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_dynamic",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--torch_tpu_device=tpu",
        "--check_dynamism_using_seed=-1",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=1",  # No need to test many samples to catch dynamism bugs.
    ],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 32,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

# See go/tt-devguide#updating-gpu-goldens for how to update the GPU goldens.
torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
    ],
    main = "ops_test.py",
    nolocal = "slow test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    # Each shard generates one golden file that contains one slice of the
    # golden data. Therefore the smaller the count, the bigger each file
    # will be.
    #
    # We should minimize the shard count without making a golden file larger
    # than 100 MB (a limit imposed by GitHub). The problem with a large shard
    # count is that it makes updating the golden files more tedious as it's
    # still done manually.
    #
    # If we adjust this, we must also adjust the data attribute in the
    # ops_test_vs_gpu target.
    shard_count = 3,
    tags = [
        "requires-gpu-sm80",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_unit_test",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "ops_unit_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//scipy",
    ],
)

# Tests the tier-3 compilation cache.
torch_tpu_py_test(
    name = "ops_unit_test_tier3_cache",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
        "ASAN_OPTIONS": "detect_leaks=0",
    },
    main = "ops_unit_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//scipy",
    ],
)

# Tests the tier-3 compilation cache without local backup compilation.
torch_tpu_py_test(
    name = "ops_unit_test_tier3_cache_no_backup",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
        "ASAN_OPTIONS": "detect_leaks=0",
    },
    main = "ops_unit_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//scipy",
    ],
)

torch_tpu_py_test(
    name = "compiler_option_test",
    srcs = ["compiler_option_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "compiler_option_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compiler_options",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "index_put_op_test",
    srcs = ["index_put_op_unit_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "index_put_op_unit_test.py",
    shard_count = 20,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "nn_test",
    srcs = ["nn_test.py"],
    tags = [
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "error_testing",
    testonly = True,
    srcs = ["error_testing.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "error_testing_test",
    srcs = ["error_testing_test.py"],
    args = [
        "--test_mode=cpu",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
    ],
)

torch_tpu_py_test(
    name = "errors_test_tpu",
    srcs = ["errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "0",
        "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "errors_test.py",
    notap = "TODO(wan): Re-enable after the March 2026 TorchTPU release.",
    shard_count = 4,
    tags = [
        "requires-tpu-v5lite",
        "wheel_test_excluded",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_test_cov",
    srcs = ["errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "errors_test.py",
    notap = "TODO(wan): Re-enable after the March 2026 TorchTPU release.",
    # TODO: Get working in OSS and enable in nightlies
    notest_oss = "Disabled as too slow and because it is failing in OSS.",
    shard_count = 4,
    tags = ["requires-tpu-v5lite"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_test_cpu",
    srcs = ["errors_test.py"],
    args = ["--test_mode=cpu"],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    main = "errors_test.py",
    notap = "TODO(wan): Re-enable after the March 2026 TorchTPU release.",
    shard_count = 4,
    tags = ["wheel_test_excluded"],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_cpp_context_test",
    srcs = ["errors_cpp_context_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "booleans_test",
    srcs = ["booleans_test.py"],
    # args = [],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "backwards_test",
    srcs = ["backwards_test.py"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "untyped_storage_test",
    srcs = ["untyped_storage_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "view_test",
    srcs = ["view_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "zero_size_test",
    srcs = ["zero_size_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "sync_test",
    srcs = ["sync_test.py"],
    main = "sync_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "set_test",
    srcs = ["set_test.py"],
    main = "set_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "execution_mode_test",
    srcs = ["execution_mode_test.py"],
    main = "execution_mode_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "dynamism_test",
    srcs = ["dynamism_test.py"],
    env = {"ASAN_OPTIONS": "detect_leaks=0"},  # TODO: fix the leak.
    main = "dynamism_test.py",
    tags = [
        # "Skipped CI on v7 TPU: Fatal crash in XLA XPrecisionCombineEmitter on v7 TPU due to SublaneBytes alignment assertion (32 vs 512) for dynamic int64 shapes."
        "excluded-on-tpu-v7x",
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "compilation_cache_stress_test",
    srcs = ["compilation_cache_stress_test.py"],
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "dynamism_model_test",
    srcs = ["dynamism_model_test.py"],
    main = "dynamism_model_test.py",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "precision_test",
    srcs = ["precision_test.py"],
    main = "precision_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/precision",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "eager_debug_test",
    srcs = ["eager_debug_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "allocator_repro_test",
    srcs = ["allocator_repro_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_package_end()
