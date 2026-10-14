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
# adivinpatel04
# ghpvnist
# zhanyong-wan
# haifeng-jin
# cbasile-g
# willfroom
# arfaian

load(
    "//build_files:build_defs.bzl",
    "DEFAULT_TORCH_EXAMPLE_DEPS",
    "OSS_REQUIRES_V5_X4",
    "TEST_COV_ENV",
    "if_oss",
    "torch_tpu_oss_torch_example",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
    "tpu_gen",
)
load(
    "//shims/py_rules:pytype.bzl",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = ["//:license"],
)

exports_files([
    "cpp_style_test_data.cc",
])

# ops_test records its GPU goldens on two architectures, because neither one
# covers every op. Blackwell (SM100) is the default and holds the bulk of the
# data. Hopper (SM90) holds the test case for ops such as _scaled_mm_v2 with
# recipes that only cuBLAS on Hopper implements.
#
# Per go/forge-for-ml, Blackwell and later are requested with a platform rather
# than a requires-gpu-* tag. These GPUs are hosted on both x86 and Arm, and only
# a platform also tells blaze which architecture to build the test for. The
# partitioned (1/7 MiG) variant suffices, as golden generation uses one GPU.
_GPU_GOLDEN_PLATFORM = "//buildenv/platforms/forge/gpu/b200"

_GPU_GOLDEN_TAG_SM90 = "requires-gpu-sm90"

# Hopper is the default Forge GPU pool, so they request it by tag and stay off
# scarce Blackwell capacity.
_REQUIRES_GPU_FOR_NON_OPS_TEST = "requires-gpu-sm90"

# TODO: change this back to 16 after the OSS v5 worker pool is back to normal.
_OSS_SAMPLE_LIMIT = "--max_samples_per_op_dtype=5"

# Knob for controlling how many tests to include in the OSS presubmit build.
# Set this to True to skip less important tests in OSS presubmit (they
# will still run in postsubmit).
_OSS_PRESUBMIT_OVERLOADED = False

# This can be used as the value for the nopresubmit_oss parameter in
# torch_tpu_*_test to skip the test in OSS presubmit if and only if
# the OSS presubmit is overloaded.
_WHEN_OSS_PRESUBMIT_OVERLOADED = (
    "This test is lower-priority and thus skipped in presubmit." if _OSS_PRESUBMIT_OVERLOADED else None
)

# Ops tests download GPU goldens from GCS in OSS; internally they read them from runfiles.
_OSS_GOLDEN_TAGS = if_oss(
    ["requires-net:external"],
    [],
)

# The ops_test GPU goldens, grouped by the test configuration that recorded
# them. Internally they are read from this directory. In OSS, Bazel fetches
# them from GCS into @torch_tpu_goldens (see oss/bazel/goldens.bzl), which
# exports each file individually so that a test only pulls in the goldens it
# actually needs, exactly as it does internally.
_GPU_GOLDENS = [
    "ops_test_gpu_golden0.gz",
    "ops_test_gpu_golden1.gz",
    "ops_test_gpu_golden2.gz",
    "ops_test_gpu_golden_hopper0.gz",
]

_GPU_GOLDENS_GRAD = [
    "ops_test_gpu_golden_grad0.gz",
    "ops_test_gpu_golden_grad1.gz",
    "ops_test_gpu_golden_grad2.gz",
    "ops_test_gpu_golden_grad_hopper0.gz",
]

_GPU_GOLDENS_COMPILED = [
    "ops_test_gpu_golden_compiled0.gz",
    "ops_test_gpu_golden_compiled1.gz",
    "ops_test_gpu_golden_compiled2.gz",
    "ops_test_gpu_golden_compiled_hopper0.gz",
]

_GPU_GOLDENS_DATA = if_oss(
    ["@torch_tpu_goldens//:" + f for f in _GPU_GOLDENS],
    _GPU_GOLDENS,
)

_GPU_GOLDENS_GRAD_DATA = if_oss(
    ["@torch_tpu_goldens//:" + f for f in _GPU_GOLDENS_GRAD],
    _GPU_GOLDENS_GRAD,
)

_GPU_GOLDENS_COMPILED_DATA = if_oss(
    ["@torch_tpu_goldens//:" + f for f in _GPU_GOLDENS_COMPILED],
    _GPU_GOLDENS_COMPILED,
)

torch_tpu_py_library(
    name = "oss_utils",
    testonly = True,
    srcs = ["oss_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//src/torch_tpu/_internal/utils",
    ] + if_oss(
        ["//shims/libtpu"],
        [],
    ),
)

torch_tpu_py_test(
    name = "api_registry_test",
    srcs = ["api_registry_test.py"],
    # Route to OSS TPU runners and exclude from OSS CPU runners
    # (running on CPU took 600+ sec due to emulated device init).
    tags = if_oss(["requires-tpu-v5lite"]),
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal/compiler_options",
        "//src/torch_tpu/_internal/device_utils:annotations",
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/ops:ops_lib",
        "//src/torch_tpu/_internal/pallas",
        "//src/torch_tpu/_internal/profiler:profiler_config",
        "//src/torch_tpu/_internal/utils:annotations",
        "@pypi//packaging",
    ],
)

torch_tpu_py_test(
    name = "oss_utils_test",
    srcs = ["oss_utils_test.py"],
    deps = [
        ":oss_utils",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "tcmalloc_check_test",
    srcs = ["tcmalloc_check_test.py"],
    use_tcmalloc = True,
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//src/torch_tpu:_tcmalloc_check",
        "//src/torch_tpu/_internal:warnings",
    ],
)

torch_tpu_py_test(
    name = "_versioned_so_loader_test",
    srcs = ["_versioned_so_loader_test.py"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//src/torch_tpu:_versioned_so_loader",
    ],
)

torch_tpu_py_test(
    name = "ao_offload_test",
    srcs = ["ao_offload_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "compilation_mode_test",
    srcs = ["compilation_mode_test.py"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal:env",
    ],
)

torch_tpu_py_test(
    name = "env_test",
    srcs = ["env_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        ":subprocess_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "env_utils_test",
    srcs = ["env_utils_test.py"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env_utils",
    ],
)

torch_tpu_py_test(
    name = "env_vars_test",
    srcs = ["env_vars_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        ":subprocess_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal:warnings",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "cpp_warnings_test",
    srcs = ["cpp_warnings_test.py"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal:warnings",
    ],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Decouple from torch_tpu to avoid C++ dependency in coverage tests.
    name = "module_registry",
    srcs = [
        "module_registry.py",
        "transformers_inputs.py",
    ],
    data = [
        "//perflab/huggingface_diffusers:model_configs_data",
        "//perflab/huggingface_transformers:model_configs_data",
        "//shims/model_configs:google_model_configs_data",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/etils/epath",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "@pypi//diffusers",  # build_cleaner: keep
        "@pypi//google_api_core",
        "@pypi//google_cloud_storage",
        "@pypi//timm",  # build_cleaner: keep
        "@pypi//torchvision",  # build_cleaner: keep
        "@pypi//transformers",  # build_cleaner: keep
    ],
)

torch_tpu_py_test(
    name = "module_registry_test",
    timeout = "long",
    srcs = ["module_registry_test.py"],
    data = ["//tests/compile:data/goldfish.jpg"],
    # Cannot run in OSS either: no access to the model resources.
    notap = "TODO(b/561614286): Flaky test due to unstable model cache access",
    shard_count = 6,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-net:external",
    ],
    deps = [
        ":module_registry",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//google_api_core",
        "@pypi//google_cloud_storage",
        "@pypi//pillow",
    ],
)

torch_tpu_py_library(
    name = "accelerator_test_utils",
    testonly = True,
    srcs = ["accelerator_test_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":oss_utils",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "seed_test_utils",
    testonly = True,
    srcs = ["seed_test_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal/device:_device_ops_backend",
    ],
)

torch_tpu_py_library(
    name = "subprocess_test_utils",
    testonly = True,
    srcs = ["subprocess_test_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu/_internal/device:_device_ops_backend",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

torch_tpu_py_library(
    name = "quantize_utils",
    testonly = True,
    srcs = ["quantize_utils.py"],
    deps = [
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_test(
    name = "quantize_utils_test",
    srcs = ["quantize_utils_test.py"],
    deps = [
        ":quantize_utils",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_library(
    name = "op_testing",
    testonly = True,
    srcs = ["op_testing.py"],
    data = ["goldens.pin"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":dynamism_test_utils",
        ":quantize_utils",
        ":seed_test_utils",
        ":test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/etils/epath",
        "//shims/safetensors",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/compiler_options",
        "//src/torch_tpu/_internal/utils",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//pyyaml",
    ],
)

torch_tpu_py_library(
    name = "ops_test_data",
    testonly = True,
    srcs = ["ops_test_data.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
    ],
)

torch_tpu_py_library(
    name = "dynamism_test_utils",
    testonly = True,
    srcs = ["dynamism_test_utils.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/utils",
    ],
)

# We have multiple targets for ops_test.py:
# * ops_test: TPU vs GPU (by default, we test against GPU).
# * ops_test_gen_gpu_golden: generates the golden file for the GPU backend.
# * ops_test_gen_gpu_golden_compiled: generates the golden file under torch.compile.
# * ops_test_grad_tpu_vs_gpu: gradient on TPU vs GPU.
# * ops_test_gen_gpu_golden_grad: generates the golden file for gradients.
# * ops_test_gen_gpu_golden*_hopper: the same, on Hopper, for tests tagged
#   @op_testing.record_gpu_goldens_on(op_testing.GPU_SM90).
# * ops_test_perf: TPU only, for measuring compilation/execution time.
# * ops_test_dynamic: TPU vs GPU, bounds a dimension.
# * ops_test_dynamic_compiled: TPU vs GPU, bounds a dimension under torch.compile.
# * ops_test_compiled: TPU vs GPU (torch.compile TpuBackend vs GPU inductor backend).

# We chose to manually maintain the duplicate tests instead of using a for-loop to generate the
# targets because:
#
# - The for-loop approach is incompatible with build_cleaner.
# - The for-loop approach breaks the Cider-torch_tpu_py_test integration (in particular,
#   it breaks running a single test case directly in the Cider UI).

torch_tpu_py_test(
    name = "foreach_ops_test_grad",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_categories=foreach",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_GRAD_DATA,
    main = "ops_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 15,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "nonzero_ops_test_grad",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_categories=nonzero",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_GRAD_DATA,
    main = "ops_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 1,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "searchsorted_ops_test_grad",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_categories=searchsorted",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_GRAD_DATA,
    main = "ops_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 1,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_compiled",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--use_compiled=True",
    ] + if_oss(
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_COMPILED_DATA,
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in postsubmit and nightly.",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 40,
    tags = [
        "noasan",  # TODO: fix the timeout of test_grid_sampler in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the timeout of test_grid_sampler in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=-foreach,-nonzero,-searchsorted",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_DATA,
    main = "ops_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 50,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_grad_tpu_vs_gpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_categories=-foreach,-nonzero,-searchsorted",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_GRAD_DATA,
    main = "ops_test.py",
    # Skip this in OSS presubmit when it's overloaded.
    nopresubmit_oss = _WHEN_OSS_PRESUBMIT_OVERLOADED,
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 35,
    tags = [
        "nosan",  # Rely on ops_test_grad for *san coverage.
        "ok-to-wait-for-v6e-in-presubmit",
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "foreach_ops_test",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=foreach",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_DATA,
    main = "ops_test.py",
    # Skip this in OSS presubmit when it's overloaded.
    nopresubmit_oss = _WHEN_OSS_PRESUBMIT_OVERLOADED,
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 50,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "nonzero_ops_test",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=nonzero",
        "--max_samples_per_op_dtype=10",  # To avoid timeout.
    ],
    data = _GPU_GOLDENS_DATA,
    main = "ops_test.py",
    # Skip this in OSS presubmit when it's overloaded.
    nopresubmit_oss = _WHEN_OSS_PRESUBMIT_OVERLOADED,
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 1,  # This test has only 1 test case.
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "searchsorted_ops_test",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=searchsorted",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        [_OSS_SAMPLE_LIMIT],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = _GPU_GOLDENS_DATA,
    main = "ops_test.py",
    # Skip this in OSS presubmit when it's overloaded.
    nopresubmit_oss = _WHEN_OSS_PRESUBMIT_OVERLOADED,
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 1,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

# This target measures the compilation time for each op.
#
# TODO: Document how to update the perf data in OSS. > op_perf.txt
#
# To debug the compilation performance of a specific op (e.g. grid_sample)
# and dtype (e.g. float32):
#   1. blaze test //tests:ops_test_perf -c opt --test_sharding_strategy=disabled --test_arg=TestOps.test_grid_sample --test_arg=--dtypes=float32
#   2. Search for "module @" in the test log to find the SHLOs corresponding to
#      the op invocations.
torch_tpu_py_test(
    name = "ops_test_perf",
    timeout = "long",  # 15 minutes. The test finishes in ~5 minutes, which is the default limit.
    srcs = ["ops_test.py"],
    args = [
        # go/keep-sorted start
        "--alsologtostderr",
        "--print_op_inputs=2",
        "--test_mode=perf",
        # go/keep-sorted end
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        ["--max_samples_per_op_dtype=8"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=10"],
    ),
    main = "ops_test.py",
    nolocal = "slow test",
    nopresubmit = "Benchmark.",
    nopresubmit_oss = "This slow test is enabled in nightly.",
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
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_dynamic",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--check_dynamism_using_seed=-1",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=1",  # No need to test many samples to catch dynamism bugs.
    ],
    data = _GPU_GOLDENS_DATA,
    main = "ops_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 32,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_dynamic_compiled",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--check_dynamism_using_seed=-1",
        "--use_compiled=True",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=1",  # No need to test many samples to catch dynamism bugs.
    ],
    data = _GPU_GOLDENS_COMPILED_DATA,
    main = "ops_test.py",
    nopresubmit = "Dynamic compile tests are run in postsubmit to avoid overloading the presubmit.",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 32,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "ops-test",  # Heavy ops test: see run_bazel_diff.py for details.
    ] + _OSS_GOLDEN_TAGS,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
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
    notap = "slow test - only need to run on demand for updating goldens",
    platform = _GPU_GOLDEN_PLATFORM,
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
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden_compiled",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
        "--use_compiled=True",
    ],
    env = {
        "TORCHINDUCTOR_COMPILE_THREADS": "0",
    },
    main = "ops_test.py",
    nolocal = "slow test",
    notap = "slow test - only need to run on demand for updating goldens",
    platform = _GPU_GOLDEN_PLATFORM,
    shard_count = 3,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

# Few tests record their goldens on Hopper, so one shard is enough and the
# Hopper generators are fast enough to run on TAP.
torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden_hopper",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
    ],
    main = "ops_test.py",
    nolocal = "slow test",
    shard_count = 1,
    tags = [_GPU_GOLDEN_TAG_SM90],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

# Like ops_test_gen_gpu_golden, but only generates 1 sample per (op, dtype).
# This is for quickly verifying that the logic for generating GPU goldens is
# not broken. We run this on TAP automatically, while only running the full
# ops_test_gen_gpu_golden on demand to update the goldens.
torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden_smoke",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
        "--max_samples_per_op_dtype=1",
    ],
    main = "ops_test.py",
    nolocal = "slow test",
    platform = _GPU_GOLDEN_PLATFORM,
    shard_count = 3,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_dump_dtype_exclusions",
    srcs = ["ops_test.py"],
    args = ["--test_mode=dump_dtype_exclusions"],
    main = "ops_test.py",
    # Keep this test unsharded. The exclusions are accumulated in a
    # per-process global, so a sharded run would dump partial results.
    shard_count = 1,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_library(
    name = "goldens_pin",
    srcs = ["goldens_pin.py"],
    data = ["goldens.pin"],
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//attrs",
        "@pypi//cattrs",
    ],
)

torch_tpu_py_test(
    name = "goldens_pin_test",
    srcs = ["goldens_pin_test.py"],
    deps = [
        ":goldens_pin",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
        "@pypi//hypothesis",
    ],
)

torch_tpu_py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    data = ["goldens.pin"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":goldens_pin",
        "//shims/absl_py/flags",
    ],
)

torch_tpu_py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    data = ["goldens.pin"],
    deps = [
        ":goldens_pin",
        ":seed_test_utils",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "op_testing_test",
    srcs = ["op_testing_test.py"],
    deps = [
        ":goldens_pin",
        ":op_testing",
        ":test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden_grad",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
        "--compute_grad=True",
    ],
    main = "ops_test.py",
    nolocal = "slow test",
    notap = "slow test - only need to run on demand for updating goldens",
    platform = _GPU_GOLDEN_PLATFORM,
    shard_count = 3,
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden_grad_hopper",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
        "--compute_grad=True",
    ],
    main = "ops_test.py",
    nolocal = "slow test",
    shard_count = 1,
    tags = [_GPU_GOLDEN_TAG_SM90],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_test_gen_gpu_golden_compiled_hopper",
    srcs = ["ops_test.py"],
    args = [
        "--test_mode=gen_gpu_golden",
        "--use_compiled=True",
    ],
    env = {"TORCHINDUCTOR_COMPILE_THREADS": "0"},
    main = "ops_test.py",
    nolocal = "slow test",
    shard_count = 1,
    tags = [_GPU_GOLDEN_TAG_SM90],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":quantize_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ops_unit_test",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    main = "ops_unit_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        ":oss_utils",
        ":quantize_utils",
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//scipy",
    ],
)

torch_tpu_py_test(
    name = "sparse_dense_matmul_test",
    timeout = "long",
    srcs = ["sparse_dense_matmul_test.py"],
    main = "sparse_dense_matmul_test.py",
    shard_count = 4,
    tags = [
        # Requires SparseCore accelerator, which is not available on TPU v5e and below.
        "fails-on-tpu-v5",
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/device_utils:annotations",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "sparse_gather_test",
    srcs = ["sparse_gather_test.py"],
    main = "sparse_gather_test.py",
    tags = [
        # Requires SparseCore accelerator, which is not available on TPU v5e and below.
        "fails-on-tpu-v5",
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x",
    ],
    deps = [
        ":op_testing",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "sparse_iota_test",
    srcs = ["sparse_iota_test.py"],
    main = "sparse_iota_test.py",
    tags = [
        # Requires SparseCore accelerator, which is not available on TPU v5e and below.
        "fails-on-tpu-v5",
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "sparse_map_row_test",
    srcs = ["sparse_map_row_test.py"],
    main = "sparse_map_row_test.py",
    tags = [
        # Requires SparseCore accelerator, which is not available on TPU v5e and below.
        "fails-on-tpu-v5",
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/export",
    ],
)

torch_tpu_py_test(
    name = "sparse_dense_matmul_stacking_ops_test",
    srcs = ["sparse_dense_matmul_stacking_ops_test.py"],
    main = "sparse_dense_matmul_stacking_ops_test.py",
    tags = [
        # Requires SparseCore accelerator, which is not available on TPU v5e and below.
        "fails-on-tpu-v5",
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x",
    ],
    deps = [
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "eager_device_assignment_test",
    srcs = ["eager_device_assignment_test.py"],
    main = "eager_device_assignment_test.py",
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

# Tests the tier-3 compilation cache.
torch_tpu_py_test(
    name = "ops_unit_test_tier3_cache",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    env = {
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
    },
    main = "ops_unit_test.py",
    shard_count = 30,
    tags = [
        "fails-on-tpu-v7",  # Numeric divergence in scaled_mm on TPU v7.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        ":oss_utils",
        ":quantize_utils",
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//scipy",
    ],
)

# Tests the tier-3 compilation cache without local backup compilation.
torch_tpu_py_test(
    name = "ops_unit_test_tier3_cache_no_backup",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
    },
    main = "ops_unit_test.py",
    shard_count = 30,
    tags = [
        "fails-on-tpu-v7",  # Numeric divergence in scaled_mm on TPU v7.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        ":oss_utils",
        ":quantize_utils",
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//scipy",
    ],
)

torch_tpu_py_test(
    name = "compiler_option_test",
    srcs = ["compiler_option_test.py"],
    main = "compiler_option_test.py",
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compiler_options",
    ],
)

torch_tpu_py_test(
    name = "index_put_op_test",
    srcs = ["index_put_op_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    main = "index_put_op_unit_test.py",
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    shard_count = 20,
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "fused_adam_test",
    srcs = ["fused_adam_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "fused_adamw_test",
    srcs = ["fused_adamw_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
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
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "jagged_tensor_test",
    srcs = ["jagged_tensor_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "fp4_test",
    srcs = ["fp4_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":quantize_utils",
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_library(
    name = "error_testing",
    testonly = True,
    srcs = ["error_testing.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",  # build_cleaner: keep
    ],
)

torch_tpu_py_test(
    name = "error_testing_test",
    srcs = ["error_testing_test.py"],
    args = [
        "--test_mode=cpu",
    ],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
    ],
    deps = [
        ":error_testing",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
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
        "TORCH_TPU_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "errors_test.py",
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":error_testing",
        ":oss_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "tpu_errors_test",
    srcs = ["tpu_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "0",
        "TORCH_TPU_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "tpu_errors_test.py",
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ] + if_oss(
        # TODO: b/567928058 - test_mm_with_oom_result fails on the RBE v5e
        # workers, so run it on the self-hosted runners until that's fixed.
        ["no-remote-exec"],
        [],
    ),
    deps = [
        ":error_testing",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal:env_utils",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/export",
        "//src/torch_tpu/_internal/pallas",
        "//src/torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils:hardware",
        "//tests/distributed:distributed_utils",
    ],
)

# Collects error coverage stats without verifying the error messages.
torch_tpu_py_test(
    name = "errors_test_cov",
    srcs = ["errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = TEST_COV_ENV,
    main = "errors_test.py",
    notap = "Run by error_coverage_test in the torch_tpu.error_coverage Guitar workflow.",
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        ":error_testing",
        ":oss_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils",
    ],
)

# Collects error coverage stats without verifying the error messages.
torch_tpu_py_test(
    name = "tpu_errors_test_cov",
    srcs = ["tpu_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = TEST_COV_ENV,
    main = "tpu_errors_test.py",
    notap = "Run by error_coverage_test in the torch_tpu.error_coverage Guitar workflow.",
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        ":error_testing",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal:env_utils",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/distributed",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/export",
        "//src/torch_tpu/_internal/pallas",
        "//src/torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils:hardware",
        "//tests/distributed:distributed_utils",
    ],
)

torch_tpu_py_test(
    name = "errors_test_gpu",
    # CUDA_LAUNCH_BLOCKING=1 runs ops synchronously, requiring long timeout to complete.
    timeout = "long",
    srcs = ["errors_test.py"],
    args = ["--test_mode=gpu"],
    env = {
        # Force synchronous CUDA kernel launches so device-side errors trigger
        # immediately at the exact Python op call instead of asynchronously.
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
    },
    main = "errors_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    notest_oss = True,  # Low value in maintaining for both internal and OSS and they run different versions of PyTorch.
    shard_count = 8,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "nozapfhahn",  # Requires --config=cuda; excluded from Zapfhahn coverage.
        _REQUIRES_GPU_FOR_NON_OPS_TEST,
        "wheel_test_excluded",
    ],
    deps = [
        ":error_testing",
        ":oss_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "errors_cpp_context_test",
    srcs = ["errors_cpp_context_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "booleans_test",
    srcs = ["booleans_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    # args = [],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "backwards_test",
    srcs = ["backwards_test.py"],
    tags = [
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "untyped_storage_test",
    srcs = ["untyped_storage_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "view_test",
    srcs = ["view_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "zero_size_test",
    srcs = ["zero_size_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "rng_test",
    srcs = ["rng_test.py"],
    main = "rng_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "rng_cuda_ref_test",
    srcs = ["rng_cuda_ref_test.py"],
    args = ["--backend=tpu"],
    main = "rng_cuda_ref_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":accelerator_test_utils",
        ":oss_utils",
        ":seed_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "rng_cuda_ref_test_cuda",
    srcs = ["rng_cuda_ref_test.py"],
    args = ["--backend=gpu"],
    main = "rng_cuda_ref_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    tags = [
        "nozapfhahn",  # Requires --config=cuda; excluded from Zapfhahn coverage.
        # Requires compute capability >= 7.0 for Triton / torch.compile.
        _REQUIRES_GPU_FOR_NON_OPS_TEST,
    ],
    deps = [
        ":accelerator_test_utils",
        ":oss_utils",
        ":seed_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "rng_multi_device_test",
    srcs = ["rng_multi_device_test.py"],
    args = ["--backend=tpu"],
    main = "rng_multi_device_test.py",
    tags = if_oss(
        [
            "exclusive",
            OSS_REQUIRES_V5_X4,
        ],
        ["requires-tpu-v5lite:8"],
    ),
    deps = [
        ":accelerator_test_utils",
        ":seed_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "rng_multi_device_test_cuda",
    srcs = ["rng_multi_device_test.py"],
    args = ["--backend=gpu"],
    main = "rng_multi_device_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    tags = [
        "nozapfhahn",  # Requires --config=cuda; excluded from Zapfhahn coverage.
        # Requires 2 devices for multi-device tests.
        "requires-gpu-sm90-full:2",
    ],
    deps = [
        ":accelerator_test_utils",
        ":seed_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

# Multi-process CUDA reference test kept separate because TPU PJRT locks
# devices, requiring worker spawning from an uninitialized parent runner.
torch_tpu_py_test(
    name = "rng_multiprocess_test",
    srcs = ["rng_multiprocess_test.py"],
    args = ["--backend=tpu"],
    main = "rng_multiprocess_test.py",
    tags = if_oss(
        [
            "exclusive",
            OSS_REQUIRES_V5_X4,
        ],
        ["requires-tpu-v5lite:8"],
    ) + [
        # Spawning 8 PyTorch processes in *san exceeds Forge's 28GB container
        # memory limit; this is an infra constraint rather than a code defect.
        "noasan",  # See comment above.
        "nomsan",  # See comment above.
        "notsan",  # See comment above.
    ],
    deps = [
        ":accelerator_test_utils",
        ":seed_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests/distributed:distributed_utils",
    ],
)

torch_tpu_py_test(
    name = "rng_multiprocess_test_cuda",
    srcs = ["rng_multiprocess_test.py"],
    args = ["--backend=gpu"],
    main = "rng_multiprocess_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    tags = [
        "nozapfhahn",  # Requires --config=cuda; excluded from Zapfhahn coverage.
        # Requires compute capability >= 7.0 for Triton / torch.compile.
        # Requires 2 devices for multi-device tests.
        # Multi-device sm80 is not available on Forge.
        "requires-gpu-sm90-full:2",
    ],
    deps = [
        ":accelerator_test_utils",
        ":seed_test_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:annotations",
        "//tests/distributed:distributed_utils",
    ],
)

torch_tpu_py_test(
    name = "accelerator_api_test",
    srcs = ["accelerator_api_test.py"],
    main = "accelerator_api_test.py",
    tags = ["requires-tpu-v5lite:8"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils",
        "//src/torch_tpu/_internal/utils:annotations",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "synchronize_test",
    srcs = ["synchronize_test.py"],
    main = "synchronize_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "internal_sync_test",
    srcs = ["internal_sync_test.py"],
    main = "internal_sync_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
    ],
)

torch_tpu_py_test(
    name = "set_test",
    srcs = ["set_test.py"],
    main = "set_test.py",
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "execution_mode_test",
    srcs = ["execution_mode_test.py"],
    main = "execution_mode_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_test(
    name = "inplace_buffer_donation_test",
    size = "medium",
    srcs = ["inplace_buffer_donation_test.py"],
    main = "inplace_buffer_donation_test.py",
    shard_count = 4,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "adamw_test",
    srcs = ["adamw_test.py"],
    main = "adamw_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/optim",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "patch_test",
    srcs = ["patch_test.py"],
    main = "patch_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/optim",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "dynamism_test",
    srcs = ["dynamism_test.py"],
    main = "dynamism_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "compilation_cache_stress_test",
    srcs = ["compilation_cache_stress_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_test(
    name = "rng_stress_test",
    srcs = ["rng_stress_test.py"],
    shard_count = 4,
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "dynamism_model_test",
    srcs = ["dynamism_model_test.py"],
    main = "dynamism_model_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/dynamism",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "precision_test",
    srcs = ["precision_test.py"],
    main = "precision_test.py",
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Load balancing",
    ),
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/precision",
        "//src/torch_tpu/_internal/precision:precision_impl",  # buildcleaner: keep
        "//src/torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "eager_debug_test",
    srcs = ["eager_debug_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_test(
    name = "async_transfer_test",
    srcs = ["async_transfer_test.py"],
    main = "async_transfer_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "experimental_collectives_test",
    srcs = ["experimental_collectives_test.py"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "autoload_test_tpu",
    srcs = ["autoload_tpu_test.py"],
    main = "autoload_tpu_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "import_test",
    size = "medium",
    srcs = ["import_test.py"],
    main = "import_test.py",
    shard_count = 4,
    tags = [],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "tracebacks_precedence_test",
    srcs = ["tracebacks_precedence_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/export",
    ],
)

torch_tpu_py_test(
    name = "amp_test",
    srcs = ["amp_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",  # build_cleaner: keep
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_test(
    name = "tpu_backend_config_test",
    srcs = ["tpu_backend_config_test.py"],
    # To exercise global APIs that are meant to be set once, keep `shard_count` identical to
    # the number of test cases so that each process runs exactly one test case.
    # TODO: b/517257494 - write test properly to run one test case per process without
    # relying on shard count.
    shard_count = 2,
    strict = True,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",  # build_cleaner: keep
        "//src/torch_tpu/_internal/execution_mode",
    ],
)

test_suite(
    name = "autoload_tests",
    tests = [
        "autoload_test_tpu",
    ],
)

torch_tpu_py_test(
    name = "tracing_test_tpu",
    srcs = ["tracing_test.py"],
    main = "tracing_test.py",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/tracing",
    ],
)

test_suite(
    name = "tracing_tests",
    tests = [
        "tracing_test_tpu",
    ],
)

torch_tpu_py_test(
    name = "batch_transfer_test",
    srcs = ["batch_transfer_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/batch_transfer",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_library(
    name = "native_functions_data",
    testonly = True,
    srcs = ["native_functions_data.py"],
    visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "native_functions_test",
    srcs = ["native_functions_test.py"],
    data = if_oss(
        [],
        ["//shims/torch:aten/src/ATen/RegistrationDeclarations.h"],
    ),
    deps = [
        ":native_functions_data",
        ":seed_test_utils",
        "//csrc/eager:tpu_aten_kernels",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@rules_python//python/runfiles",  # build_cleaner: keep
    ],
)

# Test for ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_regression_tpu",
    torch_example_main = "regression/main.py",
)

# Test for ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_fx_invert_tpu",
    torch_example_main = "fx/invert.py",
)

# Test for ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_fx_module_tracer_tpu",
    torch_example_main = "fx/module_tracer.py",
)

# TODO(b/543942649): Improve pytorch GAT example to of taking in datasets as an argument.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_gat_tpu",
    torch_example_args = "--dry-run",
    torch_example_main = "gat/main.py",
    deps = DEFAULT_TORCH_EXAMPLE_DEPS + [
        "@pypi//numpy",
        "@pypi//requests",
    ],
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_dcgan_tpu",
    torch_example_args = "--dataset=fake --dry-run --accel --batchSize=4 --workers=0",
    torch_example_main = "dcgan/main.py",
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_mnist_tpu",
    torch_example_args = "--epochs=1 --dry-run --batch-size=4 --test-batch-size=4",
    torch_example_main = "mnist/main.py",
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_mnist_forward_forward_tpu",
    torch_example_args = "--epochs=1 --train_size=16 --test_size=16",
    torch_example_main = "mnist_forward_forward/main.py",
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_siamese_network_tpu",
    torch_example_args = "--epochs=1 --dry-run --batch-size=4 --test-batch-size=4",
    torch_example_main = "siamese_network/main.py",
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_vae_tpu",
    torch_example_args = "--epochs=1 --batch-size=4",
    torch_example_main = "vae/main.py",
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_super_resolution_tpu",
    torch_example_args = "--upscale_factor=3 --batchSize=4 --testBatchSize=10 --nEpochs=1 --lr=0.001 --accel --threads=0",
    torch_example_main = "super_resolution/main.py",
    deps = DEFAULT_TORCH_EXAMPLE_DEPS + [
        "@pypi//six",
    ],
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_imagenet_tpu",
    torch_example_args = "--pretrained --batch-size=4 --epochs=1 --workers=0",
    torch_example_main = "imagenet/main.py",
)

# Test to ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
torch_tpu_oss_torch_example(
    name = "run_upstream_example_fast_neural_style_tpu",
    torch_example_args = "train --epochs=1 --batch-size=4 --dataset=/tmp/gcsfuse/data/fast_neural_style --save-model-dir=saved_models --accel --log-interval=1",
    torch_example_main = "fast_neural_style/neural_style/neural_style.py",
)

torch_tpu_py_test(
    name = "scan_test",
    srcs = ["scan_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:debug",
        "//src/torch_tpu/_internal/compile:scan",
        "//src/torch_tpu/_internal/compile:tpu_torch_compile",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "lstm_test",
    srcs = ["lstm_test.py"],
    shard_count = 4,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "gru_test",
    srcs = ["gru_test.py"],
    shard_count = 4,
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "fused_adagrad_test",
    srcs = ["fused_adagrad_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "fused_sgd_test",
    srcs = ["fused_sgd_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "empty_test",
    srcs = ["empty_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
    ],
)

torch_tpu_py_test(
    name = "tensor_factories_test",
    srcs = ["tensor_factories_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "fused_moving_avg_obs_fq_helper_test",
    srcs = ["fused_moving_avg_obs_fq_helper_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "large_invert_non_strided_slice_test",
    srcs = ["large_invert_non_strided_slice_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_library(
    name = "run_upstream_example",
    srcs = ["run_upstream_example.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "pybind_error_utils_test",
    srcs = ["pybind_error_utils_test.py"],
    deps = [
        ":seed_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
    ],
)

# Paths in the TorchTPU repository, and its root in runfiles and in the workspace
# of `blaze run`.
torch_tpu_py_library(
    name = "repo_paths",
    srcs = ["repo_paths.py"],
    visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "cpp_style_lib",
    testonly = True,
    srcs = ["cpp_style_lib.py"],
    visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "cpp_style_test",
    srcs = ["cpp_style_test.py"],
    data = [
        ":all_cpp_files_recursive",
        "//csrc:all_cpp_files_recursive",
    ],
    deps = [
        ":cpp_style_lib",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "cpp_namespace_test",
    srcs = ["cpp_namespace_test.py"],
    data = [
        ":all_cpp_files_recursive",
        "//csrc:all_cpp_files_recursive",
    ],
    deps = [
        ":cpp_style_lib",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "header_guard_lint_test",
    srcs = ["header_guard_lint_test.py"],
    data = [
        ":all_cpp_files_recursive",
        "//csrc:all_cpp_files_recursive",
    ],
    deps = [
        ":cpp_style_lib",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//src/torch_tpu",
    ],
)

torch_tpu_package_end()
