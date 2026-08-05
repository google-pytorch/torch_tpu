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
    "if_oss",
    "oss_target",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
    "tpu_gen",
)

package(
    default_applicable_licenses = ["//:license"],
)

torch_tpu_py_library(
    name = "oss_utils",
    testonly = True,
    srcs = ["oss_utils.py"],
    visibility = ["//:__subpackages__"],
    deps = if_oss(
        ["//shims/libtpu"],
        [],
    ),
)

torch_tpu_py_test(
    name = "oss_utils_test",
    srcs = ["oss_utils_test.py"],
    deps = [
        ":oss_utils",
        "//shims/absl_py/testing:absltest",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "_versioned_so_loader_test",
    srcs = ["_versioned_so_loader_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
        "//torch_tpu:_versioned_so_loader",
    ],
)

torch_tpu_py_library(
    name = "module_registry",
    srcs = ["module_registry.py"],
    data = [
        "//examples/huggingface_diffusers:model_configs_data",
        "//examples/huggingface_transformers:model_configs_data",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/etils/epath",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "@pypi//diffusers",  # build_cleaner: keep
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
        "//torch_tpu",
        "@pypi//pillow",
    ],
)

torch_tpu_py_library(
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
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/compiler_options",
        "//torch_tpu/_internal/device:_device_ops_backend",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/utils",
    ],
)

# We have multiple targets for ops_test.py:
# * ops_test: TPU vs GPU (by default, we test against GPU).
# * ops_test_gen_gpu_golden: generates the golden file for the GPU backend.
# * ops_test_grad_vs_cpu: gradient on TPU vs CPU.
# * ops_test_perf: TPU only, for measuring compilation/execution time.
# * ops_test_dynamic_vs_cpu: TPU vs GPU, bounds a dimension.
# * ops_test_compiled: TPU vs GPU (torch.compile TpuBackend vs GPU inductor backend).

# We chose to manually maintain the duplicate tests instead of using a for-loop to generate the
# targets because:
#
# - The for-loop approach is incompatible with build_cleaner.
# - The for-loop approach breaks the Cider-torch_tpu_py_test integration (in particular,
#   it breaks running a single test case directly in the Cider UI).

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "ops_test_grad_vs_cpu",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=-foreach,-nonzero,-searchsorted",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 35,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "foreach_ops_test_grad_vs_cpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=foreach",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 15,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "nonzero_ops_test_grad_vs_cpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=nonzero",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 1,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "searchsorted_ops_test_grad_vs_cpu",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=searchsorted",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 1,
    tags = [
        "noasan",  # TODO: fix the test timing out in asan mode.
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "ops_test_grad_v6e_vs_cpu",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=-foreach,-nonzero,-searchsorted",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    shard_count = 12,
    tags = [
        # Specifically designed to test v6e, mark as failing on other hardware.
        "fails-on-tpu-v5",
        "fails-on-tpu-v7",
        "nosan",  # Rely on ops_test_grad for *san coverage.
        # Specifically tests TPU v6e behaviors in postsubmit. Do not change to
        # other TPU versions.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "foreach_ops_test_grad_v6e_vs_cpu",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=foreach",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    shard_count = 10,
    tags = [
        # Specifically designed to test v6e, mark as failing on other hardware.
        "fails-on-tpu-v5",
        "fails-on-tpu-v7",
        "nosan",  # Rely on ops_test_grad for *san coverage.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "nonzero_ops_test_grad_v6e_vs_cpu",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=nonzero",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    shard_count = 1,
    tags = [
        # Specifically designed to test v6e, mark as failing on other hardware.
        "fails-on-tpu-v5",
        "fails-on-tpu-v7",
        "nosan",  # Rely on ops_test_grad for *san coverage.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch this to TPU-vs-GPU.
    name = "searchsorted_ops_test_grad_v6e_vs_cpu",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--compute_grad=True",
        "--test_mode=torch_tpu_vs_cpu",
        "--test_categories=searchsorted",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    shard_count = 1,
    tags = [
        # Specifically designed to test v6e, mark as failing on other hardware.
        "fails-on-tpu-v5",
        "fails-on-tpu-v7",
        "nosan",  # Rely on ops_test_grad for *san coverage.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_compiled",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--use_compiled=True",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden_compiled0.gz",
        "ops_test_gpu_golden_compiled1.gz",
        "ops_test_gpu_golden_compiled2.gz",
    ],
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
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
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
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 10,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "nonzero_ops_test",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=nonzero",
    ] + if_oss(
        # Reduce samples to avoid timeouts in OSS.
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 1,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.  # This test times out in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        ["--max_samples_per_op_dtype=16"],
        # Forge has more resources, allowing a larger sample size.
        ["--max_samples_per_op_dtype=20"],
    ),
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit_oss = "This slow test is enabled in nightly.",
    shard_count = 1,
    tags = [
        "noasan",  # This test times out in asan mode.
        "nomsan",  # This test times out in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_test_v6e",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=-foreach,-nonzero,-searchsorted",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    notest_oss = "The test is specifically for TPU v6e, but OSS build doesn't allow specifying the TPU version yet.",
    shard_count = 25,
    tags = [
        "nosan",  # Rely on ops_test_tpu_vs_gpu for *san coverage.
        # Specifically tests TPU v6e behaviors in postsubmit. Do not change to
        # other TPU versions.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "foreach_ops_test_v6e",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=foreach",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    notest_oss = "The test is specifically for TPU v6e, but OSS build doesn't allow specifying the TPU version yet.",
    shard_count = 10,
    tags = [
        "nosan",  # Rely on ops_test_tpu_vs_gpu for *san coverage.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "nonzero_ops_test_v6e",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=nonzero",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    notest_oss = "The test is specifically for TPU v6e, but OSS build doesn't allow specifying the TPU version yet.",
    shard_count = 1,
    tags = [
        "nosan",  # Rely on ops_test_tpu_vs_gpu for *san coverage.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "searchsorted_ops_test_v6e",
    timeout = "long",
    srcs = ["ops_test.py"],
    args = [
        "--test_categories=searchsorted",
        "--max_samples_per_op_dtype=20",  # Prevent long test run time.
    ],
    data = [
        "ops_test_gpu_golden0.gz",
        "ops_test_gpu_golden1.gz",
        "ops_test_gpu_golden2.gz",
    ],
    main = "ops_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    notest_oss = "The test is specifically for TPU v6e, but OSS build doesn't allow specifying the TPU version yet.",
    shard_count = 1,
    tags = [
        "nosan",  # Rely on ops_test_tpu_vs_gpu for *san coverage.
        "requires-tpu-v6e",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    # TODO: switch to TPU-vs-GPU mode.
    name = "ops_test_dynamic_vs_cpu",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_test.py"],
    args = [
        "--check_dynamism_using_seed=-1",
        "--test_mode=torch_tpu_vs_cpu",
        "--print_op_inputs=1",
        "--max_samples_per_op_dtype=1",  # No need to test many samples to catch dynamism bugs.
    ],
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
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
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
    notap = "slow test - only need to run on demand for updating goldens",
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
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
    nobuild = "TODO(b/540823026): re-enable once the test build hang is fixed.",
    nolocal = "slow test",
    notap = "TODO(b/540823026): re-enable once the test build hang is fixed.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    shard_count = 3,
    tags = [
        "requires-gpu-sm80",
    ],
    deps = [
        ":dynamism_test_utils",
        ":op_testing",
        ":oss_utils",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "ops_unit_test",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    main = "ops_unit_test.py",
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//scipy",
    ],
)

# Runs the ops unit tests on v6e TPU in postsubmit to catch v6e-specific
# issues.
torch_tpu_py_test(
    name = "ops_unit_test_v6e",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    main = "ops_unit_test.py",
    nopresubmit = "Long Forge wait times for v6e. Run in postsubmit only.",
    shard_count = 20,
    tags = [
        "nosan",  # Rely on ops_unit_test for *san coverage.
        # Specifically tests TPU v6e behaviors in postsubmit. Do not change to
        # other TPU versions.
        "requires-tpu-v6e",
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//scipy",
    ],
)

torch_tpu_py_test(
    name = "sparse_dense_matmul_test",
    srcs = ["sparse_dense_matmul_test.py"],
    main = "sparse_dense_matmul_test.py",
    notap = "Subslicing errors on tpu7x. See b/515767150",
    notest_oss = "Disabled because of no support for tpu v7 in presubmit",
    tags = [
        "requires-tpu-v7x",
    ],
    deps = [
        ":op_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:hardware",
        "//torch_tpu/ops/experimental/sparse_dense_matmul:preprocessing_config_py_pb2",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "eager_device_assignment_test",
    srcs = ["eager_device_assignment_test.py"],
    main = "eager_device_assignment_test.py",
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

# Tests the tier-3 compilation cache.
torch_tpu_py_test(
    name = "ops_unit_test_tier3_cache",
    timeout = "long",  # Needed for asan.
    srcs = ["ops_unit_test.py"],
    args = ["--test_mode=torch_tpu_vs_cpu"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
    },
    main = "ops_unit_test.py",
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/my_cache",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
    },
    main = "ops_unit_test.py",
    shard_count = 20,
    tags = [
        "nomsan",  # TODO: fix the test in msan mode.
        "notsan",  # TODO: fix the test in tsan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":op_testing",
        ":ops_test_data",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:flagsaver",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compiler_options",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "index_put_op_test",
    srcs = ["index_put_op_unit_test.py"],
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//torch_tpu",
        "//torch_tpu:_loader",
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
        "//torch_tpu",
        "//torch_tpu:_loader",
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
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//torch_tpu",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//numpy",
    ],
)

torch_tpu_py_test(
    name = "fp4_test",
    srcs = ["fp4_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_library(
    name = "error_testing",
    testonly = True,
    srcs = ["error_testing.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",  # build_cleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//torch_tpu",
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
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "tpu_errors_test.py",
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

# Collects error coverage stats without verifying the error messages.
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
    notest_oss = "Intended not to run this slow test in OSS. It's low value to test the error messages in both OSS and internal builds.",
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

# Collects error coverage stats without verifying the error messages.
torch_tpu_py_test(
    name = "tpu_errors_test_cov",
    srcs = ["tpu_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS": "1",
    },
    main = "tpu_errors_test.py",
    notest_oss = "Intended not to run this slow test in OSS.",
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/pallas",
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "errors_test_gpu",
    srcs = ["errors_test.py"],
    args = ["--test_mode=gpu"],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "0",
        # Force synchronous CUDA kernel launches so device-side errors trigger
        # immediately at the exact Python op call instead of asynchronously.
        "CUDA_LAUNCH_BLOCKING": "1",
    },
    main = "errors_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    notest_oss = "Low value in maintaining for both internal and OSS and they run different versions of PyTorch.",
    shard_count = 4,
    tags = [
        "nosan",  # *san builds are low-value for error tests.
        "requires-gpu-nvidia",
        "wheel_test_excluded",
    ],
    deps = [
        ":error_testing",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:env",
        "//torch_tpu/_internal/compile:tpu_torch_compile",  # build_cleaner: keep
        "//torch_tpu/_internal/pallas:tpu_torch_pallas",  # build_cleaner: keep
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
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
        "//torch_tpu",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
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
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/utils:test_utils",
        "//torch_tpu/eager:tpu_aten_kernels",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/eager:tpu_aten_kernels",
    ],
)

torch_tpu_py_test(
    name = "rng_test",
    srcs = ["rng_test.py"],
    main = "rng_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "synchronize_test",
    srcs = ["synchronize_test.py"],
    main = "synchronize_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
    ],
)

torch_tpu_py_test(
    name = "adamw_test",
    srcs = ["adamw_test.py"],
    main = "adamw_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/optim",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "patch_test",
    srcs = ["patch_test.py"],
    main = "patch_test.py",
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/optim",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "compilation_cache_stress_test",
    srcs = ["compilation_cache_stress_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
    ],
)

torch_tpu_py_test(
    name = "rng_stress_test",
    srcs = ["rng_stress_test.py"],
    notest_oss = "timeout in oss",
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
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
        "//torch_tpu",
        "//torch_tpu/_internal/dynamism",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/precision",
        "//torch_tpu/_internal/utils",
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
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "autoload_test_gpu",
    srcs = ["autoload_gpu_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    main = "autoload_gpu_test.py",
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "autoload_test_cpu",
    srcs = ["autoload_cpu_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    main = "autoload_cpu_test.py",
    tags = [],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "autoload_disabled_backend_test_gpu",
    srcs = ["autoload_disabled_backend_test.py"],
    main = "autoload_disabled_backend_test.py",
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "autoload_disabled_backend_test_cpu",
    srcs = ["autoload_disabled_backend_test.py"],
    main = "autoload_disabled_backend_test.py",
    tags = [],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

torch_tpu_py_test(
    name = "tracebacks_precedence_test",
    srcs = ["tracebacks_precedence_test.py"],
    args = [
        "--torch_tpu_internal_mlir_tracebacks=true",
    ],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    notest_oss = "Absl flags don't work in OSS",
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/export",
    ],
)

torch_tpu_py_test(
    name = "amp_test",
    srcs = ["amp_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/execution_mode",
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
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",  # build_cleaner: keep
        "//torch_tpu/_internal/execution_mode",
    ],
)

test_suite(
    name = "autoload_tests",
    tests = [
        "autoload_disabled_backend_test_cpu",
        "autoload_disabled_backend_test_gpu",
        "autoload_test_cpu",
        "autoload_test_gpu",
        "autoload_test_tpu",
    ],
)

torch_tpu_py_test(
    name = "tracing_test_cpu",
    srcs = ["tracing_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    main = "tracing_test.py",
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/tracing",
    ],
)

torch_tpu_py_test(
    name = "tracing_test_gpu",
    srcs = ["tracing_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1",
    },
    main = "tracing_test.py",
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/tracing",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/stream_executor:cuda_platform",
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
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/tracing",
    ],
)

test_suite(
    name = "tracing_tests",
    tests = [
        "tracing_test_cpu",
        "tracing_test_gpu",
        "tracing_test_tpu",
    ],
)

torch_tpu_py_test(
    name = "batch_transfer_test",
    srcs = ["batch_transfer_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/batch_transfer",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@rules_python//python/runfiles",  # build_cleaner: keep
    ],
)

# TODO: b/523312531 - Evaluate enabling these upstream tests in Google3 for internal developer
# parity. Currently, this is restricted to OSS CI via `oss_target` because the test runner relies on
# the raw PyTorch source tree checked out manually by the GitHub Actions workflow.
oss_target(
    torch_tpu_py_test,
    name = "run_upstream_test_tpu",
    srcs = ["run_upstream_test.py"],
    env = {"TORCH_TPU_INTERNAL_TEST_DEVICE": "tpu"},
    main = "run_upstream_test.py",
    nolocal = "This test requires downloading the raw PyTorch source tree. Exclude from wildcard runs.",
    nopresubmit_oss = "This test requires downloading the raw PyTorch source tree. Relegate to nightly to avoid adding flakiness/latency to every PR.",
    tags = [
        "requires-tpu-v5lite",
        # This test requires downloading the raw PyTorch source tree. Relegate to nightly to avoid
        # adding flakiness/latency to wheel tests.
        "wheel_test_excluded",
    ],
    deps = [
        "//shims/torch:testing_internal",  # build_cleaner: keep
        "//torch_tpu",
        "@pypi//pytest",
    ],
)

# See TODO above regarding Google3 parity and out-of-band PyTorch source tree dependencies.
oss_target(
    torch_tpu_py_test,
    name = "run_upstream_test_cpu",
    srcs = ["run_upstream_test.py"],
    env = {"TORCH_TPU_INTERNAL_TEST_DEVICE": "cpu"},
    main = "run_upstream_test.py",
    nolocal = "This test requires downloading the raw PyTorch source tree. Exclude from wildcard runs.",
    nopresubmit_oss = "This test requires downloading the raw PyTorch source tree. Relegate to nightly to avoid adding flakiness/latency to every PR.",
    tags = [
        # TODO: b/523384731 - Remove the "local" tag once upstream PyTorch is migrated natively into
        # Bazel's dependency graph via Bzlmod.
        # Ignores remote executors and forces local execution to avoid FileNotFound error as Bazel
        # refuses to upload the downloaded PyTorch source code to the RBE Cloud workers.
        "local",
        # This test requires downloading the raw PyTorch source tree. Relegate to nightly to avoid
        # adding flakiness/latency to wheel tests.
        "wheel_test_excluded",
    ],
    deps = [
        "//shims/torch:testing_internal",  # build_cleaner: keep
        "//torch_tpu",
        "@pypi//pytest",
    ],
)

# Test for ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
oss_target(
    torch_tpu_py_test,
    name = "run_upstream_example_regression_tpu",
    srcs = ["run_upstream_example.py"],
    env = {
        "TORCH_TPU_INTERNAL_EXAMPLE_PATH": "regression/main.py",
        "TORCH_TPU_INTERNAL_TEST_DEVICE": "tpu",
    },
    main = "run_upstream_example.py",
    nolocal = "This test requires downloading the raw PyTorch examples tree.",
    nonightly_oss = "Test requires downloading raw PyTorch examples.",
    nopresubmit_oss = "This test requires downloading the raw PyTorch examples tree.",
    tags = [
        "requires-tpu-v5lite",
        "wheel_test_excluded",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

# Test for ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
oss_target(
    torch_tpu_py_test,
    name = "run_upstream_example_fx_invert_tpu",
    srcs = ["run_upstream_example.py"],
    env = {
        "TORCH_TPU_INTERNAL_EXAMPLE_PATH": "fx/invert.py",
        "TORCH_TPU_INTERNAL_TEST_DEVICE": "tpu",
    },
    main = "run_upstream_example.py",
    nolocal = "This test requires downloading the raw PyTorch examples tree.",
    nonightly_oss = "Test requires downloading raw PyTorch examples.",
    nopresubmit_oss = "This test requires downloading the raw PyTorch examples tree.",
    tags = [
        "requires-tpu-v5lite",
        "wheel_test_excluded",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
    ],
)

# Test for ensure pytorch example works with torch_tpu. It functions as an integration test and as a
# documentation regression test. Tests downloading the raw PyTorch examples tree from GitHub.
# Therefore, this test is OSS only.
oss_target(
    torch_tpu_py_test,
    name = "run_upstream_example_fx_module_tracer_tpu",
    srcs = ["run_upstream_example.py"],
    env = {
        "TORCH_TPU_INTERNAL_EXAMPLE_PATH": "fx/module_tracer.py",
        "TORCH_TPU_INTERNAL_TEST_DEVICE": "tpu",
    },
    main = "run_upstream_example.py",
    nolocal = "This test requires downloading the raw PyTorch examples tree.",
    nonightly_oss = "Test requires downloading raw PyTorch examples.",
    nopresubmit_oss = "This test requires downloading the raw PyTorch examples tree.",
    tags = [
        "requires-tpu-v5lite",
        "wheel_test_excluded",
    ],
    deps = [
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "scan_test",
    srcs = ["scan_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        ":oss_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/utils:test_utils",
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
        "//torch_tpu",
        "//torch_tpu:_loader",
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
        "//torch_tpu",
        "//torch_tpu:_loader",
    ],
)

torch_tpu_py_test(
    name = "empty_test",
    srcs = ["empty_test.py"],
    tags = ["requires-tpu-v5lite"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/sync",
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
        "//torch_tpu",
        "//torch_tpu:_loader",
    ],
)

torch_tpu_py_test(
    name = "run_upstream_test",
    srcs = ["run_upstream_test.py"],
    notap = "Requires external PyTorch test directory",
    deps = [
        "//torch_tpu",
        "@pypi//pytest",
    ],
)

torch_tpu_package_end()
