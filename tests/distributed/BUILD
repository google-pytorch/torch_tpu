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
# basioli-k
# cbasile-g
# zhanyong-wan
# jialei-chen
# yilingyuan
# ghpvnist
# chizhang529

load(
    "//build_files:build_defs.bzl",
    "REQUIRES_V5_MULTI_DEVICE",
    "TEST_COV_ENV",
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

torch_tpu_py_library(
    name = "continuous_test_utils",
    testonly = True,
    srcs = ["continuous_test_utils.py"],
    deps = [
        ":distributed_utils",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "continuous_test_utils_test",
    srcs = ["continuous_test_utils_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":continuous_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "distributed_utils",
    testonly = True,
    srcs = [
        "distributed_utils.py",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal:env_utils",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "distributed_utils_test",
    srcs = ["distributed_utils_test.py"],
    shard_count = 5,
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "distributed_utils_tpu_test",
    srcs = ["distributed_utils_tpu_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "scan_collectives_test",
    timeout = "long",
    srcs = ["scan_collectives_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":continuous_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "sparse_core_collectives_test",
    timeout = "long",
    srcs = ["sparse_core_collectives_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:spmd_util",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:oss_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "compile_cache_process_groups_test",
    timeout = "long",
    srcs = ["compile_cache_process_groups_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:spmd_util",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "single_tpu_pg_test",
    srcs = ["single_tpu_pg_test.py"],
    tags = [
        # Skip this test on tpu v7 runners: 'RuntimeError: expected exactly one TPU device
        # per PyTorch process.' occurs because each TPU7x chip contains two addressable TensorCores,
        # but manual process group init expects exactly 1 TPU device per process.
        "fails-on-tpu-v7",
        "noasan",  # TODO(b/555770497): fix the test in asan mode.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:tpu_distributed",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "torchcomm_tpu_test",
    srcs = ["torchcomm_tpu_test.py"],
    nobuild_oss = True,  # torchcomms is not yet available in OSS
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = if_oss(
        [],
        ["@pypi//torchcomms"],
    ) + [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:torchcomm_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "torchcomm_multi_tpu_test",
    srcs = ["torchcomm_multi_tpu_test.py"],
    nobuild_oss = True,  # torchcomms is not yet available in OSS
    shard_count = 7,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = if_oss(
        [],
        ["@pypi//torchcomms"],
    ) + [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:torchcomm_tpu",
    ],
)

torch_tpu_py_test(
    name = "multi_tpu_test",
    srcs = ["multi_tpu_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:tpu_distributed",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "collective_ops_test",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/collective_ops_test",
    },
    shard_count = 12,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the timeout in test_broadcast_objects.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

# Like collective_ops_test, but uses a remote filesystem for the tier-3 cache.
torch_tpu_py_test(
    name = "collective_ops_remote_cache_test",
    timeout = "long",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "gs://torchtpu-shared/collective_ops_remote_cache_test",
    },
    main = "collective_ops_test.py",
    shard_count = 12,
    tags = if_oss(
        [
            "exclusive",
            # TODO: b/567928058 - RBE workers lack write access to the
            # gs://torchtpu-shared cache root used by this test.
            "no-remote-exec",
        ],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/495533335): fix out-of-order in msan mode.
        "notsan",  # TODO(b/495533335): fix out-of-order in tsan mode.
        "requires-net:external",  # Needs remote file access.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

# The device-order regression test needs a multi-core-per-chip topology (v7x),
# where rank order and device order differ. It is kept out of
# `collective_ops_test` so that target can stay on v5e (`requires-tpu-v5lite:8`).
torch_tpu_py_test(
    name = "collective_ops_device_order_test",
    srcs = ["collective_ops_device_order_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

# Guards the interaction between a Pallas kernel's Mesh and the device
# addressing of ordinary collectives. Lives here rather than in tests/pallas
# because the subject is distributed device addressing (the Pallas kernel is
# only the vehicle that installs a non-default DeviceAssignment), because it
# shares the multi-core TPU v7x topology and rank-order-vs-device-order thesis
# with `collective_ops_device_order_test` above, and because tests/pallas
# already depends on `:distributed_utils` here -- putting it there would add a
# dependency edge in the opposite direction.
torch_tpu_py_test(
    name = "pallas_mesh_collective_device_order_test",
    timeout = "long",
    srcs = ["pallas_mesh_collective_device_order_test.py"],
    env = {
        # The kernel issues a remote DMA; matches
        # //tests/pallas:pallas_communication_test.
        "TPU_PREMAPPED_BUFFER_SIZE": "0",
    },
    nopresubmit_oss = "TODO(b/568095515): Flaking in GitHub TPU v7 presubmit with START_SESSION failed.",
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Pallas mesh collective device order test for TPU v7",
    ),
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "fails-on-tpu-v5",
        "fails-on-tpu-v6",
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        # On single-core-per-chip parts rank order equals device order, so the
        # Mesh installs the default DeviceAssignment and nothing here can
        # detect a device-order fault. `_require_non_identity_device_order`
        # fails the run there rather than let it go green.
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":continuous_test_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/jax:pallas",
        "//shims/jax:pallas_tpu",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:device_mesh",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:tpu_distributed",
        "//src/torch_tpu/_internal/pallas",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "collective_errors_test",
    timeout = "long",
    srcs = ["collective_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "nosan",  # *san has low value for error tests.
    ],
    deps = [
        ":continuous_test_utils",
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/pallas",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:error_testing",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "collective_errors_test_cov",
    timeout = "long",
    srcs = ["collective_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = TEST_COV_ENV,
    main = "collective_errors_test.py",
    notap = "Run by error_coverage_test in the torch_tpu.error_coverage Guitar workflow.",
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":continuous_test_utils",
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/pallas",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:error_testing",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "alltoall_collective_test",
    srcs = ["alltoall_collective_test.py"],
    shard_count = 8,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-accel-hbm",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "fp8_collectives_test",
    srcs = ["fp8_collectives_test.py"],
    shard_count = 8,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
    ],
)

# Same tests on TPU v7x, the FP8 target. There each chip has two TensorCores and
# uneven all-to-all runs on SparseCore.
torch_tpu_py_test(
    name = "fp8_collectives_v7x_test",
    srcs = ["fp8_collectives_test.py"],
    main = "fp8_collectives_test.py",
    nopresubmit = "Same tests as :fp8_collectives_test, which runs in presubmit; TPU v7x has long Forge queues.",
    notest_oss = True,  # NOTEST_OSS_OK=Disabled in OSS due to multi-process v7x runner hang.
    shard_count = 4,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "fails-on-tpu-v5",
        "fails-on-tpu-v6",
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "all_gather_errors_test",
    srcs = ["all_gather_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "nosan",  # *san builds are low-value for error tests.
    ],
    deps = [
        ":continuous_test_utils",
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:error_testing",
    ],
)

torch_tpu_py_test(
    name = "all_gather_errors_test_cov",
    srcs = ["all_gather_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = TEST_COV_ENV,
    main = "all_gather_errors_test.py",
    notap = "Run by error_coverage_test in the torch_tpu.error_coverage Guitar workflow.",
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":continuous_test_utils",
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:error_testing",
    ],
)

torch_tpu_py_test(
    name = "scatter_test",
    srcs = ["scatter_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":continuous_test_utils",
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "scatter_errors_test",
    srcs = ["scatter_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {
        # Disables addr2line C++ stack trace symbolization to prevent deadlocks/hangs during
        # exception construction in multi-process distributed tests in OSS CI.
        "TORCH_DISABLE_ADDR2LINE": "1",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
    },
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "nosan",  # *san builds are low-value for error tests.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:error_testing",
    ],
)

torch_tpu_py_test(
    name = "scatter_errors_test_cov",
    srcs = ["scatter_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = TEST_COV_ENV,
    main = "scatter_errors_test.py",
    notap = "Run by error_coverage_test in the torch_tpu.error_coverage Guitar workflow.",
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "nosan",  # *san builds are low-value for error tests.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:error_testing",
    ],
)

torch_tpu_py_test(
    name = "subgroup_collectives_test",
    timeout = "long",
    srcs = ["subgroup_collectives_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":continuous_test_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:device_mesh",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "mixed_group_programs_test",
    timeout = "long",
    srcs = ["mixed_group_programs_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    shard_count = 3,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "subgroup_launch_order_test",
    timeout = "long",
    srcs = ["subgroup_launch_order_test.py"],
    oss_presubmit_tpu_generation = tpu_gen(
        "v7",
        reason = "Only multi-core-per-chip TPU v7x has a non-identity rank-to-device mapping",
    ),
    shard_count = 4,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "ok-to-wait-for-v7x-in-presubmit",
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:spmd_util",
        "//src/torch_tpu/_internal/pallas",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "dtensor_test",
    srcs = ["dtensor_test.py"],
    shard_count = 10,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "uneven_sharding_test",
    timeout = "long",
    srcs = ["uneven_sharding_test.py"],
    args = if_oss(
        [],
        [
            "--xla_tpu_collect_sflag_wait_stats_trace=true",
            "--xla_tpu_debug_sflag_wait_shalt_on_detection=true",
            "--xla_tpu_debug_sc_sflag_wait_timeout_ms=10000",
            "--xla_tpu_debug_sflag_wait_timeout_ms=10000",
        ],
    ),
    env = if_oss(
        {
            "LIBTPU_INIT_ARGS": " ".join([
                "--undefok=xla_tpu_debug_sc_sflag_wait_timeout_ms",
                "--xla_tpu_collect_sflag_wait_stats_trace=true",
                "--xla_tpu_debug_sflag_wait_shalt_on_detection=true",
                "--xla_tpu_debug_sc_sflag_wait_timeout_ms=10000",
                "--xla_tpu_debug_sflag_wait_timeout_ms=10000",
            ]),
            "TORCH_DISABLE_ADDR2LINE": "1",
        },
        {},
    ),
    shard_count = 8,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # Disabled due to b/497035517.
        "nomsan",  # Disabled due to b/516945438.
        "notsan",  # Disabled due to b/497035517.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "fairscale_tp_test",
    srcs = ["fairscale_tp_test.py"],
    # TODO(b/549484939): Clean this up once we switch to dynamically linked libtpu.
    # The flag --xla_jf_conv_full_precision is a registered C++ flag parsed via argv to enforce full
    # precision matmuls. In OSS, libtpu is loaded dynamically and the flag is not registered in
    # Python's absl.flags, causing absltest to crash on unknown CLI flags. Thus, we comment it out
    # in OSS and pass it through LIBTPU_INIT_ARGS instead to maintain numerical parity across both
    # environments.
    args = [
        # Use highest precision for matmuls for tests only.
        # copybara:uncomment "--xla_jf_conv_full_precision",
    ],
    env = if_oss(
        {"LIBTPU_INIT_ARGS": "--xla_jf_conv_full_precision=true"},
        {},
    ),
    main = "fairscale_tp_test.py",
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
        "@pypi//fairscale",
    ],
)

torch_tpu_py_test(
    name = "dcp_test",
    srcs = ["dcp_test.py"],
    shard_count = 2,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "ragged_dot_test",
    srcs = ["ragged_dot_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":continuous_test_utils",
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "ragged_all_to_all_test",
    timeout = "long",
    srcs = ["ragged_all_to_all_test.py"],
    shard_count = 3,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "spmd_util_test",
    srcs = ["spmd_util_test.py"],
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
    name = "spmd_util_collective_test",
    srcs = ["spmd_util_collective_test.py"],
    env = {
        # We explicitly disable the compilation cache because we rely on every rank compiling for itself so that we can definitively confirm that every rank produces the same MLIR.
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "",
    },
    # In OSS, we have to run the test exclusively to avoid the
    # "TPU initialization failed ... Device or resource busy" error.
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/555774204): fix the test in asan mode.
        "nomsan",  # TODO(b/555774204): OOM
        "notsan",  # TODO(b/555774204): OOM
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/execution_mode",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "split_collective_ops_test",
    srcs = ["split_collective_ops_test.py"],
    exec_properties = {
        "test.mem": "28g",
    },
    shard_count = 4,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/555734615): fix the test in asan mode.
        "notsan",  # TODO(b/555734615): fix the time-out in tsan mode.
        # Exactly 8 chips are required because the test hardcodes _world_size = 8 to match the
        # topology and avoid uninitialized chips in OSS CI.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile:compiler",
        "//src/torch_tpu/_internal/compile:debug",
        "//src/torch_tpu/_internal/compile:split_compiler",
        "//src/torch_tpu/_internal/compile:torch_tpu_compiled_executable",
        "//src/torch_tpu/_internal/distributed:collective_ops",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "device_mesh_errors_test",
    srcs = ["device_mesh_errors_test.py"],
    tags = [
        "nosan",  # *san builds are low-value for error tests.
    ],
    deps = [
        ":device_mesh_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal/distributed:device_mesh",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "device_mesh_test",
    srcs = ["device_mesh_test.py"],
    deps = [
        ":device_mesh_test_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal/distributed:device_mesh",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_library(
    name = "device_mesh_test_utils",
    testonly = True,
    srcs = ["device_mesh_test_utils.py"],
    deps = [
        "//src/torch_tpu/_internal/distributed:device_mesh",
    ],
)

torch_tpu_py_test(
    name = "handshake_e2e_test",
    srcs = ["handshake_e2e_test.py"],
    # TODO(b/567669731): Remove once _ZMQClient falls back from an unusable ::1.
    # In OSS Docker runners, `localhost` resolves to ::1 first but IPv6 loopback
    # is unusable, and libzmq never falls back to 127.0.0.1.
    env = if_oss(
        {"MASTER_ADDR": "127.0.0.1"},
        {},
    ),
    shard_count = 8,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,  # We need multi-chip TPU because we are running an actual e2e workflow to test handshake integration.
        "noasan",  # TODO(b/497035517)
        "nomsan",  # TODO(b/497035517)
        "notsan",  # TODO(b/497035517)
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/distributed:handshake",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "handshake_errors_test",
    srcs = ["handshake_errors_test.py"],
    tags = [
        "nosan",  # *san builds are low-value for error tests.
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal/distributed:handshake",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
        "@pypi//pyzmq",
    ],
)

torch_tpu_py_test(
    name = "handshake_test",
    srcs = ["handshake_test.py"],
    shard_count = 8,
    tags = [
        "requires-mem:28g",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:env",
        "//src/torch_tpu/_internal/distributed:handshake",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
        "@pypi//pyzmq",
    ],
)

torch_tpu_py_test(
    name = "window_tpu_test",
    srcs = ["window_tpu_test.py"],
    autoload = True,
    notest_oss = True,  # Requires PJRT RawBuffer extension not yet exported in OSS libtpu.
    shard_count = 6,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "process_group_utils_test",
    srcs = ["process_group_utils_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:process_group_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "simple_fsdp_test",
    srcs = ["simple_fsdp_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":continuous_test_utils",
        "//examples/fsdp:simple_fsdp",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_package_end()
