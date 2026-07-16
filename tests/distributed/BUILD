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
    "if_oss",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "distributed_utils",
    testonly = True,
    srcs = [
        "distributed_utils.py",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//portpicker",
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
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "multi_tpu_test",
    srcs = ["multi_tpu_test.py"],
    autoload = True,
    notest_oss = "fix hang in OSS",  # TODO
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
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "collective_ops_test",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/collective_ops_test",
    },
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 12,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the timeout in test_broadcast_objects.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

# Like collective_ops_test, but uses a remote filesystem for the tier-3 cache.
torch_tpu_py_test(
    name = "collective_ops_remote_cache_test",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
        # Alternatively, set it to a GCS directory.
        # "$DATA_ROOT/$TMP_SUBDIR/ttl=1d/collective_ops_remote_cache_test"
        "",
    },
    main = "collective_ops_test.py",
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 12,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/495533335): fix out-of-order in msan mode.
        "notsan",  # TODO(b/495533335): fix out-of-order in tsan mode.
        "requires-net:external",  # Needs remote file access.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "collective_errors_test",
    srcs = ["collective_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 12,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "nosan",  # *san has low value for error tests.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "collective_errors_test_cov",
    srcs = ["collective_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "1"},
    main = "collective_errors_test.py",
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 12,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "alltoall_collective_test",
    srcs = ["alltoall_collective_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 4,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-accel-hbm",
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "alltoall_collective_errors_test",
    srcs = ["alltoall_collective_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 8,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "alltoall_collective_errors_test_cov",
    srcs = ["alltoall_collective_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "1"},
    main = "alltoall_collective_errors_test.py",
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 8,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "all_gather_errors_test",
    srcs = ["all_gather_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 3,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "all_gather_errors_test_cov",
    srcs = ["all_gather_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "1"},
    main = "all_gather_errors_test.py",
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 3,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "scatter_test",
    srcs = ["scatter_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
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
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "scatter_errors_test",
    srcs = ["scatter_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 3,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "scatter_errors_test_cov",
    srcs = ["scatter_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "1"},
    main = "scatter_errors_test.py",
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 3,
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
        "//tests:error_testing",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
    ],
)

torch_tpu_py_test(
    name = "subgroup_collectives_test",
    srcs = ["subgroup_collectives_test.py"],
    env = if_oss(
        {"TORCH_TPU_TOPOLOGY": "2,4,1"},
        {},
    ),
    shard_count = 4,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
        # Skip this test on TPU v6 and v7. It was written targeting TPU v5. Running it on v6 and v7
        # will cause issues due to their different topology.
        "fails-on-tpu-v6",
        "fails-on-tpu-v7",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "dtensor_test",
    srcs = ["dtensor_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 10,
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
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "fairscale_tp_test",
    srcs = ["fairscale_tp_test.py"],
    args = [
        "--xla_jf_conv_full_precision",  # Use highest precision for matmuls for tests only.
    ],
    main = "fairscale_tp_test.py",
    notest_oss = "fix hang in OSS",  # TODO
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
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "@pypi//fairscale",
    ],
)

torch_tpu_py_test(
    name = "dcp_test",
    srcs = ["dcp_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 2,
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
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "ragged_dot_test",
    srcs = ["ragged_dot_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
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
        "//torch_tpu",
        "//torch_tpu/_internal/compile",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_py_test(
    name = "ragged_all_to_all_test",
    srcs = ["ragged_all_to_all_test.py"],
    nopresubmit = "Requires 4 tpu v7x chips.",
    tags = [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v7x:4",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
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
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:spmd_util",
    ],
)

torch_tpu_py_test(
    name = "spmd_util_collective_test",
    srcs = ["spmd_util_collective_test.py"],
    tags = if_oss(
        # In OSS, we have to run the test exclusively to avoid the
        # "TPU initialization failed ... Device or resource busy" error.
        ["exclusive"],
        [],
    ) + [
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/distributed:spmd_util",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/sync",
    ],
)

torch_tpu_py_test(
    name = "split_collective_ops_test",
    srcs = ["split_collective_ops_test.py"],
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/compile:compiler",
        "//torch_tpu/_internal/compile:split_compiler",
        "//torch_tpu/_internal/compile:torch_tpu_compiled_executable",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
    ],
)

torch_tpu_package_end()
