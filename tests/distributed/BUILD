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
        "//src/torch_tpu",
        "//tests:seed_test_utils",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "distributed_utils_test",
    srcs = ["distributed_utils_test.py"],
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
    nobuild_oss = "torchcomms is not yet available in OSS",
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
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal/distributed:torchcomm_tpu",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "torchcomm_multi_tpu_test",
    srcs = ["torchcomm_multi_tpu_test.py"],
    nobuild_oss = "torchcomms is not yet available in OSS",
    shard_count = 7,
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
    ],
)

torch_tpu_py_test(
    name = "multi_tpu_test",
    srcs = ["multi_tpu_test.py"],
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
        "//src/torch_tpu/_internal/distributed:tpu_distributed",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the timeout in test_broadcast_objects.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:oss_utils",
        "//tests:seed_test_utils",
    ],
)

# Like collective_ops_test, but uses a remote filesystem for the tier-3 cache.
torch_tpu_py_test(
    name = "collective_ops_remote_cache_test",
    timeout = "long",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT":
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
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:oss_utils",
        "//tests:seed_test_utils",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "alltoall_collective_errors_test",
    srcs = ["alltoall_collective_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {"TORCH_SHOW_CPP_STACKTRACES": "0"},
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
        "//tests:seed_test_utils",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
    ],
)

torch_tpu_py_test(
    name = "all_gather_errors_test_cov",
    srcs = ["all_gather_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        # TODO(b/493050035): Use a centralized location to set this flag.
        # Required in OSS: In OSS CI with multiprocess spawn, PyTorch's addr2line symbolizer can
        # crash or hang during exception unwinding and TPU driver cleanup. Disabling addr2line
        # prevents SIGSEGV crashes while still capturing traces.
        "TORCH_DISABLE_ADDR2LINE": "1",
    },
    main = "all_gather_errors_test.py",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal:testing",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "scatter_errors_test",
    srcs = ["scatter_errors_test.py"],
    args = [
        "--test_mode=tpu",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "0",
        # Disables addr2line C++ stack trace symbolization to prevent deadlocks/hangs during
        # exception construction in multi-process distributed tests in OSS CI.
        "TORCH_DISABLE_ADDR2LINE": "1",
    },
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "scatter_errors_test_cov",
    srcs = ["scatter_errors_test.py"],
    args = [
        "--test_mode=cov",
    ],
    env = {
        "TORCH_SHOW_CPP_STACKTRACES": "1",
        # Disables addr2line C++ stack trace symbolization to prevent deadlocks/hangs during
        # exception construction in multi-process distributed tests in OSS CI.
        "TORCH_DISABLE_ADDR2LINE": "1",
    },
    main = "scatter_errors_test.py",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//tests:error_testing",
        "//tests:seed_test_utils",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "uneven_sharding_test",
    srcs = ["uneven_sharding_test.py"],
    args = [
        "--xla_tpu_collect_sflag_wait_stats_trace=true",
        "--xla_tpu_debug_sflag_wait_shalt_on_detection=true",
        "--xla_tpu_debug_sc_sflag_wait_timeout_ms=10000",
        "--xla_tpu_debug_sflag_wait_timeout_ms=10000",
    ],
    notest_oss = "XLA timeout flags not available.",
    shard_count = 8,
    tags = [
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "ragged_all_to_all_test",
    srcs = ["ragged_all_to_all_test.py"],
    nonightly_oss = "This test requires TPU v7. Exclude this test from workflow nightly_tpu.yml running on TPU v6.",
    nopresubmit = "Requires 4 tpu v7x chips.",
    shard_count = 3,
    tags = [
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "nozapfhahn",  # Requires 4 TPU v7x chips and nopresubmit; excluded from Zapfhahn coverage.
        "requires-tpu-v7x:4",
        # Skip this test on TPU v5 and v6. It was written targeting TPU v7. Running it on v5 and v6
        # will cause issues due to their different topology.
        "fails-on-tpu-v5",
        "fails-on-tpu-v6",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "//src/torch_tpu/_internal/distributed:spmd_util",
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
    tags = if_oss(
        # In OSS, we have to run the test exclusively to avoid the
        # "TPU initialization failed ... Device or resource busy" error.
        ["exclusive"],
        [],
    ) + [
        "noasan",  # TODO(b/555774204): fix the test in asan mode.
        "nomsan",  # TODO(b/555774204): OOM
        "notsan",  # TODO(b/555774204): OOM
        # Exactly 8 chips are required because the test hardcodes _world_size = 8 to match the
        # topology and avoid uninitialized chips in OSS CI.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/distributed:spmd_util",
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "device_mesh_test",
    srcs = ["device_mesh_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu:_loader",
        "//src/torch_tpu/_internal/distributed:device_mesh",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "handshake_e2e_test",
    srcs = ["handshake_e2e_test.py"],
    notest_oss = "TODO(b/537411907): Re-enable in OSS once fixed.",
    shard_count = 8,
    tags = [
        "noasan",  # TODO(b/497035517)
        "nomsan",  # TODO(b/497035517)
        "notsan",  # TODO(b/497035517)
        "requires-tpu-v5lite:8",  # We need 8 chips because we are running an actual e2e workflow to test handshake integration.
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
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
    notest_oss = "TODO(b/560244886): re-enable the test once deflaked in OSS.",
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
    notest_oss = "Requires PJRT RawBuffer extension not yet exported in OSS libtpu.",
    shard_count = 6,
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
        "//src/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "process_group_utils_test",
    srcs = ["process_group_utils_test.py"],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:process_group_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_package_end()
