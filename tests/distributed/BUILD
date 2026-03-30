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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_strict_library(
    name = "distributed_utils",
    testonly = True,
    srcs = [
        "distributed_utils.py",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "//shims/torch:pytorch",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "single_tpu_pg_test",
    srcs = ["single_tpu_pg_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/api",
        "@pypi//portpicker",
    ],
)

torch_tpu_py_test(
    name = "multi_tpu_test",
    srcs = ["multi_tpu_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed:tpu_distributed",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "collective_ops_test",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/collective_ops_test",
    },
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 8,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the timeout in test_broadcast_objects.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

# Like collective_ops_test, but uses a remote filesystem for the tier-3 cache.
torch_tpu_py_test(
    name = "collective_ops_remote_cache_test",
    srcs = ["collective_ops_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
        # TODO: b/490440118 - Remove this once the bug is fixed.
        # "$DATA_ROOT/$TMP_SUBDIR/ttl=1d/collective_ops_remote_cache_test"
        "",
    },
    main = "collective_ops_test.py",
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 8,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/495533335): fix out-of-order in msan mode.
        "notsan",  # TODO(b/495533335): fix out-of-order in tsan mode.
        "requires-net:external",  # Needs remote file access.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
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
    shard_count = 8,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
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
    shard_count = 8,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "alltoall_collective_test",
    srcs = ["alltoall_collective_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 4,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-accel-hbm",
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal:testing",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "scatter_test",
    srcs = ["scatter_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests:error_testing",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "subgroup_collectives_test",
    srcs = ["subgroup_collectives_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 4,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "dtensor_test",
    srcs = ["dtensor_test.py"],
    notest_oss = "fix hang in OSS",  # TODO
    shard_count = 10,
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
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
    tags = [
        "exclusive",  # copybara:comment(oss-only)
        "noasan",  # TODO(b/497035517): fix the test in asan mode.
        "nomsan",  # TODO(b/497035517): fix the test in msan mode.
        "notsan",  # TODO(b/497035517): fix the test in tsan mode.
        "requires-tpu-v5lite:8",
    ],
    deps = [
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//tests/distributed:distributed_utils",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "@pypi//fairscale",
    ],
)

torch_tpu_py_test(
    name = "spmd_context_test",
    srcs = ["spmd_context_test.py"],
    env = {
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/tmp/spmd_context_test",
    },
    tags = [
        "requires-tpu-v5lite:8",
    ],
    deps = [
        ":distributed_utils",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/distributed:spmd_context",
        "//torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//torch_tpu/_internal/shims/pyglib/contrib/g3_multiprocessing",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/api",
    ],
)

torch_tpu_package_end()
