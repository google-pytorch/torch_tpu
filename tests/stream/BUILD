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
# benawilson
# yilingyuan
# haifeng-jin

load(
    "//build_files:build_defs.bzl",
    "REQUIRES_V5_MULTI_DEVICE",
    "if_oss",
    "torch_tpu_package_end",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "stream_test",
    srcs = ["stream_test.py"],
    args = if_oss(
        [],
        ["--deepsea_chip_config_name=mpmd"],
    ),
    tags = [
        # A device with SparseCore is required to allow for SparseCore pinning.
        "requires-tpu-v5p",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:oss_utils",
        "//tests:seed_test_utils",
    ],
)

torch_tpu_py_test(
    name = "copy_test",
    srcs = ["copy_test.py"],
    tags = [
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

# Integration tests that run the common stream usage patterns as users write
# them and assert the behavior the stream and event API promises. Expected
# failures at HEAD, each removed with the change that fixes it:
#   test_side_stream_prefetch_with_record_stream: an aten::record_stream kernel
#   test_side_stream_offload_with_record_stream: an aten::record_stream kernel
#   test_wait_event_holds_the_called_stream: wait_event() waiting on self
#   test_non_blocking_d2h_into_pageable_memory_is_synchronous: a synchronous
#     non_blocking copy into pageable memory
#   test_stream_query_reports_pending_work: a Stream.query() implementation
#   test_stream_switch_inside_torch_compile: Dynamo tracing of the stream switch
#   stream_collectives_test, test_async_op_work_tracks_completion: a Work that
#     tracks the collective's completion
#   stream_collectives_test, test_send_first_pipeline_ring: an isend that
#     returns before the peer posts its receive
torch_tpu_py_test(
    name = "stream_integration_test",
    srcs = ["stream_integration_test.py"],
    main = "stream_integration_test.py",
    run_on_accelerators = [
        "requires-tpu-v6e",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests:seed_test_utils",
    ],
)

# Three worker groups (one per test class, 8 ranks each) take ~110 s on v5e-8.
torch_tpu_py_test(
    name = "stream_collectives_test",
    timeout = "long",
    srcs = ["stream_collectives_test.py"],
    env = {
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/stream_collectives_test",
    },
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # Same multi-process harness as collective_ops_test (b/497035517).
        "nomsan",  # Same multi-process harness as collective_ops_test (b/497035517).
        "notsan",  # Same multi-process harness as collective_ops_test (b/497035517).
    ],
    deps = [
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/distributed:multiprocessing",
        "//src/torch_tpu/_internal/utils:test_utils",
        "//tests/distributed:continuous_test_utils",
        "//tests/distributed:distributed_utils",
    ],
)

torch_tpu_package_end()
