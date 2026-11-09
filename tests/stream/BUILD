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
# them and assert the behavior the stream and event API promises.
# Single-device stream patterns, run on CUDA and TPU alike; a pattern that needs
# a process group (collectives, P2P) goes to stream_collectives_test instead.
torch_tpu_py_test(
    name = "stream_integration_test",
    srcs = ["stream_integration_test.py"],
    main = "stream_integration_test.py",
    tags = ["requires-tpu-v5lite"],
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

# The same tests on CUDA: the behavior TorchTPU matches, run where it is native.
torch_tpu_py_test(
    name = "stream_integration_test_cuda",
    srcs = ["stream_integration_test.py"],
    # Inductor's compile worker pool cannot spawn in hermetic test runners.
    env = {"TORCHINDUCTOR_COMPILE_THREADS": "1"},
    main = "stream_integration_test.py",
    nolocal = "Skipping in local runs by default as it requires --config=cuda.",
    tags = [
        "nozapfhahn",  # Requires --config=cuda; excluded from Zapfhahn coverage.
        # Requires compute capability >= 7.0 for Triton / torch.compile.
        "requires-gpu-sm90",
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

# Stream patterns that need a process group: collectives and P2P on 8 ranks
# (TPU only: the multi-process harness and P2P ops are TPU).
# Three worker groups (one per test class, 8 ranks each) take ~110 s on v5e-8.
torch_tpu_py_test(
    name = "stream_collectives_test",
    timeout = "long",
    srcs = ["stream_collectives_test.py"],
    # OSS runs every test on one runner: the 8-chip slice must not be acquired
    # by two tests at once (/dev/vfio contention).
    tags = if_oss(
        ["exclusive"],
        [],
    ) + [
        REQUIRES_V5_MULTI_DEVICE,
        "noasan",  # 8-rank tests exceed Forge memory under sanitizers (b/497035517).
        "nomsan",  # 8-rank tests exceed Forge memory under sanitizers (b/497035517).
        "notsan",  # 8-rank tests exceed Forge memory under sanitizers (b/497035517).
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
