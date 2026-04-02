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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "ml_layers_test_cuda",
    timeout = "long",
    srcs = ["ml_layers_test.py"],
    args = [
        "--device=cuda",
        "--framework=pytorch",
    ],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {
        "cpp_link.mem": "16g",  # Avoid Forge OOM in CppLink.
    },
    main = "ml_layers_test.py",
    notap = "this is a manual test",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/flax_nnx",
        "//shims/torch:pytorch",
        "//tests:ops_test_data",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//flax",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "ml_layers_test_tpu",
    timeout = "long",
    srcs = ["ml_layers_test.py"],
    args = [
        "--device=tpu",
        "--framework=pytorch",
    ],
    main = "ml_layers_test.py",
    nopresubmit = "slow test",
    # TODO: Right this test in nightly builds
    notest_oss = "Slow test that also seems to be broken",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",  # Matches the CUDA version of the test for a fair comparison.
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/flax_nnx",
        "//shims/torch:pytorch",
        "//tests:ops_test_data",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//flax",
        "@pypi//jax",
    ],
)

torch_tpu_py_test(
    name = "ml_layers_test_xla_cuda",
    timeout = "long",
    srcs = ["ml_layers_test.py"],
    args = [
        "--device=xla_cuda",
        "--framework=pytorch",
    ],
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {
        "cpp_link.mem": "16g",
    },
    main = "ml_layers_test.py",
    notap = "this is a manual test",
    shard_count = 4,
    strict = True,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/flax_nnx",
        "//shims/torch:pytorch",
        "//tests:ops_test_data",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//flax",
        "@pypi//jax",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

torch_tpu_py_test(
    name = "sched_overhead_test_cuda",
    timeout = "long",
    srcs = ["sched_overhead_test.py"],
    args = [
        "--device=cuda",
    ],
    # 1gig.
    env = {"TPU_PREMAPPED_BUFFER_SIZE": "1073741824"},
    # go/cppdebugflags#increasing-your-size-limits
    exec_properties = {
        "cpp_link.mem": "16g",  # Avoid Forge OOM in CppLink.
    },
    main = "sched_overhead_test.py",
    notap = "this is a manual test",
    shard_count = 1,
    strict = True,
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "sched_overhead_test_tpu",
    timeout = "long",
    srcs = ["sched_overhead_test.py"],
    args = [
        "--device=tpu",
    ],
    main = "sched_overhead_test.py",
    nopresubmit = "slow test",
    shard_count = 1,
    strict = True,
    tags = [
        "cpu:4",  # Matches the CUDA version of the test for a fair comparison.
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/shims/xprof:xprof_session",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_package_end()
