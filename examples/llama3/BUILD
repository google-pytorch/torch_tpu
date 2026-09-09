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
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_test(
    name = "llama3_2_example",
    srcs = ["llama3_2_example.py"],
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/llama3/impl:llama3_2",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils",
        "//src/torch_tpu/_internal/utils:test_utils",
    ],
)

torch_tpu_py_test(
    name = "llama3_2_1b_hf_pipe_example",
    srcs = ["llama3_2_1b_hf_pipe_example.py"],
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "nomsan",  # 3p 'rust/tempfile' error: use-of-uninitialized-value in tempfile::file::imp::platform::reopen
        "requires-net:external",  # Needed to access remote files.
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//transformers",
    ],
)

torch_tpu_py_library(
    name = "llama3_2_1b_training_nb",
    srcs = ["llama3_2_1b_training_nb.py"],
    deps = [
        "//examples:paths",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "@pypi//datasets",
        "@pypi//tqdm",
        "@pypi//transformers",
    ],
)

# buildifier: disable=forbidden-tag-requires-gpu
torch_tpu_py_test(
    name = "llama3_2_1b_random_data_training_test_cuda",
    timeout = "long",
    srcs = ["llama3_2_1b_random_data_training_test.py"],
    args = ["--device=cuda"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    main = "llama3_2_1b_random_data_training_test.py",
    notap = "this is a manual test",
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

torch_tpu_py_test(
    name = "llama3_2_1b_random_data_training_test_tpu",
    timeout = "long",
    srcs = ["llama3_2_1b_random_data_training_test.py"],
    args = ["--device=tpu"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    exec_properties = {
        "cpp_link.mem": "20g",  # Needed for asan.
    },
    # Per go/linkularity, `sharded_dynamic` is an effective strategy to break up monolithic
    # `py_binrary` into multiple artifacts, and reliably mitigate relocation overflow errors.
    # See also go/be#py_binary.linking_mode.
    linking_mode = "sharded_dynamic",
    main = "llama3_2_1b_random_data_training_test.py",
    nonightly_oss = "Disabled during switch from opt-in to opt-out",
    nopresubmit = "slow example",
    tags = [
        "cpu:4",  # Matches the CUDA version of the test for a fair comparison.
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

# buildifier: disable=forbidden-tag-requires-gpu
torch_tpu_py_test(
    name = "llama3_2_1b_random_data_training_test_xla_cuda",
    timeout = "long",
    srcs = ["llama3_2_1b_random_data_training_test.py"],
    args = ["--device=xla_cuda"],
    data = [
        "//examples/huggingface_transformers:model_configs_data",
    ],
    main = "llama3_2_1b_random_data_training_test.py",
    notap = "this is a manual test",
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/sync",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

# buildifier: disable=forbidden-tag-requires-gpu
torch_tpu_py_test(
    name = "llama3_2_1b_training_test_cuda",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["llama3_2_1b_training_test.py"],
    args = ["--device=cuda"],
    main = "llama3_2_1b_training_test.py",
    nobuild = "This test doesn't build.",
    notap = "slow example",
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-accel-hbm",
        "requires-gpu-sm100-full",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//datasets",
        "@pypi//matplotlib:pkg",
        "@pypi//tqdm",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "llama3_2_1b_training_test_tpu",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["llama3_2_1b_training_test.py"],
    args = ["--device=tpu"],
    linking_mode = "static",
    main = "llama3_2_1b_training_test.py",
    notap = "slow example",
    tags = [
        "cpu:4",  # Matches the CUDA version of the test for a fair comparison.
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//datasets",
        "@pypi//matplotlib:pkg",
        "@pypi//tqdm",
        "@pypi//transformers",
    ],
)

# buildifier: disable=forbidden-tag-requires-gpu
torch_tpu_py_test(
    name = "llama3_2_1b_training_test_xla_cuda",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["llama3_2_1b_training_test.py"],
    args = ["--device=xla_cuda"],
    main = "llama3_2_1b_training_test.py",
    nobuild = "This test doesn't build.",
    notap = "slow example",
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/benchmarks:xprof_adapter",
        "//src/torch_tpu/_internal/compile",
        "//src/torch_tpu/_internal/execution_mode",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//datasets",
        "@pypi//matplotlib:pkg",
        "@pypi//tqdm",
        "@pypi//transformers",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

torch_tpu_py_test(
    name = "llama3_2_1b_training_peft_test",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["llama3_2_1b_training_peft_test.py"],
    notap = "slow test",
    tags = [
        "cpu:4",
        "nofastbuild",  # Measures performance.
        "requires-net:external",  # For downloading models and datasets
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//src/torch_tpu",
        "//src/torch_tpu/_internal/utils:log_utils",
        "//tests:seed_test_utils",
        "@pypi//datasets",
        "@pypi//peft",
        "@pypi//transformers",
    ],
)

torch_tpu_package_end()
