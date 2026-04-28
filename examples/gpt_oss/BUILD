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
    name = "gpt_oss_20b_hf_pipe_example",
    srcs = ["gpt_oss_20b_hf_pipe_example.py"],
    notap = "the test is flaky",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "cpu:4",
        "requires-accel-hbm",  # This test will OOM on Forge with the default 12GB limit.
        "requires-net:external",  # Needed to access remote files.
        "requires-tpu-v5p",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py:app",
        "//shims/safetensors",  # buildcleaner: keep
        "//shims/torch:pytorch",
        "//torch_tpu",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "gpt_oss_120b_random_data_training_test",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["gpt_oss_120b_random_data_training_test.py"],
    args = ["--device=tpu"],
    data = ["//examples/huggingface_transformers:model_configs_data"],
    main = "gpt_oss_120b_random_data_training_test.py",
    notap = "this is a manual test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "cpu:4",
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu",
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//transformers",
        "@rules_python//python/runfiles",
    ],
)

torch_tpu_package_end()
