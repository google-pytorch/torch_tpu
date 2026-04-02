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
)

torch_tpu_py_test(
    name = "resnet",
    srcs = ["resnet.py"],
    nopresubmit = "slow example",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/resnet/impl:resnet",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
    ],
)

torch_tpu_py_test(
    name = "resnet50_tv_test_cuda",
    srcs = ["resnet50_tv_test.py"],
    args = ["--device=cuda"],
    data = ["goldfish.jpg"],
    main = "resnet50_tv_test.py",
    notap = "This manual test validates that the code runs on cuda.",
    tags = [
        "requires-gpu-nvidia",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:tracer_utils",
        "//torch_tpu/api",
        "@pypi//pillow",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "resnet50_tv_test_tpu",
    srcs = ["resnet50_tv_test.py"],
    args = ["--device=tpu"],
    data = ["goldfish.jpg"],
    main = "resnet50_tv_test.py",
    notap = "Test passes but we don't want to run it on TAP.",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:tracer_utils",
        "//torch_tpu/api",
        "@pypi//pillow",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "resnet50_tv_tool_cpu",
    srcs = ["resnet50_tv_tool.py"],
    args = ["--device=cpu"],
    main = "resnet50_tv_tool.py",
    notap = "This is a tool, not a test.",
    tags = [
        "requires-gpu-nvidia",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:tracer_utils",
        "//torch_tpu/api",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "resnet50_tv_tool_cuda",
    srcs = ["resnet50_tv_tool.py"],
    args = ["--device=cuda"],
    main = "resnet50_tv_tool.py",
    notap = "This is a tool, not a test.",
    tags = [
        "requires-gpu-nvidia",
        "requires-net:external",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:tracer_utils",
        "//torch_tpu/api",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "resnet50_tv_tool_tpu",
    srcs = ["resnet50_tv_tool.py"],
    args = ["--device=tpu"],
    main = "resnet50_tv_tool.py",
    notap = "This is a tool, not a test.",
    tags = [
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/etils/epath",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/_internal/utils:tracer_utils",
        "//torch_tpu/api",
        "@pypi//torchvision",
    ],
)

torch_tpu_py_test(
    name = "resnet_hf_example_test_tpu",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["resnet_hf_example.py"],
    args = ["--device=tpu"],
    linking_mode = "static",
    main = "resnet_hf_example.py",
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
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "resnet_hf_example_test_cuda",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["resnet_hf_example.py"],
    args = ["--device=cuda"],
    main = "resnet_hf_example.py",
    notap = "slow example",
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-accel-hbm",
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//transformers",
    ],
)

torch_tpu_py_test(
    name = "resnet_hf_example_test_xla_cuda",
    timeout = "eternal",  # This causes the test to be skipped on TAP.
    srcs = ["resnet_hf_example.py"],
    args = ["--device=xla_cuda"],
    main = "resnet_hf_example.py",
    notap = "slow example",
    tags = [
        "cpu:4",  # For CUDA, the only choices are 1 or 4.
        "requires-gpu-sm100-full-only",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/profile_cpu:python_tracer",
        "//shims/torch:pytorch",
        "//shims/torch:tensorboard_support",  # buildcleaner: keep
        "//torch_tpu/_internal/compile:tpu_backend",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/_internal/shims/xprof:traceme",
        "//torch_tpu/_internal/sync",
        "//torch_tpu/_internal/utils:benchmarking",
        "//torch_tpu/_internal/utils:log_utils",
        "//torch_tpu/api",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//transformers",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

torch_tpu_package_end()
