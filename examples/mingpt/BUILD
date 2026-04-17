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
load("//shims/py_rules:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
)

# TPU version of the mingpt example.
torch_tpu_py_test(
    name = "mingpt_example_tpu",
    srcs = ["mingpt_example.py"],
    args = [
        "--device=tpu",
        "--xla_jf_conv_full_precision",  # Use highest precision for matmuls for tests only.
    ],
    main = "mingpt_example.py",
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "cpu:8",
        "requires-net:external",
        "requires-tpu-v5p",
    ],
    deps = [
        "//examples/mingpt/impl:mingpt",
        "//examples/mingpt/impl:mingpt_jax",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/profile_cpu:python_tracer",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
    ],
)

# XLA:CPU version of the mingpt example.
torch_tpu_py_test(
    name = "mingpt_example_xla_cpu",
    timeout = "long",  # 15 minutes
    srcs = ["mingpt_example.py"],
    args = [
        "--device=xla_cpu",
    ],
    main = "mingpt_example.py",
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-net:external",
    ],
    deps = [
        "//examples/mingpt/impl:mingpt",
        "//examples/mingpt/impl:mingpt_jax",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/profile_cpu:python_tracer",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",  # buildcleaner: keep
    ],
)

# XLA:GPU version of the mingpt example.
torch_tpu_py_test(
    name = "mingpt_example_xla_cuda",
    srcs = ["mingpt_example.py"],
    args = [
        "--device=xla_cuda",
    ],
    main = "mingpt_example.py",
    notap = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-gpu-sm60-only",
        "requires-net:external",
    ],
    deps = [
        "//examples/mingpt/impl:mingpt",
        "//examples/mingpt/impl:mingpt_jax",
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "//shims/profile_cpu:python_tracer",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/utils",
        "//torch_tpu/api",
        "@pypi//jax",
        "@pypi//numpy",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/stream_executor:cuda_platform",
    ],
)

torch_tpu_py_test(
    name = "mingpt_training_example",
    timeout = "long",  # Normally takes ~5 minutes.
    srcs = ["mingpt_training_example.py"],
    nopresubmit = "slow example",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "cpu:8",
        "requires-net:external",
        "requires-tpu-v5lite",
    ],
    deps = [
        "//examples/mingpt/impl:mingpt",
        "//shims/absl_py:app",
        "//shims/profile_cpu:python_tracer",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/execution_mode",
        "//torch_tpu/api",
        "//torch_tpu/eager:tpu_aten_kernels",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
    ],
)

pytype_strict_binary(
    name = "mingpt_decomposition_tests",
    srcs = ["mingpt_decomposition_tests.py"],
    exec_properties = {
        "cpp_link.mem": "20g",  # Needed for asan.
    },
    deps = [
        "//examples/mingpt/impl:mingpt",
        "//shims/absl_py:app",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/device:_device_module",
        "@pypi//numpy",
    ],
)

pytype_strict_library(
    name = "mingpt_runner_lib",
    srcs = ["mingpt_runner_lib.py"],
    deps = [
        "//examples/mingpt/impl:mingpt",
        "//shims/torch:pytorch",
        "//torch_tpu/api",
    ],
)

# Disabled because it fails
#
# torch_tpu_py_test(
#     name = "mingpt_example_gpu",
#     srcs = ["mingpt_example.py"],
#     args = [
#         "--native_device=cuda",
#     ],
#     main = "mingpt_example.py",
#     notap = "slow example",
#     tags = [
#         "requires-gpu-sm60-only",
#         "requires-net:external",
#     ],
#     deps = [
#         "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
#         "//shims/absl_py:app",
#         "@pypi//jax",
# #         "//shims/torch:pytorch",
#         "//torch_tpu/eager:tpu_aten_kernels",
#         "//torch_tpu/_internal/device:_device_ops_backend",
#         "//torch_tpu/_internal/device:_device_module",
#         "//examples/mingpt/impl:mingpt",
#         "//examples/mingpt/impl:mingpt_jax",
#     ],
# )

torch_tpu_package_end()
