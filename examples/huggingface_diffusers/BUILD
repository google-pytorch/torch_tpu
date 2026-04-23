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

load("@rules_python//python:py_binary.bzl", "py_binary")
load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "model_configs_data",
    srcs = glob(["model_configs/**"]),
)

torch_tpu_py_test(
    name = "sdxl_test_cpu",
    size = "large",
    srcs = ["sdxl_test.py"],
    args = ["--device=cpu"],
    data = glob(["model_configs/**"]),
    main = "sdxl_test.py",
    notap = "Fails - missing op upsample_nearest2d.out.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "ignore_for_dep=third_party.torch_tpu.api",
        "requires-accel-hbm",
        "requires-gpu-sm80",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//diffusers",
        "@pypi//transformers",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "sdxl_test_cuda",
    size = "large",
    srcs = ["sdxl_test.py"],
    args = ["--device=cuda"],
    data = glob(["model_configs/**"]),
    main = "sdxl_test.py",
    notap = "slow test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "ignore_for_dep=third_party.torch_tpu.api",
        "requires-gpu-sm80",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//diffusers",
        "@pypi//transformers",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "sdxl_test_tpu",
    size = "large",
    srcs = ["sdxl_test.py"],
    args = ["--device=tpu"],
    data = glob(["model_configs/**"]),
    main = "sdxl_test.py",
    notap = "Fails - missing op upsample_nearest2d.out.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v6e",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//diffusers",
        "@pypi//transformers",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "flux_test_cuda",
    size = "large",
    srcs = ["flux_1_schnell_test.py"],
    args = ["--device=cuda"],
    data = glob(["model_configs/**"]),
    main = "flux_1_schnell_test.py",
    notap = "slow test",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "ignore_for_dep=third_party.torch_tpu.api",
        "requires-accel-hbm",
        "requires-gpu-sm80",
        "requires-net:external",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//diffusers",
        "@pypi//transformers",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "flux_test_tpu",
    size = "enormous",
    srcs = ["flux_1_schnell_test.py"],
    args = ["--device=tpu"],
    data = glob(["model_configs/**"]),
    main = "flux_1_schnell_test.py",
    notap = "OOMs.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x:4",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//diffusers",
        "@pypi//transformers",  # buildcleaner: keep
    ],
)

torch_tpu_py_test(
    name = "wan_test_tpu",
    size = "enormous",
    srcs = ["wan2_2_ti2v_5b_diffusers_test.py"],
    args = ["--device=tpu"],
    main = "wan2_2_ti2v_5b_diffusers_test.py",
    notap = "OOMs.",
    notest_oss = "Disabled during switch from opt-in to opt-out",
    tags = [
        "requires-accel-hbm",
        "requires-net:external",
        "requires-tpu-v7x:4",
    ],
    deps = [
        "//examples:paths",
        "//shims/absl_py/flags",
        "//shims/absl_py/logging",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//torch_tpu",  # buildcleaner: keep
        "//torch_tpu/_internal/utils:log_utils",
        "@pypi//accelerate",  # buildcleaner: keep
        "@pypi//diffusers",
        "@pypi//pillow",
        "@pypi//transformers",  # buildcleaner: keep
    ],
)

py_binary(
    name = "import_hf_configs",
    srcs = ["import_hf_configs.py"],
    exec_properties = {
        "cpp_link.mem": "16g",  # Needed for asan.
    },
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/flags",
        "@pypi//huggingface_hub",
    ],
)

torch_tpu_package_end()
