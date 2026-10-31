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

# Domain experts for this package:
# zhanyong-wan
# benawilson
# agarwalt

load(
    "//build_files:build_defs.bzl",
    "torch_tpu_package_end",
    "torch_tpu_py_library",
    "torch_tpu_py_test",
)
load("//shims/pybind11:pybind.bzl", "pybind_extension")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_py_library(
    name = "_internal",
    srcs = ["__init__.py"],
)

pybind_extension(
    name = "env",
    srcs = ["//csrc/internal:env.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:env_vars",
        "//csrc/common:error_utils",
        "//csrc/common:libtpu_version",
        "//csrc/common:process",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:utils",
        "//shims/torch:torch_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

pybind_extension(
    # Internal testing utilities.
    name = "testing",
    srcs = ["//csrc/internal:testing.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        ":warnings_impl",  # Imported by testing.cc.
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:error_utils",
        "//csrc/common:pybind_error_utils",
        "//csrc/common:tpu_hardware",
        "//csrc/common:warnings",
        "//csrc/eager:device_gen_impl",
        "//csrc/eager:events_queue",
        "//csrc/eager:op_dispatcher",
        "//csrc/eager:structured_log_buffer",
        "//csrc/eager:tensor_to_buffer",
        "//csrc/ops:python_context",
        "//csrc/pjrt:pjrt_state",
        "//shims/torch",
        "//shims/torch:aten",
        "//shims/torch:aten_headers",
        "//shims/torch:torch_headers",
        "//shims/torch/c10/util:base_headers",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

torch_tpu_py_library(
    name = "env_utils",
    srcs = ["env_utils.py"],
)

torch_tpu_py_library(
    name = "warnings",
    srcs = ["warnings.py"],
    deps = [":warnings_impl"],
)

pybind_extension(
    name = "warnings_impl",
    srcs = ["//csrc/internal:warnings_impl.cc"],
    common_lib_packages = [
        "torch_tpu/csrc/common",
    ],
    py_deps = [
        "//shims/torch:pytorch",
    ],
    wrap_py_init = True,
    deps = [
        "//csrc/common:pybind_error_utils",
        "//csrc/common:warnings",
        "//shims/torch:torch_headers",
    ],
)

torch_tpu_py_library(
    name = "public_api",
    srcs = ["public_api.py"],
    # `public_api.py` references these modules as string literals in
    # `_PUBLIC_MODULES` and imports them lazily via `importlib.import_module()`
    # on first attribute access, so `build_cleaner` cannot see static `import`
    # statements for them.
    deps = [
        "//src/torch_tpu/_internal/compiler_options",  # build_cleaner: keep
        "//src/torch_tpu/_internal/device:_device_module",  # build_cleaner: keep
        "//src/torch_tpu/_internal/distributed",  # build_cleaner: keep
        "//src/torch_tpu/_internal/distributed:_distributed_module",  # build_cleaner: keep
        "//src/torch_tpu/_internal/distributed:spmd_util",  # build_cleaner: keep
        "//src/torch_tpu/_internal/ops:collectives",  # build_cleaner: keep
        "//src/torch_tpu/_internal/ops:jagged",  # build_cleaner: keep
        "//src/torch_tpu/_internal/ops:ops_lib",  # build_cleaner: keep
        "//src/torch_tpu/_internal/ops:splash_attention",  # build_cleaner: keep
        "//src/torch_tpu/_internal/optimizers:adamw",  # build_cleaner: keep
        "//src/torch_tpu/_internal/optimizers:muon",  # build_cleaner: keep
        "//src/torch_tpu/_internal/optimizers:optim",  # build_cleaner: keep
        "//src/torch_tpu/_internal/optimizers:sgd",  # build_cleaner: keep
        "//src/torch_tpu/_internal/optimizers:single_trace_trainer",  # build_cleaner: keep
        "//src/torch_tpu/_internal/pallas:mesh",  # build_cleaner: keep
        "//src/torch_tpu/_internal/pallas:pallas_lib",  # build_cleaner: keep
        "//src/torch_tpu/_internal/profiler:profiler_config",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_py_test(
    name = "public_api_test",
    srcs = ["public_api_test.py"],
    strict = True,
    deps = [
        ":public_api",
        "//shims/absl_py/testing:absltest",
        "//src/torch_tpu",  # build_cleaner: keep
        "//src/torch_tpu/_internal/utils:annotations",
    ],
)

torch_tpu_package_end()
