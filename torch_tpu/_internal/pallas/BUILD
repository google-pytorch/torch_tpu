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

load("//build_files:build_defs.bzl", "torch_tpu_package_end", "torch_tpu_py_library")
load("//shims/py_rules:pytype.bzl", "pytype_strict_library")
load("//shims/py_rules:strict.bzl", "py_library_providing_imports_info")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
        "//shims/torchtitan:__subpackages__",
        "@pypi//helion:__subpackages__",
        "@pypi//tokamax:__subpackages__",
    ],
)

# We use py_library_providing_imports_info with pallas.modspec.json to explicitly
# re-export the submodules provided by :pallas_lib. This prevents PyStrictDeps
# violations at downstream use sites that import `from torch_tpu._internal.pallas import pallas`
# without requiring them to directly depend on :pallas_lib.
py_library_providing_imports_info(
    name = "pallas",
    srcs = [
        "__init__.py",
    ],
    lib_rule = torch_tpu_py_library,
    module_specs = [
        "pallas.modspec.json",
    ],
    deps = [
        ":pallas_lib",
        # The "pallas" library doesn't work without this dependency, but it's not obvious
        # from the import. Therefore we have to tell build_cleaner to keep it.
        # With this, users of the "pallas" library don't need to add the dependency on "pallas_tpu"
        # themselves.
        "//shims/jax:pallas_tpu",  # build_cleaner: keep
    ],
)

torch_tpu_py_library(
    name = "pallas_compat",
    srcs = ["_compat.py"],
)

torch_tpu_py_library(
    name = "pallas_lib",
    srcs = ["pallas.py"],
    deps = [
        ":pallas_compat",
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/csrc/internal/pallas:tpu_torch_pallas",
        "@pypi//frozendict",
        "@pypi//jax",
    ],
)

# NOTE: We use pytype_strict_library here instead of torch_tpu_py_library to break a circular
# dependency cycle where torch_tpu_py_library automatically injects a dependency on
# //torch_tpu/_internal:testing.
pytype_strict_library(
    # PY_LIBRARY_OK=Break circular dependency cycle with _internal:testing.
    name = "pallas_kernel_generate_utils",
    srcs = ["pallas_kernel_generate_utils.py"],
    deps = [],
)

torch_tpu_package_end()
