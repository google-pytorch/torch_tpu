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

# model_registry: data-driven replacement for
# //tests:module_registry.
# Named model_registry during the migration so that the package does not shadow
# tests/module_registry.py. Must stay free of torch_tpu C++ deps (see
# //src/torch_tpu/google/tests/module_registry:no_cpp_deps_test).

load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/py_rules:py_library.bzl", "py_library")
load(
    "//shims/py_rules:pytype.bzl",
    pytype_strict_test = "pytype_strict_contrib_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# Public facade. Not testonly: non-test libraries (e.g.
# //perflab:model_helpers) depend on module_registry,
# which will depend on this target.
py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "model_registry",
    srcs = ["__init__.py"],
    type_checking = True,
    deps = [
        ":types",
    ],
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "types",
    srcs = ["types.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core",
    srcs = ["core/__init__.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_config_utils",
    srcs = ["core/config_utils.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_inputs",
    srcs = ["core/inputs.py"],
    type_checking = True,
    deps = [
        ":core_config_utils",
        ":types",
        "//shims/torch:pytorch",
    ],
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_modality",
    srcs = ["core/modality.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_shims",
    srcs = ["core/shims.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers",
    srcs = ["providers/__init__.py"],
    type_checking = True,
    deps = [
        ":providers_base",
    ],
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_base",
    srcs = ["providers/base.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_diffusers",
    srcs = ["providers/diffusers.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_timm",
    srcs = ["providers/timm.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_torchvision",
    srcs = ["providers/torchvision.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_transformers",
    srcs = ["providers/transformers.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides",
    srcs = ["overrides/__init__.py"],
    type_checking = True,
    deps = [
        ":overrides_audio",
        ":overrides_diffusion",
        ":overrides_multimodal",
        ":overrides_text",
        ":overrides_vision",
    ],
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides_audio",
    srcs = ["overrides/audio.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides_diffusion",
    srcs = ["overrides/diffusion.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides_multimodal",
    srcs = ["overrides/multimodal.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides_registry",
    srcs = ["overrides/registry.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides_text",
    srcs = ["overrides/text.py"],
    type_checking = True,
)

py_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "overrides_vision",
    srcs = ["overrides/vision.py"],
    type_checking = True,
)

pytype_strict_test(
    name = "inputs_test",
    srcs = ["inputs_test.py"],
    tags = [
        "nosan",  # PyTorch/SLEEF have data races under in-process multi-threading.
    ],
    deps = [
        ":core_inputs",
        ":types",
        "//shims/absl_py/testing:absltest",
        "//shims/absl_py/testing:parameterized",
        "//shims/torch:pytorch",
        "@pypi//transformers",
    ],
)

# This must be the last line in the BUILD file.
torch_tpu_package_end()
