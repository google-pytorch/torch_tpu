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
load(
    "//shims/py_rules:pytype.bzl",
    "pytype_strict_library",
    pytype_strict_test = "pytype_strict_contrib_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# Public facade. Not testonly: non-test libraries (e.g.
# //perflab:model_helpers) depend on module_registry,
# which will depend on this target.
pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "model_registry",
    srcs = ["__init__.py"],
    deps = [
        ":types",
    ],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "types",
    srcs = ["types.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core",
    srcs = ["core/__init__.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_config_utils",
    srcs = ["core/config_utils.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_inputs",
    srcs = ["core/inputs.py"],
    deps = [
        ":core_config_utils",
        ":types",
        "//shims/torch:pytorch",
    ],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_modality",
    srcs = ["core/modality.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "core_shims",
    srcs = ["core/shims.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers",
    srcs = ["providers/__init__.py"],
    deps = [
        ":providers_base",
    ],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_base",
    srcs = ["providers/base.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_diffusers",
    srcs = ["providers/diffusers.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_timm",
    srcs = ["providers/timm.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_torchvision",
    srcs = ["providers/torchvision.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "providers_transformers",
    srcs = ["providers/transformers.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks",
    srcs = ["quirks/__init__.py"],
    deps = [
        ":quirks_audio",
        ":quirks_diffusion",
        ":quirks_multimodal",
        ":quirks_text",
        ":quirks_vision",
    ],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks_audio",
    srcs = ["quirks/audio.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks_diffusion",
    srcs = ["quirks/diffusion.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks_multimodal",
    srcs = ["quirks/multimodal.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks_registry",
    srcs = ["quirks/registry.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks_text",
    srcs = ["quirks/text.py"],
)

pytype_strict_library(
    # PY_LIBRARY_OK=Must stay free of torch_tpu C++ deps.
    name = "quirks_vision",
    srcs = ["quirks/vision.py"],
)

pytype_strict_test(
    name = "inputs_test",
    srcs = ["inputs_test.py"],
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
