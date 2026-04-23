# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

load("//build_files:build_defs.bzl", "torch_tpu_package_end")
load("//shims/py_rules:pytype.bzl", "pytype_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

pytype_library(
    name = "tracing",
    srcs = ["__init__.py"],
    deps = [
        "//shims/absl_py/logging",
        "//shims/torch:pytorch",
        "//torch_tpu/_internal/compile:tpu_torch_compile",
        "//torch_tpu/_internal/sync:_tpu_torch_sync",
    ],
)

torch_tpu_package_end()
