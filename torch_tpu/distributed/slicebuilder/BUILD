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

# Defines a library that wraps the internal PJRT client.
# This is a workaround to avoid depending on internal components
# directly from the main torch_tpu library.

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_cc_test", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "discovery",
    srcs = ["discovery.cc"],
    hdrs = ["discovery.h"],
    deps = [
        "//shims/torch:torch_headers",
        "//torch_tpu/common:env_vars",
        "//torch_tpu/common:error_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

torch_tpu_cc_test(
    name = "discovery_test",
    srcs = ["discovery_test.cc"],
    deps = [
        ":discovery",
        "//torch_tpu/common:env_vars",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

torch_tpu_package_end()
