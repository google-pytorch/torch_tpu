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

load("@xla//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")
load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_cc_library",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
    "torch_tpu_py_test",
    "tpu_gen",
)

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

cc_embed_data(
    name = "embedded_bundle",
    srcs = ["//csrc/experimental/precompiled_kernels/aot:bundle"],
    outs = [
        "embedded_bundle.cc",
        "embedded_bundle.h",
    ],
)

torch_tpu_cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
        "//csrc/common:compilation",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:shape",
        "//csrc/eager:device_buffer",
        "//csrc/eager:traversal",
        "//csrc/experimental/precompiled_kernels:bundle_cc_proto",
        "//csrc/ops:op_names",
        "//csrc/pjrt:pjrt_state",
        "//csrc/pjrt:pjrt_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
    ],
)

torch_tpu_cc_library(
    name = "plugin",
    srcs = ["register.cc"],
    deps = [
        ":embedded_bundle",
        ":runtime",
        "//csrc/eager:materialize_common",
        "//csrc/experimental/precompiled_kernels:bundle_cc_proto",
        "@com_google_absl//absl/base:no_destructor",
    ],
    alwayslink = True,
)

torch_tpu_cc_test(
    name = "runtime_test",
    srcs = ["runtime_test.cc"],
    requires_libtpu = True,
    deps = [
        ":runtime",
        "//csrc/common:compilation",
        "//csrc/common:context_states",
        "//csrc/common:dimension_types",
        "//csrc/common:dtype",
        "//csrc/common:error_utils",
        "//csrc/common:oss_test_main",
        "//csrc/common:shape",
        "//csrc/common:status_test_utils",
        "//csrc/eager:device_buffer",
        "//csrc/eager:eager_mode",
        "//csrc/eager:traversal",
        "//csrc/ops:op_names",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
    ],
)

_TARGETS = [
    ("v6e", "v6", "requires-tpu-v6e"),
    ("v7x", "v7", "requires-tpu-v7x"),
]

# Manual: requires enabled=true and a bundle matching the attached hardware.
# Separate targets ensure activation is selected before the extension loads.
[
    torch_tpu_py_test(
        name = "python_e2e_" + mode + "_" + chip + "_test",
        srcs = ["python_e2e_test.py"],
        env = {
            "TORCH_TPU_EXPERIMENTAL_PRECOMPILED": activation,
        },
        main = "python_e2e_test.py",
        nopresubmit = "Requires optional linking and a matching bundle target",
        oss_presubmit_tpu_generation = tpu_gen(
            generation,
            reason = "Executes precompiled kernels for " + chip,
        ),
        tags = [if_oss(
            "requires-tpu-" + chip,
            internal_tag,
        )] + [
            "fails-on-tpu-" + other
            for other in [
                "v5",
                "v6",
                "v7",
            ]
            if other != generation
        ],
        deps = [
            "//shims/absl_py/testing:absltest",
            "//shims/absl_py/testing:parameterized",
            "//shims/torch:pytorch",
            "//src/torch_tpu",
            "//src/torch_tpu/_internal:testing",
        ],
    )
    for mode, activation in [
        ("enabled", "1"),
        ("disabled", "0"),
    ]
    for chip, generation, internal_tag in _TARGETS
]

torch_tpu_package_end()
