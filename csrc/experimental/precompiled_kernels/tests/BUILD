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

load(
    "//build_files:build_defs.bzl",
    "if_oss",
    "torch_tpu_cc_test",
    "torch_tpu_package_end",
    "tpu_gen",
)

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//csrc/experimental/precompiled_kernels:__subpackages__",
    ],
)

_GENERATOR = "//csrc/experimental/" + \
             "precompiled_kernels/aot:generate_bundle"

_LIBTPU = "@precompiled_libtpu//:libtpu/libtpu.so"

_TARGETS = [
    ("v6e", "v6", "v6e:2x2", "requires-tpu-v6e"),
    ("v7x", "v7", "tpu7x:2x2", "requires-tpu-v7x"),
]

# Each test gets a bundle compiled explicitly for its chip.
[
    genrule(
        name = "bundle_" + chip,
        srcs = [_LIBTPU],
        outs = ["kernels_" + chip + ".pb"],
        cmd = "TPU_SKIP_MDS_QUERY=true " +
              "TPU_LIBRARY_PATH=$(execpath %s) " % _LIBTPU +
              "$(execpath %s) --torch_tpu_internal_topology=%s --torch_tpu_internal_output=$@" %
              (_GENERATOR, topology),
        tools = [_GENERATOR],
    )
    for chip, generation, topology, _ in _TARGETS
]

# Direct PJRT execution, using the pinned libtpu rather than the stable wrapper.
# CI selects the matching generation and excludes this test on other chips.
[
    torch_tpu_cc_test(
        name = "bundle_correctness_" + chip + "_test",
        srcs = ["bundle_correctness_test.cc"],
        args = ["--bundle_path=$(rootpath :bundle_" + chip + ")"],
        data = [
            ":bundle_" + chip,
            _LIBTPU,
        ],
        env = {
            "TPU_LIBRARY_PATH": "$(rootpath %s)" % _LIBTPU,
            "TPU_SKIP_MDS_QUERY": "true",
        },
        nopresubmit = "Requires scarce TPU hardware",
        oss_presubmit_tpu_generation = tpu_gen(
            generation,
            reason = "Executes a bundle compiled for " + chip,
        ),
        requires_libtpu = False,
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
            ":bundle_" + chip,
            "//csrc/common:dimension_types",
            "//csrc/common:oss_test_main",
            "//csrc/common:status_test_utils",
            "//csrc/experimental/precompiled_kernels:bundle_cc_proto",
            "@com_google_absl//absl/flags:flag",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/strings",
            "@com_google_googletest//:gtest",
            "@xla//xla:literal",
            "@xla//xla:shape_util",
            "@xla//xla/pjrt:pjrt_api",
            "@xla//xla/pjrt:pjrt_client",
            "@xla//xla/pjrt:pjrt_layout",
            "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        ],
    )
    for chip, generation, topology, internal_tag in _TARGETS
]

torch_tpu_package_end()
