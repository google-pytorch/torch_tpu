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

"""Shim for build_defs."""

load("@rules_shell//shell:sh_test.bzl", _sh_test = "sh_test")

sh_test = _sh_test

def add_internal_filesystem_dependencies():
    return []

def process_accelerator_tags(tags):
    """process_accelerator_tags shim.

    This macro adds a generic "wild card" accelerator tag ("requires-tpu" or
    "requires-gpu") to all tests that require an accelerator, to make it easier
    to filter tests by accelerator in the bazel config.

    Args:
      tags: The tags to add to the test.
    """

    # Bazel doesn't support filtering tags with wildcards, so this aggregates
    # all the (OSS) TPU and GPU tags into requires-gpu and requires-tpu; we
    # assume that there are no tests that require both a GPU and a TPU.
    has_tpu = any([t.startswith("requires-tpu") for t in tags])
    has_gpu = any([t.startswith("requires-gpu") for t in tags])

    if has_tpu and "requires-tpu" not in tags:
        tags.append("requires-tpu")
    if has_gpu and "requires-gpu" not in tags:
        tags.append("requires-gpu")
