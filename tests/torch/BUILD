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

load("@rules_python//python:py_binary.bzl", "py_binary")
load(
    "//build_files:build_defs.bzl",
    "torch_tpu_oss_torch_test",
    "torch_tpu_package_end",
)

package(default_applicable_licenses = ["//:license"])

py_binary(
    name = "torch_test_runner",
    testonly = True,
    srcs = ["torch_test_runner.py"],
    deps = [
        "//shims/absl_py:app",
        "//shims/absl_py/testing:absltest",
        "//shims/torch:pytorch",
        "//shims/torch:testing_internal",
        "//src/torch_tpu",
    ],
)

# TODO(b/562086911): auto-generate all test targets defined in
# https://github.com/pytorch/pytorch/tree/main/test.

# go/keep-sorted start block=yes newline_separated=yes by_regex=torch_test_module\s*=\s*"([^"]+)"
torch_tpu_oss_torch_test(
    name = "dynamo_after_aot_test",
    torch_test_module = "dynamo/test_after_aot",
)

torch_tpu_oss_torch_test(
    name = "dynamo_backends_test",
    torch_test_module = "dynamo/test_backends",
)

torch_tpu_oss_torch_test(
    name = "dynamo_debug_utils_test",
    extra_deps = [
        "@pypi//jax",
    ],
    torch_test_module = "dynamo/test_debug_utils",
)

torch_tpu_oss_torch_test(
    name = "dynamo_export_test",
    torch_test_module = "dynamo/test_export",
)

torch_tpu_oss_torch_test(
    name = "dynamo_minifier_test",
    torch_test_module = "dynamo/test_minifier",
)

torch_tpu_oss_torch_test(
    name = "dynamo_modules_test",
    torch_test_module = "dynamo/test_modules",
)

torch_tpu_oss_torch_test(
    name = "dynamo_unspec_test",
    torch_test_module = "dynamo/test_unspec",
)

torch_tpu_oss_torch_test(
    name = "dynamo_utils_test",
    torch_test_module = "dynamo/test_utils",
)

torch_tpu_oss_torch_test(
    name = "inductor_utils_test",
    torch_test_module = "inductor/test_utils",
)

torch_tpu_oss_torch_test(
    name = "content_store_test",
    torch_test_module = "test_content_store",
)

torch_tpu_oss_torch_test(
    name = "fx_experimental_test",
    torch_test_module = "test_fx_experimental",
)

torch_tpu_oss_torch_test(
    name = "python_dispatch_test",
    torch_test_module = "test_python_dispatch",
)

torch_tpu_oss_torch_test(
    name = "serialization_test",
    extra_deps = [
        "@pypi//dill",
    ],
    torch_test_module = "test_serialization",
)

torch_tpu_oss_torch_test(
    name = "tensor_iterator_test",
    torch_test_module = "test_tensor_iterator",
)
# go/keep-sorted end

torch_tpu_package_end()
