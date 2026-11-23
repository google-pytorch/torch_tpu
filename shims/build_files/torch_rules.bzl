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

"""OSS wrapper of build definitions to run upstream PyTorch tests with TorchTPU."""

load("//build_files:torch_rules_common.bzl", "torch_tpu_torch_test_impl")
load("//shims/build_cleaner:build_defs.bzl", "register_extension_info")

def torch_tpu_torch_test(
        name,
        torch_test_module,
        nolocal = "The test execution is not hermetic.",
        **kwargs):
    """Creates a test target to run an upstream PyTorch test suite with TorchTPU.

    The PyTorch `test/` checkout is mounted via `@pytorch_test` (configured with
    `--repo_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/path/to/pytorch/test`) so
    each target tracks only its own test module and subsystem helpers in `data`
    runfiles for per-target Bazel remote caching and RBE execution. Passing
    `--test_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/path/to/pytorch/test` at
    test time remains supported as a runtime override.

    Args:
        name: Name of the test target.
        torch_test_module: Name of a test module under https://github.com/pytorch/pytorch/tree/main/test,
            without the .py extension, e.g. "test_torch".
        nolocal: Optional reason to exclude the test from local execution. To enable local
            execution, set to None.
        **kwargs: Additional arguments passed to `torch_tpu_torch_test_impl` macro
            defined in build_files/torch_rules_common.bzl.
    """
    module_target = "@pytorch_test//:{}_module".format(
        torch_test_module.replace("/", "_"),
    )
    existing_data = list(kwargs.pop("data", []))
    data = (
        existing_data if module_target in existing_data else [module_target] + existing_data
    )

    torch_tpu_torch_test_impl(
        name = name,
        torch_test_module = torch_test_module,
        nolocal = nolocal,
        data = data,
        **kwargs
    )

register_extension_info(
    extension = torch_tpu_torch_test,
    label_regex_for_dep = "{extension_name}",
)
