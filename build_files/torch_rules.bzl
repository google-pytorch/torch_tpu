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

"""Build definitions to run upstream PyTorch tests with TorchTPU."""

load("//build_files:build_defs.bzl", "torch_tpu_py_test")
load("//shims/build_cleaner:build_defs.bzl", "register_extension_info")

# This file is torch_tpu implementation details and should not be imported by
# other projects.
visibility(
    [
        "//...",
        # Visibility is not ignored for .bzl loads in experimental
        # unlike regular targets.
        # copybara:uncomment "//experimental/users/...",
    ],
)

def torch_tpu_torch_test(
        name,
        torch_test_module,
        tags = None,
        **kwargs):
    """Creates a Python test target to run an upstream PyTorch test suite with TorchTPU.

    The test execution is not hermetic: it is required to set up a https://github.com/pytorch/pytorch
    checkout and specify its path via `TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR` environment variable:
      $ bazel test --test_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/path/to/pytorch/test ...

    Args:
        name: Name of the test target.
        torch_test_module: Name of a test module under https://github.com/pytorch/pytorch/tree/main/test,
            without the .py extension, e.g. "test_torch".
        tags: Optional additional tags.
        **kwargs: Additional arguments passed to `torch_tpu_py_test`.
    """

    # Prefer environment variables for simplicity because the test runner
    # needs to rewrite sys.argv to conform with PyTorch testing entry point.
    env = {
        "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE": torch_test_module,
    }
    env.update(kwargs.pop("env", {}))

    tags = [
        "requires-tpu-v5lite",
        # Exclude wheel tests due to lack of a PyTorch checkout.
        "wheel_test_excluded",
    ] + (tags or [])

    torch_tpu_py_test(
        name = name,
        srcs = ["//tests/torch:torch_test_runner.py"],
        main = "//tests/torch:torch_test_runner.py",
        env = env,
        nolocal = "The test execution is not hermetic.",
        nonightly_oss = "Test in development. To be enabled to nightly once stable.",
        nopresubmit_oss = "Presubmits do not set up a PyTorch checkout to keep lightweight.",
        # Upstream tests parse sys.argv with unittest,
        # which rejects absltest test-shuffling flag.
        shuffle_tests = False,
        tags = tags,
        timeout = kwargs.pop("timeout", "long"),
        deps = [
            # go/keep-sorted start
            "@pypi//dill",
            "@pypi//hypothesis",
            "@pypi//jax",
            "@pypi//packaging",
            "@pypi//pytest",
            "@pypi//scipy",
            "//shims/torch:testing_internal",
            "//src/torch_tpu",
            "@pypi//pyyaml",
            # go/keep-sorted end
        ],
        **kwargs
    )

register_extension_info(
    extension = torch_tpu_torch_test,
    label_regex_for_dep = "{extension_name}",
)
