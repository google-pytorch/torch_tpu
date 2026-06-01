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

"""Shim for pytest_test in OSS."""

load("@rules_python//python:py_test.bzl", "py_test")

def pytest_test(name, tests = [], deps = [], args = [], **kwargs):
    """A wrapper around py_test that runs pytest on the specified test files.

    Args:
      name: The name of the test target.
      tests: The test source files containing the pytest test suite.
      deps: Additional dependencies required to run the test suite.
      args: Additional command-line arguments to pass to pytest.
      **kwargs: Additional keyword arguments passed to the underlying py_test rule.
    """
    main_file = name + "_pytest_runner.py"
    runner_content = "import sys; import pytest; sys.exit(pytest.main(sys.argv[1:] + %r))" % tests

    native.genrule(
        name = name + "_gen_runner",
        outs = [main_file],
        cmd = 'echo "%s" > $@' % runner_content,
        testonly = True,
    )

    kwargs.pop("srcs", None)

    py_test(
        # PY_TEST_OK=pytest_test shim macro wraps py_test in OSS
        name = name,
        srcs = tests + [main_file],
        main = main_file,
        deps = deps + ["@pypi//pytest"],
        args = args,
        **kwargs
    )
