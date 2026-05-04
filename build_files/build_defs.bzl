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

"""Common build definitions for torch_tpu."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_test.bzl", "py_test")
load(
    "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
    "use_pywrap_rules",
)
load("//shims/build_cleaner:build_defs.bzl", "register_extension_info")
load("//shims/build_files:build_defs.bzl", "process_accelerator_tags")
load("//shims/py_platform_test:py_platform_test.bzl", "py_platform_test")
load("//shims/py_rules:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

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

# Default C/C++ build flags for torch_tpu.
_TORCH_TPU_COPTS = [
    "-std=c++20",  # Align with pytorch.
    # Enable exceptions for reporting errors to pytorch. Without this, any
    # C++ exception thrown will immediately crash the process instead of being
    # converted to a Python exception.
    "-fexceptions",
    # Make unused variable a build error.
    "-Werror=unused-variable",
    # Use kineto backend for profiler.
    "-DUSE_KINETO",
]

_LSAN_SUPPRESSIONS = "//build_files:lsan_suppressions.txt"

def is_oss():
    """Returns whether this is an OSS version of torch_tpu."""

    return True  # copybara:comment(oss-only)
    # copybara:uncomment return False

def if_oss(oss_value, internal_value = None):
    """Returns a value based on whether this is an OSS version of torch_tpu."""
    if is_oss():
        return oss_value
    return internal_value

def oss_target(rule_func, name, **kwargs):
    """Returns a target that is defined only in OSS."""
    if is_oss():
        return rule_func(name = name, **kwargs)
    return None

def internal_target(rule_func, name, **kwargs):
    """Returns a target that is defined only in the internal version of torch_tpu."""
    if not is_oss():
        return rule_func(name = name, **kwargs)
    return None

def adjust_cc_options(copts, features):
    """Adjusts the C/C++ build options for torch_tpu.

    Args:
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.

    Returns:
        A tuple of the adjusted C/C++ compiler options and blaze features.
    """

    copts = copts or []
    copts += _TORCH_TPU_COPTS
    features = features or []

    # use_header_modules is enabled by default but incompatible with -fexceptions.
    # Therefore we must explicitly disable it here.
    features.append("-use_header_modules")
    return copts, features

def torch_tpu_cc_library(name, srcs = [], hdrs = [], copts = None, features = None, **kwargs):
    """Creates a C++ library for torch_tpu.

    Also creates a build_test for the library to ensure it is buildable.

    Args:
        name: The name of the library.
        srcs: The source files to compile. Must contain at most one .cc file.
        hdrs: The header files to export by the library. Must contain at most one .h file.
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.
        **kwargs: Any additional arguments.
    """

    if len(srcs) > 1:
        fail("torch_tpu_cc_library must contain at most one srcs file. This reduces build bloat " +
             "and prevents circular dependencies between files.")
    if len(hdrs) > 1:
        fail("torch_tpu_cc_library must contain at most one hdrs file. This reduces build bloat " +
             "and prevents circular dependencies between files.")

    copts, features = adjust_cc_options(copts, features)
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = copts,
        features = features,
        **kwargs
    )
    build_test(
        name = name + "_build_test",
        targets = [":" + name],
    )

# Enable build_cleaner to clean up deps for torch_tpu_cc_library.
register_extension_info(
    extension = torch_tpu_cc_library,
    label_regex_for_dep = "{extension_name}",
)

def _is_cuda_test(tags):
    """Returns true if the test is a CUDA test."""
    for tag in tags:
        if tag.startswith("requires-gpu-"):
            return True
    return False

# Tags that should propagate to the build_test target.
_BUILD_TEST_ALLOWED_TAGS = (
    "nobuild",
)

def _sort_inplace(a_list):
    """Sorts a list in place and returns it."""
    sorted_list = sorted(a_list)
    for i in range(len(a_list)):
        a_list[i] = sorted_list[i]
    return a_list

def _is_errors_test(name):
    """Returns True if the given name is an error test name.

    Equivalent to the regex: .*errors_test.*
    """
    return name.find("errors_test") >= 0

def _check_and_adjust_test_tags(
        name,
        is_oss,
        size,
        timeout,
        nobuild,
        notap,
        nopresubmit,
        nolocal,
        notest_oss,
        nobuild_oss,
        tags):
    """Validates and adjusts the test tags and calculates build test requirements.

    Args:
        name: The name of the test.
        is_oss: Whether the build is for an OSS version of torch_tpu. Normally, this parameter
            will be set to `is_oss()`. However, in tests for this .bzl file it can be set to the
            opposite of `is_oss()` to test the other version. This solves the problem that we
            cannot yet run the .bzl tests in OSS.
        size: The size of the test.
        timeout: The timeout of the test.
        nobuild: If given as a string, will not generate a build_test for the test. This implies
            nobuild_oss.
        notap: If given as a string, the test will be excluded from TAP, the string will be used
            as the reason, and a build_test named `<name>_build_test` will be added for the test
            to ensure it is buildable. This implies notest_oss.
        nopresubmit: If given as a string, the test will be excluded from presubmit, and the
            string will be used as the reason.
        nolocal: By default, we tag a test as "manual" if either notap or nopresubmit is
            set, so that it is excluded from local `blaze test //torch_tpu/...`
            runs. This behavior can be overridden by setting nolocal to a non-empty
            string - that will add a "manual" tag to the test regardless of notap or nopresubmit,
            and the string will be used as the reason.
        notest_oss: If a string is provided, the test will not be run in OSS,
            but it will still be built. The string should be a reason explaining
            why it was disabled.
        nobuild_oss: If a string is provided, the test will not be built nor
            run in OSS. The string provided should be a reason explaining why
            building was disabled. You should strongly prefer notest_oss to
            this!
        tags: The tags to add to the test.

    Returns:
        A struct with fields:
            create_build_test: Boolean, whether to create a build test.
            build_test_tags: List of tags for the build test.

    Side effects:
        Updates the `tags` list with desired tags and sorts it. The sorting is just to
        normalize the list for easy testing.
    """

    # Error tests should not run in *san builds.
    if _is_errors_test(name) and "nosan" not in tags:  # nosan not used in tags.
        fail(name + " is an error test. Please add a 'nosan' tag " +  # nosan not used in tags.
             "to skip *san builds as they may change some error messages and aren't a set-up " +
             "for TorchTPU users.")

    # Adjust tags for nobuild.
    if nobuild != None:
        # Enforce that nobuild is a non-empty string.
        if type(nobuild) != "string" or not nobuild:
            fail("nobuild must be a non-empty string documenting why the test " +
                 "should be skipped in build.")
        tags.append("nobuild")  # So that we know that this test shouldn't have a build_test.

    # Adjust tags for nobuild_oss and notest_oss
    #
    # This must come before the logic adding the build_test
    if nobuild_oss != None:
        if type(nobuild_oss) != "string" or not nobuild_oss:
            fail("nobuild_oss must be a non-empty string documenting why the test " +
                 "should be skipped in OSS build.")

        # nobuild_oss should only affect the OSS build.
        if is_oss and "nobuild" not in tags:
            tags.append("nobuild")

        # We want to allow both notest_oss and nobuild_oss to be set because
        # sometimes the reason for disabling build is different from the reason
        # for disabling tests (e.g. build is broken but also the tests are too
        # slow to run even if the build were working). We will still try to
        # discourage setting both values for the same root cause.
        if notest_oss == nobuild_oss:
            fail("nobuild_oss implies notest_oss, so there is no reason to set both " +
                 "with the same reason, either set only nobuild_oss or explain " +
                 "both reasons.")
        if notest_oss == None:
            notest_oss = nobuild_oss

    if notest_oss != None:
        if type(notest_oss) != "string" or not notest_oss:
            fail("notest_oss must be a non-empty string documenting why the test " +
                 "should be skipped in OSS tests.")
        tags.append("notest")

    # Adjust tags for notap.
    #
    # Whether to skip the test in local `blaze test //torch_tpu/...` runs.
    skip_local = False
    if "notap" in tags:  # NOTAP_OK=for implementing notap logic
        fail("notap must be passed as an argument to torch_tpu_cc_test, not as a tag.")
    if notap != None:
        # Enforce that notap is a non-empty string.
        if type(notap) != "string" or not notap:
            fail("notap must be a non-empty string documenting why the test " +
                 "should be skipped on TAP.")
        tags.append("notap")  # NOTAP_OK=for implementing notap logic
        if "notest" not in tags:
            tags.append("notest")  # notap implies notest_oss.

        if nopresubmit != None:
            fail("notap and nopresubmit cannot both be set.")

        # Skip in local runs as it's unreasonable to ask people to keep a notap test green.
        skip_local = True
    if (size == "enormous" or timeout == "eternal") and not notap:
        fail("Tests with size 'enormous' or timeout 'eternal' are always skipped by TAP. " +
             "Please add a 'notap' argument to torch_tpu_*_test() to make the fact explicit.")

    # Add a build_test for notap and notest_oss test unless nobuild is set.
    # Targets with notest_oss cannot be built in OSS because flag `--build_tests_only` is active.
    # Instead of removing `--build_tests_only` to include all unused binaries/libraries in OSS
    # builds, we generate a companion _build_test target for notest_oss tests.
    create_build_test = False
    build_test_tags = []
    if is_oss:
        create_build_test = "notest" in tags and "nobuild" not in tags
    else:
        create_build_test = "notap" in tags and "nobuild" not in tags  # NOTAP_OK=for implementing notap logic
    if create_build_test:
        build_test_tags = [tag for tag in tags if tag in _BUILD_TEST_ALLOWED_TAGS]

        # The torch_tpu.cuda build only runs tests with requires-gpu-* tags.
        # Therefore, to ensure that the build_test for a CUDA test is picked
        # up by the torch_tpu.cuda build, we must add a requires-gpu-* tag to the build_test.
        if _is_cuda_test(tags):
            build_test_tags.append("requires-gpu-nvidia")

    # Adjust tags for nopresubmit.
    if nopresubmit != None:
        # Enforce that nopresubmit is a non-empty string.
        if type(nopresubmit) != "string" or not nopresubmit:
            fail("nopresubmit must be a non-empty string documenting why the test " +
                 "should be skipped in presubmit.")

        # We only have a presubmit build for CUDA tests. Therefore, skipping a CUDA test
        # in presubmit means it won't run on TAP at all. To make this explicit, we don't
        # allow nopresubmit for CUDA tests - they must use notap instead.
        if _is_cuda_test(tags):
            fail("CUDA tests must use notap instead of nopresubmit, as there's no postsubmit build for CUDA.")

        if "nofastbuild" not in tags:
            # This tag causes the test to be skipped in presubmit, as we only run
            # fastbuild tests in presubmit.
            tags.append("nofastbuild")

            # Skip in local runs as it's unreasonable to ask people to keep a
            # nopresubmit test green.
            skip_local = True

    # Adjust tags for nolocal.
    if "manual" in tags:
        fail("Do not use the 'manual' tag to exclude the test from matching pattern " +
             "wildcards like //torch_tpu/... - notap or nopresubmit already " +
             "implies 'manual'. If you want to force the test to be manual, add a " +
             "'nolocal = \"<reason>\",' argument to torch_tpu_*_test() instead.")
    if nolocal != None:
        # Enforce that nolocal is a non-empty string.
        if type(nolocal) != "string" or not nolocal:
            fail("nolocal must be a non-empty string documenting why the test " +
                 "should be skipped in local runs.")
        skip_local = True
    if skip_local:
        # This tag causes the test to be skipped when a user runs
        # `blaze test //torch_tpu/...`.
        tags.append("manual")

    if "oss_ready_cpu" in tags or "oss_ready_tpu" in tags:
        fail("The 'oss_ready_cpu' and 'oss_ready_tpu' tags no longer have any " +
             "effect. Tests are enabled by default in OSS.")

    process_accelerator_tags(tags)
    _sort_inplace(tags)

    return struct(
        create_build_test = create_build_test,
        build_test_tags = build_test_tags,
    )

def check_and_adjust_test_tags_for_testing(
        is_oss,
        name = "test",
        size = "small",
        timeout = "short",
        nobuild = None,
        notap = None,
        nopresubmit = None,
        nolocal = None,
        notest_oss = None,
        nobuild_oss = None,
        tags = None):
    """Public wrapper for testing."""
    if tags == None:
        tags = []
    return _check_and_adjust_test_tags(
        name = name,
        is_oss = is_oss,
        size = size,
        timeout = timeout,
        nobuild = nobuild,
        notap = notap,
        nopresubmit = nopresubmit,
        nolocal = nolocal,
        notest_oss = notest_oss,
        nobuild_oss = nobuild_oss,
        tags = tags,
    )

def torch_tpu_cc_test(
        name,
        size = None,
        timeout = None,
        srcs = [],
        copts = None,
        features = None,
        args = None,
        linkstatic = True,
        shuffle_tests = True,
        fail_if_no_test_linked = True,
        nobuild = None,
        notap = None,
        nopresubmit = None,
        nolocal = None,
        notest_oss = None,
        nobuild_oss = None,
        tags = None,
        **kwargs):
    """Creates a cc_test for torch_tpu.

    Compared to cc_test, this sets default options for best practices.

    Args:
        name: The name of the test.
        size: The size of the test.
        timeout: The timeout of the test.
        srcs: The source files to compile. Must contain at most one .cc file.
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.
        args: The arguments to pass to the test.
        linkstatic: Whether to link statically. We link statically by default to catch duplicate
            definitions and increase accelerator utilization by reducing test run time.
        shuffle_tests: Whether to shuffle the test cases.
        fail_if_no_test_linked: Whether to fail if no tests are linked.
        nobuild: If given as a string, will not generate a build_test for the test. This implies
            nobuild_oss.
        notap: If given as a string, the test will be excluded from TAP, the string will be used
            as the reason, and a build_test named `<name>_build_test` will be added for the test
            to ensure it is buildable. This implies notest_oss.
        nopresubmit: If given as a string, the test will be excluded from presubmit, and the
            string will be used as the reason.
        nolocal: By default, we tag a test as "manual" if either notap or nopresubmit is
            set, so that it is excluded from local `blaze test //torch_tpu/...`
            runs. This behavior can be overridden by setting nolocal to a non-empty
            string - that will add a "manual" tag to the test regardless of notap or nopresubmit,
            and the string will be used as the reason.
        notest_oss: If a string is provided, the test will not be run in OSS,
            but it will still be built. The string should be a reason explaining
            why it was disabled.
        nobuild_oss: If a string is provided, the test will not be built nor
            run in OSS. The string provided should be a reason explaining why
            building was disabled. You should strongly prefer notest_oss to
            this!
        tags: The tags to add to the test.
        **kwargs: Any additional arguments.
    """

    if len(srcs) > 1:
        fail("torch_tpu_cc_test must contain at most one srcs file. This prevents build bloat " +
             "and circular dependencies between files.")

    copts, features = adjust_cc_options(copts, features)

    args = args or []
    if shuffle_tests:
        # Shuffle tests to avoid test ordering dependencies.
        args = args + ["--gunit_shuffle"]
    if fail_if_no_test_linked:
        # Fail if no tests are linked. This is to avoid having a test target that does not run any
        # tests. This can happen if the test's link options are not set correctly.
        args = args + ["--gunit_fail_if_no_test_linked"]
    tags = tags or []
    data = kwargs.pop("data", [])
    if is_oss():
        data.append("@bazel_tools//tools/bash/runfiles")
    kwargs["data"] = data

    result = _check_and_adjust_test_tags(
        name = name,
        is_oss = is_oss(),
        size = size,
        timeout = timeout,
        nobuild = nobuild,
        notap = notap,
        nopresubmit = nopresubmit,
        nolocal = nolocal,
        notest_oss = notest_oss,
        nobuild_oss = nobuild_oss,
        tags = tags,
    )
    if result.create_build_test:
        build_test(
            name = name + "_build_test",
            targets = [":" + name],
            tags = result.build_test_tags,
        )
    cc_test(
        name = name,
        size = size,
        timeout = timeout,
        srcs = srcs,
        copts = copts,
        args = args,
        features = features,
        linkstatic = linkstatic,
        tags = tags,
        **kwargs
    )

# Enable build_cleaner to clean up deps for torch_tpu_cc_test.
register_extension_info(
    extension = torch_tpu_cc_test,
    label_regex_for_dep = "{extension_name}",
)

def torch_tpu_py_library(name, srcs = [], **kwargs):
    """Creates a pytype_strict_library for torch_tpu.

    Also creates a build_test for the library to ensure it is buildable.

    Args:
        name: The name of the library.
        srcs: The source files for the library. Must contain at most one .py file.
        **kwargs: Any additional arguments.
    """

    if len(srcs) > 1:
        fail("torch_tpu_py_library must contain at most one srcs file. This prevents build bloat " +
             "and circular dependencies between files.")

    pytype_strict_library(
        # PY_LIBRARY_OK=for implementing torch_tpu_py_library.
        name = name,
        srcs = srcs,
        **kwargs
    )
    build_test(
        name = name + "_build_test",
        targets = [":" + name],
    )

# Enable build_cleaner to clean up deps for torch_tpu_py_library.
register_extension_info(
    extension = torch_tpu_py_library,
    label_regex_for_dep = "{extension_name}",
)

def _prepend_to_env(env, key, value):
    """Prepends a value to an environment variable in a dictionary.

    Args:
        env: The dictionary of environment variables.
        key: The key of the environment variable.
        value: The value to prepend.
    """
    if key in env:
        env[key] = value + ":" + env[key]
    else:
        env[key] = value

def torch_tpu_py_test(
        name,
        srcs = [],
        deps = [],
        args = None,
        shuffle_tests = True,
        autoload = True,
        extra_pywrap_deps = ["//torch_tpu/common:pywrap_torch_tpu"],
        strict = False,
        size = None,
        timeout = None,
        platform = None,
        nobuild = None,
        notap = None,
        nopresubmit = None,
        nolocal = None,
        notest_oss = None,
        nobuild_oss = None,
        tags = None,
        **kwargs):
    """Creates a py_test for torch_tpu.

    Compared to py_test, this sets default options for best practices.

    Args:
        name: The name of the test.
        srcs: The source files for the test. Must contain at most one .py file.
        deps: The dependencies to add to the test.
        args: The arguments to pass to the test.
        shuffle_tests: Whether to shuffle the test cases.
        extra_pywrap_deps: Additional pywrap dependencies to add to the test.
        strict: Whether to use pytype.
        size: The size of the test.
        timeout: The timeout of the test.
        platform: The platform to run the test on. Useful for tests on GPU as the
            requires-gpu-* tags are deprecated. If this is set, generates a py_platform_test
            as opposed to a py_test.
        nobuild: If given as a string, will not generate a build_test for the test. This implies
            nobuild_oss.
        notap: If given as a string, the test will be excluded from TAP, the string will be used
            as the reason, and a build_test named `<name>_build_test` will be added for the test
            to ensure it is buildable. This implies notest_oss.
        nopresubmit: If given as a string, the test will be excluded from presubmit, and the
            string will be used as the reason.
        nolocal: By default, we tag a test as "manual" if either notap or nopresubmit is
            set, so that it is excluded from local `blaze test //torch_tpu/...`
            runs. This behavior can be overridden by setting nolocal to a non-empty
            string - that will add a "manual" tag to the test regardless of notap or nopresubmit,
            and the string will be used as the reason.
        notest_oss: If a string is provided, the test will not be run in OSS,
            but it will still be built. The string should be a reason explaining
            why it was disabled.
        nobuild_oss: If a string is provided, the test will not be built nor
            run in OSS. The string provided should be a reason explaining why
            building was disabled. You should strongly prefer notest_oss to
            this!
        tags: The tags to add to the test.
        autoload: Enable autoload during the tests.
        **kwargs: Any additional arguments.
    """

    if len(srcs) > 1:
        fail("torch_tpu_py_test must contain at most one srcs file. This prevents build bloat " +
             "and circular dependencies between files.")

    args = args or []
    if shuffle_tests:
        # Shuffle test cases to avoid test ordering dependencies.
        args = args + ["--test_randomize_ordering_seed=random"]
    tags = tags or []
    data = kwargs.pop("data", [])
    if is_oss():
        data.append("@bazel_tools//tools/bash/runfiles")
    kwargs["data"] = data
    result = _check_and_adjust_test_tags(
        name = name,
        is_oss = is_oss(),
        size = size,
        timeout = timeout,
        nobuild = nobuild,
        notap = notap,
        nopresubmit = nopresubmit,
        nolocal = nolocal,
        notest_oss = notest_oss,
        nobuild_oss = nobuild_oss,
        tags = tags,
    )
    if result.create_build_test:
        build_test(
            name = name + "_build_test",
            targets = [":" + name],
            tags = result.build_test_tags,
        )

    # Remove internal-only attributes
    if is_oss():
        kwargs.pop("linking_mode", None)

    existing_env = kwargs.pop("env", {})

    # Opt-in to autoloading on a per-test basis
    if autoload:
        existing_autoload = existing_env.get("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
        if existing_autoload != None:
            fail("Autoload behavior is intended to be controlled by the" +
                 "autoload parameter rather than setting " +
                 "TORCH_DEVICE_BACKEND_AUTOLOAD directly")
    else:
        existing_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    # Add env and LD_LIBRARY_PATH for wheel based testing.
    # Define the necessary library paths

    # 1. Define all paths needed for the dynamic linker
    torch_lib = "../pypi_torch/site-packages/torch/lib"
    libtpu_lib = "../pypi_libtpu/site-packages/libtpu"
    wheel_lib = "../torch_tpu_py_import_unpacked_wheel/torch_tpu/_internal"
    solib_path = "../_solib_x86_64"

    # 2. Join them into a single string
    all_paths = [torch_lib, libtpu_lib, wheel_lib, solib_path]
    new_paths_str = ":".join(all_paths)

    # 3. Create the specialized environment for wheel testing
    env_with_wheel = dict(existing_env)
    _prepend_to_env(env_with_wheel, "LD_LIBRARY_PATH", new_paths_str)

    # Local Torch + Wheel test environment
    # We include local paths for LD_LIBRARY_PATH and PYTHONPATH
    env_with_local_torch = dict(existing_env)

    local_torch_lib = "torch_tpu_py_import_unpacked_wheel/torch/lib"
    local_torch_lib_alt = "__main__/torch_tpu_py_import_unpacked_wheel/torch/lib"
    local_paths = [local_torch_lib, local_torch_lib_alt]
    local_ld_path = ":".join(local_paths) + ":" + new_paths_str
    _prepend_to_env(env_with_local_torch, "LD_LIBRARY_PATH", local_ld_path)

    local_torch_python = "../local_torch/site-packages"
    unpacked_wheel_path = "../torch_tpu_py_import_unpacked_wheel"
    _prepend_to_env(env_with_local_torch, "PYTHONPATH", ":".join([unpacked_wheel_path, local_torch_python]))

    # Add LSAN suppressions for known third-party leaks (e.g., safetensors, pyo3)
    # that are outside the project's control.
    current_data = kwargs.pop("data", [])
    if _LSAN_SUPPRESSIONS not in current_data:
        current_data.append(_LSAN_SUPPRESSIONS)
    kwargs["data"] = current_data

    # We use LSAN_OPTIONS to pass the suppressions file.
    # Note: ASAN_OPTIONS=detect_leaks=1 is usually the default in ASAN configs.
    lsan_supps = "$(location %s)" % _LSAN_SUPPRESSIONS

    def _add_lsan_options(env):
        opts = env.get("LSAN_OPTIONS", "")
        if opts != "":
            opts += " "
        env["LSAN_OPTIONS"] = opts + "suppressions=" + lsan_supps

    _add_lsan_options(existing_env)
    _add_lsan_options(env_with_wheel)
    _add_lsan_options(env_with_local_torch)

    # 4. Use select to swap between the wheel and non-wheel envs
    test_env = if_oss(select({
        "//:wheel_test_with_local_torch": env_with_local_torch,
        "//:wheel_test_enabled": env_with_wheel,
        "//conditions:default": existing_env,
    }), existing_env)

    if "//torch_tpu" not in deps:
        fail("torch_tpu_py_test must include \"//torch_tpu\" in its deps to " +
             "ensure that torch_tpu is loaded.")

    deps_to_add = []
    if use_pywrap_rules():
        deps_to_add = extra_pywrap_deps

    all_deps = if_oss(
        select({
            "//:wheel_test_enabled": ["//:torch_tpu_py_import"],
            "//conditions:default": deps + deps_to_add,
        }),
        deps + deps_to_add,
    )

    if platform:
        rule = py_platform_test
        kwargs["platform"] = platform
    elif strict:
        rule = pytype_strict_contrib_test
    else:
        rule = py_test
    rule(
        name = name,
        srcs = srcs,
        args = args,
        size = size,
        timeout = timeout,
        deps = all_deps,
        env = test_env,
        tags = tags,
        **kwargs
    )

# Enable build_cleaner to clean up deps for torch_tpu_py_test.
register_extension_info(
    extension = torch_tpu_py_test,
    label_regex_for_dep = "{extension_name}",
)

def _get_subpackage_targets_named(name):
    """Gets targets with the given name from all direct subpackages.

    Example:
        If the current package is `foo` and it has subpackages `foo/bar`,
        `foo/baz`, and `foo/bar/qux`, calling
        `_get_subpackage_targets_named("my_target")` from the BUILD file in `foo`
        will return:
        `["//foo/bar:my_target", "//foo/baz:my_target"]`, but not
        `//foo/bar/qux:my_target` since it is not a direct subpackage.

    Args:
        name: The name of the target to collect from each subpackage.

    Returns:
        A list of full target labels.
    """
    current_package = native.package_name()
    targets = []
    for subpackage in native.subpackages(
        # Include all direct subpackages.
        include = ["**"],
        # Allow the include matching results to be empty.
        allow_empty = True,
    ):
        target_path = "//{}/{}:{}".format(current_package, subpackage, name)
        targets.append(target_path)
    return targets

def _define_test_suite(name):
    """Collectors all tests in the current package and subpackages recursively.

    Args:
        name: The name of the test suite.
    """

    test_targets = []

    # IMPORTANT: existing_rules() only returns rules in the current package
    # that have been seen SO FAR. Therefore, to collect all existing tests,
    # torch_tpu_package_end() must be called at the END of the BUILD file.
    for rule_name, info in native.existing_rules().items():
        if info["kind"].endswith("_test"):
            test_targets.append(":" + rule_name)

    native.test_suite(
        name = name + "_in_this_lib_",
        tests = test_targets,
        tags = [
            # Exclude the test suite from both local and TAP runs, as it's
            # only meant for collecting test targets for the notap coverage
            # test.
            "manual",
            "notap",  # NOTAP_OK=for collecting test targets only
        ],
    )

    native.test_suite(
        name = name,
        tests = ([":" + name + "_in_this_lib_"] +
                 _get_subpackage_targets_named(name = name)),
        tags = [
            # Exclude the test suite from both local and TAP runs, as it's
            # only meant for collecting test targets for the notap coverage
            # test.
            "manual",
            "notap",  # NOTAP_OK=for collecting test targets only
        ],
        visibility = ["//:__subpackages__"],
    )

def _define_cpp_filegroup(name):
    """Defines filegroups for all C++ files in the current package and subpackages.

    The filegroups created are:
    - "all_cpp_files": All C++ files in the current package.
    - "all_cpp_files_recursive": All C++ files in the current package and
    subpackages recursively.

    Args:
        name: The name of the filegroup. It has to be "all_cpp_files_recursive"
            to collect files from subpackages recursively.
    """
    if name != "all_cpp_files_recursive":
        fail("The name must be 'all_cpp_files_recursive' to collect files recursively.")
    native.filegroup(
        name = "all_cpp_files",
        srcs = native.glob([
            "**/*.cc",
            "**/*.h",
        ]),
        visibility = ["//:__subpackages__"],
    )
    native.filegroup(
        name = name,
        srcs = [":all_cpp_files"] + _get_subpackage_targets_named(
            name = name,
        ),
        visibility = ["//:__subpackages__"],
    )

def _define_py_filegroup(name):
    """Defines filegroups for all Python files in the current package and subpackages.

    The filegroups created are:
    - "all_py_files": All Python files in the current package.
    - "all_py_files_recursive": All Python files in the current package and subpackages recursively.

    Args:
        name: The name of the filegroup. It has to be "all_py_files_recursive"
            to collect files from subpackages recursively.
    """
    if name != "all_py_files_recursive":
        fail("The name must be 'all_py_files_recursive' to collect files recursively.")
    native.filegroup(
        name = "all_py_files",
        srcs = native.glob(["**/*.py"]),
        visibility = ["//visibility:public"],
    )
    native.filegroup(
        name = name,
        srcs = [":all_py_files"] + _get_subpackage_targets_named(
            name = name,
        ),
        visibility = ["//visibility:public"],
    )

# buildifier: disable=unnamed-macro
def torch_tpu_package_end():
    """Marks the end of a standard package for torch_tpu.

    This macro defines a recursive test suite named "all_tests",
    a recursive Python filegroup named "all_py_files_recursive" and
    a recursive C++ filegroup named "all_cpp_files_recursive"
    for the current package and all subpackages. It MUST be used at the END
    of every BUILD file in torch_tpu (or the "all_tests" group may not
    collect all tests in the package).
    """

    _define_cpp_filegroup(name = "all_cpp_files_recursive")
    _define_py_filegroup(name = "all_py_files_recursive")
    _define_test_suite(name = "all_tests")
