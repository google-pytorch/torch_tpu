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
    "-std=c++17",  # Align with pytorch and openxla.
    # Enable exceptions for reporting errors to pytorch. Without this, any
    # C++ exception thrown will immediately crash the process instead of being
    # converted to a Python exception.
    "-fexceptions",
]

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

def torch_tpu_py_library(name, deps = [], redirect_to_wheel = None, **kwargs):
    """Creates a Python library for torch_tpu with automated wheel redirection.

    Args:
        name: The name of the library.
        deps: The dependencies of the library.
        redirect_to_wheel: Whether to redirect this library to the wheel. If None,
            the macro will decide based on the package name.
        **kwargs: Any additional arguments.
    """

    # In OSS, this library redirects to the wheel import when wheel testing is enabled.
    # This prevents duplication and eliminates the need for manual extras lists.
    actual_deps = deps

    if redirect_to_wheel == None:
        pkg = native.package_name()

        # Only core libraries packaged in the wheel should redirect.
        # Test helpers, examples, and google benchmarks stay as source.
        redirect_to_wheel = pkg.startswith("third_party/py/torch_tpu/_internal") or \
                            pkg.startswith("third_party/py/torch_tpu/api") or \
                            pkg == "third_party/py/torch_tpu"

    if redirect_to_wheel:
        native.alias(
            name = name,
            actual = select({
                "//torch_tpu:wheel_test_enabled": "//torch_tpu:torch_tpu_py_import",
                "//conditions:default": ":" + name + "_source",
            }),
        )
        name = name + "_source"

    pytype_strict_library(
        name = name,
        deps = actual_deps,
        **kwargs
    )

def torch_tpu_cc_library(name, copts = None, features = None, **kwargs):
    """Creates a C++ library for torch_tpu.

    Args:
        name: The name of the library.
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.
        **kwargs: Any additional arguments.
    """

    copts, features = adjust_cc_options(copts, features)

    cc_library(
        name = name,
        copts = copts,
        features = features,
        **kwargs
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

# Tags that should propagate to the build_test target
# TODO(b/495860371): Switch to a set when we are using bazel >= 7.7
_BUILD_TEST_ALLOWED_TAGS = [
]

def _check_and_adjust_test_tags(name, size, timeout, nobuild, notap, nopresubmit, nolocal, tags):
    """Validates the test tags.

    Args:
        name: The name of the test.
        size: The size of the test.
        timeout: The timeout of the test.
        nobuild: If given as a string, will not generate a build_test for the test.
        notap: If given as a string, the test will be excluded from TAP, the string will be used
            as the reason, and a build_test named `<name>_build_test` will be added for the test
            to ensure it is buildable.
        nopresubmit: If given as a string, the test will be excluded from presubmit, and the
            string will be used as the reason.
        nolocal: By default, we tag a test as "manual" if either notap or nopresubmit is
            set, so that it is excluded from local `blaze test //torch_tpu/...`
            runs. This behavior can be overridden by setting nolocal to a non-empty
            string - that will add a "manual" tag to the test regardless of notap or nopresubmit,
            and the string will be used as the reason.
        tags: The tags to add to the test.
    """

    # Adjust tags for nobuild.
    if nobuild != None:
        # Enforce that nobuild is a non-empty string.
        if type(nobuild) != "string" or not nobuild:
            fail("nobuild must be a non-empty string documenting why the test " +
                 "should be skipped in build.")
        tags.append("nobuild")  # So that we know that this test shouldn't have a build_test.

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

        if nopresubmit != None:
            fail("notap and nopresubmit cannot both be set.")

        # Skip in local runs as it's unreasonable to ask people to keep a notap test green.
        skip_local = True
    if (size == "enormous" or timeout == "eternal") and not notap:
        fail("Tests with size 'enormous' or timeout 'eternal' are always skipped by TAP. " +
             "Please add a 'notap' argument to torch_tpu_*_test() to make the fact explicit.")

    # Add a build_test for notap test unless nobuild is set.
    if "notap" in tags and nobuild == None:  # NOTAP_OK=for implementing notap logic
        build_test_tags = [tag for tag in tags if tag in _BUILD_TEST_ALLOWED_TAGS]

        # The torch_tpu.cuda build only runs tests with requires-gpu-* tags.
        # Therefore, to ensure that the build_test for a CUDA test is picked
        # up by the torch_tpu.cuda build, we must add a requires-gpu-* tag to the build_test.
        if _is_cuda_test(tags):
            build_test_tags.append("requires-gpu-nvidia")

        build_test(
            name = name + "_build_test",
            targets = [":" + name],
            tags = build_test_tags,
        )

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

    process_accelerator_tags(tags)

def torch_tpu_cc_test(
        name,
        size = None,
        timeout = None,
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
        tags = None,
        **kwargs):
    """Creates a cc_test for torch_tpu.

    Compared to cc_test, this sets default options for best practices.

    Args:
        name: The name of the test.
        size: The size of the test.
        timeout: The timeout of the test.
        copts: The C/C++ compiler options to use.
        features: The blaze features to enable/disable.
        args: The arguments to pass to the test.
        linkstatic: Whether to link statically. We link statically by default to catch duplicate
            definitions and increase accelerator utilization by reducing test run time.
        shuffle_tests: Whether to shuffle the test cases.
        fail_if_no_test_linked: Whether to fail if no tests are linked.
        nobuild: If given as a string, will not generate a build_test for the test.
        notap: If given as a string, the test will be excluded from TAP, the string will be used
            as the reason, and a build_test named `<name>_build_test` will be added for the test
            to ensure it is buildable.
        nopresubmit: If given as a string, the test will be excluded from presubmit, and the
            string will be used as the reason.
        nolocal: By default, we tag a test as "manual" if either notap or nopresubmit is
            set, so that it is excluded from local `blaze test //torch_tpu/...`
            runs. This behavior can be overridden by setting nolocal to a non-empty
            string - that will add a "manual" tag to the test regardless of notap or nopresubmit,
            and the string will be used as the reason.
        tags: The tags to add to the test.
        **kwargs: Any additional arguments.
    """

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

    _check_and_adjust_test_tags(
        name = name,
        size = size,
        timeout = timeout,
        nobuild = nobuild,
        notap = notap,
        nopresubmit = nopresubmit,
        nolocal = nolocal,
        tags = tags,
    )
    cc_test(
        name = name,
        size = size,
        timeout = timeout,
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

def torch_tpu_py_test(
        name,
        args = None,
        shuffle_tests = True,
        extra_pywrap_deps = ["//torch_tpu/common:pywrap_torch_tpu"],
        strict = False,
        size = None,
        timeout = None,
        platform = None,
        nobuild = None,
        notap = None,
        nopresubmit = None,
        nolocal = None,
        tags = None,
        **kwargs):
    """Creates a py_test for torch_tpu.

    Compared to py_test, this sets default options for best practices.

    Args:
        name: The name of the test.
        args: The arguments to pass to the test.
        shuffle_tests: Whether to shuffle the test cases.
        extra_pywrap_deps: Additional pywrap dependencies to add to the test.
        strict: Whether to use pytype.
        size: The size of the test.
        timeout: The timeout of the test.
        platform: The platform to run the test on. Useful for tests on GPU as the
            requires-gpu-* tags are deprecated. If this is set, generates a py_platform_test
            as opposed to a py_test.
        nobuild: If given as a string, will not generate a build_test for the test.
        notap: If given as a string, the test will be excluded from TAP, the string will be used
            as the reason, and a build_test named `<name>_build_test` will be added for the test
            to ensure it is buildable.
        nopresubmit: If given as a string, the test will be excluded from presubmit, and the
            string will be used as the reason.
        nolocal: By default, we tag a test as "manual" if either notap or nopresubmit is
            set, so that it is excluded from local `blaze test //torch_tpu/...`
            runs. This behavior can be overridden by setting nolocal to a non-empty
            string - that will add a "manual" tag to the test regardless of notap or nopresubmit,
            and the string will be used as the reason.
        tags: The tags to add to the test.
        **kwargs: Any additional arguments.
    """

    args = args or []
    if shuffle_tests:
        # Shuffle test cases to avoid test ordering dependencies.
        args = args + ["--test_randomize_ordering_seed=random"]

    tags = tags or []
    _check_and_adjust_test_tags(
        name = name,
        size = size,
        timeout = timeout,
        nobuild = nobuild,
        notap = notap,
        nopresubmit = nopresubmit,
        nolocal = nolocal,
        tags = tags,
    )

    # Remove internal-only attributes
    kwargs.pop("linking_mode", None)  # copybara:comment(oss-only)

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
    existing_env = kwargs.pop("env", {})
    env_with_wheel = dict(existing_env)
    if "LD_LIBRARY_PATH" in existing_env:
        env_with_wheel["LD_LIBRARY_PATH"] = new_paths_str + ":" + existing_env["LD_LIBRARY_PATH"]
    else:
        env_with_wheel["LD_LIBRARY_PATH"] = new_paths_str

    # 4. Use select to swap between the wheel and non-wheel envs
    # copybara:uncomment test_env = existing_env
    # copybara:comment_begin(oss-only)
    test_env = select({
        "//:wheel_test_enabled": env_with_wheel,
        "//conditions:default": existing_env,
    })
    # copybara:comment_end

    deps_to_add = []
    if use_pywrap_rules():
        deps_to_add = extra_pywrap_deps

    current_deps = kwargs.pop("deps", [])

    all_deps = current_deps + deps_to_add

    if platform:
        rule = py_platform_test
        kwargs["platform"] = platform
    elif strict:
        rule = pytype_strict_contrib_test
    else:
        rule = py_test

    if any([tag.startswith("oss_ready_") for tag in (tags or [])]):
        # copybara:comment_begin(oss-only)
        native.filegroup(
            name = name + "_validation",
            srcs = all_deps,
            tags = ["manual", "wheel_validation"],
            visibility = ["//visibility:private"],
        )

        # copybara:comment_end
        pass

    rule(
        name = name,
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

# buildifier: disable=unnamed-macro
def torch_tpu_package_end():
    """Marks the end of a standard package for torch_tpu.

    This macro defines a recursive test suite named "all_tests" and
    a recursive C++ filegroup named "all_cpp_files_recursive"
    for the current package and all subpackages. It MUST be used at the END
    of every BUILD file in torch_tpu (or the "all_tests" group may not
    collect all tests in the package).
    """

    _define_cpp_filegroup(name = "all_cpp_files_recursive")
    _define_test_suite(name = "all_tests")
