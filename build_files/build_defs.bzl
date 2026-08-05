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
load("//bazel:supported_python_versions.bzl", "SUPPORTED_PYTHON_VERSIONS")
load("//shims/build_cleaner:build_defs.bzl", "register_extension_info")
load("//shims/build_files:build_defs.bzl", "process_accelerator_tags")
load("//shims/build_files:torch_version.bzl", "is_backend_dep", "reset_torch_config")
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
    # Enforce that member variables are initialized in the same order they are declared.
    "-Werror=reorder-ctor",
    # Enforce thread safety analysis.
    "-Werror=thread-safety-precise",
    # Use kineto backend for profiler.
    "-DUSE_KINETO",
    # By default, all compiler warnings in third_party/... headers are suppressed.
    # We must explicitly enable them for TorchTPU headers.
    "--no-system-header-prefix=third_party/py/torch_tpu/",
]

# The default ranking of which hardware generations to prioritize when automatically
# assigning an OSS presubmit runner to a test.
TPU_GENERATION_PREFERENCES = ["v5", "v7", "v6"]

def tpu_gen(generation, reason = None):
    """Helper to configure explicit TPU generation requirements.

    Args:
        generation: The TPU generation to run on, e.g. "v5", "v6", "v7".
        reason: The reason why a specific generation must be used.

    Returns:
        A dict containing the generation and reason.
    """
    return {"generation": generation, "reason": reason}

_LSAN_SUPPRESSIONS = "//build_files:lsan_suppressions.txt"
_TSAN_SUPPRESSIONS = "//build_files:tsan_suppressions.txt"

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

def _route_backend_deps_through_fixed(name, deps):
    """Routes external backend deps through the reset_torch_config transition.

    In a wheel build, routes external backend deps through the
    reset_torch_config torch_version-reset transition, so the backend is built
    in a single configuration and pywrap factors it into one shared library
    instead of one copy per PyTorch version. A bazel-only build keeps the deps
    unchanged.
    """
    if type(deps) != "list":
        # A non-list deps value -- a bare select(), or the common
        # `[...] + select({...})` (a SelectorList) -- is opaque to Starlark: it
        # cannot be iterated or decomposed, so a backend @-dep hidden inside it
        # cannot be discovered and routed through reset_torch_config. Passing
        # it through would let such a dep silently escape pinning and fragment
        # the shared XLA base, so refuse the shape outright rather than
        # miscompile. Keep every dep (backend and conditional alike) in a plain
        # list; select() is not supported for torch_tpu_cc_library deps.
        fail(
            ("torch_tpu_cc_library {}: `deps` must be a plain list so backend " +
             "@-deps can be routed to the shared XLA base, but got a select(). " +
             "Backend deps hidden in a select cannot be pinned; keep deps a " +
             "plain list.").format(name),
        )
    pinned = [d for d in deps if is_backend_dep(d)]
    if not pinned:
        return deps
    kept = [d for d in deps if not is_backend_dep(d)]
    wrapped = []
    for i, dep in enumerate(pinned):
        fixed_name = "_{}_backend_fixed_{}".format(name, i)
        reset_torch_config(
            name = fixed_name,
            dep = dep,
            visibility = ["//visibility:private"],
        )
        wrapped.append(":" + fixed_name)
    return kept + select({
        "//:wheel_build_enabled": wrapped,
        "//conditions:default": pinned,
    })

# Exported for build_defs_test.bzl.
is_backend_dep_for_testing = is_backend_dep

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

    kwargs["deps"] = _route_backend_deps_through_fixed(name, kwargs.get("deps", []))

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
        nopresubmit_oss,
        nonightly_oss,
        oss_presubmit_tpu_generation,
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
        nopresubmit_oss: If a string is provided, the test will be built but not run in the
            non-nightly OSS, and it will be built and run in the nightly OSS.
        nonightly_oss: If a string is provided, the test will be built but not run in the nightly
            OSS.
        oss_presubmit_tpu_generation: Optional integer specifying the TPU generation to run this
            test on in OSS presubmit.
        tags: The tags to add to the test.

    Returns:
        A struct with fields:
            create_build_test: Boolean, whether to create a build test.
            build_test_tags: List of tags for the build test.

    Side effects:
        Updates the `tags` list with desired tags and sorts it. The sorting is just to
        normalize the list for easy testing.
    """

    # Use the nopresubmit parameter instead of tags.
    if "nopresubmit" in tags:
        fail("Please use the `nopresubmit=\"<reason>\"` parameter instead " +
             "of the 'nopresubmit' tag so that you can document the reason.")
    if "postsubmit" in tags:
        fail("Please use the `nopresubmit=\"<reason>\"` parameter instead " +
             "of the 'postsubmit' tag so that you can document the reason.")

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

        # notest_oss should only affect the OSS build.
        if is_oss:
            tags.append("notest")

    if nonightly_oss != None:
        if type(nonightly_oss) != "string" or not nonightly_oss:
            fail("nonightly_oss must be a non-empty string documenting why the test " +
                 "should be skipped in the nightly OSS.")

        # nonightly_oss should only affect the OSS build.
        if is_oss and "nonightly" not in tags:
            tags.append("nonightly")

    if nopresubmit_oss != None:
        if type(nopresubmit_oss) != "string" or not nopresubmit_oss:
            fail("nopresubmit_oss must be a non-empty string documenting why the test " +
                 "should be skipped in the non-nightly OSS.")

        # nopresubmit_oss should only affect the OSS build.
        if is_oss and "nopresubmit" not in tags:
            tags.append("nopresubmit")

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
        create_build_test = ("notest" in tags or "nonightly" in tags or "nopresubmit" in tags) and "nobuild" not in tags
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

        if "nopresubmit" not in tags:
            # This tag causes the test to be skipped in presubmit.
            tags.append("nopresubmit")

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

    if is_oss:
        if oss_presubmit_tpu_generation != None:
            if type(oss_presubmit_tpu_generation) != "dict" or "generation" not in oss_presubmit_tpu_generation or "reason" not in oss_presubmit_tpu_generation:
                fail("oss_presubmit_tpu_generation must be a dict generated by tpu_gen().")
            if not oss_presubmit_tpu_generation["reason"]:
                fail("A reason must be provided when explicitly specifying an oss_presubmit_tpu_generation.")
            gen = oss_presubmit_tpu_generation["generation"]
            if ("fails-on-tpu-" + gen) in tags:
                fail("Test cannot specify oss_presubmit_tpu_generation generation=\"%s\" and fails-on-tpu-%s." % (gen, gen))
            tags.append("presubmit-" + gen)
        else:
            has_tpu = any([t.startswith("requires-tpu") for t in tags])
            if has_tpu:
                allocated = False
                for gen in TPU_GENERATION_PREFERENCES:
                    if ("fails-on-tpu-" + gen) not in tags:
                        tags.append("presubmit-" + gen)
                        allocated = True
                        break

                if not allocated:
                    fail("TPU tests failing on all generations in OSS should use notest_oss instead.")

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
        nopresubmit_oss = None,
        nonightly_oss = None,
        oss_presubmit_tpu_generation = None,
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
        nopresubmit_oss = nopresubmit_oss,
        nonightly_oss = nonightly_oss,
        oss_presubmit_tpu_generation = oss_presubmit_tpu_generation,
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
        nopresubmit_oss = None,
        nonightly_oss = None,
        oss_presubmit_tpu_generation = None,
        tags = None,
        requires_libtpu = None,
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
        nopresubmit_oss: If a string is provided, the test will be built but not run in the
            non-nightly OSS, and it will be built and run in the nightly OSS.
        nonightly_oss: If a string is provided, the test will be built but not run in the nightly
            OSS.
        oss_presubmit_tpu_generation: Optional integer specifying the TPU generation to run this
            test on in OSS presubmit.
        requires_libtpu: If True, wraps the C++ test inside an sh_test wrapper
            in OSS to load libtpu.so cleanly. If None (default), the value is set
            depending on whether the test is declared as running on TPUs (`True`)
            or CPUs (`False`).
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
        args = args + ["--gtest_shuffle" if is_oss() else "--gunit_shuffle"]
    if fail_if_no_test_linked:
        # Fail if no tests are linked. This is to avoid having a test target that does not run any
        # tests. This can happen if the test's link options are not set correctly.
        args = args + [
            "--gtest_fail_if_no_test_linked" if is_oss() else "--gunit_fail_if_no_test_linked",
        ]

    # Prevent the massive output of per-thread stack traces when a test
    # crashes. Such logs are usually more painful than useful. One can
    # opt out of this behavior by setting `--suppress_failure_output`
    # in a test target's `args` field, or by passing
    # `--test_arg=--nosuppress_failure_output` on the bazel command line.
    #
    # Note that we add this arg before the user-provided args so that the
    # user can override it in the `args` field (the last arg wins).
    if not is_oss():
        args = ["--suppress_failure_output"] + args
    tags = tags or []
    data = kwargs.pop("data", [])
    if is_oss():
        data.append("@bazel_tools//tools/bash/runfiles")
    kwargs["data"] = data

    cc_test_name = name
    cc_test_tags = tags
    effective_tags = list(tags)

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
        nopresubmit_oss = nopresubmit_oss,
        nonightly_oss = nonightly_oss,
        oss_presubmit_tpu_generation = oss_presubmit_tpu_generation,
        tags = effective_tags,
    )

    if requires_libtpu == None:
        # requires-tpu is a generic tag automatically applied whenever at least
        # one requires-tpu-X tag is present, where X is a TPU generation.
        requires_libtpu = is_oss() and "requires-tpu" in effective_tags

    is_wrapped = is_oss() and requires_libtpu

    # TODO(b/479574836): This is implemented with a hack to set up the libtpu
    # environment; refactor this when libtpu has an API equivalent to the Python
    # initialization logic.
    if is_wrapped:
        cc_test_name = name + "_bin"
        cc_test_tags = list(effective_tags)

        # Tag the C++ binary as manual/notest to prevent direct execution in
        # OSS; it should only run via the sh_test wrapper.
        for t in ["manual", "notest"]:
            if t not in cc_test_tags:
                cc_test_tags.append(t)

        # Define a wrapper around the C++ binary that sets up the libtpu environment
        sh_test_kwargs = {}

        # List of cc attributes we don't want to forward to sh_test.
        _INVALID_SH_TEST_ATTRS = [
            "deps",
            "copts",
            "linkopts",
            "includes",
            "defines",
            "local_defines",
            "linkstatic",
            "malloc",
            "stamp",
            "alwayslink",
            "additional_compiler_inputs",
            "textual_hdrs",
            "exec_properties",  # Handled explicitly below
            "data",  # Passed explicitly below
        ]
        for attr, value in kwargs.items():
            if attr not in _INVALID_SH_TEST_ATTRS:
                sh_test_kwargs[attr] = value

        if "exec_properties" in kwargs and kwargs["exec_properties"]:
            # Filter out C++ execution group properties (e.g. cpp_link, cpp_compile)
            # because sh_test does not have those execution groups.
            filtered_exec_properties = {
                k: v
                for k, v in kwargs["exec_properties"].items()
                if not k.startswith("cpp_link.") and not k.startswith("cpp_compile.")
            }
            if filtered_exec_properties:
                sh_test_kwargs["exec_properties"] = filtered_exec_properties

        native.sh_test(
            name = name,
            size = size,
            timeout = timeout,
            srcs = ["//ci/tools:run_cc_tpu_test.sh"],
            args = ["$(location :" + cc_test_name + ")"] + args,
            data = [
                ":" + cc_test_name,
                "//shims/libtpu:pypi_libtpu_whl",
            ] + data,
            tags = effective_tags,
            **sh_test_kwargs
        )
    else:
        cc_test_tags = effective_tags

    if result.create_build_test:
        build_test(
            name = name + "_build_test",
            targets = [":" + name],
            tags = result.build_test_tags,
        )

    # Define the C++ test target (handles compilation, compilation options, and static linking)
    cc_test(
        name = cc_test_name,
        size = size,
        timeout = timeout,
        srcs = srcs,
        copts = copts,
        args = args,
        features = features,
        linkstatic = linkstatic,
        tags = cc_test_tags,
        **kwargs
    )

# Enable build_cleaner to clean up deps for torch_tpu_cc_test.
register_extension_info(
    extension = torch_tpu_cc_test,
    label_regex_for_dep = "{extension_name}",
)

def torch_tpu_py_library(name, srcs = [], allow_multiple_srcs = False, **kwargs):
    """Creates a pytype_strict_library for torch_tpu.

    Also creates a build_test for the library to ensure it is buildable.

    Args:
        name: The name of the library.
        srcs: The source files for the library. Must contain at most one .py file unless allow_multiple_srcs is True.
        allow_multiple_srcs: Whether to allow multiple srcs files.
        **kwargs: Any additional arguments.
    """

    if len(srcs) > 1 and not allow_multiple_srcs:
        fail("torch_tpu_py_library must contain at most one srcs file. This prevents build bloat " +
             "and circular dependencies between files.")

    deps = kwargs.pop("deps", [])
    if not is_oss():
        if type(deps) == "list":
            if "//torch_tpu/_internal:testing" not in deps:
                deps = deps + ["//torch_tpu/_internal:testing"]
        else:
            deps = deps + ["//torch_tpu/_internal:testing"]

    pytype_strict_library(
        # PY_LIBRARY_OK=for implementing torch_tpu_py_library.
        name = name,
        srcs = srcs,
        deps = deps,
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
        nopresubmit_oss = None,
        nonightly_oss = None,
        oss_presubmit_tpu_generation = None,
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
        nopresubmit_oss: If a string is provided, the test will be built but not run in the
            non-nightly OSS, and it will be built and run in the nightly OSS.
        nonightly_oss: If a string is provided, the test will be built but not run in the nightly
            OSS.
        oss_presubmit_tpu_generation: Optional integer specifying the TPU generation to run this
            test on in OSS presubmit.
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

    # See the comment about suppress_failure_output in torch_tpu_cc_test
    # for more details.
    if not is_oss():
        # Python tests in OSS don't support flags, so we only do this for the
        # internal build.
        args = ["--suppress_failure_output"] + args
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
        nopresubmit_oss = nopresubmit_oss,
        nonightly_oss = nonightly_oss,
        oss_presubmit_tpu_generation = oss_presubmit_tpu_generation,
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
    kwargs.pop("imports", None)

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

    # Add environment variants for wheel and local-torch testing.
    # We define shared library (LD_LIBRARY_PATH) and Python paths (PYTHONPATH) here.

    # 1. Define sets of paths
    # Standard paths for stable PyPI/wheel mode
    std_ld = [
        "../pypi_torch/site-packages/torch/lib",
        "../rules_python++pip+torch_tpu_pypi_312_torch/site-packages/torch/lib",
        "../pypi_libtpu/site-packages/libtpu",
        "../rules_python++pip+torch_tpu_pypi_312_libtpu_nightly/site-packages/libtpu",
        "../torch_tpu_py_import_unpacked_wheel/torch_tpu/_internal",
        "../_solib_x86_64",
    ]

    # Extra paths for local development mode
    local_ld = [
        "local_torch/site-packages/torch/lib",
        "../local_torch/site-packages/torch/lib",
        "torch_tpu_py_import_unpacked_wheel/torch/lib",
        "__main__/torch_tpu_py_import_unpacked_wheel/torch/lib",
        "../+_repo_rules+local_torch/site-packages/torch/lib",
    ]

    local_py = [
        "torch_tpu_py_import_unpacked_wheel",
        "../torch_tpu_py_import_unpacked_wheel",
        "local_torch/site-packages",
        "../local_torch/site-packages",
    ]

    # 2. Build environment variants starting from base environment
    base_env = existing_env

    # Apply LeakSanitizer suppressions (propagates to copies)
    lsan_opts = [base_env.get("LSAN_OPTIONS", "")]
    lsan_opts.append("suppressions=$(location %s)" % _LSAN_SUPPRESSIONS)
    base_env["LSAN_OPTIONS"] = " ".join([opt for opt in lsan_opts if opt])

    # Apply ThreadSanitizer suppressions (propagates to copies)
    tsan_opts = [base_env.get("TSAN_OPTIONS", "")]
    tsan_opts.append("suppressions=$(location %s)" % _TSAN_SUPPRESSIONS)
    base_env["TSAN_OPTIONS"] = " ".join([opt for opt in tsan_opts if opt])

    # Apply AddressSanitizer options.
    # Suppress new_delete_type_mismatch caused by an upstream OpenXLA/TSL
    # async_value.h deletion alignment mismatch (not caused by torch_tpu).
    asan_opts = [base_env.get("ASAN_OPTIONS", "")]
    asan_opts.append("new_delete_type_mismatch=0")
    base_env["ASAN_OPTIONS"] = ":".join([opt for opt in asan_opts if opt])

    # Prepend standard library paths to base env (required for mp.spawn support)
    _prepend_to_env(base_env, "LD_LIBRARY_PATH", ":".join(std_ld))

    env_wheel = dict(base_env)

    env_wheel_versions = {}
    for v in SUPPORTED_PYTHON_VERSIONS:
        env_v = dict(env_wheel)
        std_ld_v = [
            "../pypi_torch_{}/site-packages/torch/lib".format(v.replace(".", "")),
            "../rules_python++pip+torch_tpu_pypi_{}_torch/site-packages/torch/lib".format(v.replace(".", "")),
            "../pypi_libtpu_{}/site-packages/libtpu".format(v.replace(".", "")),
            "../rules_python++pip+torch_tpu_pypi_{}_libtpu_nightly/site-packages/libtpu".format(v.replace(".", "")),
            "../torch_tpu_py_import_unpacked_wheel/torch_tpu/_internal",
            "../_solib_x86_64",
        ]
        _prepend_to_env(env_v, "LD_LIBRARY_PATH", ":".join(std_ld_v))
        env_wheel_versions[v.replace(".", "")] = env_v

    env_local = dict(env_wheel)  # Build on top of wheel paths
    _prepend_to_env(env_local, "LD_LIBRARY_PATH", ":".join(local_ld))
    _prepend_to_env(env_local, "PYTHONPATH", ":".join(local_py))

    # 3. Add LSan and TSan suppressions for known third-party leaks/races
    # that are outside the project's control.
    current_data = kwargs.pop("data", [])
    if _LSAN_SUPPRESSIONS not in current_data:
        current_data.append(_LSAN_SUPPRESSIONS)
    if _TSAN_SUPPRESSIONS not in current_data:
        current_data.append(_TSAN_SUPPRESSIONS)
    kwargs["data"] = current_data

    # 4. Use select to swap between environments
    select_dict = {
        "//:wheel_test_with_local_torch": env_local,
        "//shims/torch:use_local_torch": env_local,
    }
    for v in SUPPORTED_PYTHON_VERSIONS:
        select_dict["//:wheel_test_with_local_torch_" + v.replace(".", "_")] = env_local
        select_dict["//:wheel_test_" + v.replace(".", "_")] = env_wheel_versions[v.replace(".", "")]

    # Fallback for 3.12 if --test_wheel=True is passed without --define PYTHON_VERSION
    select_dict["//:wheel_test_enabled"] = env_wheel_versions["312"]
    select_dict["//conditions:default"] = base_env

    test_env = if_oss(select(select_dict), base_env)

    if "//torch_tpu" not in deps:
        fail("torch_tpu_py_test must include \"//torch_tpu\" in its deps to " +
             "ensure that torch_tpu is loaded.")

    deps_to_add = []
    if use_pywrap_rules():
        deps_to_add = extra_pywrap_deps

    if not is_oss():
        if type(deps) == "list":
            if "//torch_tpu/_internal:testing" not in deps:
                deps = deps + ["//torch_tpu/_internal:testing"]
        else:
            deps = deps + ["//torch_tpu/_internal:testing"]

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
        # The no-ide tag prevents IDEs from associating source files with this
        # helper filegroup instead of their actual test/library targets.
        tags = ["no-ide"],
    )
    native.filegroup(
        name = name,
        srcs = [":all_cpp_files"] + _get_subpackage_targets_named(
            name = name,
        ),
        visibility = ["//:__subpackages__"],
        # See the comment on no-ide above.
        tags = ["no-ide"],
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
        # See the comment on no-ide above.
        tags = ["no-ide"],
    )
    native.filegroup(
        name = name,
        srcs = [":all_py_files"] + _get_subpackage_targets_named(
            name = name,
        ),
        visibility = ["//visibility:public"],
        # See the comment on no-ide above.
        tags = ["no-ide"],
    )

def _define_py_wheel_test_site_packages(name):
    """Defines filegroups for Python files excluding tests, for wheel tests.

    The filegroups created are:
    - "wheel_test_site_packages": Python files in the current package,
      excluding tests.
    - "wheel_test_site_packages_recursive": Python files in the current
      package and subpackages recursively, excluding tests.

    Args:
        name: The name of the filegroup. It has to be
            "wheel_test_site_packages_recursive" to collect files from
            subpackages recursively.
    """
    if name != "wheel_test_site_packages_recursive":
        fail(
            "The name must be 'wheel_test_site_packages_recursive' to collect " +
            "files recursively.",
        )
    native.filegroup(
        name = "wheel_test_site_packages",
        srcs = native.glob(
            ["**/*.py"],
            exclude = [
                "**/*_test.py",
                "**/*_tests.py",
            ],
        ),
        visibility = ["//visibility:public"],
        # See the comment on no-ide above.
        tags = ["no-ide"],
    )
    native.filegroup(
        name = name,
        srcs = [
            ":wheel_test_site_packages",
        ] + _get_subpackage_targets_named(
            name = name,
        ),
        visibility = ["//visibility:public"],
        # See the comment on no-ide above.
        tags = ["no-ide"],
    )

# buildifier: disable=unnamed-macro
def torch_tpu_package_end():
    """Marks the end of a standard package for torch_tpu.

    This macro defines a recursive test suite named "all_tests",
    a recursive Python filegroup named "all_py_files_recursive",
    a recursive Python filegroup of non-test files named
    "wheel_test_site_packages_recursive", and a recursive C++ filegroup
    named "all_cpp_files_recursive" for the current package and all
    subpackages. It MUST be used at the END of every BUILD file in
    torch_tpu (or the "all_tests" group may not collect all tests in the
    package).
    """

    _define_cpp_filegroup(name = "all_cpp_files_recursive")
    _define_py_filegroup(name = "all_py_files_recursive")
    _define_py_wheel_test_site_packages(
        name = "wheel_test_site_packages_recursive",
    )
    _define_test_suite(name = "all_tests")

def if_cuda_dep(dep):
    """Returns the dependency list if building with CUDA, else empty."""
    if is_oss():
        return []
    return select({
        "//shims/gpu_cuda:using_config_cuda": [dep],
        "//conditions:default": [],
    })
