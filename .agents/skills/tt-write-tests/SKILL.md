---
name: tt-write-tests
description: >-
  Provides guidance and patterns for testing in TorchTPU and configuring test
  targets in BUILD and BUILD.bazel files. Use when applying Test-Driven
  Development (TDD) principles, writing unit tests in ops_unit_test.py or CI
  tools under ci/tools/, testing error conditions in errors_test.py or
  tpu_errors_test.py, verifying bounded dynamism in dynamism_test.py, adding or
  modifying test rules in BUILD or BUILD.bazel files, choosing which CI jobs run
  a test or its test tags/parameters, or choosing test doubles (fakes vs mocks,
  mock.patch).
---

# TorchTPU Testing

Cheatsheet for testing in TorchTPU.

## TDD Principles

1.  **Write/Enable Test First**: Verify it fails before implementing the new
    behavior.
2.  **Implement**: Write minimal code to make it pass.
3.  **Refactor**: Clean up code while keeping tests passing.
4.  **Repeat**: Repeat steps 1-3 until the new behavior is complete.

### TDD Enforcement Checklist

*   **Strict Verification**: When requested to verify a failure, run the
    specific test, confirm it fails, and that the failure is related to the
    missing feature or bug.
*   **Success Verification**: After the developer implements the fix, run the
    test again to verify it passes.
*   **Report Results**: Provide clear, concise output.

## Test Locations

*   **`tests/ops_test.py`**: Data-driven tests using PyTorch `op_db`. Use
    `self.do_test_op("{op_name}")`.
*   **`tests/ops_unit_test.py`**: Thorough unit tests for edge cases, dtypes,
    shapes, and non-contiguous inputs. **Always** add tests here.
*   **`tests/errors_test.py`**: Tests ensuring quality and parity (w.r.t. GPU)
    of error messages. Add tests here when the error condition occurs on both
    TPU and GPU (or crashes on GPU). Always try this file first before
    `tpu_errors_test.py`. For more details, see
    [tt-write-error-tests](../tt-write-error-tests/SKILL.md).
*   **`tests/tpu_errors_test.py`**: Tests ensuring quality of error messages for
    TPU-only behaviors (no GPU counterpart). Every test method must be decorated
    with `@et.why_tpu_only("reason")`. For more details, see
    [tt-write-error-tests](../tt-write-error-tests/SKILL.md).
*   **`tests/distributed/collective_errors_test.py`**: Tests for error handling
    in distributed collective operations across multiple TPUs. For more details,
    see [tt-write-error-tests](../tt-write-error-tests/SKILL.md).
*   **`tests/dynamism_test.py`**: Unit tests for bounded dynamism support and
    APIs (`get_dynamism_info`, `mark_dynamic`).

## Test Coverage Guidelines

When testing an operator, ensure coverage for:

*   [ ] Scalar tensors (zero-dimensional, one element).
*   [ ] Zero-element tensors (with a size-0 axis).
*   [ ] Various data types (float32, float16, int32, etc.) and proper casting.
*   [ ] Edge values (NaN, Inf, very large/small numbers).
*   [ ] Error messages for `TT_ERROR`, `TT_RET_CHECK`, and `TT_CHECK_THROW`
    macros.
*   [ ] Dynamic and bounded-dynamic shapes if applicable.
*   [ ] If updating tolerance values, ensure **more strict atol and rtol
    values** are used.
*   [ ] **Do not artificially create error tests on TPU for behaviors that GPU
    supports** (e.g., if GPU supports a dtype, TPU should too).
*   [ ] If error tests are added, ensure they are tested on both GPU and TPU in
    `errors_test.py` to confirm behavior parity. For details, see
    [tt-write-error-tests](../tt-write-error-tests/SKILL.md).

## Test Doubles

Per
["Don't Overuse Mocks"](https://testing.googleblog.com/2013/05/testing-on-toilet-dont-overuse-mocks.html):

*   **Preference order**: real implementation > fake > stub > mock.
*   **Mock only true external boundaries** (network/GitHub, Cloud Storage, RPCs,
    wall clock), and prefer a stateful fake there over call-recording mocks.
*   **Assert observable results**: every test must check a returned value or
    resulting state. `assert_called*` / `call_args` checks alone are not
    sufficient, and never assert a mocked return value back.
*   **Never stub the logic under test** (e.g. `mock.patch` of the functions
    being tested). Swapping a module-level seam (a root-path constant, or a
    boundary dependency such as `subprocess.run`) for a real temp resource or a
    stateful fake is fine. Prefer an injection parameter when the API can take
    one (e.g. `clock=`, `runner=`, `root=`, `pg=`).
*   **No `MagicMock` value objects**: use real `torch.device`,
    `types.SimpleNamespace`, `PretrainedConfig`, or `FakeTensorMode`.
*   **Use real resources where cheap**: temporary directories for the
    filesystem, real (test-scoped) environment variables, and multi-process
    `gloo` groups for `torch.distributed`.
*   **Mutation check**: when adding or rewriting a test, verify that it fails if
    the covered behavior is removed or broken.

Good in-repo examples:

*   `ci/tools/delete_cancelled_pr_runs_test.py` (`_FakeGh` stateful fake).
*   `csrc/common/error_utils_test.cc` (`ScopedFakeProcDir` real temp tree).
*   `tests/_versioned_so_loader_test.py` (real temp-dir inputs; only patches
    `sys.modules` with a `SimpleNamespace` fake).

## Examples

See [Test Examples](references/test_examples.md) for patterns and examples of
different types of tests.

## Running Tests

**NEVER run full test suites**. Use `bazel test <target>
--test_filter=<test_name> --test_sharding_strategy=disabled` to test the
following targets (list a package's test targets with
`bazel query 'tests(//tests:all)'`):

*   `//tests:native_functions_test`: tests native functions registration.
*   `//tests:ops_test`: runs `op_db` tests.
*   `//tests:ops_unit_test`: tests behaviors not covered by `op_db` tests.
*   `//tests:dynamism_test`: tests bounded dynamism

To match a CI job, add its config from `.bazelrc` (e.g. `--config=ci_cpu`).

**CUDA Tests**: Add `--config=cuda` when building or running GPU test targets.

**Debugging Failures**: Do not guess at the cause. Read the test logs (under
`bazel-testlogs/`, or the GitHub Actions logs for CI runs), and compare the
expected and actual values alongside the stack trace.

## Avoid `sys.path` Manipulation

Manipulating `sys.path` (e.g., `sys.path.insert` or `sys.path.append`) is
fragile in general and should be avoided whenever possible. Fallback
`try ... except ImportError` shims can mask real failures, and in tests—where
you control the environment in which the test is run—you can know with certainty
how modules will be imported and should configure the test invocation rather
than mutating `sys.path` in source files.

## CI Jobs & `BUILD` Test Configuration

### 1. CI Jobs

Each CI job is a Bazel config in `.bazelrc` whose `--test_tag_filters` select
the tests it runs, on the machine type bound to it in `.github/workflows/`:

*   **`ci_cpu_presubmit`** and **`ci_tpu_<generation>_presubmit`**: run on every
    pull request.
*   **`ci_cpu_nightly`** and **`ci_tpu_nightly`**: run nightly.

To see which jobs run a given test (or which tests a job runs), use
`ci/tools/list_ci_tests.py`; see `docs/developers/list_ci_tests_guide.md`.

### 2. Choosing CI Jobs in `BUILD` Files

The TorchTPU test macros in `build_files/build_defs.bzl` (e.g.
`torch_tpu_py_test`) take these attributes:

*   **Fast unit & correctness tests (Default)**: set none of them. The test runs
    in the presubmit and nightly jobs for its hardware.
*   **`nopresubmit_oss = "<reason>"`**: built but not run on pull requests; runs
    nightly. Use for slow tests that cannot be parallelized via
    `shard_count = N` (up to `50`).
*   **`nonightly_oss = "<reason>"`**: built but not run in the nightly jobs.
*   **`notest_oss = "<reason>"`**: built but never run. Strongly prefer this to
    `nobuild_oss`, which also skips building.
*   **`oss_presubmit_tpu_generation`**: the TPU generation to run the test on in
    presubmit.

Confirm the result with `ci/tools/list_ci_tests.py` after changing them.

## C++ Tests and the TPU Environment

For C++ unit tests that require `libtpu` (e.g. tests utilizing PjRt to run
computations on TPU), the build macro `torch_tpu_cc_test` automatically manages
the environment packaging.

*   **Default Inference**: By default, the macro **automatically infers**
    whether a test requires `libtpu` based on its hardware tags. If the test
    target is tagged with `requires-tpu` (a generic tag applied to all targets
    that require TPU hardware), `requires_libtpu` defaults to `True`. Otherwise,
    it defaults to `False`.

*   **Explicit Overrides**:

    *   If a C++ test runs on CPU but *still* requires `libtpu` (e.g. compiler
        tests that load PJRT but execute on CPU nodes), you must explicitly set
        `requires_libtpu = True`:

        ```python
        torch_tpu_cc_test(
            name = "split_utils_test",
            srcs = ["split_utils_test.cc"],
            requires_libtpu = True,
            ...
        )
        ```

    *   To explicitly disable wrapping, set `requires_libtpu = False`.

*   **Target Generation**: When wrapping is active, the macro automatically
    wraps the C++ test binary inside a shell script test (`run_cc_tpu_test.sh`)
    that properly configures the `TPU_LIBRARY_PATH` environment variable to
    point to the compiled `libtpu` so the test executes successfully in the TPU
    CI.
