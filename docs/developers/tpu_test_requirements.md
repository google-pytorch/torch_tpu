# Declaring TPU Requirements for Tests

`torch_tpu_py_test` and `torch_tpu_cc_test` describe a TPU test's hardware with
two parameters that answer different questions:

*   `compatible_accelerators`: which TPUs **can** the test run on? This is a
    property of the test, such as needing SparseCore or 4 chips. It only changes
    when the test or the hardware changes.
*   `run_on`: which of those TPUs **should** CI run it on in presubmit,
    postsubmit, and nightly? This is a cost and coverage decision that changes
    as capacity and priorities change.

Separating them means that compatibility stays accurate when CI scheduling
changes, that tests become eligible for new TPU models they support, and that
every deviation from the default schedule is explicit. Most tests only need
`compatible_accelerators`.

The helpers below come from `//build_files:accelerator_specification.bzl`. These
parameters replace TPU hardware and scheduling tags (such as
`requires-tpu-v5lite` and `fails-on-tpu-v7`), `run_on_accelerators`, and
`oss_presubmit_tpu_generation`, so don't combine them on the same test.

## Declaring compatibility

Build `compatible_accelerators` with `tpus()`. Each argument narrows the set of
TPUs:

```python
tpus()                       # Any TPU.
tpus(since = "v6")           # v6 and newer.
tpus(exclude = ["v7"])       # Any TPU except v7.
tpus(only = ["v5p", "v7x"])  # Only these; can't combine with since/exclude.
tpus(min_chips = 8)          # At least 8 chips on one host (v5e and v6e).
SPARSECORE_TPUS              # TPUs with SparseCore.
```

`since` takes a generation (`"v6"`); `only` and `exclude` take generations or
models (`"v5"` or `"v5e"`). Describe what the test supports, not where you want
it to run: narrowing compatibility to save CI time hides real coverage. Use
`run_on` for that.

## Scheduling CI runs

Without `run_on`, presubmit runs the test on one compatible TPU, and postsubmit
and nightly cover every compatible TPU that TorchTPU CI has runners for.

To change that, set `run_on = run_on_config(reason = "...", ...)` and choose,
for each of `presubmit`, `postsubmit`, and `nightly`:

*   `ANY_TPU`: one compatible TPU, picked automatically. Don't rely on which
    one; list TPUs explicitly if it matters.
*   `ALL_TPUS`: every compatible TPU.
*   `[tpu_config("v7"), ...]`: specific generations or models from
    `compatible_accelerators`. A generation picks one compatible model of that
    generation.
*   `[]`: skip the workflow.

`reason` is required whenever you change a default, so that readers know why the
test isn't scheduled like the others. Keep in mind:

*   Some TPUs (currently v5p) have no TorchTPU CI runners. If a test only asks
    for TPUs that TorchTPU CI doesn't have, the build fails with an error that
    explains the options.
*   Not every combination can be scheduled yet. Unsupported ones fail at build
    time with an error that says why.
*   `nopresubmit`, `nopresubmit_oss`, `nonightly_oss`, `notest_oss`, and `notap`
    work independently of `run_on`. Use them as temporary markers for broken or
    flaky tests, and `run_on` for deliberate scheduling decisions.

To check which CI jobs end up running a test, see
[Listing CI test targets](list_ci_tests_guide.md).

## Examples

A test that runs on any TPU, with the default schedule:

```python
load("//build_files:accelerator_specification.bzl", "tpus")
load("//build_files:build_defs.bzl", "torch_tpu_py_test")

torch_tpu_py_test(
    name = "add_test",
    srcs = ["add_test.py"],
    deps = ["//src/torch_tpu"],
    compatible_accelerators = tpus(),
)
```

A slow test that runs on any TPU but only needs postsubmit coverage. Its
schedule changes, not its compatibility:

```python
compatible_accelerators = tpus(),
run_on = run_on_config(
    reason = "Takes 10 minutes; postsubmit coverage is enough.",
    presubmit = [],
),
```

A test for a feature that only v7 and newer support:

```python
compatible_accelerators = tpus(since = "v7"),
```

A test whose numerics can differ between v5e and v6e, so postsubmit covers both:

```python
compatible_accelerators = tpus(only = ["v5e", "v6e"]),
run_on = run_on_config(
    reason = "Numerics differ between v5e and v6e.",
    postsubmit = [tpu_config("v5e"), tpu_config("v6e")],
),
```
