# Listing CI Test Targets

`ci/tools/list_ci_tests.py` reports which test targets run in each TorchTPU CI
job, and which CI jobs run a given test target.

## Motivation

Nothing in the repository states directly which tests a CI job runs. The answer
is derived from three places:

*   `.bazelrc` defines a `--test_tag_filters` list for each CI job (a Bazel
    config), for example `test:ci_cpu_presubmit
    --test_tag_filters=-nobuild,-notest,...`.
*   Each test target in a `BUILD` file carries `tags`.
*   `.github/workflows/` binds each job to a machine type.

Working out whether a particular test runs in a particular job therefore means
evaluating Bazel's tag-filter algebra by hand across hundreds of targets. This
tool does that evaluation for you.

It is useful when you want to:

*   Confirm that a new test's tags actually place it in the jobs you intended.
*   Explain a CI escape: was the test genuinely absent from the presubmit job?
*   See the blast radius of a tag change, by diffing counts before and after.
*   Judge how expensive a job is before adding to it.

## Quick start

```shell
$ ci/tools/list_ci_tests.py --format=count
```

```
CI Job Name            Machine Type                          Test Count
-------------------------------------------------------------------------
ci_cpu                 CPU (linux-x86-n4-16)                 841
ci_cpu_nightly         CPU (linux-x86-n4-16)                 839
ci_cpu_presubmit       CPU (linux-x86-n4-16)                 837
ci_tpu_nightly         TPU v6e (linux-x86-ct6e-180-8tpu)     148
ci_tpu_v5              TPU v5e (linux-x86-ct5lp-224-8tpu)    151
ci_tpu_v5_presubmit    TPU v5e (linux-x86-ct5lp-224-8tpu)    97
ci_tpu_v6              TPU v6e (linux-x86-ct6e-180-8tpu)     150
ci_tpu_v6_presubmit    Unknown                               0
ci_tpu_v7              TPU v7x (linux-x86-tpu7x-224-4tpu)    142
ci_tpu_v7_presubmit    TPU v7x (linux-x86-tpu7x-224-4tpu)    33
wheel_test_cpu         CPU (linux-x86-n4-16)                 821
wheel_test_tpu_v5      TPU v5e (linux-x86-ct5lp-224-8tpu)    97
wheel_test_tpu_v6      TPU v6e (linux-x86-ct6e-180-8tpu)     0
wheel_test_tpu_v7      TPU v7x (linux-x86-tpu7x-224-4tpu)    33
```

The first run pays for a `bazel query` over the repository; later runs reuse the
Bazel cache and are fast.

## Recipes

### Which CI jobs exist, and what do they filter on?

`--list_jobs` reads `.bazelrc` only. It does not run `bazel query`, so it
returns immediately.

```shell
$ ci/tools/list_ci_tests.py --list_jobs
```

```
Available CI jobs from /path/to/torch_tpu/.bazelrc:
  ci_cpu [Machine: CPU (linux-x86-n4-16)]: -nobuild,-notest,-requires-gpu,-requires-tpu,-cuda-build-test
  ci_tpu_v5_presubmit [Machine: TPU v5e (linux-x86-ct5lp-224-8tpu)]: -notest,-nobuild,-nopresubmit,-fails-on-tpu-v5,presubmit-v5
  ci_tpu_v7_presubmit [Machine: TPU v7x (linux-x86-tpu7x-224-4tpu)]: -notest,-nobuild,-nopresubmit,-fails-on-tpu-v7,presubmit-v7
  ...
```

### Which CI jobs run a given test?

Pass either a bare target name or a full label.

```shell
$ ci/tools/list_ci_tests.py --test=//csrc/ops/macros:kernel_test
```

```
Test Target: //csrc/ops/macros:kernel_test
Tags: presubmit-v7, requires-tpu, requires-tpu-v5lite
Source: /path/to/torch_tpu/csrc/ops/macros/BUILD

Runs in 6 CI job(s):
  - ci_tpu_nightly [TPU v6e (linux-x86-ct6e-180-8tpu)]
  - ci_tpu_v5 [TPU v5e (linux-x86-ct5lp-224-8tpu)]
  - ci_tpu_v6 [TPU v6e (linux-x86-ct6e-180-8tpu)]
  - ci_tpu_v7 [TPU v7x (linux-x86-tpu7x-224-4tpu)]
  - ci_tpu_v7_presubmit [TPU v7x (linux-x86-tpu7x-224-4tpu)]
  - wheel_test_tpu_v7 [TPU v7x (linux-x86-tpu7x-224-4tpu)]
```

This is the fastest way to check that a tag edit had the effect you wanted. In
the example above `presubmit-v7` is what pulls the target into
`ci_tpu_v7_presubmit`; without it the target would run only in the postsubmit
and nightly jobs.

If the target does not exist the tool prints an error and exits with `1`:

```
Error: Test target 'does_not_exist' was not found in /path/to/torch_tpu.
```

### List every test in one CI job

```shell
$ ci/tools/list_ci_tests.py --jobs=ci_tpu_v7_presubmit
```

```
CI Job: ci_tpu_v7_presubmit [Machine: TPU v7x (linux-x86-tpu7x-224-4tpu)] (33 tests)
  //csrc/ops/macros:kernel_test
  //csrc/ops/macros:logging_test
  //src/torch_tpu/_internal/performance/micro:qwen3_explore
  //tests/compile:activation_checkpointing_test
  //tests/compile:backend_serialization_test
  ...
```

### Narrow the output to a family of tests

`--tests` takes comma-separated globs, matched against both the bare target name
and the full label.

```shell
$ ci/tools/list_ci_tests.py --tests='ops_test*'
```

```
CI Job: ci_cpu [Machine: CPU (linux-x86-n4-16)] (5 tests)
  //tests:ops_test_compiled_tpu-v5lite_build_test
  //tests:ops_test_compiled_tpu-v6e_build_test
  //tests:ops_test_data_build_test
  //tests:ops_test_dump_dtype_exclusions
  //tests:ops_test_perf_build_test

CI Job: ci_cpu_nightly [Machine: CPU (linux-x86-n4-16)] (5 tests)
  ...
```

Quote globs so your shell does not expand them. A glob matching nothing produces
a warning on stderr rather than an error.

### Combine filters

`--jobs` and `--tests` compose, which is the usual way to answer "does this
family of tests run on this accelerator in presubmit?".

```shell
$ ci/tools/list_ci_tests.py \
    --jobs=ci_tpu_v5_presubmit,ci_tpu_v7_presubmit \
    --tests='//tests/compile:*'
```

## How a target is selected

The tool reproduces
[Bazel's `--test_tag_filters` semantics](https://bazel.build/reference/command-line-reference#flag--test_tag_filters):

*   A filter is **negative** if it starts with `-` (e.g. `-requires-gpu`),
    otherwise **positive** (e.g. `presubmit-v5`).
*   If the target carries *any* tag named by a negative filter, it is excluded.
    Exclusion always wins, regardless of ordering.
*   If the filter list contains *any* positive filters, the target must carry at
    least one of them to be included.
*   If there are no positive filters, everything not excluded is included.

So `ci_tpu_v5_presubmit`
(`-notest,-nobuild,-nopresubmit,-fails-on-tpu-v5,presubmit-v5`) runs a target
only if it is tagged `presubmit-v5` and is tagged none of `notest`, `nobuild`,
`nopresubmit` or `fails-on-tpu-v5`.

## Reading the output

*   **`Unknown` machine type** means no `.github/workflows/` file assigns the
    job to a runner. The job's filters are still evaluated; only the machine
    label is missing.
*   **A count of `0`** means no test target satisfies that job's filters. For a
    job with a positive filter, the usual cause is that no target carries the
    required tag.

## Appendix: all flags

<!-- mdformat off(reason: GitHub Flavored Markdown does not support multiline tables) -->

| Flag | Format | Default | Meaning |
| --- | --- | --- | --- |
| `--jobs` | Comma-separated job names | All jobs | Restrict output to these CI jobs. Unknown names produce a warning on stderr and are ignored; the run continues. |
| `--test` | Target name or `//pkg:label` | — | Invert the query: report the target's tags, defining `BUILD` file, and every CI job that runs it. Exits with `1` if the target does not exist. |
| `--tests` | Comma-separated globs | All tests | Restrict output to matching targets. Each glob (`fnmatch` syntax, so `*` and `?`) is tried against both the bare target name and the full label. Non-matching globs warn on stderr. |
| `--format` | `text` \| `count` | `text` | `text` lists every target under its job; `count` prints the one-line-per-job summary table. Ignored when `--test` is used. |
| `--list_jobs` | Flag | off | Print each job with its machine type and tag filters, then exit. Skips `bazel query`, so this is the one instant invocation. |
| `--bazel_bin` | Path | auto | Use this Bazel executable. Exits with `1` if it is missing or fails verification. |
| `-h`, `--help` | Flag | — | Print usage and exit. |

<!-- mdformat on -->
