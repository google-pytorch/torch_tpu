# Running upstream PyTorch tests against TorchTPU

This document describes how TorchTPU runs the upstream PyTorch device-generic
test suites — everything materialized by `instantiate_device_type_tests()` in
`pytorch/test/test_*.py` — against its PrivateUse1 (`tpu`) backend.

## TL;DR

```sh
# 1. Generate the target inventory from a pytorch/pytorch checkout
#    (v2.13.0, matching the wheel selected below):
bazel run --//:_torch_version=2.13.0 //tests:gen_torch_tests -- \
  --pytorch_test_dir=/path/to/pytorch/test

# 2. Run everything (what the upstream_tests.yaml workflow does):
bazel test --//:_torch_version=2.13.0 --test_timeout=3600 --keep_going \
  --test_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/path/to/pytorch/test \
  //tests/torch:all

# A single upstream suite:
bazel test --//:_torch_version=2.13.0 \
  --test_env=TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR=/path/to/pytorch/test \
  //tests/torch:test_torch

# Filter inside a suite (translated to unittest -k):
bazel test ... //tests/torch:test_torch --test_filter=cdist
```

The harness requires a torch that ships
`PrivateUse1TestBase.set_test_configs` (**torch 2.13+**, pytorch commit
`0e11b74ce4`). The repo's default pin is older, so both commands select the
2.13.0 wheel from the existing multi-ABI version matrix with
`--//:_torch_version=2.13.0` — no local torch build or source checkout is
needed; the same flag with the nightly channel's version (see
`NIGHTLY_TORCH_VERSION` in `bazel/pytorch_versions.bzl`) runs the suites
against the torch nightly wheel, which is the intended regular cadence once
2.13 is burned down.

## Design

**Principle: TorchTPU tracks the newest PyTorch it can.** The harness is
built on the extension point PyTorch designed for out-of-tree backends, not
on version-compatibility shims. On an older torch the runner exits early
with a notice.

**Principle: collect everything, then exclude.** Coverage is derived
mechanically at generation time, never hand-allowlisted, so upstream
additions cannot silently fall out of scope. There is no API that lists the
tests `instantiate_device_type_tests` would create (confirmed with the
OpenReg maintainers; extending `run_test.py --dry-run` upstream is the
long-term answer), so discovery *asks the framework itself* instead of
parsing text — see `gen_torch_tests.py` below.

The pieces, all under `tests/`:

- **`gen_torch_tests.py`** — generates `tests/torch/BUILD` (not checked in;
  gitignored) from a pytorch checkout, one `torch_tpu_oss_torch_test` target
  per eligible module, named after the file: `test_nn.py` → `test_nn`,
  `nn/test_pooling.py` → `nn_test_pooling`. Discovery is programmatic: each
  upstream `test_*.py` (whole tree except `distributed/`, which is deferred:
  its suites need multi-process/multi-device orchestration) is imported in a
  fresh subprocess with a fake python-only `tpu` PrivateUse1 backend
  registered (no hardware, no torch_tpu) and
  `PYTORCH_TESTING_DEVICE_ONLY_FOR=tpu`; the probe counts the
  `DeviceTypeTestBase` subclasses `instantiate_device_type_tests()` actually
  materialized. A module is eligible iff that count is nonzero — the
  framework applies its own `only_for`/`except_for` rules, so there are no
  text heuristics to drift. Newly added upstream modules are wired
  automatically with default kwargs; a module can only *leave* coverage by
  becoming ineligible upstream or by an explicit deny-list entry. An
  unprobeable module that mentions `instantiate_device_type_tests` and is
  classified nowhere fails generation, forcing a human decision.

- **`torch_test_targets.py`** — the only hand-maintained state:
  `TARGET_KWARGS` (per-module `shard_count`, `extra_deps`, `args` where the
  defaults don't fit) and `OMITTED_TEST_MODULES` (deliberate exclusions,
  each with a reason: restricted instantiation, another backend's own
  self-tests, suites that compile C++ extensions at import,
  onnxruntime-dependent suites). The deny-list overrules discovery.

- **`run_torch_tests.py`** — the driver. Executes the unmodified upstream
  file (named by `TORCH_TPU_INTERNAL_TORCH_TEST_MODULE`) as `__main__` via
  `runpy.run_path`, exactly like `python test/test_torch.py` upstream (the
  driver replicates what the CPython launcher would add to `sys.path`, since
  `run_path` itself does not). Running the file — rather than importing its
  classes — is required: `instantiate_device_type_tests()` strips the
  generic tests from the template class at import time, so the instantiated
  per-device classes only exist in that module's namespace, and running the
  module also triggers its own `if __name__ == "__main__": run_tests()`
  entry point. The driver adds what plain unittest lacks under Bazel: the
  sharding protocol (below), `--test_filter` → `-k`, always-verbose output
  (a backend C++ CHECK failure aborts the process; the last started test in
  the log is the culprit), and "no tests ran" treated as success (exclusions
  may legitimately empty a module). It guards on
  `PrivateUse1TestBase.set_test_configs` being available and exits early
  otherwise.

  *Result readability*: the driver routes upstream `--save-xml` output
  (JUnit XML via unittest-xml-reporting, upstream's own reporting path)
  into Bazel's `XML_OUTPUT_FILE`, merging the per-class reports into one
  file. CI runs use `--config=resultstore_base` (BES), so ResultStore /
  Build Explorer shows a per-test tree — names like
  `TestSortAndSelectTPU.test_msort_tpu_bfloat16` with timings —
  instead of one opaque pass/fail per shard.

  *Sharding*: with `shard_count = N`, Bazel starts N independent runner
  processes, each told `TEST_TOTAL_SHARDS=N` and its `TEST_SHARD_INDEX`.
  Each process loads the full suite, then filters unittest's loader so a
  test runs iff `md5(qualified test id) % N == shard index` — a
  deterministic, disjoint, complete partition needing no coordination
  (stdlib `hash()` is per-process randomized and would not agree across
  shards). Every process must also touch `TEST_SHARD_STATUS_FILE` — on
  every exit path, including the version-guard early exit — or Bazel
  rejects the shard. In CI, `parallel_accelerator_execute.sh` assigns each
  shard process its own TPU chip, so shards run concurrently.

- **`torch_test_configs.py`** — the TorchTPU capability envelope: the
  *only* place that decides which tests run inside a module, applied by
  `install(module)` before the upstream module is imported (the
  configuration is consumed at import time; reference usage: pytorch
  `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py`).
  Every entry is a process-crasher with its cause; ordinary failures stay
  visible. Four layers:
  1. *Backend dtype support* — every OpInfo gets `dtypesIf["tpu"]` (the
     per-backend dtype declaration `OpInfo.supported_dtypes()` is built
     around) equal to its dtypes minus `_CRASHING_DTYPES` (complex128
     globally) and the module's `_MODULE_CRASHING_DTYPES` (e.g. complex64
     in `test_decomp`, whose decompositions compute complex64 references
     in complex128 on device); `@dtypes` tests are filtered the same way.
     Excluded variants are *never instantiated*. Parametrizers that take
     dtype lists straight from their decorator (`@optims`, ...) are
     covered by the driver's loader-level backstop on the name token.
  2. *`_TEST_EXCLUSIONS`* → `set_test_configs(test_exclusions=...)`: whole
     generic methods or classes (`"*"`), keyed by the generic class name
     (torch 2.13 matches on the generic member, so one entry covers every
     op/dtype variant of that method).
  3. *`_OP_OVERRIDES`* → `set_test_configs(op_overrides=...)`: per-op
     `DecorateInfo(unittest.skip)` entries, optionally scoped to a generic
     class, test name and dtypes — upstream's own mechanism for op_db
     suites (e.g. `cdouble` everywhere; `nn.functional.*` only in
     `TestDecomp`; float64 `nn.functional.*` only in the gradient suites).
     Expanded from the `_OP_OVERRIDE_RULES` table.
  4. *`_TEST_NAME_EXCLUSIONS`* — the residue `set_test_configs` cannot
     express: tests in plain (non-device-generic) classes and
     device-generic variants parametrized by module_db or layouts.
     Regexes over `ClassName.test_name`, applied in the driver's loader.

  Class naming: `install()` runs `PrivateUse1TestBase.setUpClass()` up
  front, so every instantiated class is `<Generic>TPU`. Upstream renames
  the base's device type lazily on the first instantiated test; without
  the up-front call the first class in a module is `FooPRIVATEUSE1` and
  later ones `FooTPU`, and `op_overrides` (device-matched against "tpu")
  silently misses the first class.

- **`torch_tpu_oss_torch_test`** (in `//build_files:build_defs.bzl`,
  mirroring `torch_tpu_oss_torch_example`) — the macro each generated target
  instantiates. TPU-only: CPU runs of upstream suites exercise stock
  PyTorch, not TorchTPU, so none are generated.

The upstream sources are **not** part of the Bazel graph: the
`upstream_tests.yaml` workflow checks out `pytorch/pytorch` (`pytorch-ref`
dispatch input, default `v2.13.0`), selects the matching wheel
(`torch-version` input → `--//:_torch_version`), runs the generator, and
then `bazel test //tests/torch:all` with
`TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR` pointing at the checkout's `test/`
directory.

### Running against a specific torch wheel

`--//:_torch_version=<v>` pins the whole dev build — the C++ glue *and*, via
`torch_py_select` in `shims/torch/torch_wheel_select.bzl`, the Python torch
package — to one of the versions in the existing multi-ABI matrix
(`bazel/pytorch_versions.bzl`). The extra-version pip hubs are torch-only
locks, so the rest of the environment (numpy, scipy, ...) comes from the
default lock; torch's own Python deps are supplied from there too. This is
what makes "the 2.13 wheel just works" true for local testing: no
`local_torch` build is required. (`--config=local_torch` still works for
running against an arbitrary local torch build, e.g. unreleased trunk.)

## How PrivateUse1 instantiation works

1. `import torch_tpu` registers the `tpu` PrivateUse1 backend and the
   `torch.tpu` device module.
2. `common_device_type` adds `PrivateUse1TestBase` to the active test bases
   whenever `torch.tpu.is_available()` is true.
3. `instantiate_device_type_tests(TestFoo, globals())` creates
   `TestFoo<DEVICE_TYPE>` — `TestFooTPU` here, because the harness resolves
   the base's device type to the backend name up front (upstream does it
   lazily in `setUpClass`, which is why the suffix would otherwise flip
   from `PRIVATEUSE1` to `TPU` after the first class). Tests are named
   `test_bar_tpu[_dtype]` and receive `device="tpu:0"`.
   `PYTORCH_TESTING_DEVICE_ONLY_FOR=tpu` (set by the Bazel target)
   restricts instantiation to the TPU.
4. `set_test_configs` (torch 2.13+) is consumed during that instantiation,
   so excluded variants never exist.

## Triage workflow

1. `bazel test ... //tests/torch:<name> --test_timeout=3600`
2. **Crash** (SIGABRT/CHECK/SIGSEGV/OOM): the last started test in the
   verbose log is the culprit. Whole method/class → `_TEST_EXCLUSIONS`;
   one op (optionally one class/test/dtype) → `_OP_OVERRIDE_RULES`; a
   dtype the suite's own machinery upcasts everywhere →
   `_MODULE_CRASHING_DTYPES`; plain-class or module_db/layout variants →
   `_TEST_NAME_EXCLUSIONS`. Annotate the cause. **Failure**: fix the op;
   never hide it.
3. Re-run until the suite completes; remove entries as the backend improves.

Known backend CHECK-crash bugs this surfaced (each aborts the whole
process; file and link from the config):

1. libtpu `xprecision_rewriter.cc:2100`: fatal "Unsupported CVT X{64|128}
   expansion" on any lowering reaching complex128 — the dominant crash
   cause, also reached from complex64/float64 tests that upconvert
   internally. Until fixed (or until the pin gains per-dtype exclusion
   rules), affected tests are excluded whole.
2. libtpu `xprecision_rewriter.cc:6184`: tuple-shape CHECK
   (test_scatter_gather_ops).
3. libtpu `xprecision_rewriter.cc:3270`: `operand->shape().IsTuple()` CHECK
   (test_torch).
4. torch_tpu `tensor_to_buffer.cc:205`: DLPack import of a PrivateUse1
   tensor not owned by the TPU allocator (test_dlpack; whole class
   excluded).
5. torch_tpu `device_rt.cc:236`: fatal "PjRtClient is null" when TPU
   devices are busy — concurrent processes abort instead of reporting
   unavailability. Also fatal in `DataLoader` forked workers
   (test_dataloader hangs its shard).
6. torch_tpu `AdaptiveAvgPool2dHelper`: SIGFPE (division by a zero
   dimension) — nn/test_pooling.
7. libtpu `xla::jellyfish` `ElementPackingFactor`: SIGILL on a sub-byte
   dtype lowering — nn/test_pooling.
8. torch_tpu `GetOutputDtypeFromTensorAndScalar`: SIGSEGV — test_foreach
   scalar-overload sweeps.

## Status

Measured 2026-08-25 on a TPU v7 dev box (4 chips), torch 2.13.0 wheel +
pytorch v2.13.0 sources, TorchTPU at `dec6f4f8`, with the capability
envelope in `torch_test_configs.py` and **no TorchTPU code changes**.

**Coverage.** 935 non-`distributed/` `test_*.py` probed; **92 instantiate
device-generic tests for tpu** — 91 wired, 1 deliberately omitted (OpenReg's
own self-test). 109 files are unprobeable in the hermetic environment
(optional deps, `GRAPH_EXECUTOR`, import-time C++ extension builds); none is
both eligible and unclassified.

**Test cases.** 183,239 device-generic cases instantiate with no envelope.
The envelope withholds **25,430 (13.9%)** as known process-crashers, leaving
**157,809** eligible; the same files add 8,583 plain-class cases.

**Result of the run** (348 shards):

| | collected | skipped by upstream | ran | passed | failed |
|---|---|---|---|---|---|
| device-generic | 142,994 | 49,625 | 93,369 | **61,010** | 32,359 |
| plain classes | 7,500 | 2,142 | 5,358 | 5,139 | 219 |
| **total** | **150,495** | 51,767 | **98,728** | **66,150 (67.0%)** | **32,578** |

150,495 of the 157,809 eligible cases returned a verdict (**95.4%**). The
gap is **29 shards of 348 (8.3%) that still die mid-run**, each taking the
rest of its shard with it — down from **203 of 324 (63%)** before the
envelope existed. Suite level: 91 targets → 18 pass, 73 fail (a target fails
on its first failing case, so the per-case numbers are the real signal).

**Every one of the 32,578 failures is classified** (each signature reviewed,
then coverage- and consistency-audited):

| category | cases | share | what it is |
|---|---|---|---|
| unimplemented op | 26,104 | 80.1% | a missing TPU kernel, reported cleanly |
| unsupported dtype | 1,749 | 5.4% | complex32 creation, per-op int gates (`bmm`/`dot` int32) |
| backend defect | 1,705 | 5.2% | wrong device propagation, MLIR compile failures, wrong values |
| missing feature | 1,228 | 3.8% | sparse layout construction, dynamic shapes, data-dependent `Materialize()` |
| numerical diff | 1,144 | 3.5% | tolerance-scale mismatches |
| incompatible test | 556 | 1.7% | asserts CPU/CUDA specifics (`_ConvBackend` enums, exact error text, profiler labels) |
| infra/flaky | 92 | 0.3% | DataLoader timeouts, environment |

The unimplemented bucket is concentrated: nested-tensor jagged conversions
(`_padded_dense_to_jagged_forward` 4,020, `_jagged_to_padded_dense_forward`
737), sparse (`_validate_compressed_sparse_indices` 1,952, `_to_sparse`
1,899, `_to_sparse_csr` 685), then special functions (`special_i1e` 571,
`special_bessel_j1` 436), `_linalg_svd` 423 and `copysign` 412. Sparse and
nested layouts together account for roughly 9,600 of the 26,104 — i.e. **two
layout features are ~37% of all failures**, and ~200 distinct aten ops make
up the rest.

The 29 surviving crashes are: 10 c128 CVT (libtpu, the residue the envelope
cannot pre-empt because the upcast happens inside a lowering), 8 LLVM
lowering errors in decompositions, 9 hangs/timeouts (multiprocessing and
DataLoader children that cannot re-init the TPU), 2 other. They are named
one-by-one in `torch_test_configs.py`; further progress needs backend fixes,
not more exclusions — every exclusion costs real coverage.

**Burn-down by leverage:** (1) sparse + nested-tensor layout support removes
~9.6k failures; (2) the `docs/libtpu_c128_cvt_crash.md` fix retires the
largest crash family and lets the envelope give complex128 back; (3) the
~200-op long tail; (4) device-propagation defects (`index_put_`, `mm`:
"tensor is expected to be on tpu, got cpu").

- Google3 parity (b/523312531): targets are OSS-only.
