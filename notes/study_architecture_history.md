# torch_tpu: architecture and history brief

Repository: google-pytorch/torch_tpu, checkout at `/Users/joinal/pytorvhtpu/upstream/torch_tpu`,
detached at `origin/main` = `214c68c83` (2026-11-05, "[CI] Run RBE presubmit on fork PRs").
Study date: 2026-11-06. All paths below are relative to the repo root unless absolute.

## 0. One caveat that shapes every history number

`origin/main` was rewritten on 2026-10-23. Its root commit `d5ec4cc94` ("Fix or suppress `[&]`
lambda captures across non-test C++ code in TorchTPU", Zhanyong Wan) is a single full-tree
snapshot (2136 files, 1,893,109 insertions) and `git merge-base` finds no common ancestor with
the pre-rewrite lineage. `origin/main` therefore holds only 346 commits (2026-10-23 to 2026-11-05).

The pre-rewrite lineage still exists on the `fork` remote:

| Ref | Commits | Range | Note |
|---|---|---|---|
| old mainline tip `ca7da6a87` (reachable via `fork/chizz/contributing-issue-claiming`) | 4506 | 2026-02-11 .. 2026-10-22 | root `370ee5920` "Start of public TorchTPU history" (Paul Ganssle) |
| `fork/release/0.2.0` | 4071 | .. 2026-10-19 | tracks old main until `23b09f731` 2026-10-12, then cherry-picks |
| `fork/release/0.1.2` | 4037 | .. 2026-10-12 | cherry-picks only after branch cut |
| `fork/release/2026-09-22` | 3530 | .. 2026-10-03 | dated release branch |
| `fork/main` | 180 | 2026-10-23 .. 2026-10-30 | same new root; 166 commits behind `origin/main` |

The "last ~600 commits" window below is therefore stitched: 346 on `origin/main` plus the last 254
mainline commits of the old lineage (`git log ca7da6a87 --first-parent --committer=Copybara-Service -254`,
2026-10-14 .. 2026-10-22). At the observed velocity (150-270 commits/week) 600 commits span only
~3.5 weeks, not Sept-Nov; a per-week table back to 2026-09-01 is given for the longer view.

## 1. Architecture map

### 1.1 Terminology anchor (from `docs/`)

Sources: `docs/how_torchtpu_works.md`, `docs/intro_to_torchtpu.md`, `docs/dynamic_shapes.md`,
`docs/feature_set.md`, `docs/torchtpu_execution_model.md`, `docs/custom_kernels.md`.

- Out-of-tree PyTorch backend registered as `PrivateUse1`, renamed to `tpu`, autoloaded on
  `import torch` (`src/torch_tpu/_loader.py`). Distributed backend name is `tpu_dist`.
- Every ATen op dispatched to TPU becomes a Deferred Op: a StableHLO builder closure plus static
  op parameters used for the cache key. The builder is `stablehlo/integrations/cpp/builder/MlirBuilder.h`
  (upstream StableHLO; included by 368 files under `csrc/`). Results are `at::Tensor`s whose storage
  points at a `DeviceBufferRef` in state `kDeferred`.
- At a sync point the runtime traverses the deferred-op DAG, picks what to materialize, computes a
  fingerprint, and compiles via PJRT (`PjRtClient`) or hits the compilation cache.
- Compilation cache tiers: tier 1 in-process (always on), tier 2 host shared memory (on for
  distributed), tier 3 remote filesystem/GCS (opt-in). Code: `csrc/common/compilation_cache.*`,
  `tier2_compilation_cache.*`, `tier3_compilation_cache.*`.
- Execution modes (`csrc/eager/eager_mode.h` enum `EagerMode`, env vars in `csrc/common/env_vars.h`):
  - Strict eager = DEFER_NEVER (default): one op per executable, async.
  - Debug eager: `TORCH_TPU_LAUNCH_BLOCKING=1` (kDeferNeverAndLaunchBlocking), sync + bounds checks.
  - Optimized eager = DEFER_AND_FUSE: `TORCH_TPU_DEFER_AND_FUSE=1`, fuse op sequences until a
    materialization point (`.cpu()`, `.item()`).
  - Compiled mode: `torch.compile(model, backend="tpu")`; whole FX graph traced with all ops deferred
    into one StableHLO program; no TorchInductor. C++: `csrc/internal/compile/`.
- Bounded dynamism: SymInt dims lowered to bounded StableHLO types; range split into buckets of
  `bucket_size` (default 128, the TPU lane width), `num_dynamic_buckets` default 1, max 8; one Dynamo
  range guard across all buckets; runtime picks the bucket executable and pads. Unbounded (data
  dependent) shapes such as `nonzero` go through "unbounded dynamic kernels" (precompiled Mosaic
  kernels, `csrc/experimental/precompiled_kernels/`, eager only).
- Custom kernels: Pallas via `torch.tpu.pallas.jax_op` (`src/torch_tpu/_internal/pallas/pallas.py`),
  prebuilt Tokamax kernels, experimental Helion, and Kiln: an internal C++ "Pallas-shaped kernel
  language for authoring Mosaic TPU kernels" (`csrc/internal/mosaic/kiln/kiln.h`).
- Multi-ABI wheel: one shared XLA/MLIR base library plus one thin pybind "glue" `.so` per PyTorch
  version (`env_2_12_0.so`, ...). `src/torch_tpu/_versioned_so_loader.py` installs a `sys.path_hooks`
  finder that serves the newest glue whose version is <= the installed torch. Versions come from
  `bazel/pytorch_versions.bzl`.

### 1.2 Directory map

| Path | Responsibility | Central files |
|---|---|---|
| `csrc/common/` | Shared runtime: compilation to PjRt executables, 3-tier compilation cache, cache keys and fingerprints, dtype/shape/layout utils, env var registry, error utils, TPU hardware/libtpu version probes, unbounded dynamic kernel registry. | `compilation.{h,cc}`, `compilation_cache.{h,cc}` (+`tier2_`/`tier3_`), `cache_key.{h,cc}`, `env_vars.h`, `error_utils.{h,cc}`, `unbounded_dynamic_kernel_registry.*` |
| `csrc/eager/` | The eager runtime: op dispatch, deferred-op DAG traversal, materialization, device buffers, events/streams, ATen kernel registration for PrivateUse1, AMP policies, host allocator. | `op_dispatcher.{h,cc}`, `traversal.{h,cc}` (1485 lines), `materialize.{h,cc}`, `tpu_aten_kernels.cc` (1874), `events_queue.*`, `device_buffer*.{h,cc}`, `eager_mode.h` |
| `csrc/ops/` | ~170 per-op directories (`<op>/<op>_aten_kernels.cc`) lowering ATen to StableHLO, plus shared builders. Largest: `gru/gru_aten_kernels.cc` (7123), `lstm/lstm_aten_kernels.cc` (6438), `foreach_aten_kernels.cc` (3463). | `op_builder_utils.{h,cc}`, `op_names.h` (2764), `view_decomposition/{decomposition,inversion}.cc`, `binary/unary/nullary_aten_kernels.cc`, `scan_builder.*`, `custom_kernels.*`, `copy_from/tpu_to_cpu.cc` |
| `csrc/pjrt/` | PJRT client lifecycle, execute (incl. layout validation and unbounded-dynamic routing), H2D/D2H transfers. | `pjrt_client.*`, `pjrt_execute.*`, `pjrt_transfer.*`, `pjrt_state.*` |
| `csrc/distributed/` | `c10d::Backend` for TPU (`tpu_dist`): collectives, handshake, RDMA windows, slice builder. | `process_group_tpu.{h,cc}` (2967), `allreduce/allgather/alltoall/reduce_scatter.*`, `handshake.*`, `window_tpu.*` |
| `csrc/api/` | Public C++ handle to a tensor's device buffer (query/materialize/sync without exposing eager internals). | `tensor_buffer.{h,cc}` |
| `csrc/mlir/` | TorchTPU MLIR optimization pipeline (`torch-tpu-optimization-pipeline`), custom passes, instrumentation. | `pipeline.*`, `passes/`, `registration.*` |
| `csrc/internal/` | The pybind11 extension modules and Python-facing subsystems: `compile/` (Dynamo backend: `tpu_torch_compile.cc` 2460, `compiled_mode.*`, `dispatch_scan.*`), `dynamism/` (`dynamism.*`, `dynamism_ops.*`, `dynamism_py.cc`), `mosaic/` (`op_builders.*`, `kiln_custom_call.*`, `kiln/{core,lib,mlir,passes}`), `pallas/pallas_py.cc`, `device/`, `execution_mode/`, `precision/`, `profiler/` (xprof plugin), `compiler_options/`, `distributed/tpu_distributed.cc`, `tracing/`, `batch_transfer/`, `sync/`, `env.cc`, `testing.cc`, `warnings_impl.cc`. | `compile/tpu_torch_compile.cc`, `compile/compiled_mode.cc`, `mosaic/kiln/kiln.h`, `dynamism/dynamism.cc` |
| `csrc/experimental/` | `batch_transfer/` and `precompiled_kernels/` (Kiln/Mosaic `nonzero_indices` unbounded-dynamic kernel, kernel registration, `mosaic_custom_call`). | `precompiled_kernels/nonzero_indices.cc`, `kernel_registration.cc` |
| `src/torch_tpu/` | The Python package. `__init__.py` installs the versioned loader hook, configures libtpu, imports `_loader`, checks tcmalloc. `_internal/` holds all Python subsystems; the public surface is staged via `_internal/public_api.py` ("tt_api"). | `_loader.py` (408), `_versioned_so_loader.py` (415), `_internal/compile/_backend.py` (1304), `_internal/compile/split_compiler.py`, `_internal/device/_device_module.py` (1322), `_internal/distributed/handshake.py` (1665), `_internal/pallas/pallas.py` (1830) |
| `shims/` | Bazel shims that map Google-internal build targets to OSS equivalents: `torch/` (lib and header `select()` by `//:_torch_version`), `libtpu/` (alias to `@pypi//libtpu`), `pybind11/` (glue transition), `jax/`, `flax/`, `absl_py/`, `etils/`, `safetensors/`, `torchprime/`, `torchtitan/`, `gpu_cuda/`, `mpm/`, `pkg/`, `py_rules/`, `build_files/`. | `torch/BUILD.bazel`, `torch/torch_wheel_select.bzl`, `pybind11/pybind.bzl`, `build_files/torch_version.bzl` |
| `bazel/` | Dependency pins and wheel machinery: XLA pin, PyTorch version matrix, Python versions, wheel version rule, per-version glue packages, pywrap extension list, local torch repo rule, patches for curl/grpc/rules_ml_toolchain/xla. | `xla_revision.bzl`, `xla_repo.bzl`, `pytorch_versions.bzl`, `supported_python_versions.bzl`, `wheel_version.bzl`, `glue.bzl`, `glue_packages.bzl`, `pywrap_extensions.bzl`, `torch_local_repo.bzl` |
| `build_files/` | Starlark macros used by every BUILD (`torch_tpu_py_test`, `if_oss`, RBE exec properties). | `build_defs.bzl`, `wheel.bzl` |
| `ci/` | Wheel build orchestration and CI tooling: Kokoro wrapper, shared wheel script, wheel layout tests, bazel-diff impacted-test selection, lint wrappers, PyTorch upstream test generation, metrics export. | `build_wheel.sh`, `build_wheels.sh`, `wheel/{pywrap_mapper.bzl,wheel_mapper.py,wheel_structure_test.py}`, `tools/{run_bazel_diff.py,list_ci_tests.py,gen_torch_tests.py,torch_test_module_utils.py}` |
| `.github/workflows/` | 22 workflows: `presubmit.yml` (CPU, RBE via bazel-diff, TPU v5 newest-deps, RBE fallback), `postsubmit.yml`, `nightly_tpu.yml`, `nightly_benchmarks.yml`, `nightly_unpinned_deps.yml`, `build_wheel.yml`, `release_candidate_tests.yml`, `upstream_tests.yaml`, `upstream_examples.yaml`, `out-of-tree-ci.yml`, `tsan.yml`, `lint.yml`, `turn-tracker*.yml`, PR hygiene jobs. | `presubmit.yml`, `build_wheel.yml`, `release_candidate_tests.yml` |
| `tests/` | ~130 top-level entries plus `compile/` (40), `distributed/` (42), `dynamism/` (15), `pallas/`, `torch/` (runner for PyTorch device-generic tests), `api/` (tt_api stage scanning), `model_registry/`, `optimizers/`, `device/`, `stream/`, `export/`, `profiler/`. | `ops_unit_test.py`, `ops_test.py`, `op_testing.py`, `errors_test.py`, `tpu_errors_test.py`, `error_coverage_data.yaml`, `torch/torch_test_configs.py` |
| `docs/` | mkdocs site (`mkdocs.yml` nav: Get started / In-depth / Case studies / Analyze your workload), ReadTheDocs (`.readthedocs.yaml`), `developers/` (api_design_guide, api_policy, python_support_policy, list_ci_tests_guide), `notebooks/`. | `how_torchtpu_works.md`, `torchtpu_execution_model.md`, `public_apis.md` |
| `perflab/` | Benchmark harness and registry (moved from `examples/` on 2026-09-25), HF diffusers and distributed model cases, known failures, MFU/throughput export. | `cases.py`, `benchmark_registry.pbtxt`, `known_failures.py`, `flops.py` |
| `examples/` | `FLUX_optimization/`, `fsdp/`, `muse_glimmer_30b/`, `simple_pytorch/`, `tensor_parallel/`. | |
| `requirements/` | Locked environments per Python (`requirements_3_11..3_15.txt`, `_cuda_`), torch-only locks per glue version (`requirements_torch_<ver>.txt`, `_nightly`), `lock_environments.sh`. | |
| `tools/` | `scan_external_api_uses.py`, `turn_tracker.py`, `setup_clangd.py`. | |
| root | `MODULE.bazel` (bzlmod; module version 0.2.1), `WORKSPACE` (legacy path still present), `.bazelversion` 9.3.1, `.bazelrc` (32 KB of configs: RBE regions, wheel_nightly/wheel_release, tsan, local_torch), `pyproject.toml`, `noxfile.py` (lint, wheel characterization, `intermediate_patch_abi`), `AGENTS.md`/`CLAUDE.md`, `CONTRIBUTING.md`, `visibility.bzl`. | |

Every pybind extension must be listed in `bazel/pywrap_extensions.bzl` (`PYWRAP_EXTENSIONS`),
which feeds both the Bazel-only aggregator `//csrc/common:pywrap_torch_tpu` and each per-version
glue `pywrap_library` generated by `bazel/glue.bzl` into `@torch_tpu_glues`.

## 2. History analysis

### 2.1 Volume per week (stitched old mainline + `origin/main`, Copybara commits only)

| ISO week | Dates | Commits |
|---|---|---|
| 2026-W36 | Aug 31 - Sep 6 | 121 |
| 2026-W37 | Sep 7 - 13 | 159 |
| 2026-W38 | Sep 14 - 20 | 167 |
| 2026-W39 | Sep 21 - 27 | 215 |
| 2026-W40 | Sep 28 - Oct 4 | 219 |
| 2026-W41 | Oct 5 - 11 | 260 |
| 2026-W42 | Oct 12 - 18 | 272 |
| 2026-W43 | Oct 19 - 25 | 128 (113 old lineage to Oct 22; rewrite on Oct 23: 12; Oct 24-25: 3) |
| 2026-W44 | Oct 26 - Nov 1 | 187 |
| 2026-W45 | Nov 2 - 5 (Thu) | 144 |

Total since 2026-09-01: 1872 commits. Weekday rate on `origin/main` is 29-45 commits/day;
weekends 1-2. Every one of the 346 `origin/main` commits has committer `Copybara-Service`.

### 2.2 Top authors, 600-commit window (346 new + 254 old, 2026-10-14 .. 2026-11-05)

66 distinct authors; 32 commits from non-Google, non-bot addresses.

| # | Author | Commits | Area (directories most touched; counts are file-touches in the `origin/main` window unless noted) |
|---|---|---|---|
| 1 | Zhanyong Wan | 64 | `csrc/ops` 793, `perflab` 207, `src/torch_tpu` 193, `csrc/common` 173, `tests` 151. Cross-cutting: C++ hygiene (root commit), error-coverage test regime, warnings format, API staging, symbolic-shape tests. Listed as a quality champion in `CONTRIBUTING.md`. |
| 2 | Gunhyun Park | 56 | `csrc/ops` 352, `tests` 27. Op lowerings and their standardization (IsBool/IsComplex/IsIntegral/SanitizeOptionalTensor), SDPA routing, bounded dynamism in glu, dropout lowering, perflab memory accounting. |
| 3 | Aditya Patel | 45 | `tests` 36, `csrc/ops` 32, `csrc/common` 14. Op input validation and error tests (`[Ops][cdist]`, xlogy, addmm.dtype, exponential_, splash_sdpa), single-device init (#4164). |
| 4 | Ben Wilson | 30 | `csrc/eager` 58, `csrc/ops` 24, `csrc/internal` 11, `csrc/pjrt` 9. Eager runtime: materialization modes, streams/events, async assertion errors, pinned non-blocking transfers, FP4 D2H. |
| 5 | Claudio Basile | 29 | `csrc/eager` 37, `csrc/ops` 27, `csrc/common` 10, `perflab` 8. Numerical precision (opmath `InferComputationDtype`, `CheckAtToOpMathType` lint), `torch.tpu.pallas.verifier`, `@jax_op` mesh support, hardware specs. Quality champion. |
| 6 | Zac Mustin | 27 | `tests/distributed` 18, `tests` 10. Converting distributed tests to `TpuMultiProcContinuousTest`. |
| 7 | Chi Zhang | 24 | `tests/torch` 17, `csrc/common` 10, `ci` 7. Enabling PyTorch device-generic upstream test modules (test_nn, test_autograd, test_torch, flex_attention), CONTRIBUTING, torch 2.14/2.15 test upgrades. |
| 8 | Will Froom | 21 | `csrc/internal` 51 (Kiln), `csrc/ops` 14. Kiln DSL (typed refs, Concat, windowed Load, value-returning If), flash attention unaligned lengths, unbounded kernels restricted to eager. |
| 9 | Francisco Rangel | 20 | `csrc/ops` 7, `tests/torch` 6, `csrc/common` 5, `.github` 3. Upstream test configs, out-of-tree CI and CRCR reporting, Float8_e5m3fnu, nightly torch pin. |
| 10 | Jay Gan | 19 | `perflab` 44, `csrc/ops` 13. perflab layer configs/builders, benchmark sharding, microbenchmark names. |
| 11 | Luke Boyer | 16 | `perflab` 95, `csrc/mlir` 9. perflab export (MFU, throughput, batch sweeps, workload declarations), MLIR CSE guard, grouped_mm. |
| 12 | XLA Green CLs (bot) | 15 | `bazel/xla_revision.bzl`, `MODULE.bazel`. Daily XLA pin bumps. |
| 13 | Melissa Weber Mendonça | 15 | `docs`, `mkdocs.yml` (external, via GitHub PRs). Docs rewrite, ReadTheDocs setup, FAQ. |
| 14 | Yiling Yuan | 13 | `tests`, `tests/api`, `csrc/eager`, `tools`. Public API surface (tt_api stages, Stream/Event classes, `scan_external_api_uses`). |
| 15 | Ryo Suzuki | 13 | `csrc/common` 30, `csrc/experimental` 24, `tests` 16. Unbounded dynamic kernels (nonzero), layout utils, cache stats, rollbacks. |
| 16 | Lee Howes | 13 | `docs`, `examples` (old lineage). Release docs/examples cleanup. Quality champion. |

Other `origin/main` contributors worth knowing: Theotime Combes (10; Kiln `EmitPipeline`, nonzero
vectorization, TORCH_TRACE cache hits), Paul Ganssle (9; build/release: Bazel 9, torch 2.15 glue,
Python 3.15), Shilpa Kesavan (7; op tests), Tudor Pipernea (6; flash/flex attention VMEM, module
naming), ML Velocity Otto (bot, 6; API refactors), Niklas Vangerow (5), Emilio Cota (5; CI/RBE,
MODULE.bazel), Manish Khanna (4; dynamism tests), Haitong Tian (4; ops). In the old tail: jparkerh
(Parker Holloway; `_internal`, `csrc/eager`, release manager for 0.1.2/0.2.0), Tushar Agarwal
(20 in Oct 12-22; `torch_tpu/`, tests), Jonathan Kim (release-candidate workflows), Pedro Goncalves
Mokarzel (examples), Haifeng Jin (tests, shutdown/teardown, OOM errors).

### 2.3 Themes (keyword match on `origin/main` subjects; a commit may match several)

| Theme | Commits (of 346) | Representative subjects |
|---|---|---|
| ops / aten / lowerings | 71 | `650d3084f` [Ops][nonzero_static] Implement aten::nonzero_static; `d86cca284` Standardize on IsBool across operator lowerings; `483b1336f` [Ops][linalg.lu] Replace host-sync pivot loop |
| tests (incl. upstream device-generic) | 76 | `0cf635c97` Re-enable multi-device PyTorch device-generic tests on TPU; `31c42eb83` Add ops_test_static target and @also_run_static; `503ce6b9c` Shard ops_test_symbolic_shapes to 50 |
| errors / warnings / precision / dtype | 50 | `b83f80db2` Add CheckAtToOpMathType presubmit lint rule; `c25e28003` [errors] Translate XLA OOM to torch.OutOfMemoryError; `b1be8ed1b` Use `//...//` for warning category prefixes |
| CI / presubmit / RBE / nightly / lint | 48 | `214c68c83` [CI] Run RBE presubmit on fork PRs; `4562ff0d6` Run device-generic tests against nightly wheel in out-of-tree-ci.yml; `dae987dce` Run error_coverage_test in OSS presubmit |
| kernels: Kiln / Mosaic / Pallas / SparseCore / flash | 44 | `339220dc4` Add windowed Load to Kiln; `9f62e55a6` Type Kiln refs by memory space; `ad93e3a74` [kiln] Add EmitPipeline (1/N); `9997c7800` Unaligned seq lens in Flash Attention; `0ae56f19b` SparseCore vector subcore DMA tests |
| dynamism (bounded + unbounded) | 31 | `ca9c7d519` Enable auto-loading unbounded dynamic kernels by default; `7d3195b3a` Bounded dynamism in glu; `e5d3657ec` Dynamic shape testing with gradients under torch.compile; `35eebc91a` Vectorize nonzero |
| perf / perflab / benchmarks | 29 | `1bf9cb694` Export MFU from nightly benchmarks; `a172b384b` Add a batch sweep mode; `5cda5569b` Shard perflab benchmarks by stable hash |
| eager / defer / materialize / streams | 26 | `4308a3581` Remove MaterializationBlockMode; `8c2a368d8` Record async assertion errors on applicable stream only; `ab218d636` Replicate CUDA pinning for non-blocking transfers; `0118cfa3e` Skip redundant per-launch PjRtBuffer checks |
| compile / Dynamo / AOT | 13 | `e20d8b5c7` Compiled-mode view base independent of storage offset; `1182deb79` Clone only returned placeholders a compiled graph writes to; `f4a3712d7` Keep fallback-keyed AOTAutogradCache entries out of on-disk cache |
| distributed | 12 | `2813679a3` Skip MegaScale PJRT client when world fits in one slice; `0065c447d` Default handshake port 36423 -> 21423; `b6ab2fd19` FSDP2 split_with_sizes_copy without per-rank views |
| docs | 9 | `7ec6acd1e` Encourage linking PRs to issues; `988b22f12` XLA_FLAGS vs LIBTPU_INIT_ARGS |
| build / Bazel / wheel / ABI / Python | 8 | `1dcb164b7`, `1ef180cdf`, `4aa6f0852`, `aaba9aa5e`, `179a0c065` |
| XLA pin bumps (bot) | 8 | `273cd6e63` and 7 more |
| rollbacks | 2 | `cf7fcd7b4` Automated g4 rollback of changelist 1014154139; `a3879659b` [rollback] Lower dropout mask generation ... |

Directory churn (file-touches, 600 window): `csrc/ops` 1601, `perflab` 531, `docs` 389,
`src/torch_tpu` 365, `csrc/internal` 363, `csrc/common` 345, `csrc/eager` 272, `examples` 189,
`tests/distributed` 150, `ci` 88, `tests/compile` 83, `bazel` 67, `csrc/pjrt` 63,
`csrc/experimental` 63, `.github` 61, `shims` 51, `csrc/distributed` 45.

### 2.4 Major structural changes (hash, date, lineage)

| Date | Hash | Change |
|---|---|---|
| 2026-02-11 | `370ee5920` | "Start of public TorchTPU history" (old lineage root) |
| 2026-04-28 | `da889965c` | Bazel 8.6.0; bzlmod opt-in via `--config=bzlmod` |
| 2026-06-17 | `0941d4936` | Centralize supported Python versions (`bazel/supported_python_versions.bzl`) |
| 2026-07-22 | `8249ac5c1` | Build a multi-ABI wheel: one thin glue per PyTorch version (`f51f8e1e8` adds torch 2.13.0; `ff02f5d25` optional nightly glue; `4e31538b9` 07-27 fix internally) |
| 2026-08-19 | `2d0ce00c5`, `bdc49bb9d` | CI smoke tests against dependency upper bounds; nightly suite against unpinned deps |
| 2026-09-10 | `eff277701` | Restructure layout: Python package moved to `src/` via Copybara |
| 2026-09-15 | `886e0f0f5` | Default Python baseline 3.12 -> 3.14 |
| 2026-09-23 | `4e1249f0d` | Last human XLA bump ("relocated StableHLO builder targets"); daily bot bumps start `8eee4b6ac` 09-24 |
| 2026-09-24 | `864cd3e06` | PyTorch 2.14.0 added to the wheel matrix |
| 2026-09-24/25 | `956a1a982`, `5bd5ccccb` | Remove deprecated `singlehost_wrapper` / `multihost_wrapper` launchers |
| 2026-09-25 | `fdbcabaef` | e2e benchmarks and harness moved to top-level `perflab/` |
| 2026-09-29 | `7fc646c5d` | Python version support policy documented |
| 2026-10-01 | `f3ffd3cd0` | Version 0.1.2 |
| 2026-10-02..13 | `83bfff900`, `49a74e48c`, `87bf01c24`, `5d1d91f58` | tt_api: move `torch_tpu/ops`, optimizers to `_internal`, expose under `torch.tpu.*`; every API must declare a stage |
| 2026-10-08/13 | `f4d8fd44f`, `757f4f2c1` | ABI high-water-mark utility and wheel presubmit job |
| 2026-10-09 | `2d486f1aa`/`9ffa32db4`, `e80f003b6`/`926463b08` | torch 2.15.0 RC in ABI matrix; free-threaded 3.14t wheel builds |
| 2026-10-11/12 | `0cd072ee8`, `23b09f731` | main to 0.2.1; release/0.2.0 set to 0.2.0; `1950bccfc` torch `<=2.15.0` upper bound on the release branch |
| 2026-10-12 | `89b629d55` | Remove 2.15.0 RC from main wheel matrix |
| 2026-10-13 | `23685e178` | libtpu floor 0.0.41 -> 0.0.50, obsolete gates removed |
| 2026-10-16 | `5170049a4`, `1b1e44b57` | RBE for Kokoro wheel builds; ReadTheDocs setup (external PR #5334) |
| 2026-10-17/19 | `005903d0e` (tag `v0.2.0`), `ddf88b974` (tag `0.2.0-docs`), `a50392789` | 0.2.0 release; README points at PyPI |
| 2026-10-23 | `d5ec4cc94` | History rewrite: new single-root `origin/main` |
| 2026-10-27 | `aaba9aa5e` | `MODULE.bazel`: work around yanked c-ares 1.34.6.bcr.1 (CVE-2025-62408) |
| 2026-11-02 | `1dcb164b7` | Bazel 8.6.0 -> 9.3.1, rules_python 1.8.5 -> 2.4.5, `simpleapi_download`, drop 4 rules_python patches, `rules_shell`; Fixes #4647 |
| 2026-11-03 | `4bfc10b6c` | Minimum JAX/jaxlib 0.11.0 |
| 2026-11-04 | `1ef180cdf` | torch 2.15.0 promoted to standard release glue |
| 2026-11-04 | `ca9c7d519`, `2cf50f8d9` | Unbounded dynamic kernels auto-loaded by default; restricted to eager |
| 2026-11-05 | `4aa6f0852` | Python 3.15 and 3.15t (`torch>=2.15.0` on 3.15); rules_python now 2.4.6 in `MODULE.bazel` |
| 2026-11-05 | `25c4e2047` | Pinned upstream torch nightly 2.16.0.dev20261102 |
| 2026-11-05 | `214c68c83` | RBE presubmit runs on fork PRs |

XLA pin cadence: 36 bumps between 2026-09-23 and 2026-11-04, one per calendar day including
weekends, skipping only 10-08, 10-12, 10-17, 10-24..26 (rewrite) and 11-02. Each bump edits
`bazel/xla_revision.bzl` and the `archive_override(module_name = "xla", ...)` block of `MODULE.bazel`
together. Current pin: `968e6be1d3b04ca6c49b00e00dc08c9b181dc72e` (XLA commit of 2026-11-04).

## 3. Commit message conventions on `origin/main`

Measured over all 346 commits:

| Marker | Commits |
|---|---|
| `PiperOrigin-RevId:` trailer | 346 (100%) |
| "Generated from cl/NNN" line | 346 |
| `COPYBARA_INTEGRATE_REVIEW=https://github.com/google-pytorch/torch_tpu/pull/N from user:branch sha` | 20 (every GitHub PR import) |
| `Closes #N` (PR number, appended by the import) | 20 |
| `Fixes #N` (issue) | 18 |
| `## Description` / `## Test Plan` headings (PR template kept verbatim) | 17 |
| `b/NNNNNNNNN` internal bug ids | 15 |
| `Assisted-by:` | 2 |
| `Co-authored-by`, `Signed-off-by`, `Reviewed-by`, `Change-Id`, `GitOrigin-RevId` | 0 |

Subject line: imperative, sentence case, usually no trailing period; average 67 characters,
97 of 346 exceed 72, longest 151. About 20% carry a bracket prefix: `[Ops][<op>]` (26),
`[torch_tpu]` (11), `[kiln]` (6), `[TorchTPU MXU Improvement]` (4), `[torch_tpu perflab]`/`[perflab]`
(8), plus one-offs `[CI]`, `[docs]`, `[errors]`, `[rollback]`, `[tt_api]`, `[torchtpu/issues/4164]`.
A few use `Kiln:`, `distributed:`, `MODULE.bazel:` colon prefixes. One subject includes a GitHub PR
number in parentheses (`2cc64d89f` "... exponential_ lowering. (#5807)"). One old-lineage import
leaked the template into the subject (`ac978415c` 2026-10-13, subject "## Description").

Body: one to three prose paragraphs describing previous behaviour, the change, and why; internal
authors reference `b/` bugs, external imports keep the PR template. Rollbacks use
"Automated g4 rollback of changelist N." or "[rollback] <original subject>". Bot XLA bumps use a
fixed template. The pre-rewrite lineage additionally had GitHub-committed merge commits
("Merge pull request #N from google-pytorch/<user>/cherry-pick-pr-N-release-0.x") for release
branches only.

Five representative messages, quoted in full:

**(a) External PR import, `1e71c2b51`** (Andrew `<pullin@berkeley.edu>`, 2026-11-05)

```
Bound the per-thread cache of scalar device buffers

## Description

Fixes #6036

`MakeBuffer` cached a device buffer for every distinct scalar in a thread-local map that never shrank, so every new value, such as a scheduled learning rate, leaked a device buffer. Now a miss with 1024 cached entries (about 1 MiB) clears the map before inserting. Buffers that pending ops or tensors still use stay alive through their own references, and the hit path is unchanged.

Cycling through more than 1024 distinct scalars now misses on every lookup (1.19x per-op time with DEFER_NEVER and 1.45x with DEFER_AND_FUSE for a 1500-value cycle), and a clear takes up to 4 ms. In DEFER_AND_FUSE, a fused region that uses a scalar before and after a clear gets an extra graph input, which can add compiles.

## Test Plan

Wheels built from main at a8a3ccc84 with and without this change, on a v5e-1 (torch 2.14.1, libtpu 0.0.51), in DEFER_NEVER and DEFER_AND_FUSE:
- `accelerator_api_test`: the new `test_distinct_scalars_do_not_accumulate_device_buffers` fails without the change (4001 and 4002 allocations, limit 2048) and passes with it. The new `test_queued_op_keeps_scalar_buffer_after_cache_clear` passes on both.
- The repro in #6036: +19999 -> +543 live allocations. 3000 steps of AdamW + CosineAnnealingLR: +9032 -> +872, with the same XLA compile count and step time within noise.
- `nox -s lint`: clean.

Closes #6037

Generated from cl/1014592898

COPYBARA_INTEGRATE_REVIEW=https://github.com/google-pytorch/torch_tpu/pull/6037 from apullin:fix-scalar-buffer-cache 3274e76c3bd97fa04870f73058eeaa7fe45fa7a1
PiperOrigin-RevId: 1014592898
```

**(b) Internal op fix, `35a3fa373`** (Haitong Tian, 2026-11-05)

```
[Ops][nan_to_num] Preserve -0.0 signbit in EncodeParamCacheKey(double)

Stop normalizing `-0.0` to `+0.0` in `EncodeParamCacheKey(double)` so floating-point parameters with distinct IEEE-754 zero signbits produce distinct compilation cache fingerprints. Previously, calling ops such as `torch.nan_to_num` with `0.0` and `-0.0` replacement values collided in the HLO cache and returned stale zero signbits.

Generated from cl/1015066310

PiperOrigin-RevId: 1015066310
```

**(c) Structural build change, `1dcb164b7`** (Paul Ganssle, 2026-11-02)

```
Update Bazel to 9.3.1 and rules_python to 2.4.5

Upgrade OSS Bazel from 8.6.0 to 9.3.1 and `rules_python` from 1.8.5 to
2.4.5, and enable `rules_python`'s `simpleapi_download` wheel downloader
so Bazel downloads PyPI wheels directly via the repository cache instead
of invoking `pip` at repo-rule time for every wheel.

Drop all four local `rules_python` patches:
- `rules_python_scope.patch`: Bazel 9 natively supports the `scope`
  attribute on Starlark build-setting rules (`bool_flag`,
  `string_flag`).
- `rules_python_versions.patch`: `rules_python` 2.4.5's
  `get_release_info` supports all `python-build-standalone` release
  dates configured in `python/versions.bzl`.
- `rules_python_pip_version.patch`: superseded by `rules_python` 2.4.5,
  which pins `pip` 26.2.2 and `packaging` 26.3.
- `rules_python_freethreaded.patch`: under Bzlmod, `3.14` and `3.14t`
  share the `3.14` toolchain registration and select free-threaded
  wheels via `--@rules_python//python/config_settings:py_freethreaded=yes`
  with `target_platforms = ["{os}_{arch}", "{os}_{arch}_freethreaded"]`.

Notable build and dependency changes:
- Bump `rules_cc` (0.2.18 -> 0.2.25), `protobuf` (33.5 -> 33.6), and
  `highway` (1.3.0 -> 1.4.0), add `rules_shell` 0.8.0, and migrate
  `sh_test` / `sh_binary` loads to `@rules_shell` for Bazel 9
  compatibility.
- Pass `--incompatible_autoload_externally=+@rules_cc,+@rules_java,+@rules_shell,+@com_google_protobuf`
  in `.bazelrc` for transitive dependencies (`@rules_ml_toolchain`,
  `@com_github_grpc_grpc`, `@rules_go`, `@protoc-gen-validate`,
  `@cel-spec`) that still rely on autoloaded symbols.
- Route `@pypi//torch:...`, `@pypi//libtpu:whl`, and
  `@pypi_torch_<ver>//torch:...` references through the `rules_python`
  hub repositories rather than spoke repositories. The hub aliases
  branch on `rules_python`'s `python_version` setting (including
  `3.14t`), which also fixes extra PyTorch versions (`pypi_torch_<ver>`)
  to resolve the wheel matching the active Python version rather than
  hardcoding `cp314`.
- Delegate `bazel/torch_wheel.json`'s `additive_build_content` to
  `@@//bazel:pip_parse_targets.bzl` (syncing `pip_parse_targets.bzl` to
  the `torch_wheel.json` target definitions, which did not carry the
  unused `@com_google_protobuf//:protobuf` dependency) and exclude
  PyTorch's bundled `google/protobuf/**` headers from `torch_headers` so
  they do not shadow `@com_google_protobuf` in sandboxed/remote builds.

Verified with Copybara export and Bazel 9.3.1 RBE builds/queries on
Python 3.14 and 3.14t (`//ci/wheel:torch_tpu_wheel`,
`//csrc/common:dtype`, `//ci/tools/...`, `//build_files:build_defs_test`).

Fixes https://github.com/google-pytorch/torch_tpu/issues/4647

Generated from cl/1012238398

PiperOrigin-RevId: 1012238398
```

**(d) Bot dependency bump, `273cd6e63`** (XLA Green CLs, 2026-11-04)

```
Update XLA dependency to use revision http://github.com/openxla/xla/commit/968e6be1d3b04ca6c49b00e00dc08c9b181dc72e

Updates the pinned OpenXLA revision to https://github.com/openxla/xla/commit/968e6be1d3b04ca6c49b00e00dc08c9b181dc72e (committed on 2026-11-04T05:34:04.695704+00:00).

Generated from cl/1013631807

PiperOrigin-RevId: 1013631807
```

**(e) Rollback, `cf7fcd7b4`** (Ryo Suzuki, 2026-11-05)

```
Automated g4 rollback of changelist 1014154139.

Rolling back changelist 1014154139 as it is breaking github presubmits.
Breaking OSS presubmits: https://github.com/google-pytorch/torch_tpu/actions/runs/42042122493/job/127295341183

Generated from cl/1014685530

PiperOrigin-RevId: 1014685530
```

## 4. Release and version info

Tags in the clone: `v0.2.0` -> `005903d0e` (2026-10-17, release/0.2.0 lineage, "Merge pull request
#5545 ... cherry-pick-pr-5544-release-0.2.0"); `0.2.0-docs` -> `ddf88b974` (2026-10-19, docs links
rewritten to the tag); `build-base` -> `988b22f12` (2026-11-02, on the new lineage; looks like a local
working tag rather than an upstream release).

Release branches exist only on the `fork` remote (`origin` exposes only `main` in this clone):
`release/0.1.2`, `release/0.2.0`, `release/2026-09-22`. They take cherry-picks through GitHub PRs
named `<user>/cherry-pick-pr-<N>-release-<ver>` merged by GitHub (committer `GitHub`), unlike `main`
where everything arrives through Copybara. The 0.2.0 branch pinned `torch<=2.15.0` (`1950bccfc`) and
received RC qualification workflows (`5dd9dbb72`, `77ba251aa`, `8138c0ba2`).

Version sources:

| Item | Where | Current value |
|---|---|---|
| Package version | `pyproject.toml` `[project] version`; read by the `torch_tpu_version_repo` rule in `bazel/wheel_version.bzl`, suffixed with `WHEEL_VERSION_EXTRAS` (`.dev<ts>` for `--config=wheel_nightly`, `rcN` or empty for `--config=wheel_release`, see `ci/build_wheel.sh`); mirrored by hand in `MODULE.bazel` `module(version = ...)` | `0.2.1` |
| libtpu floor | `pyproject.toml` `"libtpu>=0.0.50"` with a "DO NOT change the minimum libtpu version without approval" comment; locked `libtpu==0.0.50` in every `requirements/requirements_3_*.txt`; Bazel alias `shims/libtpu/BUILD.bazel` -> `@pypi//libtpu`; runtime gating in `csrc/common/libtpu_version.*` | `>=0.0.50` (was `>=0.0.41` until `23685e178` 2026-10-13) |
| XLA pin | `bazel/xla_revision.bzl` (`XLA_COMMIT`, `XLA_SHA256`) consumed by `bazel/xla_repo.bzl` `http_archive` (WORKSPACE path), duplicated in `MODULE.bazel` `archive_override(module_name = "xla", integrity, strip_prefix, urls, patches = [remove_stablehlo_source_location.patch])` (bzlmod path). Also `archive_override`s for `rules_ml_toolchain` and `googletest`. | `968e6be1d3b0...` |
| PyTorch matrix | `bazel/pytorch_versions.bzl`: `DEFAULT_TORCH_VERSION`, `EXTRA_PYTORCH_VERSIONS`, `WHEEL_TORCH_VERSIONS_BY_PYTHON`, `NIGHTLY_TORCH_VERSION`, `INTERMEDIATE_PATCH_VERSIONS`, `SENTINEL_TORCH_VERSION`. Data-only so it is also parsed by `requirements/lock_environments.sh`; `MODULE.bazel` copies `EXTRA_PYTORCH_VERSIONS` by hand. Per-version torch-only locks `requirements/requirements_torch_<ver>.txt`; pip hubs `@pypi_torch_<ver>`; selection by `//:_torch_version` flag in `shims/torch/torch_wheel_select.bzl`; `//:include_nightly_glue` excludes the nightly glue under `wheel_release`. | default `2.11.0`; extras `2.12.0`, `2.13.0`, `2.14.0`, `2.15.0`; Python 3.15 ships only `2.15.0`; nightly `2.16.0`; intermediate `2.12.1`; `pyproject`: `torch>=2.11.0` (<3.15), `torch>=2.15.0` (>=3.15) |
| Python versions | `bazel/supported_python_versions.bzl` `SUPPORTED_PYTHON_VERSIONS`, `DEFAULT_PYTHON_VERSION`; copied by hand into `MODULE.bazel` (checked by `ci/tools/sync_module_bazel_versions.py`); `LINT.IfChange` links to `pyproject.toml` `requires_python`, `README.md`, `docs/developers/python_support_policy.md`; comment lists 8 workflows/scripts to update too | `3.11`, `3.12`, `3.13`, `3.14`, `3.15`, `3.14t`, `3.15t`; default `3.14` |
| Bazel | `.bazelversion`; `MODULE.bazel` `bazel_dep`s | Bazel `9.3.1`; `rules_python` 2.4.6, `rules_cc` 0.2.25, `protobuf` 33.6, `grpc` 1.78.0, `abseil-cpp` 20260526.0, `googletest` override at `28e9d1f2` |
| JAX | `pyproject.toml` (optional deps) | `>=0.11.0` since `4bfc10b6c` 2026-11-03 |

## 5. Hot spots (600-commit window)

Most frequently changed files:

| Changes | File |
|---|---|
| 91 | `tests/ops_unit_test.py` |
| 46 | `tests/ops_test.py` |
| 46 | `tests/errors_test.py` |
| 43 | `tests/BUILD` |
| 36 | `tests/distributed/BUILD` |
| 31 | `tests/tpu_errors_test.py` |
| 30 | `csrc/common/BUILD` |
| 28 | `csrc/eager/BUILD` |
| 26 | `tests/error_coverage_data.yaml` |
| 25 | `tests/op_testing.py` |
| 23 | `csrc/internal/mosaic/kiln/kiln_test.cc` |
| 20 | `MODULE.bazel` |
| 19 | `tests/torch/torch_test_configs.py` |
| 18 | `csrc/ops/copy_from/tpu_to_cpu.cc` |
| 17 | `csrc/ops/BUILD` |
| 16 | `tests/torch/BUILD` |
| 16 | `perflab/tests/BUILD` |
| 16 | `csrc/eager/tpu_aten_kernels.cc` |
| 16 | `bazel/xla_revision.bzl` |
| 15 | `tests/dynamism_test_utils.py` |
| 14 | `perflab/known_failures.py` |
| 13 | `tests/compile/BUILD` |
| 13 | `csrc/ops/copy_from/BUILD` |
| 13 | `csrc/experimental/precompiled_kernels/nonzero_indices.cc` |
| 13 | `csrc/eager/events_queue.cc` |

Files with the most distinct authors:

| Authors | File |
|---|---|
| 63 | `tests/ops_unit_test.py` |
| 35 | `tests/errors_test.py` |
| 29 | `tests/ops_test.py` |
| 21 | `tests/tpu_errors_test.py` |
| 20 | `tests/error_coverage_data.yaml` |
| 16 | `tests/BUILD` |
| 11 | `csrc/internal/mosaic/kiln/kiln_test.cc` |
| 10 | `csrc/ops/copy_from/tpu_to_cpu.cc` |
| 9 | `docs/source/index.py` |
| 8 | `tests/compile/BUILD`, `pyproject.toml`, `docs/notebooks/intro_to_torchtpu.ipynb` |

Within `origin/main` alone the same pattern holds (`ops_unit_test.py` 53 changes / 41 authors;
`errors_test.py` 39 / 32; `error_coverage_data.yaml` 26 / 20), and the most-touched non-test C++
is `csrc/ops/copy_from/tpu_to_cpu.cc` (11 / 7), `csrc/eager/tpu_aten_kernels.cc` (11),
`csrc/internal/compile/compiled_mode.cc` (10), `csrc/eager/op_dispatcher.cc` (10),
`csrc/eager/execution_task.cc` (9), `csrc/eager/events_queue.cc` (9),
`csrc/eager/device_buffer_utils.cc` (9), `csrc/common/compilation_cache.cc` (9), `csrc/common/env_vars.h` (9).

Most changed test files: `tests/ops_unit_test.py` 91, `tests/ops_test.py` 46, `tests/errors_test.py` 46,
`tests/tpu_errors_test.py` 31, `tests/error_coverage_data.yaml` 26, `tests/op_testing.py` 25,
`csrc/internal/mosaic/kiln/kiln_test.cc` 23, `tests/torch/torch_test_configs.py` 19,
`tests/dynamism_test_utils.py` 15, `csrc/eager/events_queue_test.cc` 13,
`csrc/internal/mosaic/kiln/kiln_e2e_test.cc` 12, `csrc/eager/materialize_test.cc` 8,
`csrc/common/error_utils_test.cc` 8, `csrc/eager/op_dispatcher_test.cc` 7,
`csrc/common/compilation_cache_tiered_test.cc` 7.

Reading: (1) nearly every op change lands a case in `tests/ops_unit_test.py` and an expected error
string in `tests/errors_test.py` / `tests/tpu_errors_test.py` / `tests/error_coverage_data.yaml`;
the error-coverage baseline became a presubmit gate on 2026-10-27 (`dae987dce`, `2ba901bca`).
(2) `csrc/eager/` (dispatcher, events queue, execution task, device buffers) and
`csrc/ops/copy_from/tpu_to_cpu.cc` (D2H path) are the riskiest runtime files: many authors, many
small changes, and several external PRs touched lifetime/ordering there. (3) Kiln is the most
active new subsystem (`kiln_test.cc` 23 changes, 11 authors). (4) `MODULE.bazel` and
`bazel/xla_revision.bzl` churn is bot-driven.

## 6. External contributions that landed

Identification: Copybara imports keep `COPYBARA_INTEGRATE_REVIEW=.../pull/N from <user>:<branch>`
and append `Closes #N`; authorship is preserved. Googlers also use the GitHub PR path for CI, docs
and test-infra work, so "imported PR" is broader than "external author".

### 6.1 `origin/main` (2026-10-23 .. 2026-11-05): 20 imported PRs, 13 by external authors

| Date | Hash | Author | PR | Area | Change |
|---|---|---|---|---|---|
| 11-05 | `1e71c2b51` | Andrew (apullin, berkeley.edu) | #6037 | `csrc/eager`, `tests/accelerator_api_test.py` | Bound per-thread scalar device-buffer cache (Fixes #6036) |
| 11-04 | `e14e4d1a1` | hx (meta.com) | #5913 | `csrc/ops`, tests | Early return in `split_with_sizes_copy` for zero rows |
| 11-04 | `a75b08238` | Andrew | #5956 | `csrc/ops`, `tests/dynamism_test.py` | Bounded dynamic dims in avg_pool backward |
| 11-03 | `ced69732e` | dependabot[bot] | #6116 | `.github` | GitHub Actions bumps |
| 11-03 | `968fd65c0` | Vardhman Gupta | #5778 | `csrc/ops`, error tests | `torch.dot` length validation |
| 11-03 | `4d1158294` | Andrew | #5962 | `csrc/ops`, `tests/torch` | leaky_relu gradient at 0 |
| 11-03 | `86b465657` | Kuba Sołtys (qooba.net) | #5950 | `src/torch_tpu`, tests | Contiguous outputs from splash_sdpa fake kernels |
| 11-02 | `e20d8b5c7` | Ivan Shivalov (intexcor) | #6019 | `csrc/internal/compile`, `tests/compile` | Compiled-mode view base independent of storage offset (Fixes #4409) |
| 11-02 | `ab12194bf` | Rohit Sroch | #5717 | `docs` | Notebook fix |
| 10-30 | `dbdccfafe` | Saloni Garg | #5846 | `src/torch_tpu`, `tests/dynamism` | View decomposition base larger than view storage |
| 10-29 | `1182deb79` | hx (ethannxie) | #5864 | `src/torch_tpu`, `tests/compile` | Clone only returned placeholders a compiled graph writes to |
| 10-29 | `b6ab2fd19` | hx | #5913 | `csrc/ops`, tests | FSDP2 `split_with_sizes_copy` without per-rank views |
| 10-29 | `5731cd519` | Saloni Garg | #5847 | `src/torch_tpu`, `tests/dynamism` | View decomposition IndexError on 0-d views |
| 10-29 | `7f679816b` | Andrew | #5953 | `csrc/eager`, `csrc/ops`, tests | Record autograd for avg_pool1d |
| 10-27 | `4d7b0d2a6` | Parit Kansal | #5715 | `csrc/ops`, tests | elu/celu for small negative inputs |
| 10-27 | `d3fb8d0c7` | Snehal Reddy | #5854 | `csrc/pjrt`, `csrc/distributed`, `csrc/experimental` | Unify raw DMA buffer lifetime with self-retaining futures |
| 10-26 | `5bb2a40b2` | Snehal Reddy | #5843 | `csrc/pjrt` | Keep DeviceBufferList alive during async ToLiteral in StartD2H |
| 10-26 | `0c34e98a6` | Chi Zhang (gmail) | #5889 | `CONTRIBUTING.md` | PR-to-issue linking instructions |
| 10-26 | `d58efbe56` | Paul Ganssle (Google) | #3895 | `.github`, `ci` | Fail cleanup job when a run cannot be deleted |
| 11-05 | `6a02fcb04` | Chi Zhang (Google) | #6279 | `CONTRIBUTING.md`, `docs` | CI workflow approval screenshot |

Other `origin/main` commits with GitHub references but no PR import: `99010d9a8` "[torchtpu/issues/4164]
Default to single-device TPU initialization when WORLD_SIZE is unset or 1" (Aditya Patel) and
`2cc64d89f` "... exponential_ lowering. (#5807)" (Aditya Patel, Fixes #5807). Issues fixed in the
window: #6036, #6084, #5961, #5952, #5949, #5830, #5808, #5807, #5773, #5727, #5672, #5671, #4896,
#4892, #4843, #4409, #4141, #4647 (as a URL).

### 6.2 Old lineage, 2026-09-01 .. 2026-10-22: 51 imported PRs

Dominant external contributors and the kinds of change accepted:

- Melissa Weber Mendonça (external, docs): ~22 PRs, all under `docs/` and `mkdocs.yml`: ReadTheDocs
  setup (#5334), FAQ (#5461), rewrites of every In-depth page (#5340, #5404, #5427, #5428, #5429,
  #5430, #5432, #5344, #5351, #5354, #5360), precision docs merge (#4498), docs-review agent skill
  (#4168), CLs explained in CONTRIBUTING (#4180).
- Will Constable (Meta): experimental `all_gather_dim`/`reduce_scatter_dim` collectives (#4225,
  `csrc/distributed`, `csrc/eager`, `csrc/ops`, Python, tests), submodule metadata in split
  compiler (#4154), top-level donations in compilation cache keys (#3614, `csrc/eager`).
- Jack Oh: scan body operands as placeholders (#3301), copy loop-body helper functions (#3404,
  `csrc/internal`, `csrc/ops`).
- Yin Lin: `elapsed_time()` for TPU events (#4400; later relanded as `78af37376`).
- Haifeng Jin: SIGSEGV during interpreter teardown fixed by clearing compiled executables (#3639).
- Googlers via GitHub: Chi Zhang (upstream PyTorch tests to 2.14/2.15 #5054/#5196, actionlint
  #3552, auto-labeler #4046, logo #4048, dependabot re-enable #3443), Paul Ganssle (workflow
  pinning #3904, cancel-in-progress fixes #3893/#3868, `--local_test_jobs` #3695).
- dependabot: GitHub Actions group bumps (#2518 twice).

Pattern: accepted external changes are narrow and test-backed: single-op numerical or validation
fixes with `ops_unit_test.py`/`errors_test.py` cases, adding bounded-dynamism support to one op,
view-decomposition and compiled-mode argument fixes in `src/torch_tpu/_internal/compile`, buffer
lifetime fixes in `csrc/pjrt`, documentation, and CI hygiene. Larger external features (Meta's
experimental collectives) land as `experimental` APIs. The process (`CONTRIBUTING.md`): oncall
triage, domain-expert review, quality-champion review (`cbasile-g`, `LeeHowes`, `vladbelous`,
`zhanyong-wan`), `pull ready` label, Copybara creates the internal CL, PR is closed (not merged)
and the commit appears on `main` under the original author.
