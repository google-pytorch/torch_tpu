# How torch_tpu maintainers review external PRs (study, 2026-11-06)

Source: `gh pr list` (1000 most recent PRs, back to #5256), plus per-PR
reviews / inline comments / issue comments / events via `gh api` for 60 PRs
whose author is not google*/copybara/dependabot. Read-only. Quotes are verbatim
(trimmed with `...` where long). Days-to-first-review = first non-author,
non-bot review, inline comment or issue comment after PR creation.

Note on authorship: the "external" filter is on login only. Several of these
authors are Googlers or close collaborators (chizhang529, zhanyong-wan,
jparkerh, jonathan1920, Snehal-Reddy, yinlin09, jack8558, saloni-garg,
melissawm/Quansight docs contractor, pianpwk, weifengpy). They are kept in the
table because they show the same process, but the theme catalogue weights
true outsiders (apullin, ethannnnnn, Kaap10, ParitKansal, chenmiaoming,
intexcor, petkokp, qooba, capemox, stomrex77, SeverinVisionary, joinalahmed).

Of the 60 PRs: 25 landed via copybara, 1 merged on GitHub (release branch),
15 closed without landing (duplicate / superseded / rejected / internal-only),
19 open. Median days to first review across PRs with any review: about 1.5;
for op PRs assigned to adivinpatel04/yilingyuan/ghpvnist the typical wait
after the Monday triage was 2-5 days; two PRs waited 11-13 days
(#5717 docs, #5691 layer_norm).

## 1. PRs examined

| PR | Author | Area | Reviewers (non-author) | Outcome | Days to 1st review | Size |
|---|---|---|---|---|---|---|
| 6279 | chizhang529 | docs/CONTRIBUTING | zhanyong-wan | landed (copybara 6a02fcb04) | 0.2 | +3/-1 |
| 6277 | joinalahmed | eager runtime (scalar cache) | none yet | open, unassigned | - | +73/-19 |
| 6265 | apullin | ops (var/std, dynamism) | arfaian assigned | open | - | +181/-52 |
| 6228 | apullin | ops (flash attention kernel) | tudorpip | open, approved by domain expert | 0.5 | +111/-9 |
| 6220 | melissawm | docs tooling | LeeHowes | open | 0.1 | +2637/-8070 |
| 6203 | jack8558 | compile (StaticCompiler) | none | open | - | +512/-29 |
| 6199 | Kaap10 | ops (bincount validation) | benawilson, zhanyong-wan | open, 2 rounds done | 1.5 | +118/-11 |
| 6173 | apullin | ops (index_add) | adivinpatel04 assigned | open | - | +208/-17 |
| 6165 | lazyprgmr (Googler) | examples/docs | cota | closed: "Piper is our source of truth" | 0.1 | +2018/-0 |
| 6130 | joinalahmed | ops (logaddexp) | - | closed by author after pacing request | - | +213/-1 |
| 6110 | Zhichenzzz | RL / distributed runtime | jparkerh assigned | open | - | +305/-5 |
| 6106 | joinalahmed | compile (split compiler) | apullin, benawilson, basioli-k | open, approved x2, not yet pull ready | 2.4 | +67/-1 |
| 6105 | joinalahmed | ops (max_unpool) | basioli-k | open, 1 round addressed | 4.0 | +518/-4 |
| 6102 | joinalahmed | eager runtime | apullin, cota | closed as duplicate of #6037 | 2.4 | +111/-0 |
| 6097 | pianpwk | compile (scan) | arfaian | open, changes requested | 4.9 | +61/-16 |
| 6074 | petkokp | ops (linalg_cross, new op) | agarwalt, cota, ghpvnist | open, pull ready, internal lint churn | 3.5 | +458/-1 |
| 6066 | ethannnnnn | ops (split_with_sizes_copy) | ghpvnist, vladbelous | landed (copybara e14e4d1a1) | 3.7 | +21/-5 |
| 6037 | apullin | eager runtime (scalar cache) | benawilson, zhanyong-wan | landed (copybara 1e71c2b51) | 6.0 | +73/-1 |
| 6020 | intexcor | ops (group_norm fwd) | yilingyuan | open, changes requested | 4.1 | +173/-0 |
| 6019 | intexcor | compile (view base offset) | benawilson, zhanyong-wan | landed (copybara e20d8b5c7) | 1.1 | +47/-12 |
| 6011 | Kaap10 | ops (addmv validation) | ghpvnist, zhanyong-wan | pull ready (import pending) | 0.6 | +56/-0 |
| 5981 | yinlin09 | distributed | none | open | - | +18/-1 |
| 5962 | apullin | ops (leaky_relu backward) | adivinpatel04, zhanyong-wan | landed (copybara 4d1158294) | 1.5 | +36/-4 |
| 5959 | weifengpy | examples (FSDP) | none | open, CLA failing | - | +1218/-0 |
| 5956 | apullin | ops (avg_pool backward, bounded dynamism) | yilingyuan, zhanyong-wan | landed a75b08238, then rolled back (torch 2.15 postsubmit) | 1.8 | +448/-18 |
| 5953 | apullin | ops (avg_pool1d autograd) | yilingyuan, zhanyong-wan | landed (copybara 7f679816b) | 1.0 | +166/-1 |
| 5950 | qooba | custom ops (splash_sdpa fake kernels) | chizhang529, zhanyong-wan | landed (copybara 86b465657) | 2.8 | +93/-4 |
| 5913 | ethannnnnn | ops (FSDP2 split_with_sizes_copy perf) | pgmoka, vladbelous | landed (copybara b6ab2fd19) | 2.6 | +130/-22 |
| 5912 | ethannnnnn | ops (cross_entropy lowering) | adivinpatel04, cbasile-g | open, QC round done | 2.6 | +109/-8 |
| 5911 | ethannnnnn | custom ops (splash_sdpa) | - | closed, superseded by internal #5898 | - | +286/-29 |
| 5910 | ethannnnnn | ops (exponential_) | adivinpatel04 | closed, internal PR #5899 preferred | 1.6 | +36/-1 |
| 5889 | chizhang529 | docs/CONTRIBUTING | zhanyong-wan | landed (copybara 0c34e98a6) | 0.1 | +1/-2 |
| 5864 | ethannnnnn | compile (FX pass) | basioli-k, saloni-garg, vladbelous | landed (copybara 1182deb79) | 0.1 | +534/-10 |
| 5854 | Snehal-Reddy | PJRT runtime | benawilson | landed (copybara d3fb8d0c7) | 0.2 | +224/-289 |
| 5847 | saloni-garg | compile (view decomposition) | LizLian | landed (copybara 5731cd519) | 2.9 | +60/-0 |
| 5846 | saloni-garg | compile (view decomposition) | LizLian | landed (copybara dbdccfafe) | 3.0 | +50/-0 |
| 5845 | Snehal-Reddy | distributed (P2P) | jparkerh assigned, ghpvnist nudged | open, no review in 12 days | 8.3 | +91/-58 |
| 5844 | Snehal-Reddy | PJRT runtime | benawilson | closed, folded into #5854 | 1.0 | +70/-16 |
| 5843 | Snehal-Reddy | PJRT runtime | benawilson | landed (copybara 5bb2a40b2) | 1.1 | +9/-4 |
| 5838 | allenwang2333 | hardware utils (TPU v4) | cbasile-g | closed: unsupported hardware | 1.5 | +73/-16 |
| 5828 | chenmiaoming | ops (_int_mm, new op) | adivinpatel04, zhanyong-wan | pull ready, 4 review rounds | 2.7 | +533/-2 |
| 5798 | renato-umeton | docs (translations) | LeeHowes | closed: unmaintainable | 2.9 | +40330/-2 |
| 5778 | Kaap10 | ops (dot validation) | ghpvnist, zhanyong-wan | landed (copybara 968fd65c0) | 3.3 | +24/-20 |
| 5776 | capemox | ops (gcd, new op) | adivinpatel04 | open, 2 rounds, waiting on reviewer | 3.2 | +619/-81 |
| 5745 | chizhang529 | docs/CONTRIBUTING | zhanyong-wan | landed (copybara bddc1ca46) | 0.1 | +19/-2 |
| 5734 | SeverinVisionary | ops (binary alpha opmath) | adivinpatel04, cbasile-g | closed: maintainer landed own CL stack | 4.0 | +424/-35 |
| 5720 | ParitKansal | ops (batch_norm running stats) | adivinpatel04, cbasile-g | open, approved x2 | 4.4 | +428/-31 |
| 5717 | rohitsroch | docs (marimo) | LeeHowes | landed (copybara ab12194bf) | 11.4 | +12/-12 |
| 5715 | ParitKansal | ops (elu/celu) | yilingyuan, zhanyong-wan | landed (copybara 4d7b0d2a6) | 5.4 | +68/-12 |
| 5714 | ParitKansal | ops (gelu erfc) | adivinpatel04, zhanyong-wan | pull ready | 4.6 | +62/-36 |
| 5691 | stomrex77 | ops (layer_norm mixed dtype) | ghpvnist | open, first review after 13 days | 13.2 | +79/-14 |
| 5690 | stomrex77 | ops | - | closed by author (dup of 5691) | - | +79/-14 |
| 5682 | ParitKansal | nested tensor utils | cbasile-g | closed: landed internally as ec6d8cd31 | 5.4 | +125/-11 |
| 5596/5587/5586/5461/5429 | melissawm | docs | LeeHowes | landed same day | 0.0-0.2 | small |
| 5568 | xevimirobruix | docs | LeeHowes | closed: folded into maintainer CL | 0.1 | +237/-144 |
| 5536 | junjieqian | README | LeeHowes | closed: folded into maintainer CL | 0.9 | +2/-2 |
| 5495 | jonathan1920 | CI / release branch | pganssle-google | merged on GitHub (release/0.2.0) | 0.1 | +32/-18 |

## 2. Themed catalogue of reviewer asks

### 2.1 Tests: where they go and what kind

The single most repeated instruction. Op tests live in `tests/ops_unit_test.py`
(TPU vs CPU via `self.assert_close_tpu_vs_cpu`), error tests in
`tests/errors_test.py` / `tests/tpu_errors_test.py` with `et.assert_raises_message(tpu=..., gpu=...)`,
and no new standalone test files or Bazel targets.

- adivinpatel04, #5828: "Please do not add a separate `tests/int_mm_test.py` file and `int_mm_test` Bazel target ... Op unit tests belong in `tests/ops_unit_test.py` (using `self.assert_close_tpu_vs_cpu`), and error validation tests belong in `tests/errors_test.py` (using `et.assert_raises_message` with both `gpu=` and `tpu=` patterns ...)"
- adivinpatel04, #5828 (high-level pointers): "1) We require coverage of Aten lowerings in ops_test.py to ensure numerical stability. Currently the golden generation is done internally for numerics diff, so we ask that you instead add tests to ops_unit_tests.py for now to test against cpu and ensure comprehensive numerical parity prior to asking for a review."
- adivinpatel04, #5734: "Per the repo testing guidelines, unit tests in `tests/ops_unit_test.py` should compare TPU against CPU dynamically using `self.assert_close_tpu_vs_cpu(fn, rtol=0, atol=0)` ... rather than hardcoding magic float constants"
- yilingyuan, #6020: "This is not enough to cover test cases. Could you move the tests to ops_unit_test.py and compare TPU against CPU with more coverage? For example, 32 and bf16 inputs with mixed and non-mixed params, with and without weight/bias, empty input..."
- chizhang529, #5950: "could you move these two tests next to the other `test_splash_sdpa_*` tests? That keeps the splash coverage in one place."
- agarwalt, #6074: "I think we can remove the tests here [a C++ test file] as the new tests ops_unit_test and dynamism_test appear to cover the cases already."
- adivinpatel04, #5776: "nit: could we also include a non-contiguous tensor input case (e.g., a transposed `.t()` or sliced `[::2]` view) ... to cover non-contiguous inputs alongside broadcasting, 0D, and empty shapes?"

Test structure (compile/runtime side, basioli-k and LizLian):

- basioli-k, #5864: "can we make this a parametrized test based on these ops?" / "an e2e test shouldn't be inside this file. This file should just verify the pass." / "Could we maybe to a torch.compile test and check if the HBM usage increases (test should be failing before your change and passing with your change)."
- basioli-k, #5864: "You should use _UNANNOTATED_ALIAS_OPS and _SET_SOURCE_OPS somehow, if those get extended this test won't make sure we use them and we'd like to avoid duplication"
- LizLian, #5847: "We prefer one behavior per test. This test checks three ... Suggest splitting it"
- LizLian, #5847: "Redundant assertion: `self.assertIs(base, view)` and `utils.assert_close(base, view)`. Keep assertIs and drop assert_close."
- zhanyong-wan, #5828: "Style: avoid conditionals based on the test case parameter. Just remove `case` from the parameters and split the test case into 3."
- zhanyong-wan, #5828: "I'd remove this test case as it tests an implementation detail that users don't care about."
- arfaian, #6097: "Please avoid patching internal implementation details which can change without notice."
- adivinpatel04, #5912: against a 1.24 GB allocation plus a 30 s wall-clock assertion in a unit test; asked instead for a small shape and "asserting directly on the lowered StableHLO text via `tpu_torch_compile.serialize_mlir_text(tpu_torch_compile.build_mlir(...))` (`self.assertNotIn("stablehlo.gather", mlir_text)`)".

### 2.2 Error coverage (error_coverage_test) and GPU-parity error tests

Every `TT_RET_CHECK` / `TT_CHECK_THROW` / `TT_ERROR` in `csrc/` must be hit by
an error test, or carry `// ERROR_COV_INFEASIBLE=<why>`. Error tests supply
both `tpu=` and `gpu=` strings; GPU strings must match real CUDA output,
sometimes gated on torch version. `tests/error_coverage_data.yaml` exists in
the public tree (`google/tools/error_coverage_data.yaml` internally).

- yilingyuan, #6020: "These 5 `TT_CHECK_THROW` are not covered, so `error_coverage_test` will fail. Please cover them in errors_test.py/tpu_errors_test.py"
- adivinpatel04, #5776: "`error_coverage_test` scans all `csrc/` files for `TT_ERROR` / `TT_RET_CHECK` / `TT_CHECK_THROW` macros and fails if any new file has an uncovered error macro without an `// ERROR_COV_INFEASIBLE=...` annotation."
- benawilson, #5844: "To satisy error coverage requirements, either add a test to errors_test_tpu exercising ValidateContiguousBase or remove one TT_RET_CHECK from this file"
- adivinpatel04, #5720: "`ValidateIsFloating(input, "input")` is now duplicated in 3 places ... leaving 2 uncovered error checks in this file."
- adivinpatel04, #5828 (pointers): "Error coverage: You may inspect the CUDA lowering for this op for example and find that there are certain conditions under which they throw an error. Please add that to errors_test.py ... We do not want to regress error coverage"
- agarwalt, #6074: "Could you please also fix the GPU error messages? ... Expected the gpu message: `"cross" not implemented for 'Bool'` but got: `"cross_cuda" not implemented for 'Bool'`"
- adivinpatel04, #5828: "In PyTorch >= 2.15, `_int_mm_out_cuda` ... switched to `check_mm_shapes` ... Please gate the `gpu=` strings on `utils.get_torch_version() >= (2, 15)`"
- pgmoka, #5913: "The oen request is to add a new test under errors_test.py for coverage" (provided the full test body, with `message_reviewed_by="pgmoka"`).
- ghpvnist, #6011/#5778: inline `message_reviewed_by="gunhyun",` on each new assertion (reviewer wants their LDAP recorded as the message reviewer).

### 2.3 Error message wording (quality-champion pass)

zhanyong-wan reviews error strings line by line at the QC stage; benawilson
and adivinpatel04 also rewrite them at the domain stage. House style:
lowercase `expected ..., got ...`; no op-name prefix (TT_KERNEL adds
`op(): `); `[]` for shapes; `1D`/`2D`; `vs` between mismatched values;
PyTorch concepts only, never StableHLO/XLA; `NotImplementedError` for dtype
gaps to match GPU.

- benawilson, #6199: "You don't need to specify "bincount"; the TT_KERNEL wrapper will already prepend with "bincount()" in any error messaging."
- benawilson, #6199: "Split into two checks and return errors specific to the error case"
- zhanyong-wan, #6199: "The message should focus on what is wrong - don't let the reader guess which part of the constraint is violated. `expected integral inputs, got float32`"
- zhanyong-wan, #6199: "Please raise NotImplementedError instead to match GPU behavior."
- zhanyong-wan, #6199: "For consistency, say `got` instead of `but got`." / "for consistency, please say 1D and 3D instead." / "`2 vs 5`"
- zhanyong-wan, #5828: "Nit: `[32, 64] and [32, 16]` - we always use `[]` for shapes."
- zhanyong-wan, #5828/#6199: "please use a `"""..."""` string without line break, so that it's easy for tools to update the message as needed."
- zhanyong-wan, #5956: "The error message should not reference implementation details like stablehlo, as we don't expect TorchTPU users to understand what they mean. Usually we validate the op inputs inside TT_KERNEL directly, before invoking the builder."
- zhanyong-wan, #6011: "Grammar nit: "arguments shapes" => "argument shapes"" / "simplify "have compatible shapes" to "be compatible"? Then we can drop " which is not compatible" as it's obvious."
- ghpvnist, #6011: "Can we make this error message more intuitive like the gpu mode? Something along the lines of `expected arguments shapes [5], [2, 3], [3] to have compatible shapes, got 5 vs 2 ...`"
- adivinpatel04, #5828: "Error messages must start with lowercase `expected` ... Prefer `ToString(...)` over streaming `c10::Device` directly"

### 2.4 Use the existing helper / follow the neighbouring kernel

- basioli-k, #6105: "Can we try to use using BuildScatterShlo from scatter.h" / "nit: use NumElements from utils.h" / "instead of using expect_int we can use ConcreteSizeToDimensions on output_size from aten_utils.h" / "Can we validate input dtypes? E.g. call ValidateNotLowPrecisionFloat or reject boo/complex types"
- adivinpatel04, #5912: "Let's use `Dimensions` instead of `std::vector<int64_t>` ... the repo's `CheckIntVec` check disallows `std::vector<int64_t>` in favor of `Dimensions`"
- adivinpatel04, #5828: "Per the C++ op implementation guidelines, please use `Dimensions` ... instead of raw C arrays ... and extract the StableHLO builder into a standalone `BuildIntMmShlo` function in the anonymous namespace (matching `BuildMmShlo` in `csrc/ops/mm/mm_aten_kernels.cc`)"
- adivinpatel04, #5734: "`MakeTensor(const at::Scalar&, ...)` is deprecated in favor of `PromoteScalar`"
- adivinpatel04, #5720: "Please rebase onto `main` and keep `IgnoreInCacheKey(momentum, ...)`" / "Do not call `at::native::resize_output(tensor, buf.dimensions())` inside a `TT_KERNEL` ... use `internal::AssignBufferToOutput(...)`"
- benawilson, #6199: "Avoid redispatching through `at::` operations; this requires a fairly expensive roundtrip through the torch dispatcher. Slightly better would be call into `AtenAMinMaxOut` directly."
- cbasile-g, #5734 (closing): maintainers landed `InferComputationDtype` and "#6149: Add lint rule and op implementation guidance for `InferComputationDtype`" (a presubmit lint `CheckAtToOpMathType` now rejects direct `at::toOpMathType`).
- agarwalt, #6074 (internal lint, after approval): "files under torch_tpu/csrc/ops/<op>/ should be named *_kernels.cc or *_kernels.h ... combine the implementation under *_kernels.cc and *_kernels.h (i.e. 2 files)"

### 2.5 Structure: out= variants, validation split, complexity, naming

- basioli-k, #6105: "This can just return absl::Status and we can populate the out tensor so that the out variants below can be used by the non out variants" / "nit: can this use the out variant?"
- basioli-k, #6105: "Function name makes it unclear that this returns output dimensions as well. Please split this into `absl::Status ValidateMaxUnpoolInputs` and `absl::StatusOr<Dimensions> ComputeOutputDimensions`"
- adivinpatel04, #5828: "Please split `out` validation into `ValidateIntMmOutInputs(self, mat2, out)` (called only by `AtenIntMmOut`) so `IntMm` only runs `ValidateIntMmInputs(self, mat2)` once."
- adivinpatel04, #5912: "Adding this `if / else` branch ... bumps its cognitive complexity. Could we extract a small `SelectTargetLogProbs(...)` helper that uses a `switch` ... That keeps `BuildNllLossForwardOutShlo` under the complexity limit and gives us compiler exhaustiveness checking on the enum."
- basioli-k, #5864: "_write is too generic, function is very specific. _add_into_node or something more relevant" / "please give a descriptive function name" / "why do some params get a lambda and some get a helper? Please consistent naming"
- adivinpatel04, #5714: "per the C++ style guidelines on `auto` when the type is not explicitly stated in the initializer, could we use explicit `mlir::MlirOp` types instead of `auto`"
- adivinpatel04, #5720: "Nit: mark `const int64_t num_features = ...;` (and immutable locals in the helpers above) as `const`."
- yilingyuan, #5956: "This file now has 3 anonymous namespace blocks. Could you move `HasBoundedDimension` up into the first anonymous namespace block"
- cbasile-g, #5912: "Nit (IWYU / `misc-include-cleaner`): could you add `#include "mlir/IR/Types.h"` ... for `mlir::Type`?"

### 2.6 Buffer allocation, donation, dispatcher round-trips (perf hygiene)

- adivinpatel04, #5828: "calling `MakeEmptyTensor` allocates an empty `int32` device buffer and then `IntMmOut` ... immediately allocates a second ... `AtenIntMm` can directly `return MakeTensor(std::move(buf))` (single allocation)"
- adivinpatel04, #5720: "Switching from `DispatchOpOut` to `DispatchOp<...>(..., {.enable_autodonation = false})` disables buffer donation completely ... call `internal::AutoDonateInPlaceBuffers` before `DispatchOp`"
- adivinpatel04, #5720: "3 throwaway device buffer allocations per training step"
- adivinpatel04, #5734: "dispatching a separate `UnaryOp(kToCopy)` is redundant" when promoted dtype equals computation dtype.
- benawilson, #6037: accepted a full cache clear but noted "A full clear does have some minor risk of unnecessary recompilation ... the complexity cost of an LRU cache or similar is probably not justified."

### 2.7 Recompilation / cache keys

- adivinpatel04, #5720: "`TT_KERNEL` automatically hashes every non-`PromotedScalar`, non-`IgnoreInCacheKey` scalar in its argument tuple into `param_keys` ... every change to `momentum` still changes `param_keys` and forces a cache miss and recompilation on every step" and asked for "a compile-count assertion ... to verify that running 3 steps does not recompile".

### 2.8 Dtype coverage and CPU/CUDA parity

- adivinpatel04, #5828: "Dtype coverage. We want to ensure that dtypes supported by CPU/CUDA are also supported for TPU. If your lowering doesn't support a specific dtype and there isn't a non hardware agnostic reason, its probably wrong."
- adivinpatel04, #5776: "`c10::isIntegralType(dtype, /*includeBool=*/false)` ... returns `true` for `uint16`, `uint32`, and `uint64`, whereas `BuildGcdShlo` only supports `I8`, `UI8`, `I16`, `I32`, and `I64`"
- adivinpatel04, #5776: INT_MIN parity: CPU and CUDA return `gcd(-128, 6) == -2`; "Please replace the hardcoded TPU-only assertion with `self.assert_close_tpu_vs_cpu` across all signed integer dtypes"
- yilingyuan, #6020: "Please confirm the return dtypes of mean and rstd. CPU group_norm ... so mean/rstd dtype should be f32 when types are mixed."
- ghpvnist, #5691: "check with `TT_RET_CHECK` that any defined `weight`/`bias` share the same dtype and match either `input.scalar_type()` or `at::kFloat`"
- agarwalt, #6074: "No need for this exhaustive list of dtypes, I think it should be fine to test for a few of these types."
- adivinpatel04, #5828: asked why uint8 `self` (CPU-supported) was excluded; accepted scoping to int8 "it matches CUDA ... Tracking the CPU-only `uint8` `self` case separately is the right call."

### 2.9 Tolerances

- adivinpatel04, #5720: "`bfloat16` tolerances should really stay within `rtol <= 1e-2, atol <= 1e-2`" and widening a tuple's tolerance "relaxes the check on the `float32` outputs ... all the way to `3e-2`/`2e-2`"
- agarwalt, #6074: "Just checking that these tolerances are the minimal/tightest possible needed?"
- adivinpatel04, #5714: cast expected values to float32 so `assert_close` "also verify that `out` and `grad` have `dtype == torch.float32`"

### 2.10 Bounded dynamism

- yilingyuan, #6020: "`group_norm` isn't skipped in `dynamism_test_utils.py`, so `ops_test_dynamic` will now hit this kernel and it will likely fail. Please either support bounded dynamism or add a skip entry with a TODO."
- ghpvnist, #5691: main "switched to `BroadcastIfNeeded(...)` to preserve bounded dynamic dimensions" - rebase and keep those calls.
- #5956 (apullin) is the model for adding bounded-dynamism support to a backward: follow `ConvolutionLike`, run on upper-bound static shapes, mask, re-bound. It landed, then was rolled back because a dynamic-shape test "failed under PyTorch 2.15 ... on main postsubmit"; "The affected tests currently don't run on PRs from forks, so the failure only surfaced after merge." (yilingyuan)

### 2.11 Autograd and GPU golden data

- yilingyuan, #5953: "Since ops_test requires regenerating GPU goldens, let's get this merged first and I'll follow up internally to update the goldens and enable check_grad=True in ops_test.py."
- adivinpatel04, #5776: "`self.do_test_op("gcd", max_samples_per_op_dtype=6)` in `tests/ops_test.py` looks good for now. we will generate goldens later."
- adivinpatel04, #5776: "The domain experts list in `csrc/ops/gcd/BUILD` (matching `lcm`) can have adivinpatel04 and ghpvnist."
- cbasile-g, #5912: non-blocking note that the `torch.compile` path still decomposes via `at::native::cross_entropy_loss_symint` - "No need to expand scope here".

### 2.12 Comments and documentation in code

- agarwalt, #6074: "please consider adding a brief line or so of comments with the equation, as it makes it easier to read and understand for future readers."
- chizhang529, #5950: "Document why contiguous, e.g. `# Kernel outputs are allocated via torch.empty, which are contiguous whatever the input strides.`"
- basioli-k, #5864: seven separate "please add a comment explaining why ..." requests on one pass.
- LizLian, #5847: "Google style asks comments to explain the code as it is now, not its history." / "Add a TODO: record the follow-up work in the code. `# TODO: lizliang - ...`" / "Add a `Raises:` section"
- LizLian, #5847: "Keep the docstring to what the test checks" and test names that "state the behavior being checked".

### 2.13 Scope: small PRs, split, one thing per PR

- yilingyuan, #5956: "can you reduce this large PR or split it into smaller ones if possible?" then, after rework: "On the "small PR" guidance: it isn't a strict line count. The point is that each PR does one thing and includes its tests, and this one does exactly that, so no need to trim further."
- adivinpatel04, issue #6111 (eight ops in one branch): "your PR and issue is going to get rejected as its constructed right now. Please come up with 1 issue per op at a time. I also took a quick scan at your code changes and the code quality is not acceptable."
- cota, #6165: "before dropping a big PR or CL, please discuss with us first what you want to do and why it is important for the project."
- cbasile-g, #5912: "could you change `Fixes #5829` to `Part of #5829` in the PR body so merging doesn't auto-close the issue?"
- adivinpatel04, #5734: negotiated the split between the external PR and an internal PR (#5882) explicitly, then the maintainer landed a 5-CL stack and closed the external PR: "we can close this PR".

### 2.14 Pacing: one op PR at a time, don't flood

- adivinpatel04, issue #6118: "you have #6103 out for one op implementation. Please hold off on creating any more issues or opening a PR for this until you merge that one. We will review them sequentially so you can intake op implementation feedback."
- cota, issue #6114: "like most OSS projects we are getting a lot of code coming our way, so please be mindful of our time and try to spread your submissions over time ... a big flood of issues/PR's is problematic for us."
- CONTRIBUTING.md: "we may limit the number of PRs a contributor can open at one time." Observed: GitHub blocks a 4th open PR for new contributors (apullin, #4784: "I have 3 PRs open, so GitHub won't let me open another one yet.").
- apullin's pattern that works: post "The fix is on my fork: <compare link>" on the issue, open the PR when a slot frees ("PR up: #6173").

### 2.15 Issue first, claim it, link it

- CONTRIBUTING.md: "For anything more complicated than trivial fixes, please open an issue first" and "PRs without a linked issue are reviewed as time allows."
- adamomainz, issue #4784 (to apullin asking about etiquette): "very happy for you to put up a PR. Feel free to put up the PR, tag the issue and add me as the reviewer"
- WillFroom, issue #6150: "Thanks @apullin for the report and proposed fix! @tudorpip implemented this logic so I'll let him check & respond" - the implementer then approved #6228 within an hour of being requested.
- saloni-garg, #5847: proactively apologised for not commenting first ("I had reproduced the bug ... and had a small fix ready, so I opened this PR instead of commenting on the issue first") - accepted.
- Duplicates are closed in favour of the earlier one, regardless of quality: cota, #6102: "Closing this PR as a duplicate. @joinalahmed feel free to suggest improvements to #6037." adivinpatel04, #5910: "Rejecting this PR, I got in a PR before this one was created with more in-pattern testing/impl parity".

### 2.16 Documentation and examples policy

- cota, issue #6114: "We keep in-tree docs/ and examples/ tightly focused on core first-party TPU training, inference, and distributed scaling workflows that we can continuously test in our internal CI without introducing additional third-party library dependencies like Captum ... This write-up is better fit as a standalone community repo ... or external blog post."
- LeeHowes, #5798: "This is going to be impossible to maintain and we would have noone to validate the translations ... I am rejecting this PR for the time being."
- LeeHowes, #5717: "we are planning to migrate away from marimo and treat the markdown as the source of truth."
- cota, #6165: "Case-study numbers should come from reproducible TPU execution (backed by the benchmark/checkpoint harness in-repo), and official docs must be authored via Marimo sources in docs/source/*.py and wired into oss/mkdocs.yml (strict: true)."
- LeeHowes folds small outside docs fixes into his own CL and closes the PR (#5568, #5536).

### 2.17 Hardware / product scope

- cbasile-g, #5838: "TPUv4 detection was intentionally not included since it is not a supported TPU." / "We don't test on it and don't plan to support it, so I wonder what would that buy us."

### 2.18 Rebase, merge conflicts, CI

- chizhang529, #5950: "Overall LGTM, thanks! Please rebase onto `main` and resolve merge conflicts."
- vladbelous, #6066: "looks like this change has merge conflicts (ops_unit_test.py). Could you please rebase this and I'll try landing it again."
- zhanyong-wan, #6011: approved then "Please fix the CI errors." (clang-format).
- adivinpatel04, #5734: "Please run the formatter on `tests/ops_unit_test.py`." (2-space vs 4-space continuation).
- zhanyong-wan, #5828: "Could you push the small fix to the PR branch? Thanks." (author offered to let them patch the CL; they preferred the PR branch).
- Fork PRs do not run TPU CI: "The affected tests currently don't run on PRs from forks" (yilingyuan, #5956); authors must ask for workflow approval ("Could you approve the CI runs ?", chenmiaoming).

### 2.19 PR description / commit expectations

- Template: `## Description` and `## Test Plan` (`.github/pull_request_template.md`). Reviewers read the Test Plan: saloni-garg on #5864 noted torch 2.14.1 "is a version your test plan does not list".
- `Fixes #N` vs `Part of #N` matters (cbasile-g, #5912).
- Descriptions that reviewers praised include root cause, numbers and the alternative considered: cbasile-g, #5912: "really clean root-cause analysis, benchmarks, and fix"; pgmoka, #5913: "I have ran tests on my end and there has been no numerical change ... pretty consistent performance improvement on my benchmarks too."
- CONTRIBUTING.md on AI: "you may add an `Assisted-by:` tag in your commit message. AI assistants should not be considered co-authors on any commits." Also "'an AI wrote it' is not an acceptable answer."

### 2.20 CLA

- google-cla[bot] posts on first PRs ("you'll need to sign a Contributor License Agreement"). #5959 (weifengpy) has sat 9 days with no review and a failing CLA check. Memory note: a `Co-Authored-By: ... <noreply@anthropic.com>` trailer fails the CLA bot; use `Assisted-by:`.

### 2.21 Performance measurement expectations

- Perf PRs were accepted when they carried a before/after number in the description and a test that fails before / passes after: #5864 ("forward holds 32.3 MiB with main's pass ... 0.28 MiB with this change"), #5913 (benchmarks zip), #5912 (compile time table). basioli-k asked for the regression test to be a real `torch.compile` memory check, not a pass-level unit test.
- For scope-limited perf claims, maintainers ask for the mechanism, not a timing assertion (#5912: "won't catch regressions on TPU generations where the gather form already compiles under 30s").

## 3. Process signals

### The pipeline (CONTRIBUTING.md "Code Reviews", observed in events)

1. Bot comments "Thanks for contributing to TorchTPU, @author! :fire: To ensure smooth code reviews, we kindly ask authors and reviewers to follow the process documented in .../CONTRIBUTING.md#code-reviews. Happy hacking!" (`.github/workflows/on_pr_ready.yaml`). Turn tracker adds `stat:awaiting-reviewers`.
2. Oncall triage assigns a domain expert via `review_requested`. Observed oncalls: benawilson (week of Oct 26-30, batch-assigned ~15 PRs on Mon Oct 26 16:24-17:33 UTC and Oct 28 13:05-14:04), cota (Nov 2 16:44-16:50), ghpvnist (Nov 2-5, also pings idle reviewers: "@agarwalt could you provide some updates?"). Triage happens in a batch, so a PR opened on a Friday waits until Monday.
3. Domain expert reviews; approves; requests zhanyong-wan / cbasile-g / vladbelous / LeeHowes (quality champions; CONTRIBUTING lists cbasile-g, LeeHowes, vladbelous, zhanyong-wan) and sets them as assignee. zhanyong-wan, #5715: "Per the PR review process, please add me after the domain expert (Yiling) approves the PR."
4. QC reviews style/error strings; approves; adds `pull ready` label. Copybara imports as an internal CL needing two Googler approvals; new pushes need re-approval ("@adivinpatel04 , I think you need to re-approve this.", zhanyong-wan, #5828).
5. Copybara closes the PR (not merged) and references the main commit: `copybara-service[bot] closed <sha>`. Internal presubmit (lint, error coverage, torch 2.15) can bounce the PR back after `pull ready` (#6074 filename lint, #5828 duplicate test entry, #5962 "Working on some internal build issues that block this PR").
6. Post-submit failures are rolled back and the author offered the fix (#5956).

### Labels

- `stat:awaiting-reviewers` / `stat:awaiting-requester` flipped by `tools/turn_tracker.py` on every comment/review (requester = PR author; responders = assignees + reviewers). Third-party comments are ignored by the tracker.
- `pull ready`: QC approval, triggers Copybara import. `ci:torch-tests`: path label for `tests/torch/**`. `stale` after 30 idle days on assigned items, closed 14 days later (`daily_project_management.yml`).

### Explicit policy statements

- CONTRIBUTING.md: "Let us assign owners and reviewers: Please don't assign or request specific people as assignees or reviewers ... unless you have arranged this with them beforehand."
- CONTRIBUTING.md: "Small, focused PRs ... A concise 100-line diff gets reviewed far faster than a 1,000-line rewrite." "All non-trivial PRs must link to the relevant GitHub issue." "We prioritize reviews by the priority of the linked issue."
- CONTRIBUTING.md "Claiming an Issue" (added by #5745 after "duplicate work due to internal and external contributors independently fixing the same GitHub issue"): check assignee/comments/linked PRs; "Comment before you start"; "If several fixes for the same issue are proposed anyway, maintainers may choose the one that best addresses the issue."
- cota, issue #6036: "We've marked this as low priority. This means we acknowledge the issue, but can't commit to a specific timeline" - yet the PR still landed in 7 days once assigned.
- cota, #6165: "For Googlers like yourself, Piper is our source of truth rather than GitHub fork PRs."
- The internal "op implementation guidelines" (`implementing_ops/SKILL.md`, referenced by cbasile-g's commit b83f80db2 and adivinpatel04's "Per the C++ op implementation guidelines") is not in the public tree; `.agents/skills/docs-review/SKILL.md` is. adivinpatel04, #5828: "Lowering ops as an external contributor is a bit of a tricky game for TorchTPU currently. We envision it becoming better very soon." Also #6111: "We understand its difficult to implement op lowerings without some concrete guidelines and we are working on that".

## 4. Reviewer map (who reviews what)

| Login | Role observed | Areas | Style |
|---|---|---|---|
| adivinpatel04 | ops domain expert; most active on external op PRs (#5828, #5776, #5720, #5734, #5714, #5912, #5962, #5910, #6173) | ATen lowerings, dtype/opmath, cache keys, buffer donation, test conventions | Very long inline comments with exact code to paste, PyTorch source line refs; CHANGES_REQUESTED first round; approves 2-4 days after fixes; enforces pacing on issues |
| yilingyuan | ops domain expert (#5956, #5953, #5715, #6020) | pooling, norms, dynamism, error coverage | Short, specific; asks for split or dynamism skip; follows up on rollbacks |
| ghpvnist | ops domain expert + oncall (#6011, #5778, #5691, #6074 nudges) | validation ops, layer_norm | Rewrites error strings to match GPU; wants `message_reviewed_by="gunhyun"` |
| benawilson | eager runtime / PJRT domain expert and Oct oncall (#6037, #6019, #6106, #5854, #5843, #5844, #6199) | op_dispatcher, buffers, compile view base, bincount | Fast approvals (hours) on runtime fixes; thoughtful limitation notes; error-coverage asks |
| basioli-k | compile domain expert (#5864, #6105, #6106) | split_compiler, FX passes; also reviewed max_unpool kernel structure | Many nits per round, "please add a comment", parametrize tests, use existing helpers |
| arfaian | compile (#6097, #6265) | scan, var/std | "avoid patching internal implementation details" |
| LizLian | compile views (#5847, #5846) | view_decomposition | Google style for comments/docstrings/TODOs, one behaviour per test |
| cbasile-g | QC + numerics owner (#5912, #5720, #5734, #5682, #5838) | opmath/alpha precision, nested tensors, hardware scope | Will land his own CL stack if an external PR overlaps his work; asks `Part of #N` |
| vladbelous | QC (#5864, #5913, #6066) | FSDP2 kernels, compile passes | Lands quickly, even before nits are fixed ("looks like I was too quick to land") |
| zhanyong-wan | QC on nearly every op PR (#6199, #6011, #5828, #5956, #5962, #5953, #5950, #5778, #5715, #5714, #6037, #6019) | style, error message wording, test structure | Approves within ~1 day of being added; CHANGES_REQUESTED with "Thanks!" for wording nits; adds `pull ready` |
| LeeHowes | docs owner and QC (#6220, #5798, #5717, #5568, #5536, melissawm PRs) | docs, README, notebooks | Same-day approvals for Quansight docs; rejects translations/third-party docs; folds small fixes into his own CL |
| cota | triage lead / policy (#6102, #6165, #6074 QC approval, issues #6114, #6036, #6118) | duplicates, scope, contribution policy | Closes duplicates and out-of-scope work with a policy explanation |
| agarwalt | ops (#6074) | linalg | Light nits, then hit internal lint |
| tudorpip, WillFroom | attention kernels (#6228, issue #6150) | flash/splash attention | Implementer reviews the fix to their own code; fast LGTM |
| pgmoka | FSDP perf (#5913) | split_with_sizes_copy | Runs benchmarks himself, supplies the test to add |
| chizhang529 | custom ops / docs (#5950, CONTRIBUTING) | splash_sdpa Python, process docs | Rebase + comments + test placement |
| jparkerh | distributed/RL (#6110, #5845 assigned) | ProcessGroupTpu | Slow: #5845 unreviewed 12 days |
| pganssle-google | CI/release (#5495, #6279) | workflows, wheels | "We only execute this in one way, we don't need fallbacks." |
| apullin (external) | reviews other externals' PRs (#6106: found a regression; #6102: flagged duplicate) | - | Maintainers act on his findings |

## 5. What gets an external PR merged here (12 lines)

1. One issue, one op/bug, one PR; comment on the issue first, post a fork compare link if your PR slots are full, and never have more than one op-lowering PR in flight.
2. Link `Fixes #N` (or `Part of #N` if the issue should stay open); use the `## Description` / `## Test Plan` template with torch/libtpu versions and the exact `bazel test` targets you ran.
3. Keep the diff to the fix plus its tests; "each PR does one thing and includes its tests" is the real rule, not a line count.
4. Tests go in `tests/ops_unit_test.py` via `self.assert_close_tpu_vs_cpu` (default tolerances, bf16 within 1e-2), next to the existing tests for that op; no new test files or Bazel targets; parametrize instead of repeating.
5. Every new `TT_RET_CHECK`/`TT_CHECK_THROW`/`TT_ERROR` needs an `errors_test.py`/`tpu_errors_test.py` case with both `tpu=` and real `gpu=` strings (version-gated if CUDA changed), or an `// ERROR_COV_INFEASIBLE=` note.
6. Error strings: lowercase `expected X, got Y`, no op-name prefix, `[]` shapes, `1D`, `vs`, PyTorch vocabulary only, `NotImplementedError` for dtype gaps; validate in the kernel before the builder.
7. Reuse the shared builders and helpers (`BuildScatterShlo`, `NumElements`, `Dimensions`, `PromoteScalar`, `InferComputationDtype`, `IgnoreInCacheKey`, `AssignBufferToOutput`, `BroadcastIfNeeded`); never call `at::` ops or `resize_output` inside `TT_KERNEL`; implement `.out` once and route the functional variant through it.
8. Match CPU/CUDA dtype coverage and semantics (INT_MIN, mixed dtypes, 0-d, empty, non-contiguous) and either support bounded dynamism or add the `dynamism_test_utils.py` skip with a TODO.
9. Explain the "why" in code comments and docstrings written for the present state of the code, and add `TODO: <ldap> - ...` for deferred work when the reviewer asks.
10. Expect two passes: domain expert (correctness, structure) then quality champion (zhanyong-wan: wording and test style); answer every inline thread with what changed and the commit, re-request review, and push follow-up fixes to the PR branch rather than letting them patch the CL.
11. Rebase promptly when asked, run the formatter, and remember fork PRs skip TPU CI, so state the TPU generation and torch version you tested on, including torch 2.15 if the test touches dynamic shapes.
12. Do not send third-party-library docs/examples, translations, unsupported-hardware support or multi-op batches; for anything architectural, discuss on an issue first, and sign the CLA with an `Assisted-by:` (not `Co-Authored-By:`) trailer if an AI helped.
