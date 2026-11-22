# torch_tpu: how to contribute so it lands (synthesis, 2026-11-06)

Distilled from four read-only studies of google-pytorch/torch_tpu at origin/main
214c68c83 and of 60 external PRs. Details with `path:line` citations:

- [study_contributor_rules.md](study_contributor_rules.md) — written rules, lint markers, test conventions, CI
- [study_code_conventions.md](study_code_conventions.md) — kernel/Python/commit style from the code; §10 skeleton for a new op; §11 anti-patterns
- [study_pr_review_feedback.md](study_pr_review_feedback.md) — what reviewers actually ask, with quotes; reviewer map
- [study_architecture_history.md](study_architecture_history.md) — directory map, velocity, hot spots, landed external changes

## 1. Process (CONTRIBUTING.md)

- Pipeline: bot "Review Process" comment → oncall (benawilson Oct; cota/ghpvnist Nov) assigns a domain expert → expert approves and adds a quality champion (zhanyong-wan on almost every op PR; cbasile-g, vladbelous, LeeHowes) → `pull ready` label → Copybara CL needing two Googler approvals → PR is **closed**, commit lands under the original author. Median first review ~1.5 days; op PRs wait for the Monday batch (2–5 days).
- **Every push after approval forces the champion to re-approve the import.** Never push cosmetic commits after approval. (This is why #6106's attribution lines stay until it lands.)
- Never request reviewers or assign yourself. One issue per op, one op PR at a time, "spread your submissions over time"; maintainers may cap open PRs (CONTRIBUTING.md:258-262). Duplicates are closed in favour of the earlier PR regardless of quality; maintainers may land their own CL over an overlapping external PR.
- Link an issue (`Fixes #N`) on every non-trivial PR; reviews are prioritised by the issue's priority. PR body = template `## Description` / `## Test Plan`. Optional `Assisted-by:` is allowed by policy; AI co-authors are not. (Our rule: no attribution at all.)
- Out of scope by policy: third-party-library docs/examples, translations, unsupported hardware, Googler fork PRs.

## 2. What reviewers ask for, ranked by frequency

1. **Test placement and form.** Op tests go in `tests/ops_unit_test.py` via `assert_close_tpu_vs_cpu(lambda device: ...)`, `parameterized.named_parameters`, default tolerances (bf16 ≤ 1e-2), no hardcoded expected constants, next to related tests; no new test files or Bazel targets. GPU goldens for `ops_test.py` are generated internally after merge.
2. **Error coverage.** Every `TT_RET_CHECK` / `TT_CHECK_THROW` / `TT_ERROR` needs an `errors_test.py` case with real `tpu="""..."""` and `gpu="""..."""` strings (version-gated if CUDA changed), or `// ERROR_COV_INFEASIBLE=<why>`. Reviewers want `message_reviewed_by=<ldap>`. `tests/error_coverage_data.yaml` is a presubmit gate since 2026-10-27 (dae987dce).
3. **Error wording.** Lowercase "expected X, got Y"; no op prefix (TT_KERNEL adds `op(): `); shapes as `[2, 3]`; "1D"; PyTorch vocabulary, not StableHLO; `NotImplementedError` (kPythonNotImplementedError) for dtype gaps; validate inside TT_KERNEL before building; no trailing period. Debug builds crash on violations.
4. **Reuse helpers**: BuildScatterShlo, NumElements, Dimensions (not std::vector), CopyIntVector, PromoteScalar, InferComputationDtype (not toOpMathType), IgnoreInCacheKey, AssignBufferToOutput, BroadcastIfNeeded, SafeWrapDim, ResizeTensorIfShapeDiffers. Never redispatch `at::` ops or `resize_output` inside TT_KERNEL; implement `.out` once and route functional variants through it; split Validate vs Compute; explicit types over `auto`; const locals; IWYU includes; exact buildifier-sorted BUILD deps.
5. **Parity**: CPU/CUDA dtype coverage ("if your lowering doesn't support a specific dtype … it's probably wrong"), 0-d, empty, non-contiguous, INT_MIN, mixed dtypes; bounded dynamism supported or a `dynamism_test_utils` skip with TODO; comments explain present-state "why"; deferred work as `TODO: ldap -`.

## 3. Code shape for a new kernel (see study_code_conventions.md §10 for the skeleton)

- 2026 Apache header; guard `TORCH_TPU_CSRC_OPS_<DIR>_<FILE>_H_` with `#endif  // GUARD`; includes: own header / std / one sorted quoted block; `namespace torch_tpu { namespace { helpers } Aten* }`; only `error`/`internal`/`kiln`/anonymous nested namespaces; no `torch_tpu::` qualifiers inside.
- `TT_KERNEL(OpName::kX, param_keys|_, (args), {...})`; `PromoteScalar(..).AvoidPromoting(kOne)` before the macro; `UnaryOpOut/BinaryOpOut/DispatchOp<N>` with designated-initializer options; `absl::bind_front(BuildXShlo, params...)` with MlirOp last; `MakeTensor(std::move(buf))`.
- Validation order for out= kernels (cumsum is canonical): device → dtype → SafeWrapDim → ResizeTensorIfShapeDiffers → numel()==0 return → 0-D → NotImplemented checks → dispatch. Functional variants return MakeEmptyTensor on any zero dim before bf16/f16 rejection.
- Registration: `csrc/ops/op_names.h` enum + name string; `ImplStable<OpName::kX>(m, AtenX)` in `csrc/eager/tpu_aten_kernels.cc`; dep in `csrc/eager/BUILD`; remove from `tests/native_functions_data.py` UNREGISTERED_OPS.
- Error codes → Python: kInvalidArgument/others → RuntimeError; kPythonNotImplementedError → NotImplementedError; kIndexError/kPythonIndexError → IndexError; kResourceExhausted → torch.OutOfMemoryError.
- Python: ruff, 2-space, 80 cols, py3.11 typing, google docstrings; implementation under `_internal`, one public path via `public_api.py`; `annotations.experimental/stable/deprecated`; knobs `TORCH_TPU_INTERNAL_*`; custom ops `torch.ops.tpu._foo` re-exported as `torch.tpu.ops.foo`.

## 4. Lint and CI you must pass

- GitHub lint: addlicense header; clang-format (Google, changed lines); ruff (pyproject.toml, line 80, indent 2, mccabe 18, max-args 5); buildifier 8.5.1; directory-layout manifest; actionlint. BUILD/bzl wrap at 80 (AGENTS.md).
- Bazel style tests: header guards, namespace rule, `NON_TORCH_TPU_OK=` marker; macros add `-Werror=unused-variable -Werror=reorder-ctor -Werror=thread-safety-precise`.
- BUILD macros: one src per `torch_tpu_cc_library/cc_test/py_test`; py_test deps include `//src/torch_tpu`; tags `requires-tpu-v5lite|v5p|v6e|v7x[:chips]` (never bare `requires-tpu`); `*errors_test*` targets need `nosan`; no `manual/notap/nopresubmit` tags, use string params with reasons; every BUILD ends with `torch_tpu_package_end()`.
- ~60 Google-internal `FOO_OK=<reason>` markers (CRASH_OK, ALLOW_CC_TARGETS, ABSLTEST_OK, ASSERT_RAISES_OK, CPP_THREAD_LOCAL_OK, GETENV_OK, REF_CAPTURE_OK, STATUS_CODE_OK, …) are enforced on the imported CL after approval (`ci/custom_lint_rules.METADATA`); a `pull ready` PR can bounce there. Enumerated in study_contributor_rules.md.
- Presubmit: bazel-diff picks impacted tests for ci_cpu / ci_tpu_v5 / ci_tpu_v7 / ci_rbe; "Presubmit on TPU v5 (newest deps)" runs tt_apis_test and error_coverage_test. Labels `ci:disable-bazel-diff`, `ci:local-torch`, etc. opt into more. Check placement with `ci/tools/list_ci_tests.py --test=//pkg:target`. zizmor and CLA run org-side; stale bot closes after 30+14 days.
- Commit messages: imperative subject ≤ ~70 chars, optional `[Ops][op]` prefix, prose body, `Fixes #N`; no trailers survive import.

## 5. Where the project is going (for picking work)

- 150–270 commits/week, all via Copybara; 66 authors in the Oct–Nov window. Active fronts: Kiln C++ kernel DSL, unbounded dynamic kernels (nonzero) on by default, opmath precision standardisation, upstream PyTorch device-generic tests, perflab metrics.
- Hot files: tests/ops_unit_test.py (91 changes/63 authors), errors_test.py, ops_test.py, error_coverage_data.yaml; riskiest C++: copy_from/tpu_to_cpu.cc, eager/tpu_aten_kernels.cc, events_queue.cc, op_dispatcher.cc, compiled_mode.cc.
- Accepted external change shape: narrow, test-backed op/validation fixes, one-op dynamism support, compile/view-decomposition fixes, pjrt buffer lifetime fixes, docs, CI hygiene; bigger features land as `experimental`.
- Ownership: adivinpatel04/yilingyuan/ghpvnist/agarwalt = ops; benawilson = eager runtime/PJRT; basioli-k/arfaian/LizLian = compile; cbasile-g = numerics; tudorpip/WillFroom = attention/Kiln; jparkerh = distributed; LeeHowes = docs; cota = triage/policy.

## 6. Pre-PR checklist (short form)

1. Issue exists and is linked; no other open PR of ours in the same reviewer group.
2. Branch rebased on origin/main (zizmor fires otherwise); one commit, imperative subject, no attribution.
3. Kernel follows §3; `.out` implemented once; helpers reused; no `at::` redispatch.
4. Every error macro covered in errors_test.py (`tpu=` and `gpu=`, `message_reviewed_by`) or marked infeasible; error_coverage_data.yaml regenerated if counts change.
5. Unit tests in ops_unit_test.py next to related ops; dtype parity with CPU/CUDA; 0-d/empty/non-contiguous.
6. BUILD: exact deps, 80 cols, correct `requires-tpu-*` tag, buildifier clean.
7. Ran the bazel targets on a TPU (ops_unit_test, errors_test with `--test_mode=tpu`, ops_test filter) and recorded commands + results in `## Test Plan`.
8. After approval: do not push again unless asked.
