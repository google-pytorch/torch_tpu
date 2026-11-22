# torch_tpu contributor rules and enforcement: reference

Source: google-pytorch/torch_tpu checkout at `/Users/joinal/pytorvhtpu/upstream/torch_tpu`
(detached at origin/main, 2026-11-06). All paths below are relative to that root;
`path:line` cites the line where the rule is written. "Absent" means the file does not
exist in the checkout.

Two enforcement layers exist. GitHub CI (lint.yml, presubmit.yml, bazel tests) runs on
the PR. A second layer, `ci/custom_lint_rules.METADATA`, is the Google-internal presubmit
that runs on the Copybara-imported CL after approval (CONTRIBUTING.md:113-125); its
`FOO_OK=<reason>` markers are what the inline justification comments in the tree are for.
Violations there surface only after GitHub review, so follow them up front.

---

## 1. Contributor process

### CONTRIBUTING.md
- Philosophy: "small, well-tested contributions that address a single clear issue" over
  sprawling PRs (CONTRIBUTING.md:16-18).
- Open an issue first for anything beyond trivial fixes; link the PR to it
  (CONTRIBUTING.md:26-30).
- Bug report must have: what happened vs expected + full traceback, MRE, PyTorch/torch_tpu
  version + TPU generation (CONTRIBUTING.md:36-41).
- Google CLA required; a CLA bot checks PRs (CONTRIBUTING.md:47-56). No CLA workflow
  lives in `.github/workflows/` (it is a GitHub App, not a repo workflow).
- Claiming an issue: check assignee/comments/linked PRs, comment before starting, a
  maintainer *may* assign; duplicate fixes: maintainers pick one (CONTRIBUTING.md:60-73).
- Community guidelines: Google OSS conduct (CONTRIBUTING.md:77-79).
- Review flow (CONTRIBUTING.md:85-149):
  - All submissions reviewed via GitHub PRs (85-88). Approved PRs are copied internally
    by Copybara; the PR is *closed*, not merged, and a separate commit lands on `main`
    (90-93).
  - Step 1: Google oncall triages, adds a **domain expert** reviewer (95-96).
  - Step 2: domain expert approves, adds a **quality champion** as reviewer AND assignee;
    champions are `cbasile-g`, `LeeHowes`, `vladbelous`, `zhanyong-wan` (97-110).
  - Step 3: quality champion reviews for style/quality, approves, applies `pull ready`
    label (111-112).
  - Step 4: `pull ready` triggers the bot to create the internal CL; external-author CLs
    need approvals from **two different Googlers**; if the expert approved an older
    revision they must re-approve (113-128).
  - Steps 5-6: presubmit failures fixed by the domain expert, by patching the CL or
    pushing to the PR branch (129-149).
  - Click "Re-request review" when ready for another round (153-154).
- After every PR update the quality champion must re-approve the Copybara import
  (`import/copybara` check -> Code-Review+2) and GitHub approvals do not carry over to new
  commits (CONTRIBUTING.md:165-183). Practical consequence: avoid pushing cosmetic commits
  after approval.
- Fork PRs: workflows need maintainer approval each time you push or someone reviews;
  unapproved runs expire after 30 days (CONTRIBUTING.md:185-218).
- Major changes/new public APIs: RFC discussion first; prefer extending PyTorch mechanisms;
  new APIs start internal or experimental, never stable (CONTRIBUTING.md:220-237).
- AI-generated code policy (CONTRIBUTING.md:239-262): author fully responsible (242-243);
  must understand it (245-247); must validate it (248-249); "unreviewed, automated LLM
  output ... will be closed" (250-253); optional `Assisted-by:` tag, AI must not be a
  co-author (254-256); maintainers "may limit the number of PRs a contributor can open at
  one time" (258-262).
- Expectations of contributors (CONTRIBUTING.md:280-296): "do one thing well. A concise
  100-line diff gets reviewed far faster than a 1,000-line rewrite" (280-281); all
  non-trivial PRs must link an issue (282-284); `Fixes #123` in description, review
  priority follows linked-issue priority (285-289); shepherd CI and reviews (290-291);
  do **not** assign or request reviewers yourself (292-296).
- Security: private vulnerability reporting form, not public issues (CONTRIBUTING.md:
  300-310); SECURITY.md says use http://g.co/vulnz (SECURITY.md:5-11).

### README.md
- Contributing section only points at CONTRIBUTING.md (README.md:214-219).
- Python support 3.11-3.15 incl. free-threaded; guarded by `LINT.IfChange(python_version_support)`
  that also names `pyproject.toml:requires_python`, `bazel/supported_python_versions.bzl`,
  `docs/developers/python_support_policy.md` (README.md:35-48). Change all four together.
- Build: `bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel`; `--config=no_rbe`
  must come *after* `wheel_common` (README.md:79-110).

### AGENTS.md / CLAUDE.md
- BUILD/bzl files: act as Bazel/Starlark expert, **wrap to 80 columns** (AGENTS.md:3-6).
- Requirements: never edit `requirements_3_*.txt`; add to `pyproject.toml` (`dependencies`
  or the right `[project.optional-dependencies]` group) then run
  `./requirements/lock_environments.sh` (AGENTS.md:8-16).
- Bazel repos: `bazel query //external:all`; only define a repo in WORKSPACE if missing
  (AGENTS.md:18-25).
- Docs: use the docs-review skill (AGENTS.md:27-29).
- PR body: build from `.github/pull_request_template.md`, keep `## Description` and
  `## Test Plan` headings, drop HTML comments; `gh pr create --body` does not apply the
  template (AGENTS.md:30-34).
- CLAUDE.md imports AGENTS.md and adds "remove all mannered prose" (CLAUDE.md:1-7).

### .github/pull_request_template.md
- Two required sections: `## Description` (what and why) and `## Test Plan` (tests run,
  how to reproduce, environment, e.g. `bazel test ...`) (pull_request_template.md:1-8).

### docs-review skill (`.agents/skills/docs-review/SKILL.md`)
- Google Developer Documentation Style Guide (16-18); second person, active voice, present
  tense, American English, conditions before instructions (34-42); no "simply/easy/just"
  (28-29); sentence-case headings, imperative not gerund (69-71); code in backticks, UI in
  bold, descriptive link text (49-56); word list: JAX not Jax, TorchTPU not torchtpu,
  Google Cloud not GCP, Compute Engine not GCE, Cloud Storage not GCS, Hugging Face, do not
  mention PyTorch/XLA (81-90); notebooks are Marimo `.py` under docs/notebooks with `mo.md`
  cells, no `mo.code()` (62-65).

### docs/developers/*
- README.md lists api_design_guide, api_policy, list_ci_tests_guide (docs/developers/README.md:9-16).
- **api_design_guide.md**
  - APIs are global (process) or local (thread region) (17-18).
  - Prefer reusing existing PyTorch API; prefer `torch.accelerator` over `torch.tpu` (20-25).
  - Global knobs: Python API, properties over getters/setters, last write wins (28-33);
    optional env var; TorchTPU env vars start `TORCH_TPU_INTERNAL_` (internal) or
    `TORCH_TPU_` (user-facing) (43-48); Python API beats env var (49-50); C++ flags only for
    Google-internal knobs, named `torch_tpu_internal_*` (51-55); **never** Python
    `flags.DEFINE_*` in library code (56-58).
  - Local knobs: Python context managers; local overrides global (59-68).
  - Read flags/env once and memoize: `ReadFlagOnce()`, `GetEnvOnce()` in C++ (71-77).
  - Every public dotted path (no `_` component) is an API; adding/removing/changing needs
    API-owner approval (84-90). Each object has **at most one public path** (93-96).
    Implement in `_internal` private modules and re-export only public names (105-110).
    **Instance attributes of public classes must start with `_`**; use properties;
    `tt_apis_test` enforces; dataclass fields exempt (111-116).
  - Module layout `torch.tpu.{compiler,distributed,exceptions,ops,optimizers,pallas(.verifier),profiler}`
    (128-140); new public module needs API-owner approval (142); no public APIs in
    `torch_tpu` or `torch.backends.tpu` (144-148); all custom ops in `torch.tpu.ops`;
    C++ ops register as `torch.ops.tpu._foo` and are re-exported `foo = torch.ops.tpu._foo`
    (155-174).
- **api_policy.md**
  - "try to avoid new APIs; when we really need one, limit access" (3-4).
  - Stages internal -> experimental -> stable -> deprecated (35-53); deprecate for at least
    2 minor releases before removal (49-51).
  - Naming: internal flags `torch_tpu_internal_`, env `TORCH_TPU_INTERNAL_`, Python `_`
    prefix (62-66); experimental/stable env `TORCH_TPU_`, warn once on use (67-74); **all**
    flags must be internal (94-95).
  - Marking: env vars via `kEnvVarToStage` in `csrc/common/env_vars.h` (88-93); Python
    callables via decorators in `torch_tpu._internal.utils.annotations` (96-100);
    attributes via `__tt_api_stages__` (101-106); pybind types via wrapper functions
    (107-111); custom ops via `ImplInternal`/`ImplExperimental`/`ImplStable`/`ImplDeprecated`
    (114-119).
  - `docs/public_apis.md` is generated and "a test fails when it's out of date" (127-129).
- **python_support_policy.md**: matches PyTorch's matrix for the minimum supported PyTorch
  (3-6); currently 3.11-3.15 incl. 3.14t/3.15t, in a `LINT.IfChange` block (25-34).
- **list_ci_tests_guide.md**: `ci/tools/list_ci_tests.py --test=//pkg:target` shows which
  CI jobs run a target (75-100); `--list_jobs` reads `.bazelrc` only (58-73); tag-filter
  algebra: negative filter excludes, any positive filter requires one of them (159-175).

### PR bots and labels (`.github/workflows`)
- **"Review Process" comment**: `on_pr_ready.yaml` (`On PR Ready`, job
  `Post Review Process Comment`) fires on `pull_request_target` opened/ready_for_review
  (17-25), skips drafts and `copybara-service[bot]` (33-36), posts: "Thanks for contributing
  to TorchTPU, @author! :fire: To ensure smooth code reviews, we kindly ask authors and
  reviewers to follow the process documented in .../CONTRIBUTING.md#code-reviews. Happy
  hacking!" (49-55). Posted every time the PR becomes ready (47-48).
- **Turn tracker**: `turn-tracker.yml` (15-68) runs `tools/turn_tracker.py` on `main`;
  review events relayed by `turn-tracker-pr-review.yml` (15-35). Labels:
  `stat:awaiting-reviewers` (new/reopened/ready PR, reviewer change, or author comments),
  `stat:awaiting-requester` (a reviewer/assignee comments or reviews),
  `stat:awaiting-assignee` (issues) (tools/turn_tracker.py:50-61, 88-90). Drafts, closed
  items, bots, third parties, self-assigned items ignored (63-71).
- **Path labeler**: `label-pr.yml` (15-41) + `.github/labeler.yml:18-20`: changes under
  `tests/torch/**` get `ci:torch-tests` (which in turn enables upstream_tests.yaml).
- **Stale**: `daily_project_management.yml`: `stale` after 30 days inactivity, closed 14
  days later (57-58), only assigned items (69), any comment removes the label
  (`remove_stale_label.yml:19-59`); messages at 92-136.
- **On close**: `on_pr_close.yaml` cancels and deletes in-flight runs (15-78).
- CODEOWNERS: **absent** (`find -iname CODEOWNERS*` returns nothing). Domain experts are
  listed as comments at the top of BUILD files instead (tests/BUILD:15-22,
  tests/api/BUILD:15-17, tests/dynamism/BUILD:15-17). `tests/api/OWNERS` is referenced by
  tests/api/tt_apis_test.py:44 but does not exist in the OSS checkout.
- `.github/ISSUE_TEMPLATE/`: **absent**.
- Dependabot: GitHub Actions only, monthly, grouped, 7-day cooldown; pip updates disabled
  (.github/dependabot.yml:19-57).

---

## 2. Formatting and lint configuration

### Config files present / absent
- `.clang-format`: `BasedOnStyle: Google`, nothing else (.clang-format:22-25; comment 15-17
  says it enforces the Google C++ Style Guide).
- `.clangd`: skips indexing `*_mlir_embed.cc` (.clangd:4-9).
- `pyproject.toml` `[tool.ruff]` (128-147): `preview = true`, `line-length = 80`,
  `indent-width = 2`, `target-version = "py311"`, excludes `perflab/**`, `docs`, `examples`,
  `**/docs/notebooks/**`.
  - `[tool.ruff.lint]` select (181-212): `B9`, `C`, `F`, `W`, `YTT`, `ASYNC`, `E101`,
    `E112`, `E113`, `E115`, `E117`, `E225`, `E227`, `E228`, `PLE`, `UP006`, `UP007`,
    `UP035`, `UP042`, `UP045`; ignore (150-178): `C408 C417 C420 C901 B901 B903 B904 B905
    D100 D104 D105 D107 PLR6301 PLR2004 PLC2701`.
  - pydocstyle `convention = "google"` (214-215); mccabe `max-complexity = 18` (217-218);
    pylint `max-args = 5`, `max-statements = 50`, `max-returns = 6` (221-224).
  - per-file-ignores: `**/tests/**` and `**/*_test.py` drop `D`, `PLR0913`, `PLR2004`,
    `PLR6301`; `**/examples/**` drop `D`, `PLR2004` (226-230).
  - Only `[tool.ruff*]` and `[tool.uv*]` sections exist; no `[tool.pylint]`, `pyink`,
    `black`, `isort`, `mypy`, `pytype` sections.
- `.pylintrc`, `.clang-tidy`, `.editorconfig`, `.pre-commit-config.yaml`, buildifier config
  file: **absent**. Buildifier settings live in the script (below).
- Column limit for BUILD/bzl: 80 (AGENTS.md:6).

### What `nox -s lint` runs (lint.yml:104-112, noxfile.py:90-97)
- `ci/tools/clang_format.sh lint`: `git diff -U0 BASE_SHA -- "*.cc" "*.h" | clang-format-diff.py -p1 -style=file`
  (clang_format.sh:41); only changed lines are checked (5-9); escape hatch
  `// clang-format off` / `// clang-format on` (62-66); fix with `nox -s format` (58-60).
- `ci/tools/ruff_lint.sh lint`: `ruff check` on added/modified `.py` files vs BASE_SHA
  (ruff_lint.sh:22,30); fix: `nox -s format` (36); suppression hint printed by the script:
  trailing `# ruff: ignore[rule]` (38-39; used e.g. tests/errors_test.py:28).
- `ci/tools/buildifier_lint.sh lint`: buildifier **8.5.1** (18),
  `-mode=check -lint=warn -warnings=-out-of-order-load` on changed `BUILD`, `BUILD.bazel`,
  `*.bzl`, `*.star` (46-64).
- `ci/tools/check_directory_layout.py`: root dirs must be exactly `bazel build_files ci
  csrc docker docs examples perflab requirements shims src tests tools`; `src/torch_tpu`
  must contain only `_internal` (check_directory_layout.py:26-45). Adding a new top-level
  directory fails lint.
- `nox -s actionlint` (lint.yml:109-112): actionlint 1.7.12 with shellcheck on all of
  `.github/` (actionlint.sh:19,36-43); self-hosted runner labels must be declared in
  `.github/actionlint.yaml:21-27`; suppress shellcheck with `# shellcheck disable=SCxxxx`
  (actionlint.sh:49-53).
- Also in lint.yml: license headers `go run github.com/google/addlicense -check -s=only .`
  (lint.yml:52-55) -> every new file needs the Apache-2.0 SPDX-style header; commit
  messages must not contain `[NO DESCRIPTION SET]` (lint.yml:56-78); `nox -s test_ci_tools`
  runs unit tests of `ci/tools/*` (lint.yml:114-127, noxfile.py:141-154).
- Local equivalents: `nox -s lint`, `nox -s format` (noxfile.py:90-106).

### C++ compile flags enforced by the macros (build_files/build_defs.bzl:46-63)
- `-std=c++20`, `-fexceptions`, `-Werror=unused-variable`, `-Werror=reorder-ctor`,
  `-Werror=thread-safety-precise`, `-DUSE_KINETO`, warnings enabled for TorchTPU headers.

### Bazel-run style tests (tests/BUILD:3095-3137, 3205-3224)
- `cpp_namespace_test` and `header_guard_lint_test` lint every `.cc/.cpp/.h` under
  `//csrc` and `//tests` (data `all_cpp_files_recursive`), excluding `*test_data*` files
  (tests/cpp_style_lib.py:61-64,171-187) and venv/hidden dirs (tests/repo_paths.py:128-138).
- **Header guards** (cpp_style_lib.py:83-94, 215-300): guard = `TORCH_TPU_` + path relative
  to repo root (leading `torch_tpu/` stripped), non-alphanumerics -> `_`, upper-cased, plus
  trailing `_`, e.g. `csrc/common/env_vars.h` -> `TORCH_TPU_CSRC_COMMON_ENV_VARS_H_`
  (cpp_style_test.py:265-281). Requires `#ifndef GUARD`, `#define GUARD` as the next
  non-blank line, and a last line `#endif  // GUARD` (two spaces) (cpp_style_lib.py:
  232-298).
- **Namespaces** (cpp_style_lib.py:303-431):
  - Every top-level namespace must be `torch_tpu` (anonymous top-level and `mlir::torch_tpu`
    are errors) (383-394, 410-414).
  - Under `torch_tpu`, only `error`, `internal`, `kiln` are approved nested names; only the
    innermost may be anonymous (303, 325-343).
  - `torch_tpu` may only be the outermost segment (327-331).
  - Escape hatch: trailing `// NON_TORCH_TPU_OK=<reason>` on the line that opens the
    namespace; **reason must be non-empty** (305-317, 416-423). It exempts the root check and
    the "torch_tpu must be outermost" check, but **not** an unapproved nested name under
    torch_tpu (cpp_style_test.py:250-261).
  - Regexes expect `namespace x {` and `}  // namespace x` closers (32-33).
- `markdown_tables_test`: published `.md` files must not use g3doc multi-line table rows
  (lines starting and ending with `:`); wrap tables in `<!-- mdformat off(reason: ...) -->`
  / `<!-- mdformat on -->` (tests/markdown_tables_test.py:15-21, 40-51, 107-125).
- `cpp_warnings_test`: C++ `TT_WARN`/`TT_WARN_ONCE` categories and their `//TorchTpu/...//`
  prefixes are public API (tests/cpp_warnings_test.py:15, 27-42).
- `api_registry_test`: `dir(torch.tpu)` has no `_` names; public symbols are re-exports of
  `_internal` objects with stage metadata; C++ ops exist as `torch.ops.tpu._foo` only
  (tests/api_registry_test.py:52-66, 159-223; BUILD tests/BUILD:210-230).
- `tests/api:tt_apis_test` (OSS only, tag `requires-newest-deps`): fails if `tt_apis.yaml`
  or `docs/public_apis.md` is out of date, if a public API lives outside `torch.tpu`, has
  more than one public path, lacks a stage, or has "experimental" in its name
  (tests/api/tt_apis_test.py:15-30, 58-60; tests/api/BUILD:90-122). Update with
  `bazel run //tests/api:scan_tt_apis` on a TPU VM after
  `cp requirements/requirements_3_14_latest.txt requirements/requirements_3_14.txt`, or
  copy the files from the failed run's `bazel-testlogs-*` artifact
  (tests/api/scan_tt_apis.py:1917-1928). Pinned newest torch `2.14.0`
  (tt_apis_test.py:56). `exposed_internal_apis_test` does the same for internal APIs and
  env vars mentioned in docs/examples (tests/api/exposed_internal_apis_test.py:15-21).

### Google-internal presubmit rules (`ci/custom_lint_rules.METADATA`)
Format: `prohibited_regexp` lines fail unless they match `suppression_regexp`
(typically `FOO_OK=<reason>` on the same line); `disable_tags` are `TAG=<reason>` in the
CL description; most rules are `check_delta_only` (METADATA:15-32). Full list of marker
comments the checks recognize:

Python linked into the library (`torch_tpu/*.py`, excluding tests/docs/examples/ci/tools; 61-82):
- `PY_ENVIRON_OK=<reason>`: allows `os.environ/getenv/putenv/unsetenv`; otherwise use
  `torch_tpu._internal.env.get_*_env_once()` / `env_utils.set_str_env()` (84-111).
- `RAW_WARNINGS_OK=<reason>`: allows `warnings.warn*`; otherwise use
  `torch_tpu._internal.utils.tt_warnings.warn` (graph-breaks under torch.compile) (113-145).

Op file naming (`csrc/ops/`):
- New files must not end in `kernel.cc/.h` (use `kernels`); CL tag
  `ALLOW_SINGULAR_KERNEL_FILENAME` (148-168).
- Files in `csrc/ops/<op>/` must be `*_kernels.cc|h`; CL tag `ALLOW_ANY_OP_IMPL_FILENAME`
  (170-192).
- Every `*kernels.cc` must contain `TT_KERNEL` (a comment mentioning it suffices); CL tag
  `NO_TT_KERNEL_OK` (194-220).

All C++ (`*.cc|cpp|h`; 222-734):
- `TF_ASSERT_OK_AND_ASSIGN`/`ASSERT_OK_AND_ASSIGN` banned -> `TT_ASSERT_OK_AND_ASSIGN` (226-239).
- `C10_TOSTRING_OK`: allows `c10::toString`; use `torch_tpu::ToString` (241-256).
- `DEPRECATED_MUTEX_LOCK_OK`: allows `MutexLock lock(&m)`; use `lock(m)` (258-272).
- `IGNORE_IN_CACHE_KEY_LEGACY_OK` (CL tag): "Legacy" reason in `IgnoreInCacheKey` (274-291).
- `#include` before the copyright notice is an error (293-305).
- `THREAD_LOCAL_NO_DESTRUCTOR_OK=<reason>`: `thread_local absl::NoDestructor` (307-323).
- `CPP_THREAD_LOCAL_OK=<reason>`: any `thread_local`; C++ threads are not 1:1 with Python
  threads, use ThreadLocalDebugInfo for per-Python-thread state (325-342).
- `MAYBE_WRAP_DIM_OK=<reason>`: `at::maybe_wrap_dim`; use `SafeWrapDim()` (344-358).
- `ALLOW_LOG_MACROS=<reason>`: bare `LOG/VLOG/CHECK/DCHECK/...`; use `ABSL_*` (360-398).
- `ENDL_OK`: `<< std::endl;` (400-414).
- `UNINITIALIZED_TENSOR_OK`: `at::Tensor foo;`; use `MakeEmptyTensor` (416-437).
- Error API: `ASSIGN_OR_RETURN`, `RETURN_IF_ERROR`, `RET_CHECK*`, `TF_*`, `TORCH_CHECK*`,
  `absl::*Error`, `throw` banned; use `TT_ASSIGN_OR_RETURN`, `TT_RETURN_IF_ERROR`,
  `TT_RET_CHECK`, `TT_CHECK_THROW`, `TT_ERROR`, `TT_THROW_IF_ERROR`, `TT_ASSIGN_OR_THROW`;
  any `//` or `/*` comment on the line suppresses (439-466).
- `NON_TT_MACRO_OK=<reason>`: `#define` names must start `TT_` or `TORCH_TPU_` (468-490).
- `INT_VEC_OK`: `std::vector<int64_t>`; use `Dimensions`/`Strides`/`Indices` (492-511).
- `VEC_OK`: `.vec()`; use `CopyIntVector()` (513-531).
- `STATUS_CODE_OK`: `absl::StatusCode::`; use `error::kFoo` (533-553).
- `LEADING_COLONS_OK`: `::absl::`, `::std::`, `::mlir::` etc. (555-580).
- `USING_DECL_OK`: `using absl|std|llvm|mlir|stablehlo|xla::` (582-600).
- `MLIR_SIGNED_INT_OK=<reason>`: `isSigned*`; use `isSignless` (602-622).
- `DEFAULT_CAPTURE_OK`: `[=]` lambda captures (624-650).
- `AT_INFER_SIZE_OK`: `at::infer_size`; use `InferSize()` (652-670).
- `AT_TO_OP_MATH_TYPE_OK=<reason>`: `at::toOpMathType`; use `InferComputationDtype()` (672-693).
- `STD_FUNCTION_OK`: `std::function<`; use `absl::AnyInvocable` (695-713).
- `ABSL_STRING_VIEW_OK`: `absl::string_view`; use `std::string_view` (715-733).
- `AT_TO_ACCUMULATE_TYPE_OK=<reason>`: `at::toAccumulateType`; use `ToAccumulateType` (1671-1692).
- `EXPLICIT_PRECISION_OK=<reason>`: `Precision::DEFAULT|HIGH|HIGHEST` (1198-1223).

Non-test C++ only (`*.cc|cpp|h` not `*_test.*`; 736-1196):
- `TORCH_WARN_OK=<reason>`: `TORCH_WARN*`; use `TT_WARN`/`TT_WARN_ONCE` (741-761).
- `CHECK_NAME_OK=<reason>`: `absl::Status Check*()` functions; use `Validate*` (763-780).
- `HASH_OK=<reason>`: `std::hash`/`absl::Hash`; consider fingerprinting (782-806).
- `VOID_CAST_OK=<reason>`: `(void)` / `static_cast<void>` (808-832).
- `IGNORE_ERROR_OK=<reason>`: `.IgnoreError()` (834-852).
- `NONSTANDARD_FLAG_NAME_OK=<reason>`: `ABSL_FLAG` names must start `torch_tpu_internal_` (854-871).
- `STRING_LITERAL_ENV_VAR_NAME_OK=<reason>`: `setenv("...")`; env vars must be defined in
  `csrc/common/env_vars.h` (873-893).
- `SETENV_OK=<reason>`: `setenv`/`unsetenv`; use `SafeSetEnv<>()`/`SafeUnsetEnv<>()` (895-919).
- `GETENV_OK=<reason>`: `getenv(`; use `GetEnvOnce<>()` (921-940).
- `GET_FLAG_OK=<reason>` / `SET_FLAG_OK=<reason>`: `absl::GetFlag`/`absl::SetFlag`; use
  `GetFlagOnce<>()` / `SafeSetFlag()` (942-984).
- `CRASH_OK` (reason optional if self-evident): `FATAL`, `ABSL_CHECK*`, `ABSL_QCHECK*`,
  `TT_ASSIGN_OR_CRASH`; "for catching torch_tpu bugs instead of user errors" (986-1031).
- `RAW_OP_NAME_OK`: error strings beginning `"name(): "`; `TT_KERNEL` adds the op name (1033-1055).
- `M_IMPL_OK=<reason>`: `m.impl(`; use `ImplExperimental/ImplStable/ImplDeprecated/ImplInternal` (1057-1077).
- `STD_PAIR_OK=<reason>`: `std::pair`; use named structs (1079-1100).
- `PY_SUFFIX_OK=<reason>`: functions named `*Py(`; use `Py*` prefix (1102-1122).
- `TORCH_THREAD_LOCAL_STATE_GUARD_OK=<reason>`: `c10::DebugInfoGuard`/`at::ThreadLocalStateGuard` (1124-1147).
- `RAW_PYBIND_OK=<reason>`: `PYBIND11_MODULE`, `py::class_`, `py::module_`; use
  `TT_PYBIND11_MODULE` / `PyBindClass<T>` (1149-1169).
- `REF_CAPTURE_OK=<reason>`: `[&]` lambda captures (1171-1195).

BUILD / .bzl (1225-1422):
- `NOTEST_OSS_OK=<reason>` / `NOBUILD_OSS_OK=<reason>`: any `notest_oss =` / `nobuild_oss =`
  (1230-1269).
- `"//api` dependency banned (1271-1282).
- `nomsan|notsan|noasan|nosan` tags need a `# reason` comment on the same line (1284-1298).
- `NOTAP_OK=<reason>`: literal `"notap"` tag (1300-1318).
- `ALLOW_CC_TARGETS=<reason>`: raw `cc_library(`/`cc_test(` (same line or first line
  after); use `torch_tpu_cc_library`/`torch_tpu_cc_test` (1320-1339).
- `PY_LIBRARY_OK=<reason>` (first line after `py_library(`): raw `py_library`,
  `pytype_library`, `pytype_strict_library`; use `torch_tpu_py_library` (1341-1370).
- `PY_TEST_OK=<reason>`: raw `py_test`, `py_strict_test`, `py_platform_test`; use
  `torch_tpu_py_test` (1372-1393).
- Every BUILD file must end with `torch_tpu_package_end()`; CL tag
  `NO_TORCH_TPU_PACKAGE_END_OK` (1396-1422; macro doc build_defs.bzl:1713-1733).

Python tests and misc (1424-1669):
- Error tests (`*error*test*.py`): `tpu=`/`gpu=` messages must be triple-quoted strings
  (1427-1440), not wrapped in parentheses (1442-1454), no backslashes except raw strings
  (1456-1472).
- `ASSERT_RAISES_OK=<reason>`: `assertRaises*`/`assert_raises_message` in any
  `tests/*_test.py` that is not an `*errors_test.py`; error tests belong in
  `tests/errors_test.py` (or `tpu_errors_test.py`, `*_errors_test.py`) (1475-1500).
- `RELATIVE_IMPORT_OK=<reason>`: `from .` imports outside examples/prototypes (1503-1526).
- `PYTHON_FLAG_OK=<reason>`: `flags.DEFINE_*` outside tests/benchmarks/examples (1528-1555).
- `ABSLTEST_OK=<reason>` / `UNITTEST_OK=<reason>`: `absltest.TestCase`,
  `unittest.TestCase`, `parameterized.TestCase`; use `RepeatableTest` or
  `VaryingSeedInPostsubmitTest` from `tests/seed_test_utils.py` (1557-1589;
  seed_test_utils.py:36-47).
- `TORCH_ASSERT_CLOSE_OK=<reason>`: `torch.testing.assert_close` / `torch.allclose` in
  internal tests; use `_internal/utils/utils.py` `assert_close` (1591-1614).
- `PYTHON_LOGGING_OK=<reason>`: `import logging`; use `from absl import logging` (1616-1642).
- `INIT_DEVICE_OK=<reason>`: `_init_device` calls (legacy) (1644-1669).
- Copyright year in existing files must not change; CL tag `ALLOW_COPYRIGHT_YEAR_UPDATE`
  (1694-1721). New files use the year of first publication.
- `ALLOW_IRREVERSIBLE_TRANSFORM` (CL tag): `copybara:strip|insert` tags (34-55).

Markers seen in the tree but with no checker in the OSS checkout: `TOLERANCE_OVERRIDE_OK`
(tests/ops_test.py:604,761,956) and `EXCLUDE_DTYPES_OK=<reason>` (tests/ops_test.py:2793);
their rule is internal-only. Treat them as required when adding tolerance overrides to
`ACCURACY_OVERRIDES_VS_GPU` or `exclude_dtypes=` in ops_test.py.

Other repo conventions enforced by tooling: `# go/keep-sorted start` / `# go/keep-sorted end`
blocks must stay sorted (used in tests/native_functions_data.py:27, tests/ops_test.py:220,
tools/turn_tracker.py:93, build_defs_test.bzl:21-31, METADATA:70-82); no keep-sorted
checker exists in the OSS repo (it is an internal formatter), so sort by hand.

---

## 3. Test conventions

### Bazel macros (`build_files/build_defs.bzl`)
- `torch_tpu_cc_library`: at most one `.cc` in `srcs` and one `.h` in `hdrs` (296-301);
  `deps` must be a plain list, no `select()` (232-246); creates `<name>_build_test`
  (317-320).
- `torch_tpu_cc_test`: one `srcs` file (750-752); adds `--gtest_shuffle` and
  `--gtest_fail_if_no_test_linked` (757-765); `linkstatic = True` default (683,711-712);
  in OSS, tests tagged `requires-tpu*` are wrapped in an `sh_test` using
  `ci/tools/run_cc_tpu_test.sh` to load libtpu (809-878).
- `torch_tpu_py_library`: one `srcs` file unless `allow_multiple_srcs = True` (932-934).
- `torch_tpu_py_test`: one `srcs` file (1079-1081); adds
  `--test_randomize_ordering_seed=random` (1084-1086); **`deps` must include
  `"//src/torch_tpu"`** (1241-1243); do not set `TORCH_DEVICE_BACKEND_AUTOLOAD` in `env`,
  use `autoload=` (1123-1131); `exclusive` tag sets `TORCH_TPU_EXCLUSIVE_TEST=1` (1120-1121);
  `run_on_accelerators = ["requires-tpu-v5lite", "requires-tpu-v6e"]` generates
  `<name>_tpu-v5lite`, `<name>_tpu-v6e` targets; v6e/v7x targets get
  `nopresubmit = "Avoid long Forge queue."` automatically (1058-1073, 1284-1316).
- Shared tag rules (`_check_and_adjust_test_tags`, 363-636):
  - Generic `requires-tpu` tag is disallowed; name a generation: `requires-tpu-v5lite`,
    `requires-tpu-v5p`, `requires-tpu-v6e`, `requires-tpu-v7x`, optionally `:<chips>`
    (269-274, 73-78). `REQUIRES_V5_MULTI_DEVICE` = `requires-tpu-v5lite:4` in OSS (186-189).
  - Never use tags `nopresubmit`, `postsubmit`, `notap`, `manual`, `oss_ready_*`; use the
    string parameters `nopresubmit="<reason>"`, `notap="<reason>"`, `nolocal="<reason>"`,
    `nobuild="<reason>"`, `nopresubmit_oss="<reason>"`, `nonightly_oss="<reason>"`,
    booleans `notest_oss`/`nobuild_oss` (429-434, 443-448, 470-486, 492-510, 580-584,
    596-598).
  - Any test whose name contains `errors_test` must carry the `nosan` tag (356-361, 437-440).
  - `size = "enormous"` or `timeout = "eternal"` require `notap=` (508-510).
  - GPU (`requires-gpu-*`) and TPU tags are mutually exclusive (600-605).
  - OSS presubmit generation is auto-picked from `TPU_GENERATION_PREFERENCES =
    ["v5", "v7", "v6"]`, skipping `fails-on-tpu-<gen>` tags, adding `presubmit-<gen>`
    (67, 607-626); override with `oss_presubmit_tpu_generation = tpu_gen("v7", "reason")`
    (146-156). A TPU test failing on all generations must use `notest_oss` (625-626).
  - `shard_count` is clamped in OSS by chips per runner (v5: 8, v6: 8, v7: 4) and to 1 for
    `exclusive` (81-85, 107-144).
- `torch_tpu_package_end()` must be the last line of every BUILD file (1713-1733; enforced
  by METADATA:1396-1422).
- Domain experts are listed as a comment at the top of each BUILD (tests/BUILD:15-22).

### Op tests (`tests/ops_test.py`, `tests/op_testing.py`)
- ops_test: "For each op, we test that the result on TPU matches the golden result recorded
  on GPU" (ops_test.py:15-19); variable naming `golden_*` vs `tpu_*` (147-150).
- Test classes derive from `op_testing.TorchTpuTestBase` (ops_test.py:168;
  op_testing.py:6440) or `TorchTpuVsCpuTestBase` for unit tests (ops_unit_test.py:67;
  op_testing.py:8027). Base classes derive from `seed_test_utils.RepeatableTest`.
- Each test method calls `self.do_test_op(op_name, ...)` **at most once** (op_testing.py:
  7882-7891); args: `exclude_dtypes`, `exclude_inplace_dtypes`, `check_out_variant`,
  `check_grad`, `check_device`, `check_value`, `check_dtype`, `check_op_failures`,
  `check_inplace_op_failures`, `skip_output_indices`, `skip_if`, `max_samples_per_op_dtype`,
  `variant_test_name` (7823-7877). Do not pass `check_value=CheckValueMode.LOOSE` (it is the
  default and raises) (7902-7905).
- Ops come from PyTorch's OpInfo db (`common_methods_invocations`, `op_db`)
  (op_testing.py:50-60, 78-79; ops_unit_test.py:68).
- Modes via `--test_mode`: `torch_tpu_vs_gpu` (default), `gen_gpu_golden`, `perf`,
  `dump_dtype_exclusions` (op_testing.py:132-147). Other flags: `--dtypes`,
  `--max_samples_per_op_dtype`, `--compute_grad`, `--use_compiled`,
  `--ignore_accuracy_overrides`, `--check_dynamism`, `--dynamism_plan_filter`,
  `--check_symbolic_shapes` (149-200); `--test_categories` with `@category(...)` decorator
  (ops_test.py:38-64).
- Tolerances: `ACCURACY_OVERRIDES_VS_GPU` dict keyed op -> dtype -> {rtol, atol}, criterion
  `abs(tpu - gpu) <= atol + rtol * abs(gpu)`, kept in a keep-sorted block (ops_test.py:
  206-230); overrides must be > 0 (op_testing.py:111-129).
- GPU goldens: recorded on Blackwell SM100 by default, Hopper SM90 via
  `@op_testing.record_gpu_goldens_on(op_testing.GPU_SM90)` (op_testing.py:92-105,
  5387-5414; tests/BUILD:55-70). Golden `.gz` files are **not checked in**; `tests/goldens.pin`
  (bucket `torchtpu-goldens`, sha256 per file) is the source of truth and Bazel fetches
  them as `@torch_tpu_goldens` (tests/test_utils.py:30-32, 82-88, 106-111; tests/goldens.pin:
  1-10; bazel/goldens.bzl:15-21). `split_gpu_goldens` splits them per test case
  (tests/split_goldens.bzl:41-60; tests/split_gpu_goldens.py:15-20). Adding a new op test
  therefore needs new goldens generated with `ops_test_gen_gpu_golden*` targets
  (tests/BUILD:1203-1298) and a goldens.pin update by maintainers.
- ops_test target: `run_on_accelerators` v6e+v5lite, `shard_count = 50`, tags `noasan
  nomsan notsan ops-test` + `requires-net:external` in OSS, `--max_samples_per_op_dtype=5`
  in OSS (tests/BUILD:73, 88-91, 767-804). `ops-test` tagged targets are "heavy" and handled
  specially by `ci/tools/run_bazel_diff.py` (tests/BUILD:790).
- `ops_unit_test`: hand-written cases vs CPU, `shard_count = 50`, v6e+v5lite
  (tests/BUILD:1491-1525).
- `also_run_static` decorator asserts no unbounded-dynamic kernel lookups in
  `ops_test_static` (ops_test.py:67-109).

### Dynamism tests
- `tests/dynamism_test_utils.py` discovers bounded-dynamic candidates and marks inputs;
  modes `EAGER`, `COMPILE`, `COMPILE_GRAD` (15-22, 45-56). `--check_dynamism` runs every
  op test across marking plans (op_testing.py:184-197).
- `tests/dynamism/BUILD` targets follow the standard pattern: `torch_tpu_py_test`, tag
  `requires-tpu-v5lite`, deps incl. `//src/torch_tpu`, `//tests:seed_test_utils`
  (31-46). Eager dynamism needs `EagerMode.DEFER_AND_FUSE` (tests/dynamism_test.py:73-76).

### Error tests (`tests/errors_test.py`, `tests/error_testing.py`)
- Required flag `--test_mode` in {`tpu`, `gpu`, `cov`}; `cov` only collects coverage and
  does not verify messages (error_testing.py:43-57, 61-68).
- Use `with et.assert_raises_message(ExcType, tpu="""...""", gpu="""...""",
  message_reviewed_by=None):` (274-296). `gpu` defaults to `tpu` (309-310); messages match
  **exactly** (string) or `fullmatch` (compiled regex, discouraged) (139-187); subprocess
  variant `assert_subprocess_raises_message` (333).
- `ErrorTestBase` bans `assertRaisesRegex` (819-829); in gpu mode each test runs in a
  subprocess (880-946).
- TPU-only errors: `TpuOnlyErrorTestBase` requires `@et.why_tpu_only("reason")` on every
  method and forbids `gpu=` (949-998, 306-307); `TpuOnlyErrorTestBaseNoCheckingWhy` and
  `TpuOnlyDistributedErrorTestBase` relax the decorator (949-953, 1000-1007).
  `@gpu_crashes_or_raises("reason")` tests belong in errors_test.py (491-526, 986-990).
- Placement: errors that occur on both GPU and TPU -> `tests/errors_test.py`; TPU-specific
  -> `tests/tpu_errors_test.py`; distributed -> `*_errors_test.py` (METADATA:1492-1496).
- Message style: triple-quoted, no parentheses, no backslashes (METADATA:1427-1472).
- BUILD: `errors_test_tpu` (`--test_mode=tpu`, env `TORCH_SHOW_CPP_STACKTRACES=0`,
  `TORCH_TPU_ENABLE_DEBUG_CHECKS=1`, tags `nosan`, `requires-tpu-v5lite`, shard 4)
  (tests/BUILD:1874-1903); `errors_test_cov` (`--test_mode=cov`, `env = TEST_COV_ENV`,
  `notap`) (1953-1981); `errors_test_gpu` (`notest_oss = True  # NOTEST_OSS_OK=...`)
  (2025-2046). The failure note says `*errors_test` failures from PyTorch version drift are
  "not currently enforced for new changes" (error_testing.py:207-215, 243-248).

### Error coverage ledger (`tests/error_coverage_data.yaml`)
- `error_coverage_test` scans C++ for `TT_ERROR`, `TT_RET_CHECK`, `TT_CHECK_THROW`
  (error_coverage_lib.py:82-86), runs `errors_test_cov`, `tpu_errors_test_cov`,
  `distributed/collective_errors_test_cov` (45-51), and compares per-file
  `{remaining, total, infeasible}` to the yaml (error_coverage_data.yaml:15+;
  error_coverage_lib.py:522).
- Fails on: more uncovered macros than baseline (regression), fewer than baseline
  (outdated), or a macro marked infeasible that is actually covered
  (error_coverage_test.py:15-29; error_coverage_lib.py:680-760). A new error macro in a
  file therefore needs a new error test **or** `// ERROR_COV_INFEASIBLE=<why>` on the same
  line (reason required by regex `\bERROR_COV_INFEASIBLE\s*=.+`) (15-26, 228-229).
- Update command in OSS (`update_command`, 607-629):
  `cp requirements/requirements_3_14_latest.txt requirements/requirements_3_14.txt && bazel test --config=ci_tpu_v5_newest_deps --run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh" //tests:error_coverage_test --test_arg=--print_coverage_data --test_output=all`
  then paste the YAML between `BEGIN_YAML`/`END_YAML` below the license header (648-653).
- BUILD tags: `exclusive`, `requires-newest-deps`, `fails-on-tpu-v7`, `nofastbuild`,
  `nosan`, `requires-tpu-v5lite:8` (tests/BUILD:3163-3203). It runs in the
  "Presubmit on TPU v5 (newest deps)" job on every PR (presubmit.yml:269-305).

### Op registration consistency (`tests/native_functions_test.py`)
- Invariant: `ops_requiring_registration + REGISTRATION_OVERRIDES == registered_ops +
  UNREGISTERED_OPS` (disjoint unions) (native_functions_test.py:119-129, 174-240).
- `UNREGISTERED_OPS`: ops that need a PrivateUse1 kernel but have none; "temporary list
  that should be emptied" (native_functions_data.py:15-27). Registering a new aten op ->
  **remove it from `UNREGISTERED_OPS`** or the test fails (native_functions_test.py:
  214-220).
- `REGISTRATION_OVERRIDES`: ops registered although not required; each entry must be
  commented with why (native_functions_data.py:394-404); `REGISTRATION_OVERRIDES_REGEX`
  e.g. `_foreach_.*` (558-565). Both lists are keep-sorted.
- Needs `RegistrationDeclarations.h`, provided only internally; in OSS the test skips
  (tests/BUILD:2784-2802; native_functions_test.py:147-153). The invariant is enforced on
  the internal CL.

### Other test rules
- Test classes derive from `seed_test_utils.RepeatableTest` (or `MultiProcessRepeatableTest`),
  not raw `absltest.TestCase` (METADATA:1557-1589; seed_test_utils.py:36-47).
- Prefer `_internal/utils/utils.py` `assert_close` over `torch.testing.assert_close`
  (METADATA:1591-1614).
- Upstream PyTorch test modules run via `tests/torch/torch_test_runner.py` (requires
  `TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR`, `TORCH_TPU_INTERNAL_TORCH_TEST_MODULE`, torch >=
  2.14) (15-58); triggered on PRs only with label `ci:torch-tests`, auto-applied for
  `tests/torch/**` changes (.github/labeler.yml:18-20, upstream_tests.yaml:66).

---

## 4. CI

### Workflows that run on every PR to main (trigger `pull_request`)
| Workflow (file) | Check name(s) | What it does |
|---|---|---|
| `lint.yml` "Lint checks" | "Lint check", "Run code linters", "Test CI tools" | addlicense header check, `[NO DESCRIPTION SET]` commit check, `nox -s lint`, `nox -s actionlint`, `nox -s test_ci_tools` (lint.yml:40-127) |
| `presubmit.yml` "CI - Presubmit" | "Presubmit on CPU", "Presubmit on linux-x86-ct5lp-224-8tpu", "Presubmit on linux-x86-tpu7x-224-4tpu", "Presubmit on RBE", "Presubmit on TPU v5 (newest deps)", aggregator **"Presubmit"** | bazel-diff selects impacted tests per config `ci_cpu_presubmit`, `ci_tpu_v5_presubmit`, `ci_tpu_v7_presubmit` (v6 disabled, 84-87); RBE job runs `ci_rbe_presubmit` (CPU + single-chip v5e) with self-hosted fallback (151-267); newest-deps job runs `bazel test --config=ci_tpu_v5_newest_deps //...` after `cp requirements_3_14_latest.txt requirements_3_14.txt` (269-305). The aggregator "Presubmit" is "checked by Copybara" (103-149). |
| `build_wheel.yml` | "Build and Test Wheels" aggregator (382-383) | builds wheels and runs wheel smoke tests when bazel-diff says wheel inputs changed; forced by labels `ci:force-wheel-build`/`ci:disable-bazel-diff` (42-43, 81-88) |
| `on_pr_ready.yaml`, `turn-tracker*.yml`, `label-pr.yml`, `on_pr_close.yaml` | bots | see section 1 |

- Label `ci:disable-bazel-diff` forces all tests instead of impacted ones
  (presubmit.yml:98, 227, 264).
- `pull_request` + `labeled` triggers gated on labels (job skipped otherwise):
  `ci:postsubmit` (postsubmit.yml:56), `ci:tsan` (tsan.yml:54), `ci:torch-tests`
  (upstream_tests.yaml:66), `ci:torch-examples` (upstream_examples.yaml:56),
  `ci:local-torch` (local_torch.yml:63,129), `ci:tpu-nightly` (nightly_tpu.yml:57),
  `ci:unpinned-deps` (nightly_unpinned_deps.yml:54), `ci:crcr-nightly`
  (out-of-tree-ci.yml:50), `ci:run_all_versions`, `tpu-nightly` (build_wheel.yml:86-87).
- Schedules/postsubmits: postsubmit every 4h and on push (postsubmit.yml:18-34);
  nightly_tpu 22:15 LA (nightly_tpu.yml:18-22); tsan daily 08:00 UTC (tsan.yml:17-18);
  nightly_unpinned_deps 09:00 UTC; upstream tests/examples daily.
- Fork PRs: workflows wait for maintainer approval (CONTRIBUTING.md:185-218); turn tracker
  and comment bots use `pull_request_target` so they still run (on_pr_ready.yaml:17-25).

### Bazel CI configs (`.bazelrc`)
- `ci` base: `--build_tests_only`, remote cache, resultstore, `--keep_going`,
  `--test_output=errors` (316-321).
- `ci_cpu_presubmit`: `--test_tag_filters=-nobuild,-notest,-requires-gpu,-requires-tpu,-nopresubmit,-cuda-build-test,-requires-newest-deps` (351-352).
- `ci_tpu_base`: `--build_tag_filters=-nobuild,requires-tpu-v5lite,requires-tpu-v5lite:4,requires-tpu-v5lite:8,requires-tpu-v5p,requires-tpu-v5p:4,requires-tpu-v6e,requires-tpu-v7x,requires-tpu-v7x:4` (370).
- `ci_tpu_v5_presubmit`: multi-chip and `no-remote-exec` tests only:
  `requires-tpu-v5lite:4,requires-tpu-v5lite:8,requires-tpu-v5p:4,no-remote-exec,-notest,-nobuild,-nopresubmit,-fails-on-tpu-v5,-presubmit-v6,-presubmit-v7,-requires-newest-deps` (412-415).
- `ci_rbe_presubmit`: single-chip v5e + CPU on RBE: `-nobuild,-notest,-nopresubmit,-fails-on-tpu-v5,-requires-gpu,-cuda-build-test,-presubmit-v6,-presubmit-v7,-requires-tpu-v5lite:4,-requires-tpu-v5lite:8,-requires-tpu-v5p:4,-no-remote-exec,-requires-newest-deps` (524-533).
- `ci_tpu_v5_rbe_fallback`: `presubmit-v5,...` (420-421).
- `ci_tpu_v7_presubmit`: `-notest,-nobuild,-nopresubmit,-fails-on-tpu-v7,presubmit-v7,-requires-newest-deps` (449-450); `ci_tpu_v6_presubmit` similar with `presubmit-v6` (438-439) but has no runner (presubmit.yml:84-87).
- `ci_tpu_v5_newest_deps`: `requires-newest-deps,-notest,-nobuild,-fails-on-tpu-v5` (426-427); v7 counterpart (456-458).
- Nightly configs add `-nonightly` and `--test_timeout=-1,-1,-1,4800` (355-356, 387-390, 429-431, 452-454).
- `wheel_test_*`: exclude `wheel_test_excluded` (607-627).
- Use `ci/tools/list_ci_tests.py --test=//pkg:target` to confirm where a new test runs
  (docs/developers/list_ci_tests_guide.md:75-100).

### Other CI facts
- zizmor: no config file in repo; workflows carry `# zizmor: ignore[dangerous-triggers]`
  annotations (on_pr_ready.yaml:24, turn-tracker.yml:24, remove_stale_label.yml:25) and a
  "GitHub Actions Scan" run is referenced (ci/tools/delete_cancelled_pr_runs.py:181;
  build_wheel.yml:45). It runs as an org-level scan, not from this repo.
- CLA: GitHub App bot (CONTRIBUTING.md:54-56); no workflow file.
- PR size/scope policy is prose only (CONTRIBUTING.md:16-18, 280-281, 258-262); no
  automated size check exists in `.github/`.
- Required-check names are not configured in the repo (branch protection is server-side);
  the names visible in workflows are "Presubmit" (presubmit.yml:116), "Lint check",
  "Run code linters", "Test CI tools" (lint.yml:42,81,115), "Build and Test Wheels"
  (build_wheel.yml:383), plus the external `import/copybara` check (CONTRIBUTING.md:170-172).

---

## Checklist before opening a PR

1. Issue exists and is linked (`Fixes #N`); PR does one thing, ideally about 100 lines (CONTRIBUTING.md:26-30, 280-289).
2. CLA signed; do not request reviewers or assignees yourself (CONTRIBUTING.md:45-56, 292-296).
3. PR body uses `## Description` and `## Test Plan` from the template, with the exact `bazel test` commands run (pull_request_template.md:1-8, AGENTS.md:30-34).
4. Commit message has a real description (no `[NO DESCRIPTION SET]`), optional `Assisted-by:` tag, no AI co-author (lint.yml:74-77, CONTRIBUTING.md:254-256).
5. Every new file has the Apache-2.0 header with the current year; never change the year in existing files (lint.yml:52-55, METADATA:1694-1721).
6. `nox -s lint` and `nox -s actionlint` pass locally: clang-format Google style, ruff (80 cols, 2-space indent, google docstrings), buildifier 8.5.1, directory layout (noxfile.py:90-97, pyproject.toml:145-147).
7. C++: namespaces rooted at `torch_tpu` (nested only `error`/`internal`/`kiln`), header guard `TORCH_TPU_<PATH>_H_` with `#endif  // GUARD`, macros `TT_*`, no `TORCH_CHECK`/`throw`/`absl::*Error`/`CHECK`, use `TT_*` error macros and `ABSL_*` logging (cpp_style_lib.py:83-94,303; METADATA:439-490).
8. C++: no `getenv`/`setenv`/`absl::GetFlag`/`thread_local`/`std::function`/`absl::string_view`/`std::vector<int64_t>`/`.vec()`/`[=]`/`[&]`/`TORCH_WARN*`/`m.impl` without the matching `FOO_OK=<reason>` marker; crashes need `// CRASH_OK` (METADATA:325-342, 492-531, 624-650, 741-761, 854-984, 986-1031, 1057-1077, 1171-1195).
9. Python library code: no `os.environ`, `warnings.warn`, `import logging`, `flags.DEFINE_*`, relative imports; new public API only under `torch.tpu.*`, one public path, `_`-prefixed instance attributes, stage decorator applied (METADATA:84-145, 1503-1555, 1616-1642; api_design_guide.md:84-116).
10. BUILD: only `torch_tpu_cc_library/cc_test/py_library/py_test`, one src per target, `deps` includes `//src/torch_tpu`, generation-specific `requires-tpu-*` tag, reasons via `nopresubmit=`/`notap=`/`nolocal=` params not tags, `no*san` tags commented, file ends with `torch_tpu_package_end()`, 80 columns (build_defs.bzl:269-274, 296-301, 429-510, 1241-1243, 1713-1733; AGENTS.md:6).
11. Tests derive from `RepeatableTest`/`TorchTpuTestBase`; no `assertRaises*` outside `*errors_test.py`; error tests use `et.assert_raises_message(..., tpu="""...""")`, `nosan` tag, `@why_tpu_only` for TPU-only cases (METADATA:1475-1500, 1557-1589; error_testing.py:274-296, 977-998; build_defs.bzl:437-440).
12. New `TT_ERROR`/`TT_RET_CHECK`/`TT_CHECK_THROW` macros are covered by an error test or carry `// ERROR_COV_INFEASIBLE=<why>`; `tests/error_coverage_data.yaml` regenerated with the documented command (error_coverage_lib.py:15-26, 607-629).
13. New aten kernel: removed from `UNREGISTERED_OPS`, op test added with `do_test_op` once per method, tolerance overrides marked `# TOLERANCE_OVERRIDE_OK`, goldens regenerated/pinned with maintainers (native_functions_data.py:15-27; op_testing.py:7882-7891; tests/goldens.pin).
14. Public API or docs changed: regenerate `tests/api/tt_apis.yaml` and `docs/public_apis.md` with `bazel run //tests/api:scan_tt_apis` on newest deps; docs follow the docs-review word list (TorchTPU, JAX, Google Cloud) (scan_tt_apis.py:1917-1928; SKILL.md:81-90).
15. Confirm where the new test runs with `ci/tools/list_ci_tests.py --test=//pkg:target`; watch CI, re-request review after each round, and expect the quality champion to re-approve the Copybara import after every push (list_ci_tests_guide.md:75-100; CONTRIBUTING.md:153-154, 165-183).
