# torch-tpu advice for AI Agents

This document guides AI agents working in the TorchTPU repository. Follow these
rules to keep code quality high, pass CI, and minimize reviewer friction.

For BUILD or bzl files:

*   Act as an expert in Bazel, Starlark, rules_python, bazel_skylib, rules_cc
*   Wrap code to 80 columns

When updating Python requirements:

*   Do not modify the locked, resolved `requirements_3_*.txt` files directly.
*   Instead, add the dependency to the appropriate list in `pyproject.toml`.
*   Place core requirements in the main `dependencies` block, and use-case
    specific requirements in the correct `[project.optional-dependencies]` group
    (e.g., `dev`, `test`, etc).
*   Run `./requirements/lock_environments.sh` to update all of the locked
    requirements files automatically.

Instructions related to Bazel repositories:

*   Run `bazel query //external:all` to list all external repositories.
*   Run `bazel query //external:<name>` to list a single one.
*   Add `--output=build` to show the url (and thus version) of the repository
    being used.
*   Only define a repository in WORKSPACE if it doesn't exist, or errors
    indicate a newer version must be used.

When writing documentation:

*   Use the tt-write-docs skill when writing, editing, or reviewing
    documentation.
*   Wrap Markdown at 80 characters, use ATX headers (`## Header`), and use
    asterisks for unordered lists (`* Item`), for consistent rendering and
    diffs.

When creating a pull request:

*   Use the tt-create-change skill when committing a change, or opening,
    describing, or updating a pull request.

## References

*   [CONTRIBUTING.md](CONTRIBUTING.md): how contributions and code reviews work,
    and the policy on AI-generated code.
*   [Developer documentation](docs/developers/README.md): API design guidelines
    and developer tooling.

## Core Philosophy: High-Quality ML Framework

*   **Performance + Clarity**: We are building a high-performance ML framework.
    Optimization is critical, but logic MUST remain clear and maintainable. Do
    not sacrifice readability for micro-optimizations unless backed by
    benchmarks and documented clearly.
*   **Rigorous Testing**: It is vital that **all ops are properly tested**.
    Every new feature or optimization must include comprehensive tests (unit
    tests and integration tests) to ensure correctness across edge cases. Use
    test-driven development (TDD) for new features or behavior changes:
    1.  Write a test that fails because the behavior doesn't exist yet.
    2.  Write the minimal code to make the test pass.
    3.  Refactor and optimize the code while keeping the test passing.

## Code Style & Conventions

Changes are checked against the rules in `ci/custom_lint_rules.METADATA`
(license headers, macro naming, kernel registration, and C++ best practices),
and CI runs `nox -s lint` (clang-format, ruff, buildifier, and the directory
layout check). In addition, follow these conventions:

*   **Style Guides**: Ensure that the code conforms to the
    [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
    and the
    [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
*   **API Design**: When adding or modifying TorchTPU APIs, follow
    [the API design guide](docs/developers/api_design_guide.md) and
    [the API policy](docs/developers/api_policy.md).
*   **Small PRs**: Per
    [Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html),
    prefer small and focused pull requests that each solve one problem. When
    possible, split a large change into a series of pull requests.
*   **Format Before Pushing**: Run `nox -s format` before every push, and make
    sure `nox -s lint` passes.
*   **Log Macros**: Use log macros that start with `ABSL_` (e.g. `ABSL_LOG`
    instead of `LOG`, `ABSL_CHECK` instead of `CHECK`, etc).
*   **Const**: Make a variable or function parameter `const` unless there's a
    need to mutate it or move from it. Exception: `MlirOp` must not be `const`,
    as it acts as a reference.
*   **Boolean Parameters**: Per
    [Tip of the Week #94](https://abseil.io/tips/94), use enum parameters as
    opposed to `bool` to improve call-site readability.
*   **Searchable Identifiers**: Spell out every identifier in full in code,
    comments, regexes, and docs, so that searching for it finds every use. Don't
    synthesize identifiers (e.g. by string concatenation, token pasting, or
    `getattr(obj, f"{prefix}_bar")`) or split them across optional regex groups.
    For example, write the regex `FooBar|Bar` instead of `(Foo)?Bar`, so that it
    shows up in a search for `FooBar`.
*   **Fine-grained Build Rules**: To avoid build bloat and circular
    dependencies, each python build rule should have only one `.py` source file,
    and each C++ build rule should have at most one `srcs` file and at most one
    `hdrs` file.
*   **No `sys.path` Manipulation**: Avoid manipulating `sys.path` or adding
    `try ... except ImportError` fallback shims in Python scripts and tests.

## Behavior & Execution Workflow

### Search Strategy

*   **Indexed, Scoped Search**: Use `rg` (ripgrep) or `git grep` to locate
    files, types, or methods. Avoid unscoped `find` or recursive `grep` over the
    whole tree, which also walk the large `bazel-*` output directories.

### Efficient Communication

*   **No "Chain-of-Thought" in Comments**: Do NOT leave verbose "stream of
    consciousness" comments in the code (e.g., "I wonder if I should
    broadcast...").
    *   *Rule*: Comments should explain *why* code exists, not the history of
        how you got there.
*   **Concise Summaries**: When updating task lists or artifacts, be brief.
*   **Concise Writing for Humans**: PR descriptions, issue reports, and review
    comments and replies are skimmed by busy readers:
    *   **Lead with the point**: The first sentence carries the key message.
    *   **High level only**: State what, why, and impact. Omit implementation
        walkthroughs, lists of files or functions touched, debugging history,
        anything evident from the diff, and details that are only marginally
        useful. Link to logs instead of inlining them.
    *   **No filler**: No preamble, praise, hedging, restated summaries, or
        recaps.
    *   **Targets**: PR description body: 1–3 sentences or a few one-line
        bullets. Review comment: one issue, 1–3 sentences, requested change
        first. Reply: one line (e.g. "Done."), or 1–2 sentences if disagreeing.
        Issue report: symptom as title, minimal repro, expected vs. actual,
        suspected cause if known.
    *   **Self-review before publishing**: Reread the draft as a casual reader,
        cut whatever breaks these rules, then publish.

### Build & Verify

*   **Always Build**: Run `bazel build //path/to:target` before claiming
    completion.
*   **Configs**: Add `--config=cuda` to build against CUDA-enabled PyTorch. To
    reproduce a CI run, use its config from `.bazelrc` (e.g.
    `bazel test --config=ci_cpu //tests:target`).

### Autonomy and Long-Running Tasks

*   **Autonomy**: Never prompt the user to check if you should continue running
    a long job (e.g., build, test, migration). You have full autonomy to
    proceed.
*   **Long-Running Tasks**: Run operations expected to take a long time in the
    background if your environment supports it, and check back on them instead
    of blocking on them.
*   **User Intervention**: Only stop execution and ask the user if they
    explicitly ask to stop, ask for a status update, or if a critical
    unrecoverable error occurs.

### Command Safety

*   **No Global Builds/Tests**: Never run commands that consume excessive
    resources, such as `bazel test //...` or building the entire repository,
    unless explicitly scoped.
*   **Timeouts**: Forbid running commands that might hang indefinitely waiting
    for input without setting a timeout or running in non-interactive mode. Git
    operations that can hit merge conflicts (e.g. rebasing or merging) are a
    common case: run them non-interactively so a conflict fails fast instead of
    waiting on a prompt.
*   **Never Hide Output**: Do not pipe long-running or potentially interactive
    commands through `tail`, `head`, or similar filters. They buffer the output,
    hiding prompts, progress and errors until the command exits.
*   **Monitor, Don't Assume**: Never assume a running command will work itself
    out. Read its output as it arrives, and check back on it often (every few
    minutes for commands that normally finish quickly). If a command goes quiet,
    promptly check whether it is still making progress (e.g. its CPU time and
    state via `ps`) or is blocked on input or a lock. If it is stuck, kill it,
    inspect and clean up any partial state it left (e.g. an unfinished rebase),
    and fix the root cause rather than waiting longer.
*   **Sequential Bazel Runs (No Concurrency)**: Never run multiple `bazel`
    (build or test) commands concurrently in the same workspace. If one is
    active, wait for it to finish (or kill it) before starting a new one.

## Bypassing Checks (Use Sparingly)

If you have a legitimate reason to violate a rule, use the suppressions defined
in `ci/custom_lint_rules.METADATA`:

*   `// UNINITIALIZED_TENSOR_OK` (if `at::Tensor x;` is truly safe)
*   `// INT_VEC_OK` (if `std::vector<int64_t>` is required by an external API)
*   `// NON_TT_MACRO_OK=<reason>` (for third-party macro compatibility)

## Writing Agent Instructions

*   **Single Source of Truth (Canonical Ownership)**: Every workflow or rule
    must have **exactly one canonical home**:
    *   Repo-wide C++/Python style and conventions: `AGENTS.md`
    *   Procedural workflows, domain-specific guides, templates, and checklists:
        find and extend the most appropriate existing skill under
        `.agents/skills/<skill>/SKILL.md` (or create a new specialized skill if
        none fits, and add it to the skill index below)
*   **The "1-Sentence Trigger + Pointer" Rule**: Because skills (`SKILL.md`) are
    loaded on demand, non-canonical files (`AGENTS.md` or sibling skills) may
    include **at most a 1-sentence trigger** stating *when* a rule applies (or
    what not to run prematurely), followed immediately by a link to the
    canonical `SKILL.md`. **Never** duplicate multi-step procedures, templates,
    or checklists outside their canonical file.
*   **Skill Naming**: Every TorchTPU skill has one name of the form
    `tt-<verb-phrase>`, used as its frontmatter `name` and its directory name
    (`.agents/skills/<name>/`). `nox -s lint` checks the name and directory
    parts of this rule with `ci/tools/check_agent_skills.py`.
    *   **`tt-` prefix**: Skill names share one namespace across everything an
        agent loads, and a same-named skill from elsewhere silently shadows
        ours. The prefix keeps TorchTPU skills unique.
    *   **Verb phrase**: after the prefix, name the task the skill performs,
        starting with a verb in the imperative (e.g. `tt-write-tests`,
        `tt-review-code`, `tt-fix-composite-ops`), not a noun or gerund (not
        `tt-testing`, `tt-code-review`, or `tt-fixing-composite-ops`).
    *   **Lowercase words joined by hyphens, never underscores** (e.g.
        `tt-implement-ops`): the [Agent Skills spec](https://agentskills.io)
        allows only `a-z`, `0-9` and `-` in a skill name and requires the name
        to equal the skill's directory name.

## Skills

Before starting a task, check whether one of these skills covers it. If so, read
its `SKILL.md` and follow it.

*   [`tt-create-change`](.agents/skills/tt-create-change/SKILL.md): committing a
    change, and opening, describing, or updating a pull request, including
    addressing review comments.
*   [`tt-fix-composite-ops`](.agents/skills/tt-fix-composite-ops/SKILL.md):
    migrating a composite ATen operator to a native one that lowers directly to
    StableHLO.
*   [`tt-write-docs`](.agents/skills/tt-write-docs/SKILL.md): writing, editing,
    or reviewing narrative documentation or README content.
*   [`tt-write-error-tests`](.agents/skills/tt-write-error-tests/SKILL.md):
    writing and placing tests for user-facing errors (`errors_test.py`,
    `tpu_errors_test.py`, `collective_errors_test.py`).
*   [`tt-write-tests`](.agents/skills/tt-write-tests/SKILL.md): writing tests
    and configuring test targets.
