---
name: tt-review-code
description: >-
  Reviews TorchTPU code for architectural complexity, C++ nuances, and readability compliance.
  Similar to a senior engineer review.
  Use when evaluating implementations for unnecessary abstractions, checking for subtle C++ bugs (like lambda capture issues or copies), spotting moved files that were recorded as a delete plus an add instead of a rename, and ensuring style compliance (naming, logs, enums) and documentation quality.
---

# TorchTPU Code Review

This skill provides comprehensive guidance for reviewing code in TorchTPU across
Complexity, C++ Nuances, and Readability (C++ and Python).

## Manual Review Gates

You MUST manually evaluate the following gates.

### 1. Formatting

*   Verify formatting (`nox -s format`) and lint (`nox -s lint`) per
    [tt-create-change](../tt-create-change/SKILL.md).

### 2. Complexity & Performance

*   **StableHLO Lowering Performance**:
    *   Research if a JAX lowering exists, and thoroughly analyze the sequence
        of StableHLO operations to suggest lowering that is more efficient or
        has mathematical expansions that lead to more accurate precision.
    *   Assume JAX lowering as the ground truth for performant code.
*   **Architectural Simplification & Anti-Bloat**:
    *   Identify all custom logic, new files, and new abstractions in the pull
        request.
    *   Flag unnecessary abstractions and logic.
    *   For agent skills and tooling, flag scripts or taxonomies that encode
        semantic judgment the calling agent can follow in a few lines of
        `SKILL.md` prose.
    *   Treat whole-change bloat as a blocking issue: if the diff is too large
        to review cleanly, ask to cut low-value or speculative code first and
        split what remains into smaller pull requests.
*   **Conventions**: check the change against the code style rules in
    `AGENTS.md` ("Code Style & Conventions"), in particular const correctness,
    enum-instead-of-`bool` parameters, no `sys.path` manipulation or
    `ImportError` fallback shims in Python files, and comments that explain
    *why* rather than *how*.
*   **`BUILD` Test Targets & CI Jobs**: When the pull request adds or modifies
    test rules in `BUILD` files or CI job configs in `.bazelrc`, verify
    compliance with
    [tt-write-tests](../tt-write-tests/SKILL.md#ci-jobs--build-test-configuration).
*   **Test Doubles**: When the pull request adds or modifies tests that use
    mocks, fakes, or `mock.patch`, flag mock overuse per
    [tt-write-tests](../tt-write-tests/SKILL.md#test-doubles).

### 3. C++ Nuance Review

Check for deep C++ pitfalls and correct library usage:

*   **Avoid Copies**: Prefer passing by `const &` for large inputs. Use
    `std::move` for transferring ownership. Check loop variables.
*   **Lambda Captures**: Verify captures in deferred execution contexts don't
    outlive referenced objects. Be careful with `[&]` and `[this]`.
*   **Null Pointers**: Check for potential dereferences. Use `absl::Status` or
    `absl::StatusOr`.
*   **Logging**: Use of `ABSL_LOG` macros. Do NOT use `LOG` macros.
*   **Initialization**: Ensure all primitive types and pointers are initialized.
*   **Custom Utilities**: Prefer TorchTPU specific utilities over raw Torch,
    MLIR, `absl`, or other standard library APIs: `csrc/common/aten_utils.h`,
    `csrc/common/error_utils.h`, `csrc/common/to_string.h`,
    `csrc/common/utils.h`, `csrc/ops/op_builder_utils.h`, and
    `csrc/eager/op_dispatcher.h` (`DispatchOpOut`).
*   **Error Messages**: When the change adds or modifies user or internal error
    checks, verify compliance with the error message rules in
    [tt-write-error-tests](../tt-write-error-tests/SKILL.md#error-message-format).
*   **Buffer Donation**:
    *   Verify that all out-variant (`Aten*Out`) and in-place (`Aten*Inplace`)
        ops use `DispatchOpOut` instead of `DispatchOp`.
    *   Confirm that internal buffer donation functions are never called
        manually in op kernels.
    *   Confirm that destination buffers are donated so that TPU memory is
        efficiently reused rather than reallocated.

### 4. Readability & Style Review

While formatting and lint are checked by `nox -s format`, `nox -s lint`, and CI,
you should still verify:

*   **Style Guides**:
    *   Review that the code follows the
        [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
    *   Review that the code follows the
        [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
*   **Actionable Feedback**: Provide specific, actionable comments on how to
    improve readability.
*   **Concise Writing**: Write comments per "Concise Writing for Humans" in
    `AGENTS.md`, and ask authors to trim pull request descriptions that violate
    it.

### 5. Documentation & Agent Instructions Review

Verify:

*   The code is adequately commented so that the API contract, code intent, and
    tricky implementation details are clear.
*   The pull request description is clear and summarizes the change well. It
    should clarify the motivation of the change and its impact.
*   The docs under `docs/` are in sync with the code. E.g. if the pull request
    changes a documented feature or environment variable, the page that
    documents it should be updated to reflect the change.
*   Any new or modified TorchTPU API surface conforms to
    `docs/developers/api_design_guide.md` and `docs/developers/api_policy.md`.
*   **Agent Instructions Single Source of Truth**: When the pull request
    modifies `AGENTS.md` or `.agents/skills/*/SKILL.md`, enforce the **Single
    Source of Truth** and **"1-Sentence Trigger + Pointer"** rules in
    `AGENTS.md` ("Writing Agent Instructions"). Multi-step workflows, CLI paths,
    templates, and checklists must live **only** in their canonical file. Flag
    any duplicated procedure in a non-canonical file and require replacing it
    with at most a 1-sentence trigger plus a link to the canonical owner.

### 6. Moved File Review

Files are frequently relocated (e.g. an op kernel moved to another directory, a
test file renamed). A move **MUST** be recorded as a rename in git rather than
done as a delete plus a copy-paste, so that history is preserved and the diff
shows only the delta from the original.

*   **How to verify whether a file is already recorded as moved**:

    *   Run `git diff --find-renames --name-status <base>...HEAD` (e.g. with
        `origin/main` as `<base>`), or check the pull request's "Files changed"
        tab on GitHub:

        *   If the file is listed as renamed (`R<score>` in the output), **the
            move is ALREADY recorded**. Do **NOT** complain, report an
            unrecorded move, or ask the author to redo the move.
        *   A file is an addition only when listed as added (`A`).

*   Only when a file is confirmed to be an addition (`A`, not `R`):

    *   If the pull request deletes a file and adds a similar one at a new path
        as a plain addition, it is an unrecorded move. Require the author to
        record the move with `git mv`, committing the move separately from large
        edits so that git detects the rename (see Step 1 in
        [tt-create-change](../tt-create-change/SKILL.md)), and name the
        suspected source file in your comment.
    *   To locate a plausible source file, scan for structural similarity (same
        boilerplate, license header, includes, class or test-case skeleton,
        near-identical helper functions) and use `rg` or `git grep` on
        distinctive lines.

### 7. Pull request description review

*   Verify that the pull request title, description, and commit messages follow
    the TorchTPU guidelines in [tt-create-change](../tt-create-change/SKILL.md).
*   Validate both the form of the description (the sections of
    `.github/pull_request_template.md`) and the content, making sure that it
    accurately describes the changes legibly for external readers.
*   **The description must not be summary-only.** Below the one-line summary
    (the title), the description must have at least one paragraph saying what
    the change does and why, written for a reader who sees only the pull
    request. Treat this as a binary check rather than a judgement call: if the
    description body is empty, or merely restates the summary, post an
    unresolved comment asking for the motivation and impact.
*   **All links must be accessible to external readers.** A reference or URL in
    the description that they can't open is a blocking finding; ask for it to be
    removed or replaced with a public one (e.g. a GitHub issue).

## Validation

1.  Complete each review gate.
2.  If any of the gates fail:
    -   Create a detailed list of gates that failed for feedback.
    -   Reset your state to step 1 and give feedback to the calling agent.
3.  Your review is complete when you have no additional feedback from the gates.
