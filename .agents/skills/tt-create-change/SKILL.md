---
name: tt-create-change
description: >-
  Prepares and describes a TorchTPU change as a GitHub pull request (PR).
  Use when committing a TorchTPU change, recording moved or renamed files, opening or updating a pull request, writing or revising its description, or addressing reviewer comments on a pull request (keeping the change scoped to the comments).
  Don't use for general git usage.
---

# Creating a TorchTPU Change

This skill provides the template, guidelines, and checks for preparing a
TorchTPU change and describing it as a GitHub pull request.

--------------------------------------------------------------------------------

## Procedural Steps

> [!IMPORTANT] **Format, push, and open the pull request early.**
>
> Formatting takes seconds, so run it before every push (Step 4). Then push and
> open the pull request right away, as a draft if it is not ready for review, so
> that reviewers and CI can start. Run slower builds and tests while CI runs,
> and push their fixes as follow-up commits.

### Step 1. Record Moves with Git

> [!IMPORTANT] **Never relocate a file by deleting it and adding a copy
> elsewhere.** Record the move so that git keeps the file's history.

*   **Move or rename** — relocating an op, splitting a directory, renaming a
    test: use `git mv <old> <new>`.
*   **Keep history detectable**: git detects renames by content similarity. If a
    moved file also needs large edits, commit the move first and the edits in a
    separate commit, so that `git log --follow` and the pull request diff show a
    rename plus a small delta instead of a deleted and an added file.
*   **Verify** before pushing: `git status` lists the file as `renamed`.

### Step 2. Write the Pull Request Description

Use a one-line summary in the imperative mood as the pull request title (e.g.
"Add a native lowering for torch.logaddexp"). Build the body from
`.github/pull_request_template.md`, keeping its `## Description` and
`## Test Plan` headings and dropping the HTML comment placeholders.
`gh pr create --body` does not apply the template, so fill it in yourself:

```
## Description

{high_level_description_of_the_change_its_motivation_and_its_impact}

## Test Plan

{tests_you_ran_and_how_to_reproduce_them_e.g._bazel_test_commands}
```

Follow these guidelines:

*   **Summarization Style**:
    *   Explain the motivation and impact of the change and what it does at a
        high level.
    *   Do **NOT** write a detailed, line-by-line log of *how* the
        implementation works, as this is redundant with the code itself.
    *   Keep the description concise, professional, and objective, per "Concise
        Writing for Humans" in `AGENTS.md`.
*   **References**: link related GitHub issues (e.g. `Fixes #123`). Do not
    reference links that external readers cannot open.
*   **Commit messages**: give every commit a real message. CI rejects commits
    whose message contains `[NO DESCRIPTION SET]`.
*   **Agent Attribution**: if an AI assistant made meaningful contributions, you
    may add an `Assisted-by:` line to the commit message, per the policy on
    AI-generated code in `CONTRIBUTING.md`. Never list an AI assistant as a
    co-author. The author stays fully responsible for the change.

### Step 3. Updating an Existing Pull Request Description

> [!IMPORTANT] **Never blindly overwrite an existing description.** It
> accumulates context over time: the author may have reworded it, and reviewers
> may have requested changes to it.

When amending the description of a pull request that already exists:

1.  **Read the current description first** (e.g. `gh pr view <number>`).
2.  **Revise, do not regenerate.** Start from the existing text and apply only
    the edits that the change actually requires (e.g. updating the summary for
    newly added files). Preserve all author-written wording.
3.  **Apply the revision from a file** (e.g.
    `gh pr edit <number> --body-file <file>`), so that the whole description is
    reviewed before it is replaced.
4.  **Confirm** with `gh pr view <number>` that no content was dropped.

### Step 4. Formatting, Checks, and Rebasing

1.  **Before every push**: run `nox -s format` and make sure `nox -s lint`
    passes. Lint also checks agent skill names and the directory layout.
2.  **After pushing**: build and test what you changed, e.g.
    `bazel test --config=ci_cpu //tests:<target>` (see `.bazelrc` for the CI
    configs), and check the GitHub Actions results on the pull request. Push
    fixes as new commits, running `nox -s format` again first.
3.  **Rebasing**: after rebasing onto `main`, run `nox -s format` again, since
    files changed by the rebase may need formatting too, then push with
    `git push --force-with-lease`.

--------------------------------------------------------------------------------

## Addressing Review Comments

> [!IMPORTANT] **When responding to comments on a pull request, change only what
> the comments require.** The reviewer re-reads the changes since their last
> review; every unrequested edit in them makes it harder to confirm that their
> comments were addressed and to spot regressions. Scope creep turns a quick
> re-review into a fresh review.

*   **Minimal edit per comment**: for each unresolved thread, make the smallest
    change that resolves it. If a comment's intent is ambiguous, ask the user
    (or reply on the thread, if asked to) instead of guessing broadly.
*   **No drive-by changes**: do not refactor, rename, reformat, reorder, reword
    comments or docstrings, add features or tests, or fix unrelated issues that
    no comment asks for, even when you notice them. Allowed exceptions are only
    the edits mechanically required by a requested change or by this skill:
    updating callers or tests of something the reviewer asked to change,
    `nox -s format` output (Step 4), and build or test fixes needed for the pull
    request to pass again.
*   **Stay inside the pull request's diff**: when a comment flags a pattern
    ("use X instead of Y"), apply it to the other instances in lines this pull
    request already changes, not to untouched code elsewhere in the file or
    package (unless the comment explicitly asks for that).
*   **Report, don't fix, out-of-scope findings**: list anything else worth
    changing in your summary to the user as a proposed follow-up pull request or
    issue.
*   **Check the delta before pushing**: diff your new commits against what the
    reviewer last saw and revert every hunk you cannot trace to a comment or to
    one of the allowed exceptions above. In your summary, map each comment to
    its edit and justify any edit that is not tied to a comment.
*   **Re-request review**: when the pull request is ready for another look, use
    the "Re-request review" button next to the reviewer's name.
*   **Description**: edit the description only if a comment asks for it or the
    requested change makes it inaccurate; follow
    [Step 3](#step-3-updating-an-existing-pull-request-description).

--------------------------------------------------------------------------------

## Verification and Validation Checklist

Before requesting review, run the following self-validation steps:

-   [ ] **Format Code**: Did you run `nox -s format` before every push,
    including after a rebase, and does `nox -s lint` pass?
-   [ ] **Moved Files**: Is every relocated file recorded as a rename (check
    `git status`), rather than a delete plus an add?
-   [ ] **Template**: Does the description follow
    `.github/pull_request_template.md`, with a filled-in `## Description` and
    `## Test Plan`?
-   [ ] **Description Preservation**: If you updated an existing description,
    did you read the current one and revise it instead of overwriting it?
-   [ ] **Review-Comment Scope**: If this push addresses reviewer comments, does
    every hunk since the reviewer's last look trace to a comment or an allowed
    exception in [Addressing Review Comments](#addressing-review-comments)?
-   [ ] **Guideline Compliance**: Does the description comply with every
    guideline in [Step 2](#step-2-write-the-pull-request-description)?
-   [ ] **Tests and CI**: Did you run the relevant `bazel` tests, and are the
    GitHub Actions checks on the pull request green?
