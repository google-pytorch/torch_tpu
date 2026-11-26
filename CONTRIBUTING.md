# Contributing to TorchTPU

Thank you for your interest in contributing to TorchTPU, we are happy to welcome
contributions! Before contributing, we ask that you read this document to
understand our process. The goal of this document is to describe how we work
together and to help us collaborate effectively, respect one another's time and
build a high-quality library.

We are looking to keep friction as low as possible for contributors while still
assuring maximum quality for our users. If any part of this process feels like
arbitrary box-checking that is not serving that goal, please open an issue so
that we can work on it.

## Philosophy on Contributions

We value your time and energy, and we ask that you value ours in return. We much
prefer small, well-tested contributions that address a single clear issue over
sprawling pull requests that try to solve multiple problems at once.

## Issue Reports

While pull requests fixing bugs or implementing features are always welcome,
they are by no means required. **Submitting a clear, well-isolated bug report or
feature issue is in itself a valuable contribution.**

For anything more complicated than trivial fixes, please open or comment on an
existing issue (or [an RFC](#proposing-major-changes-and-api-design) for major
architectural or API changes) *before* writing code. In particular, prior
alignment is required for new operators, edits to hot-path lowerings, new APIs
or shared utilities, and changes exceeding roughly 300 non-generated, non-test
lines. Aligning upfront lets maintainers confirm the approach, avoid duplicate
work, and agree on how to split larger changes into focused PRs. Linking your PR
to that issue also helps us prioritize its review (see
[What We Expect From Contributors](#what-we-expect-from-contributors)).

### What Makes a Great Bug Report

If you run into a bug or crash, please include:

*   **What happened vs. what you expected**: Clear reproduction steps and the
    full error traceback.
*   **Minimal Reproducible Example (MRE)**: A short script isolating the bug.
    Removing unrelated model logic makes diagnosis much faster.
*   **Environment details**: Your PyTorch version, `torch_tpu` version and TPU
    hardware generation if applicable.

## Before You Begin

### Sign Our Contributor License Agreement

Contributions to this project must be accompanied by a Google Contributor
License Agreement (CLA). You (or your employer) retain the copyright to your
contribution; signing simply gives us permission to redistribute your changes.

*   Visit <https://cla.developers.google.com/> to see your agreements or sign a
    new one.

If you have already signed the Google CLA for another open-source project, you
likely do not need to sign it again. A CLA bot checks pull requests
automatically and guides you if action is needed.

### Claiming an Issue

To help avoid duplicate work, please follow these practices before starting on
an existing issue:

*   **Check for existing work**: Check whether the issue has an assignee, read
    its comments, and look for linked pull requests. If someone is assigned to
    it, has said they are working on it, or has already opened a pull request
    referencing it, someone is likely already working on it.
    -   If an assigned issue has had no recent activity, feel free to ask in a
        comment whether it is free to pick up.
*   **Comment before you start**: If the issue is unassigned, leave a comment on
    it to express your interest in working on it. A maintainer *may* then assign
    the issue to you.
*   **Overlapping fixes**: If several fixes for the same issue are proposed
    anyway, maintainers may choose the one that best addresses the issue.

### Review Our Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
We treat all contributors with respect, warmth, and professionalism.

## Contribution Process

### Code Reviews

All submissions, including submissions by project members, require review. We
use
[GitHub pull requests](https://help.github.com/articles/about-pull-requests/)
(PRs) for this purpose.

#### How a PR lands

A PR goes through four hand-offs. At each one, exactly one person owns the next
step:

1.  **Triage**: a Google **oncaller** assigns a Google **domain expert** as the
    PR's reviewer.
2.  **Review**: the **domain expert** reviews the PR, makes sure CI passes on
    its latest commit, and approves it. Once approved, a bot picks a Google
    **quality champion** at random (excluding the approver) and adds them as a
    reviewer and the assignee.
3.  **Quality**: the **quality champion** reviews the PR for style and quality,
    approves it, and adds the `pull ready` label.
4.  **Land**: a bot (Copybara) copies the PR into Google's internal repository
    as a change. Once two Googlers have approved it and all review comments on
    it are resolved, the change lands, the bot commits it to `main`, and the PR
    is closed rather than merged.

Two rules hold throughout:

*   The PR's `stat:*` label shows whose turn it is: `stat:awaiting-reviewers`
    (the reviewers' and assignee's) or `stat:awaiting-requester` (the author's).
*   Approvals apply to a specific commit. After the author pushes a new commit,
    a Googler approves the PR again before it can move forward.

<img src="docs/developers/images/pr_roles.png" alt="Where the reviewer and assignee roles appear in a PR" width="400px">

#### Details

*   **Author**: Keep the PR small and linked to an issue. After you address
    comments, click **Re-request review** to hand the PR back. Keep CI green.
*   **Domain expert**: On PRs from some forks, GitHub doesn't run CI until you
    approve the workflow runs; review the changes first, since approving runs
    the PR's code. Don't approve the PR until the full set of checks has run on
    the latest commit and passed. If the internal change later needs fixes,
    you're usually the one who makes them.
*   **Quality champion**: Review in the open, so contributors can learn the
    project's standards. If the PR's author isn't a Googler, the internal change
    needs a second Googler's approval besides yours; a domain expert's GitHub
    approval of the latest commit counts.
*   **Anyone involved**: If the PR shouldn't land (e.g. it's obsolete or a
    duplicate), close it with a comment explaining why.

When you're not sure what to do next, for example when CI doesn't run, an import
fails, or the internal change fails presubmit, see the
[pull request FAQ](docs/developers/pr_faq.md).

## Proposing Major Changes and API Design

While most changes start with an issue, major architectural changes or new
public APIs require a Request for Comments (RFC) discussion before
implementation begins.

*   **API Philosophy**: We aim to make TorchTPU feel native to PyTorch. When
    contributing, prefer to extend standard PyTorch mechanisms (such as
    `torch.distributed` and `torch.compile`) rather than inventing
    hardware-specific wrappers. See
    [TorchTPU API Guidelines](docs/developers/api_design_guide.md) for more
    details.
*   **Stability Lifecycle**: New public APIs progress from internal, to
    experimental, and finally to stable. Once marked stable, APIs undergo a
    formal deprecation period before removal. To follow this process, create new
    APIs as internal — or experimental if intended to be used by consumers of
    TorchTPU — rather than stable. See
    [TorchTPU API Policy](docs/developers/api_policy.md) for more details.

## Policy on AI-Generated Code

We neither encourage nor discourage the use of AI assistants for generating
code, but we do have one firm policy no matter how the code was generated: **The
author is fully responsible for the code they submit.** This means:

*   **You must understand it**: If a reviewer asks why an operation is written a
    certain way, "an AI wrote it" is not an acceptable answer. Never submit code
    that you cannot explain and maintain yourself.
*   **You must validate it**: Ensure the code is correct, well-tested, and free
    of subtle bugs, regressions, or licensing violations.
*   **No automated low-effort submissions**: Pull requests that appear to be
    unreviewed, automated LLM output waste reviewer time and will be closed.
    When a PR shows systemic un-audited generation, maintainers will close it or
    pause review with high-level feedback pointing back to this policy rather
    than giving line-by-line fixes. Please expect to spend at least as much time
    preparing your code for submission as you expect maintainers to spend
    reviewing it.
*   **Optional attribution**: If an AI assistant provided meaningful help, you
    may add an `Assisted-by:` tag in your commit message. AI assistants should
    not be considered co-authors on any commits.

To minimize the maintenance burden imposed by people who do *not* follow these
rules, we may limit the number of PRs a contributor can open at one time. If
this causes problems for you and you have a record of consistently making
high-quality contributions, please reach out to us and we will try to find a
solution.

## Review Expectations

A constructive review process relies on mutual respect for time and attention.

### What You Can Expect From Maintainers

*   **Thoughtful feedback**: We review code for correctness, clarity, and long-
    term maintainability, focusing on the problem rather than the person.
*   **Alignment before line-by-line review**: When a PR lacks required prior
    alignment, maintainers will close it or pause review with high-level
    feedback pointing back to this guide rather than giving line-by-line fixes.
*   **Commitment to main branch stability**: We follow a strict revert-on-
    breakage policy so our main branch remains dependable for everyone.
*   **Up-to-date contribution documentation**: All requirements for contribution
    should be well-documented and clearly explained. If you encounter unclear or
    missing documentation, we encourage you to file an issue about it.

### What We Expect From Contributors

*   **Small, focused PRs**: Submit pull requests that do one thing well, and
    split larger approved changes into focused PRs. A concise 100-line diff gets
    reviewed far faster than a 1,000-line rewrite.
*   **Clear context**: Explain what your change does and why it is needed. All
    non-trivial PRs must
    [link to the relevant GitHub issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).
*   **Link your PR to an issue**: Even for small changes, we encourage you to
    link your PR to the issue it addresses, e.g. with `Fixes #123` in the PR
    description. We prioritize reviews by the priority of the linked issue, so
    PRs that address higher-priority issues get more prompt attention, while PRs
    without a linked issue are reviewed as time allows.
*   **Shepherd your change**: Keep an eye on CI checks, address reviewer notes,
    and let us know if you get blocked or need to step away.
*   **Let us assign owners and reviewers**: Please don't assign or request
    specific people as assignees or reviewers of your issues and PRs unless you
    have arranged this with them beforehand. A Google oncaller triages new
    issues and PRs and assigns them to the people with the best matching
    expertise, while balancing the load across the team.

## Reporting Security Vulnerabilities

If you discover a potential security vulnerability in TorchTPU, please report it
privately rather than opening a public GitHub issue.

*   **How to report**: Use GitHub's private vulnerability reporting form at
    <https://github.com/google-pytorch/torch_tpu/security/advisories/new> (or
    navigate to the **Security** tab of our repository and select **Report a
    vulnerability**).
*   **Scope**: High memory usage, crashes caused by invalid inputs, or numerical
    divergence from untrusted models are normal bugs rather than security
    vulnerabilities. Running untrusted model code is equivalent to running
    untrusted arbitrary Python code.
