# Pull request FAQ

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This FAQ helps you take the next step on a TorchTPU pull request (PR) when
you're not sure what it is. Each answer is a short recipe for one situation. For
the overall review process, see
[Code reviews in CONTRIBUTING.md](../../CONTRIBUTING.md#code-reviews).

The answers name the role that acts:

*   **Author**: the person who opened the PR.
*   **Oncaller**: the Google engineer who triages new PRs.
*   **Expert**: the Google domain expert who reviews the PR first.
*   **Champion**: the Google quality champion, who reviews the PR for style and
    quality and is its assignee.
*   **Googler**: any of the oncaller, expert, or champion.

## Open a PR

### Do I need an issue before opening a PR?

**Author**: For a trivial fix, no. For anything bigger, open an issue first so
that maintainers can agree on the shape of the solution before you write code,
and link your PR to it (e.g. `Fixes #123` in the PR description). Reviews are
prioritized by the priority of the linked issue, so PRs without one are reviewed
as time allows.

### The CLA check failed. What do I do?

**Author**: Sign the Google Contributor License Agreement at
<https://cla.developers.google.com/>, using the same email address as your
commits. The CLA bot re-checks the PR automatically. If you've already signed,
make sure every commit in the PR uses an email address covered by your
agreement, then push again.

### Can I choose my own reviewer?

**Author**: Not unless you've arranged it with that person beforehand. The
oncaller assigns a domain expert based on the code your PR touches and the
team's current review load, so leave the reviewer and assignee fields empty.

### My PR has no reviewer yet. What do I do?

**Author**: Wait for the oncaller to triage it; new PRs are triaged on business
days. If a few business days pass without a reviewer, leave a comment on the PR
asking for one. Make sure the PR isn't a draft: draft PRs aren't triaged.

### Do draft PRs get reviewed?

**Author**: No. Draft PRs aren't triaged or reviewed. When your PR is ready,
click **Ready for review**; the PR then enters triage like a new PR.

## Get CI to run

### The PR says "Workflow(s) awaiting approval". What do I do?

**Expert or champion**: GitHub doesn't run CI on PRs from some forked
repositories until a maintainer approves the runs. First, in the **Files
changed** tab, check that the PR doesn't change `.github/workflows/` or other CI
scripts in a way that could misuse the repository's secrets or CI machines. Then
click **Approve workflows to run** in the **Conversation** tab. Approve again
each time the author pushes new commits or someone submits a review, because
both can start new runs.

### No checks failed. Did CI pass?

**Expert**: Not necessarily. On PRs that need workflow approval, the checks list
shows only the few workflows that don't need approval, so it can look clean
while CI hasn't run at all. Before you approve the PR, make sure the full set of
checks (e.g. `CI - Presubmit` and `Lint checks`) ran on the latest commit and
passed.

### CI failed. Who fixes it?

**Author**: You do, unless the failure is unrelated to your change. Open the
failed check's details, fix the problem, and push a new commit. If you think the
failure is flaky or unrelated, say so in a PR comment; a maintainer can re-run
the check or confirm the failure is pre-existing. **Expert**: Help the author if
the failure is hard to diagnose, and don't approve the PR until CI passes.

### My workflow runs expired. How do I re-run them?

**Author**: GitHub expires workflow runs that aren't approved within 30 days: it
marks them as failed, and they can't be approved or re-run anymore. To start new
runs, push a new commit (an empty commit is fine). A maintainer can also close
and reopen the PR.

## Review the PR

### Whose turn is it?

**Everyone**: Check the PR's `stat:*` label, which a bot updates as people
respond. `stat:awaiting-reviewers` means the turn is with the reviewers and the
assignee; `stat:awaiting-requester` means it's with the author. A comment or
review from a reviewer hands the turn to the author, and a comment, review, or
new review request from the author hands it back.

### I addressed the comments. How do I hand the PR back?

**Author**: Push your changes, reply to each comment (or resolve the ones that
need no reply), then click **Re-request review** (the circular arrows) next to
each reviewer's name. That notifies them and gives the turn back to the
reviewers.

### My reviewer hasn't responded. What do I do?

**Author**: First check that it's the reviewers' turn
(`stat:awaiting-reviewers`) and that you've re-requested review. If a few
business days pass after that, leave a comment on the PR mentioning the
reviewer.

### When does the expert hand off to a champion?

**Expert**: Once you're satisfied with the PR and CI passed on its latest
commit, approve it, then add a quality champion as both a reviewer and the
assignee. The champions are `cbasile-g`, `LeeHowes`, `vladbelous`, and
`zhanyong-wan`. To balance their load, pick the one who is assignee on the
fewest
[open PRs](https://github.com/google-pytorch/torch_tpu/pulls?q=is%3Apr+is%3Aopen+-author%3Aapp%2Fcopybara-service+sort%3Aupdated-asc+draft%3Afalse).

<img src="images/pr_roles.png" alt="Where the reviewer and assignee roles appear in a PR" width="400px">

### What do the expert and the champion each check?

**Expert and champion**: The expert checks that the change is correct, well
tested, and the right approach for the code it touches, and that CI passes. The
champion reviews the PR for style, readability, and overall code quality, in the
open, so that contributors can learn the project's standards. The champion
doesn't need to repeat the expert's correctness review.

### The author stopped responding. What happens?

A bot marks a PR `stale` after 30 days without activity and closes it 14 days
later; any comment resets the clock. **Expert or champion**: If the change is
valuable, you may finish it yourself in a new PR that credits the original
author, or close the PR with a comment explaining why.

### The PR shouldn't land. How do I close it?

**Author, oncaller, expert, or champion**: If the PR is obsolete, duplicates
another PR, or proposes a change the project doesn't want, close it without
landing it and leave a comment explaining why. For a duplicate, link the PR or
issue it duplicates.

### The review found a big design problem. What now?

**Expert**: Explain the problem and the direction you'd prefer in a PR comment,
and agree on a plan with the author before they rewrite the PR. If the new
approach is substantially different, ask the author to open a new PR (linked to
the same issue) and close the old one.
