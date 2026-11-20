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
assignee. The champions are listed in
[`.github/quality_champions.txt`](../../.github/quality_champions.txt). To
balance their load, pick the one who is assignee on the fewest
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

## Import the PR

### When does the import happen?

**Champion**: Google's Copybara bot imports a PR into an internal change once
the PR has the `pull ready` label, a Googler has approved its latest commit,
**and** a quality champion is assigned (`assignee-is-quality-champion` status is
green). After you approve the PR, add the `pull ready` label. If you forget, any
Googler may add it once you've approved. The import usually starts within
minutes; the PR's `import/copybara` check shows its status.

### How do I check if a PR has been imported?

**Googler**: The bot usually creates or updates the internal change within about
five minutes of the import being triggered. To find it, search for
`torch_tpu/pull/123` (with your PR number) in Critique, Google's internal code
review tool: the internal change's description includes an `Imported from:` line
with the PR's URL. If nothing shows up after 15 minutes, see
[The import didn't happen](#the-import-didnt-happen-how-do-i-find-out-why).

### The author pushed new commits after approval. What now?

**Expert or champion**: Approvals apply to a specific commit, so a new commit
stops the import until a Googler approves the PR again. Review the new commits
and approve the PR again on GitHub. That approval re-triggers the import, and
the bot updates the existing internal change.

### `import/copybara` says "Waiting for internal safe review approval". What do I do?

**Googler**: The bot skips its internal safe review when a Googler has approved
the PR's latest commit, so first make sure that's the case. If the check still
waits (e.g. it's left over from before the latest approval), click the check's
**Details** link and vote **Code-Review+2** on the internal safe-review page
that opens. The page is Google-internal, so external authors can't do this step.

<img src="images/copybara_import_approval.png" alt="Pending import/copybara checks" width="600px">

### The import didn't happen. How do I find out why?

**Googler**: First check the conditions in
[When does the import happen?](#when-does-the-import-happen): the PR isn't a
draft, has the `pull ready` label, has a Googler approval on its latest commit,
and has a quality champion assigned (`assignee-is-quality-champion`). Then open
the details of the PR's `import/copybara` check, if it has one, which says why
an import is blocked, e.g. a missing `pull ready` label, no Googler approval of
the latest commit, a missing `assignee-is-quality-champion` status, or a merge
conflict. Fix that cause and the import retries. The check doesn't appear on
every PR, so if it's missing, look for the internal change (see
[How do I check if a PR has been imported?](#how-do-i-check-if-a-pr-has-been-imported)).
If the conditions are met but there's no internal change after an hour, or the
change exists but was never sent for review, start the import yourself (see the
next answer).

### How do I start an import myself?

**Googler**: Use this when an import should have happened but didn't, or when it
failed partway: the bot created the internal change but didn't send it for
review, so the change has no reviewers and never gets approvals, runs presubmit,
or lands. The bot doesn't retry such an import on its own, and approving the PR
again doesn't restart it unless the PR has new commits. Start the import by hand
as described in
[Starting a GitHub PR import manually](http://go/tt-devguide#starting-a-github-pr-import-manually)
in the TorchTPU developer guide (a Google-internal link that works only for
Googlers). The bot then updates the existing internal change (or creates one)
and sends it for review.

### The PR conflicts with `main`. What do I do?

**Author**: Rebase your branch on the latest `main`, resolve the conflicts, and
push. The new commit needs a Googler's approval again before the PR can be
imported.

## Land the internal change

### The internal change fails presubmit. Who fixes it?

**Expert**: You do, usually. For a fix the author should learn from, or one that
belongs in the PR, ask the author to push it to the PR (or push it yourself if
you can), approve the new commit, and let the bot update the internal change.
For an internal-only fix, such as a build file or an internal test, patch the
internal change instead (see the next answer).

### How do I patch the internal change and land it myself?

**Expert**: Patch the bot's internal change into your own workspace with
Copybara's patch workflow (the internal change's Copybara comment shows how),
make your fixes there, and get the champion's approval. Your copy is authored by
you, so it needs only that one approval. After it lands, comment on the PR with
the commit SHA from `main` and close the PR. The bot deletes its original
internal change later.

### Who approves the internal change?

**Expert and champion**: When the author isn't a Googler, the internal change
needs approvals from two different Googlers. A Googler's GitHub approval of the
PR's latest commit counts as one. So if the expert approved an older commit, the
expert also approves the internal change (or approves the latest commit on
GitHub again), and the champion approves the internal change.

### It's approved but hasn't landed. What's blocking it?

**Googler**: The internal change lands automatically once it has the required
approvals, passes presubmit, and has no unresolved comments, so check those
three on the internal change. A presubmit failure needs a fix (see "The internal
change fails presubmit"), and each unresolved comment needs a reply and
resolution. If Copybara has stopped retrying the submit, approve the internal
change again to restart it.

### My PR was closed, not merged. Is that normal?

**Author**: Yes. The project's source of truth is Google's internal repository,
so Copybara lands your change there, then commits it to `main` and closes the PR
instead of merging it. Look for a commit on `main` that mentions your PR number.

### My change was rolled back. What now?

**Author**: The project rolls back changes that break the main branch, so that
`main` stays dependable. The rollback commit explains what broke. Fix the
problem, then open a new PR that reapplies your change with the fix, and link
the rolled-back PR.

### How is my authorship credited?

**Author**: The commit on `main` lists you as its author, using the name and
email address of your PR commits. Make sure those match the account that signed
the CLA.

## Find the answers for your role

*   **Author**: [issues](#do-i-need-an-issue-before-opening-a-pr),
    [CLA](#the-cla-check-failed-what-do-i-do),
    [no reviewer](#my-pr-has-no-reviewer-yet-what-do-i-do),
    [CI failures](#ci-failed-who-fixes-it),
    [handing the PR back](#i-addressed-the-comments-how-do-i-hand-the-pr-back),
    [unresponsive reviewer](#my-reviewer-hasnt-responded-what-do-i-do),
    [conflicts](#the-pr-conflicts-with-main-what-do-i-do), and
    [closed, not merged](#my-pr-was-closed-not-merged-is-that-normal).
*   **Oncaller**:
    [closing unwanted PRs](#the-pr-shouldnt-land-how-do-i-close-it).
*   **Expert**:
    [approving CI runs](#the-pr-says-workflows-awaiting-approval-what-do-i-do),
    [checking CI](#no-checks-failed-did-ci-pass),
    [handing off to a champion](#when-does-the-expert-hand-off-to-a-champion),
    [new commits](#the-author-pushed-new-commits-after-approval-what-now),
    [presubmit failures](#the-internal-change-fails-presubmit-who-fixes-it), and
    [patching](#how-do-i-patch-the-internal-change-and-land-it-myself).
*   **Champion**:
    [what to check](#what-do-the-expert-and-the-champion-each-check),
    [starting the import](#when-does-the-import-happen),
    [failed imports](#the-import-didnt-happen-how-do-i-find-out-why), and
    [approving the internal change](#who-approves-the-internal-change).
