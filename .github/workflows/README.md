<!-- Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# CI overview

A rough map of what runs where, and notes on the parts that are easy to get
wrong. This is not exhaustive; the workflow files are the source of truth.

## Workflows

| Workflow | Runs on | Notes |
| --- | --- | --- |
| `presubmit.yml` | PR | The main blocking presubmit. |
| `postsubmit.yml` | push to main, schedule, PR | Consumes wheels built by `build_wheel.yml`. |
| `build_wheel.yml` | push to main, PR | Builds the wheels; publishes `gcs_wheel_path`. The only workflow here that runs on every push to main. |
| `lint.yml` | PR | Formatting and lint. |
| `local_torch.yml` | PR, nightly | Builds against a local PyTorch nightly. |
| `tsan.yml` | PR, nightly | ThreadSanitizer build and tests. |
| `nightly_tpu.yml` | PR, nightly | Longer-running TPU tests. |
| `nightly_unpinned_deps.yml` | PR, nightly | Builds with dependency pins relaxed. |
| `nightly_benchmarks.yml`, `run_benchmarks.yml` | schedule, manual | Benchmarks via the `google-ml-infra/bap` reusable workflow. |
| `upstream_tests.yaml`, `upstream_examples.yaml` | PR (labelled), nightly | Upstream PyTorch tests and examples. |
| `out-of-tree-ci.yml` | `repository_dispatch` | Triggered by upstream PyTorch pushes. |
| `on_pr_close.yaml` | PR closed | Cancels and deletes leftover runs; see below. |
| `docs.yaml` | push | Publishes the Marimo notebook HTML to Pages. |
| `daily_project_management.yml` | schedule | Issue and project board housekeeping. |

## Concurrency and cancellation

Most workflows set a `concurrency` group so a newer run supersedes an older
one on the same branch, with `cancel-in-progress` deciding whether the older
run is killed or left to finish.

Two spellings are in use, and they mean the same thing:

- `cancel-in-progress: ${{ github.ref_name != 'main' }}`
- `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` (`postsubmit.yml`)

**Use `github.ref_name`, not `github.ref`.** `github.ref` is the fully
qualified ref — `refs/heads/main` on main — so comparing it to `main` is always
true and cancels everywhere, including the branch you meant to protect. This
was the bug fixed in #3893; it had been cancelling roughly 90 of every 100
`build_wheel` runs on main.

Note this guard only bites where a workflow actually runs on a non-PR ref.
`build_wheel.yml` is the only one triggered by every push to main; for the
nightly workflows it protects scheduled runs, which execute against main.

**Open question:** none of the above explains *why* in-progress runs on main
should be left alone. The guard and its comment both arrive in the initial
squashed commit (`Start of public TorchTPU history`), so the original reasoning
isn't recoverable from this repository, and it isn't written down anywhere
else. If you know it, please replace this paragraph. See the discussion on
#3893.

## Deleting cancelled runs on PR close

`on_pr_close.yaml` cancels in-flight runs when a PR closes and then deletes
the cancelled ones, via `ci/tools/delete_cancelled_pr_runs.py`.

This exists because Copybara updates a PR immediately before merging it, so the
PR shows as *merged* rather than *closed*. That update triggers a full
presubmit run that nobody wants, and a cancelled run otherwise lingers as a
failed check on the commit.

A periodic sweep of all cancelled runs would be simpler but is not safe: some
cancellations are real signal, and nothing currently distinguishes "cancelled
because this run should never have started" from "cancelled for some other
reason."

Note the script talks to the REST API through `gh api` rather than the
`gh run` subcommands, which resolve a run's parent workflow and therefore fail
with HTTP 404 for runs whose workflow is no longer in the repository.

## Container images

Jobs run in `ml-build` from Artifact Registry, pinned by digest with the
matching tag in a trailing comment:

```yaml
container:
  image: "us-docker.pkg.dev/.../ml-build@sha256:..."  # Tag infrastructure-public-image-...
```

**These pins are maintained by hand.** Dependabot cannot update them: the
`github-actions` ecosystem only touches `uses:` references, and the `docker`
ecosystem only fetches Dockerfiles, Containerfiles, and YAML that looks like a
Kubernetes resource (it requires both `apiVersion` and `kind`), so it never
looks at `jobs.<id>.container.image`. Adding a `docker` entry to
`.github/dependabot.yml` would appear to cover these and would not.

They do drift as a result. Keep every workflow on the same digest when you bump
one.

## zizmor

Workflows are scanned by zizmor through an org-level workflow that is not in
this repository. Two things to know:

- It only scans workflows your PR touches, so a file nobody edits can carry
  findings indefinitely. Editing several workflows at once will surface
  unrelated pre-existing findings.
- Medium and High findings fail the `zizmor-output` check. Most can be
  suppressed with `# zizmor: ignore[rule-name]`, but the mandatory set cannot:
  `bot-conditions`, `insecure-commands`, `known-vulnerable-actions`,
  `template-injection`, `unpinned-uses`, `unredacted-secrets`,
  `unsound-condition`.

Prefer fixing over suppressing where the fix is small.
