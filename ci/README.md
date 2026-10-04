# TorchTPU CI scripts

This directory holds the scripts behind the TorchTPU continuous integration. The
GitHub Actions workflows in `.github/workflows/` call them, and the `nox`
sessions in `noxfile.py` let you run most of them locally (`nox -l` lists the
sessions).

## CI network robustness

Every CI run downloads from PyPI and GitHub, and either can reset a connection
for a moment. One file, `ci/tools/network_policy/policy.env`, defines how `uv`,
`pip`, and `curl` retry and time out, and every job inherits it from a single
activation step. You don't configure retries per step, per tool, or per
workflow.

### How a job inherits the policy

Every job that checks out the repository runs the composite action right after
its first `actions/checkout` step:

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
- uses: ./.github/actions/network-policy
```

The action runs `ci/tools/network_policy/activate.sh`, which exports the policy
to every later step of the job through `GITHUB_ENV`, including the steps of
composite actions the job calls (`noxfile.py` runs the same script for local
`nox` sessions):

-   `uv` and `pip` read `UV_HTTP_RETRIES`, `UV_HTTP_TIMEOUT`, `PIP_RETRIES`, and
    `PIP_TIMEOUT` natively. This also covers the `uv pip install` calls inside
    `nox` sessions.
-   `curl` reads a `.curlrc` that the script renders from the `DOWNLOAD_*`
    values; `CURL_HOME` points at it.

If a job checks out the repository under another path (`with: path:`), run the
script directly, because GitHub Actions resolves local actions only from the
workspace root:

```yaml
- run: <path>/ci/tools/network_policy/activate.sh
```

### Change a value

Edit `ci/tools/network_policy/policy.env`. Each key has a comment that says
which tool reads it. Locally, `noxfile.py` only fills in the keys you haven't
set, so your own `UV_*`/`PIP_*` settings survive.

Don't set these values anywhere else. `ci/tools/network_policy_test.py`, part of
the `test_ci_tools` nox session, fails when a workflow, action, script, or
`noxfile.py` hard-codes a retry or timeout number, passes `--retry`-style flags,
or disables curl's configuration file.

### Add a job

1.  Check out the repository and add `- uses: ./.github/actions/network-policy`
    as the next step.
2.  If the checkout uses `sparse-checkout`, include `ci/tools` and
    `.github/actions/network-policy` in it.
3.  If the job never checks out the repository, add it to
    `_JOBS_WITHOUT_CHECKOUT` in `ci/tools/network_policy_test.py` together with
    the reason.

### What the policy doesn't cover yet

The policy reaches `uv`, `pip`, and `curl` only. `git` and `gh` calls, `apt`,
the Python scripts that use `urllib`, the Kokoro and Docker flows, and Bazel
(which has its own retry defaults and mirrors) run with their default behavior.
Retry network operations only: a test, a build, or a write to a remote system
never runs twice, because a retry there hides a real failure or duplicates a
side effect.
