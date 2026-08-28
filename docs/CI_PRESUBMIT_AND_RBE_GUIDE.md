# Developer Guide: CI Presubmits, TPU v5 Bypass & RBE Modes

This document explains how to control CI presubmit test executions in `torch_tpu`, including running Remote Build Execution (RBE) as a shadow run or replacing self-hosted Cloud TPU v5 presubmits with the RBE setup.

---

## 1. Quick Reference: Which Label Should I Use?

Every label needs write access to the repo, which is the point: skipping
hardware tests should take more than editing a PR description.

| Goal | How to Trigger | What Happens |
| :--- | :--- | :--- |
| **Normal presubmits** | Default | Runs CPU, TPU v5 (`linux-x86-ct5lp-224-8tpu`), and TPU v7. |
| **Relay shadow run** | Label `ci:relay-tpu-v5`, then a Googler runs `scripts/relay_presubmit_pr.sh` | Normal presubmits still run. The relay covers the same single-chip v5 targets and reports under its own advisory check. Never blocks the PR. |
| **Relay replacement** | Label `ci:replace-tpu-v5`, then a Googler runs the same script with `--mode replacement` | Drops TPU v5 from the presubmit matrix. The required ct5lp check goes pending until the relay reports, so the relay's verdict gates the PR. CPU and TPU v7 still run. |
| **Full-RBE shadow run** | Label `run-rbe` | Advisory run of the CPU and TPU v5e suites through RBE workers. The TPU leg needs worker pools that do not exist yet. |
| **Standalone bypass** | Label `ci:bypass-tpu-v5` | Drops TPU v5 and runs nothing in its place. For runner outages, or PRs that don't touch v5 code. |
| **Manual dispatch** | Actions UI or `gh workflow run` | `bypass-tpu-v5: true` on `presubmit.yml`, or `hardware_path` + `mode` on `test_rbe_opt_in.yml`. |

> [!IMPORTANT]
> The relay runs on a Googler's corp workstation, not in GitHub Actions. A
> hierarchical firewall above `rbe-tpu-oss` blocks port 22 from every GitHub
> runner. Section 5 has the evidence and the one command to run.

---

## 2. Execution Modes Explained

### Mode 1: Relay shadow run (`ci:relay-tpu-v5`)
- **Use case**: Try the relay on your PR while the ct5lp presubmit stays the
  source of truth. This is the mode to start with.
- **What happens**:
  - `presubmit.yml` runs CPU, TPU v5, and TPU v7 as usual.
  - A Googler runs `scripts/relay_presubmit_pr.sh --pr <PR> --mode shadow`. It
    borrows the standing v5e fleet in `rbe-tpu-oss`, runs the `presubmit-v5`
    single-chip targets, posts the verdict under `TPU v5e relay (shadow)`, and
    comments the full report on the PR.
  - Relay failures are advisory and don't block the merge.
- **What it covers**: the same targets the ct5lp job runs, minus the ones tagged
  `requires-tpu-v5lite:8`. The relay leases one v5litepod-1 VM per test action,
  so a multi-chip test can never pass there. That is 57 of the 77
  `presubmit-v5` targets.

### Mode 2: Relay replacement (`ci:replace-tpu-v5`)
- **Use case**: The self-hosted TPU v5 runners are backed up, or you're
  validating the relay as the gating path for v5.
- **What happens**:
  - `presubmit_job_matrix.sh` leaves `linux-x86-ct5lp-224-8tpu` out of the matrix.
  - `presubmit.yml`'s bypass notice job stands down. It publishes under
    `TPU v5 Bypass Notice` instead of claiming the ct5lp check name,
    because an always-green check next to a real one hides a red relay run.
  - `relay_handoff` publishes `Presubmit on linux-x86-ct5lp-224-8tpu` as
    **pending**, so the PR shows it is waiting on a person rather than missing
    a check.
  - A Googler runs `scripts/relay_presubmit_pr.sh --pr <PR> --mode replacement`,
    which resolves that same check to the relay's real verdict.

> [!WARNING]
> The required check stays pending until someone runs the relay. That is the
> intended failure mode: a gating check that never touched hardware has
> verified nothing, so it must not go green on its own.

> [!NOTE]
> Multi-chip coverage does not move with the label. The 20 targets tagged
> `requires-tpu-v5lite:8` run on neither path while `ci:replace-tpu-v5` is on.
> Don't leave the label on a PR that touches collective ops.

### Mode 3: Full-RBE shadow run (`run-rbe`)
- **Use case**: Exercising the RBE path once TPU-attached worker pools exist.
- **What happens**: `run_tests` builds and tests through RBE with
  `--config=ci_tpu_v5_full_rbe`, on the same self-hosted runner and application
  default credentials the normal presubmits use. Advisory only. The TPU leg
  cannot gate a PR, because `//bazel/platforms:rbe_tpu_v5e` asks for a worker
  pool nobody has provisioned.

### Mode 4: Standalone bypass (`ci:bypass-tpu-v5`)
- **Use case**: The PR only touches CPU, TPU v7, or docs, or the v5 runners are
  down.
- **What happens**: TPU v5 is skipped, nothing runs in its place, CPU and TPU v7
  run as usual. The bypass notice job claims the ct5lp check name so branch
  protection stays satisfied.

---

## 3. Driving it from the command line

```bash
# Relay, advisory. Check out the PR head first.
git fetch origin pull/<PR>/head && git checkout FETCH_HEAD
scripts/relay_presubmit_pr.sh --pr <PR> --mode shadow \
  --zone europe-west4-b --zone us-south1-a --zone us-west1-c --zone us-west4-a

# Relay, gating. --add-label applies ci:replace-tpu-v5 for you.
scripts/relay_presubmit_pr.sh --pr <PR> --mode replacement --add-label \
  --zone europe-west4-b --zone us-south1-a --zone us-west1-c --zone us-west4-a

# See the plan without touching hardware.
scripts/relay_presubmit_pr.sh --pr <PR> --dry-run

# Run presubmits with TPU v5 bypassed and nothing in its place.
gh workflow run presubmit.yml -f bypass-tpu-v5=true

# Full RBE instead of the relay.
gh workflow run test_rbe_opt_in.yml -f hardware_path=full-rbe -f test_suite=tpu_v5e_only
```

The fleet spans four zones and the driver only looks in `europe-west4-b` unless
told otherwise. Naming all four is the difference between 8 VMs and 28, which
is the difference between 20 minutes and 7.

> [!IMPORTANT]
> Until this lands on `main`, none of these scripts exist on your branch. Pull
> them across without moving HEAD, so the verdict still belongs to your code:
> ```bash
> git fetch origin pull/<PR>/head && git checkout FETCH_HEAD
> git fetch origin feat-rbe-presubmit-relay
> git checkout FETCH_HEAD -- scripts/ ci/tools/ .bazelrc
> ```

Adding a label by hand, since `gh pr edit --add-label` does not work on this
repo:

```bash
gh api repos/google-pytorch/torch_tpu/issues/<PR>/labels \
  -f 'labels[]=ci:relay-tpu-v5'
```

### One run at a time, on all of it

The fleet is 28 chips and a run takes seven minutes, so the fastest way through
two runs is back to back on everything, not side by side on half each. A run
therefore takes the whole fleet or none of it, and the next person stands in
line:

```
[spot_tpu_fleet] Waiting for the fleet: 1 run(s) ahead of you
[spot_tpu_fleet] Attached 28/28 VM(s) in /tmp/tpu_pool_you
```

Each VM carries a claim naming the pool that holds it, and the line itself is a
set of tickets on one VM, ordered by arrival. `relay_presubmit_pr.sh` waits by
default.

| Situation | What happens |
| :--- | :--- |
| Somebody else is mid-run | You queue, and go when they detach |
| You got half the fleet | It is handed straight back. Two runs each holding half would wait on each other forever |
| A waiter is killed | Its ticket goes stale after two minutes and the line moves on |
| A run crashed holding VMs | Claims age out after 4 hours (`--claim-ttl`), or `--force-claim` takes them now |
| You would rather fail than wait | `--no-wait` on the driver, or plain `attach` without `--wait` |

> [!NOTE]
> This is the right shape only while the fleet is small. Once there is enough
> capacity for several suites at once, stop queueing and split the fleet
> instead: run N suites on M VMs each.

### If the run comes back 56/57

Check whether the one red target is this:

```
FAIL: test_kernel_compiled_mode_donation_invalidates_deferred_op
      (__main__.TestPallasKernels)
AssertionError: [] has length of 0, expected 1.
```

That one is a known flake on single-chip v5e, tracked in b/561202917. It fails
about a third of the time and has nothing to do with your change. Re-run. Any
other failure is worth reading.

---

## 4. Who can run the relay, and what they need

The relay runs on a **corp workstation or cloudtop**, not on a GitHub runner.
Section 5 explains why. What a Googler needs:

| Requirement | How to get it | Check it |
| :--- | :--- | :--- |
| Membership of `torchtpu-dev@google.com` or `cloud-tpus-dev-team@google.com` | go/membership, or ask in the group | `gcloud compute tpus tpu-vm list --project=rbe-tpu-oss --zone=europe-west4-b` |
| `corp-ssh-helper` on PATH | Standard on cloudtop and corp workstations | `command -v corp-ssh-helper` |
| `gcloud` application default credentials | `gcloud auth application-default login` | `gcloud auth list` |
| The `gh` CLI, authenticated | go/gh-cli | `gh auth status` |
| Write access to the repo | Needed to set labels and post statuses | `gh api repos/google-pytorch/torch_tpu --jq .permissions` |

Both groups hold `projects/rbe-tpu-oss/roles/torchTpuCiRelay`, a custom
borrow-only role:

| Permission | Why |
| :--- | :--- |
| `tpu.nodes.list`, `tpu.nodes.get` | `attach` finds READY VMs and reads their addresses |
| `tpu.nodes.update` | `gcloud compute tpus tpu-vm ssh` writes the caller's public key into the **node's own** metadata |
| `tpu.locations.*`, `tpu.operations.*` | Reading zones and polling operations |
| `serviceusage.services.use` | Billing attribution on API calls |

It deliberately leaves out `tpu.nodes.create` and `tpu.nodes.delete`, so
"borrow, never own" holds in IAM and not just by convention. Nobody running the
relay can destroy the fleet.

To add another group:

```bash
gcloud projects add-iam-policy-binding rbe-tpu-oss \
  --member='group:YOUR-GROUP@google.com' \
  --role='projects/rbe-tpu-oss/roles/torchTpuCiRelay'
```

### The build phase talks to two things that are not `rbe-tpu-oss`

The relay only borrows TPUs from `rbe-tpu-oss`. The bazel build in front of it
reaches two other services on your application default credentials.

| Service | What for | If you cannot reach it |
| :--- | :--- | :--- |
| `projects/tensorflow-testing/instances/default_instance` | Compile actions, via `--config=ci_tpu_v5_relay` | Pass `--bazel-config=""` and build locally. First build is slow, later ones are warm. |
| ResultStore / BES on `ml-oss-rbe-testing` | Build event upload, dragged in by `--config=resultstore_base` | Already off. `relay_presubmit_pr.sh` passes `--bes_backend=` unless you ask for `--upload-results`. |

> [!IMPORTANT]
> Only the CI service account can write to the ResultStore instance. Left on,
> the upload fails **after** the build finishes, bazel still exits non-zero, and
> a green run gets reported as an error. That is why the driver turns it off by
> default. Do not add `--upload-results` unless you know your account can write
> there.

---

## 5. Why the relay runs on your workstation

A hierarchical firewall policy above `rbe-tpu-oss` denies all ingress from
`0.0.0.0/0` at priority 31. The allow rules above it list Google corp and relay
netblocks plus RFC1918. The project's own `default-allow-ssh` rule sits at
priority 1000 in the VPC, which is never reached — hierarchy rules are
evaluated first.

So:

| From | Port 22 to a fleet VM |
| :--- | :--- |
| Corp workstation or cloudtop | **works** — `/etc/ssh/ssh_config` proxies through `corp-ssh-helper --proxy-mode=grue` |
| Anything inside the VPC (RFC1918) | **works** — priority 24 allows it |
| GitHub-hosted runner | blocked |
| The repo's self-hosted runners in `ml-velocity-actions-production` | blocked — different VPC, so traffic arrives from a public NAT address |

Confirm it yourself. A raw TCP connect fails even from a corp machine, while
`ssh` to the same address succeeds, because only the latter picks up the
ProxyCommand:

```bash
ip=$(gcloud compute tpus tpu-vm list --project=rbe-tpu-oss \
  --zone=europe-west4-b --limit=1 \
  --format='value(networkEndpoints[0].accessConfig.externalIp)')
python3 -c "import socket,sys; socket.create_connection((sys.argv[1],22),10)" "$ip"   # times out
ssh -o BatchMode=yes "$(whoami)@${ip}" true                                           # works
```

This also rules out Workload Identity Federation as a fix. A GitHub-hosted
runner could hold a perfectly valid GCP token and still not reach port 22, so
federating an external identity provider into the project would buy nothing.
The WIF pool, provider bindings and repository variables that used to be here
have been removed.

### The one-time infrastructure

**1. A standing v5e fleet.** The relay attaches to VMs that already exist; it
never creates them. Bring the fleet up from a workstation or a long-lived job:

```bash
scripts/spot_tpu_fleet.sh up --size 8 --on-demand \
  --zone europe-west4-b --pool /tmp/tpu_pool \
  --deadline-minutes 0     # 0 disables the auto-teardown deadline
```

Check what is running with `scripts/spot_tpu_fleet.sh status --pool /tmp/tpu_pool`.

> [!CAUTION]
> These VMs bill until deleted. A standing fleet is a standing cost. Tear it
> down with `scripts/spot_tpu_fleet.sh down --pool /tmp/tpu_pool --zone <zone>`
> when the experiment ends, and check for strays with
> `scripts/spot_tpu_manager.sh reap --dry-run`.

**2. Borrowing capacity you did not create.** `attach` defaults to VMs named
`spot-tpu-v5e-*`. To borrow from a reservation whose VMs are named something
else, widen the search:

```bash
scripts/spot_tpu_fleet.sh attach --pool /tmp/tpu_pool \
  --name-prefix '' --accelerator-type v5litepod-1 --zone europe-west4-b
```

`--name-prefix` only applies to `attach`. The teardown path stays pinned to
`spot-tpu-v5e-`, so a widened search can never widen a delete.

**3. Branch protection.** Nothing to change. A replacement run posts a commit
status under the existing `Presubmit on linux-x86-ct5lp-224-8tpu` context, so
the required check you already have keeps working and now carries the relay's
verdict.

**4. The labels.** Create them if they do not exist:

```bash
gh label create ci:relay-tpu-v5 --repo google-pytorch/torch_tpu \
  --description "TPU v5e SSH relay runs as an advisory shadow check"
gh label create ci:replace-tpu-v5 --repo google-pytorch/torch_tpu \
  --description "Drop the ct5lp presubmit and gate on the TPU v5e relay instead"
```

### Running it

Check out the PR head, then point the driver at the PR:

```bash
git fetch origin pull/<PR>/head && git checkout FETCH_HEAD

# Advisory. Reports under "TPU v5e relay (shadow)". Never blocks the PR.
scripts/relay_presubmit_pr.sh --pr <PR> --mode shadow

# Gating. Reports under "Presubmit on linux-x86-ct5lp-224-8tpu".
scripts/relay_presubmit_pr.sh --pr <PR> --mode replacement --add-label
```

What the driver does, in order:

1. Refuses to start unless `corp-ssh-helper` is present and the working tree is
   at the PR's head commit. Reporting a verdict for code you did not run is
   worse than not reporting one. `--skip-head-check` overrides.
2. For a replacement run, checks the PR carries `ci:replace-tpu-v5`, or adds it
   with `--add-label`. Without that label `presubmit.yml` still schedules the
   real ct5lp runner and both would report under the same check name.
3. Posts a `pending` status so the PR shows the run is under way.
4. `attach`es to the fleet, runs the suite, and `detach`es. It never calls `up`
   or `down`, so it cannot create or delete anyone's hardware.
5. Reads `presubmit_summary.json` and posts `success`, `failure`, or `error`,
   then upserts a PR comment with the full report. It does not trust the exit
   code alone: a run that dies before writing a summary reports `error`, not a
   pass.

Useful flags: `--zone` (repeatable), `--jobs N`, `--pool DIR`, `--dry-run`,
`--no-report` to run the suite without touching the PR.

### What GitHub still does

`relay_handoff` in `test_rbe_opt_in.yml` needs no GCP access at all. On a
replacement PR it publishes the `Presubmit on linux-x86-ct5lp-224-8tpu` check as
**pending**, with the exact command in the job summary, so the PR reads as
"waiting on a person" instead of "missing a check". It leaves a status alone if
that commit already has a verdict, so adding a label later cannot knock a
finished run back to pending.

Fork PRs get a read-only token, so the pending status cannot be published there.
The job logs a warning instead, and a maintainer runs the relay.

---

## 6. Running the suite on Spot TPUs locally

`presubmit.yml` and `test_rbe_opt_in.yml` are the CI paths. To run the same test
targets against real Spot TPU v5e VMs from a workstation, use the relay:

```bash
# Single VM, one test at a time (a v5litepod-1 has one chip).
scripts/run_presubmit_v5_relay.sh

# See what would run without provisioning anything.
scripts/run_presubmit_v5_relay.sh --dry-run

# Spread the suite over a fleet. Each test leases one VM for its lifetime.
scripts/spot_tpu_fleet.sh up --size 8 --pool /tmp/tpu_pool
scripts/run_presubmit_v5_relay.sh --session-pool=/tmp/tpu_pool --jobs 8
scripts/spot_tpu_fleet.sh down --pool /tmp/tpu_pool --zone europe-west4-b

# Borrow a fleet somebody else brought up, then hand it back.
scripts/spot_tpu_fleet.sh attach --pool /tmp/my_pool --zone europe-west4-b
scripts/run_presubmit_v5_relay.sh --session-pool=/tmp/my_pool --jobs 8
scripts/spot_tpu_fleet.sh detach --pool /tmp/my_pool
```

On a machine with a cold Bazel output base, add `--bazel-config ci_tpu_v5_relay`
so the compile actions go out to RBE. Plain `--config=ci` will not do: it sets
`--remote_download_minimal`, and `stage_relay_base.sh` builds the base cache by
reading the runfiles trees off `bazel-bin`, which minimal downloads never
materialise.

> [!CAUTION]
> `spot_tpu_fleet.sh down` is not optional if you brought the fleet up yourself.
> Spot VMs bill until deleted. Check for strays with
> `scripts/spot_tpu_manager.sh reap --dry-run`.

> [!NOTE]
> Spot capacity is the real limit, not quota. As of this writing only
> `europe-west4-b` has a v5e reservation with room; `us-central1-a` regularly
> returns "no more capacity" and `us-east5-a` has no reservation at all. Asking
> for 12 VMs can easily get you 4, so `up` is resumable — re-run it to fill the
> gaps.

---

## 7. SSH Relay Architecture

We use a Bazel `--run_under` wrapper (`ci/tools/relay_test_runner.sh`) that intercepts test actions on the host, leases a Cloud TPU v5e VM from a pool via `flock`, and runs the test in a sandbox on the remote VM (`ci/tools/remote_tpu_executor.sh`).

### Performance and the Base Cache

A real 57-target run expands to 330 test actions holding 7,895s of test body time. With 8 VMs, a perfect floor is 987s. The original wall time was 3,015s — a 32.7% efficiency caused by 49.2s of relay overhead per action. The pool averaged 2.58 active VMs; adding more VMs does not speed up the run if they stay locked in I/O.

The overhead was mostly the `libpywrap_torch_tpu_common.so` extension module. At 493 MB out of a 499 MB payload, it was identical for every test but got packaged and shipped repeatedly.

`ci/tools/stage_relay_base.sh` now pushes `_main/csrc` alongside the shared C++ solibs into the base cache on the VM. `ci/tools/remote_tpu_executor.sh` symlinks `${BASE_CACHE}/csrc` directly into the sandbox workspace root on a cache miss, and `relay_test_runner.sh` excludes it from the per-action payload. The payload packing step dropped from 14.63s (120 MB gzipped) to 0.066s (200 KB). The base cache layer stamp already hashes file size and mtime, so a rebuilt extension module re-stages automatically.

### Bazel Execution Strategy

`.bazelrc` now uses `--strategy=TestRunner=local` to pin only test actions to the relay runner. The previous flag, `--spawn_strategy=standalone,local`, inadvertently forced compilation and linking jobs onto the host instead of sending them to RBE.

### Watchdogs and Quarantine

We removed the in-guest shutdown watchdogs from `spot_tpu_manager.sh`. Guest-side `shutdown -h` tells the TPU service to reboot the VM, not stop billing. Reboots wipe `/tmp` (deleting the base cache) and kill the SSH control master.

Before, a broken VM failed tests in milliseconds, released its `flock` instantly, and won the race for the next lease. We saw one rebooted VM instantly fail 64 shards while seven healthy VMs sat idle.

Now `relay_test_runner.sh` writes a `.quarantine` marker next to the session file of any VM answering without a base cache or SSH connection. The fleet scripts skip it from then on.

To catch orphaned billing, `scripts/spot_tpu_fleet.sh up` arms a detached host-side deadline (default 180 min, disable with `--deadline-minutes 0`). The timeout runs `down`, which sweeps both zones explicitly for unknown VMs named `spot-tpu-v5e-*`.

### Unrunnable Multi-Chip Targets

The relay leases one `v5litepod-1` VM per action. Multi-chip targets will hang until they hit their timeout, then fail. The relay excludes `requires-tpu-v5lite:8` by default. 20 of the 77 v5 targets need 8 chips, leaving 57 targets holding the 330 shards.
