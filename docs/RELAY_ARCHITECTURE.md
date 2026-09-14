# TPU v5e relay: architecture

How the relay gets a bazel test onto a physical Cloud TPU v5e chip, and why it
is built the way it is. Audience is engineers who need to change it. For how to
*run* it, see [CI_PRESUBMIT_AND_RBE_GUIDE.md](CI_PRESUBMIT_AND_RBE_GUIDE.md).

## The problem

The `presubmit-v5` suite is 57 targets that need real v5e silicon. The
GitHub-hosted runner gets one 8-chip host and takes about an hour, most of it
compiling. Compilation is embarrassingly parallel and does not need a TPU. Test
execution needs a TPU and almost no CPU.

So: send the compiles to a few hundred RBE workers, and fan the test actions out
across a standing fleet of single-chip TPU VMs. 57 targets in about 7 minutes on
28 chips.

## Three tiers

```
  tensorflow-testing RBE            driver (workstation or GCE VM)       rbe-tpu-oss
  ~200 concurrent workers           exactly one                          28 v5litepod-1 VMs
  ──────────────────────            ──────────────────────────────       ──────────────────
  compile actions          ───────▶  bazel test                 ──ssh──▶  execute test binary
  C++, LLVM, XLA, protos   artifacts  --run_under=relay_test_runner.sh    on a real chip
  ephemeral, stateless               --strategy=TestRunner=local          standing, persistent
                                     --local_test_jobs=<vm count>
```

The RBE workers never contact a TPU. They have no idea TPUs are involved. Every
byte that reaches a chip goes through the driver.

### Why the driver is the hub

`--config=ci_tpu_v5_relay` is `ci_tpu_v5_presubmit` plus one flag:

```
common:ci_tpu_v5_relay --config=ci_tpu_v5_presubmit
common:ci_tpu_v5_relay --remote_download_outputs=all
```

The stock presubmit config inherits `--remote_download_minimal` from `:ci`,
which leaves outputs on the RBE cluster. The relay needs the runfiles trees on
local disk so it can tar them to the VMs, so it asks for all of them.

That is the whole config delta. Matching the presubmit's RBE instance and tag
filters is deliberate: a shadow run is only meaningful if the only variable is
where tests execute.

Consequence worth naming: **every byte a VM receives leaves one machine's
uplink.** The weight is not the per-test payload — that has been a few hundred
KB since `_main/csrc` moved into the base layer. It is base staging:

| Layer tarball | Size, compressed | Moves when |
| :--- | ---: | :--- |
| `deps_*.tar.gz` | ~694 MB | `MODULE.bazel.lock` changes |
| `solib_*.tar.gz` | ~405 MB | any code change |

Sizes measured off `/tmp/torch_tpu_relay/base_tarballs` on a driver. The
per-fleet numbers are those times 28, not a measured transfer: ~31 GB out of
one uplink when both layers are cold, ~11 GB on an ordinary change where only
`solib` moves. It has not bitten us at 28 chips. It is the first thing that
breaks if the fleet grows, and it is why a GCE driver inside the VPC has more
headroom than a corp workstation going through `corp-ssh-helper`.

## One test action, end to end

Bazel believes it is running tests locally. `--run_under` replaces the test
binary with [`relay_test_runner.sh`](../ci/tools/relay_test_runner.sh):

| # | Step | Mechanism |
| :-- | :--- | :--- |
| 1 | Lease a VM | `flock -n` on `<session>.env.lock`, first unlocked file in `$TPU_SESSION_POOL` wins |
| 2 | Check the socket | `ssh -O check`; reconnect if dead, quarantine + classify if that fails |
| 3 | Probe the base cache | `test -d`, catches a rebooted VM before spending a payload on it |
| 4 | Push the executor | `cat >`, guarded by a `sha256sum` comparison |
| 5 | Probe the payload cache | `test -d` over the existing master, a few ms against ~20s of tar |
| 6 | Stream the payload | `tar -czhf - . \| ssh ... remote_tpu_executor.sh` |
| 7 | Run | `timeout --foreground` around the ssh, `$TEST_TIMEOUT` or 900s |
| 8 | Collect and release | One round trip pulls `test.exitcode` + `test.xml`, moves the sandbox aside, `setsid rm -rf` in the background; the lock drops when the fd closes |

Concurrency is `--local_test_jobs=<vm count>`, so bazel never has more test
actions in flight than there are chips to lease.

### No daemons

There is no server on either side. Every hop rides an OpenSSH ControlMaster:

```bash
setsid ssh -M -N -f -S "$control_path" -o ControlPersist=4h \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$remote"
```

`setsid` matters. Whoever provisioned the VM has usually exited by the time
tests run, and a master left in that process group dies with it.

The only thing installed on a TPU VM is
[`remote_tpu_executor.sh`](../ci/tools/remote_tpu_executor.sh), a shell script.

Session files in the pool carry `TPU_IP`, `SSH_USER`, `SSH_CONTROL_PATH`,
`TPU_NAME`, `TPU_ZONE`.

## Caching: three layers, three lifetimes

| Layer | Contents | Key | Path | Lifetime |
| :--- | :--- | :--- | :--- | :--- |
| Base | Hermetic CPython, pip wheels (torch + libtpu), shared solibs, `_main/csrc` | Content hash, split `deps` / `solib` | `/tmp/torch_tpu_relay/base` | Across runs |
| Payload | One target's own runfiles | Per-run id | `/tmp/torch_tpu_relay/payloads/<key>` | One run |
| Sandbox | One shard's working tree | — | `/tmp/torch_tpu_relay/sandboxes/<id>` | One action |

**Base**, staged by [`stage_relay_base.sh`](../ci/tools/stage_relay_base.sh).
A couple of GB every test shares. Two layers because they move at different
rates: `deps` when `MODULE.bazel.lock` moves, `solib` on every code change.
Content-addressed with a symlink directory per run — a fixed path was fine for
one person, but on a shared pool the second run would overwrite the libraries
the first run's tests are executing against.

The single biggest win is excluding `_main/csrc`. Without the exclusion,
`libpywrap_torch_tpu_common.so` is **493 MB of a 499 MB payload**, identical for
every test and re-sent by every test. That 499 MB is the pre-exclusion size and
is why the base layer exists — it is not what ships now. Commit `61204558` moved
`_main/csrc` into the solib layer and `gru_test`'s payload went **499 MB →
189 KB**, gzip **14.7s → 0.2s**. Per-test payloads today are a few hundred KB.

**Payload.** Every shard of a target ships the same tree and the big targets
have 30-50 shards, so the tree is unpacked once and `cp -al` hardlinks the rest.
About 20s down to under a second.

**Sandbox.** Base is symlinked in at the sandbox root, not copied.

### The payload is three concatenated tars

The runfiles tree mixes link types that need different handling:

1. Everything outside the venv, **dereferenced** (`-h`). Bazel points these at
   absolute paths in the local output base, meaningless on the VM.
2. The venv itself, **links intact**. Its `site-packages` is relative links up
   into the dependency repos, which resolve once base is linked in.
3. The few absolute links inside the venv that do not point at a staged repo —
   `pyvenv.cfg`, the bazel `.pth` files. Real content, must be dereferenced.

> [!WARNING]
> The three sets must stay strictly disjoint. GNU tar 1.34, which the TPU image
> ships, will not replace an existing symlink with a regular file — it keeps the
> link and still exits 0. First archive to write a name wins, so (2) has to skip
> everything (3) is about to send.

Deliberately *not* in set 3: the ~41 venv links into `external/` naming torch
and libtpu. Dereferencing those drags 1.7 GB through every test action; the
executor repoints them at the base cache instead.

## Fleet lifecycle

> [!WARNING]
> **The standing fleet is on-demand, not Spot, despite the names.**
> `VM_NAME_PREFIX` is hardcoded to `spot-tpu-v5e-` and does not change when `up`
> is given `--on-demand`, so on-demand VMs still come out named `spot-*`. Do not
> infer the scheduling model from the name. Check it:
>
> ```
> gcloud compute tpus tpu-vm describe <name> --project=rbe-tpu-oss \
>   --zone=<zone> --format='value(schedulingConfig)'
> ```
>
> Empty means on-demand. As of 2026-09-14 all 28 VMs return empty and have been
> up continuously since 2026-09-11.

This matters for two reasons. Runs are more reliable than a Spot fleet would
suggest, so a mid-run disappearance should be treated as a real anomaly worth
investigating rather than normal background noise. And the fleet costs
on-demand rates while sitting idle, which is a standing bill, not a
per-run one.

The preemption handling below is still correct and worth keeping — `up` creates
Spot VMs by default, so a fleet built without `--on-demand` genuinely can be
reclaimed.

[`spot_tpu_fleet.sh`](../scripts/spot_tpu_fleet.sh) separates creating hardware
from borrowing it:

- `up` / `down` create and delete VMs. `up` arms a detached deadline so a fleet
  cannot outlive a crashed orchestrator.
- `attach` / `detach` borrow a fleet somebody else brought up. Never creates,
  never deletes, never arms a deadline.

**Everything in CI and every relay driver uses attach/detach only.** A finished
job must not be able to delete hardware another job is using.

**Claims.** `attach` leaves a marker on each VM it takes and walks past VMs
somebody else holds. Without it, two people attaching at once both take the
whole fleet and run two tests per chip. Claims age out after 4h.

**Queue.** With `--wait`, attach is all-or-nothing and stands in line. A run is
seven minutes, so waiting for all 28 chips beats splitting them and both going
slowly. The line lives on one VM, ordered by arrival; a waiter that goes quiet
for two minutes loses its place.

**Guards on the widened search.** `--name-prefix` widens which VMs attach will
consider, but never widens what `down` deletes — the orphan sweep stays pinned
to the creation prefix. `--accelerator-type` defaults to `v5litepod-1` because
the relay leases one chip per action; attaching a v5p or a multi-chip host burns
a full timeout per test and reports as an ordinary failure.

## Address modes

```
--ip-mode != auto          -> use it verbatim
corp-ssh-helper on PATH    -> external   (short-circuits, no probe)
metadata server answers    -> internal
otherwise                  -> external   (and the error names what is missing)
```

The ordering is load-bearing. **A cloudtop is itself a GCE VM whose metadata
server answers, but it is not in `rbe-tpu-oss`'s VPC.** Probe first and every
cloudtop run resolves to internal and hangs. There is a test named for this.

Internal mode does not fall back to the external address. From inside the VPC
that address is firewall-denied, so a fallback trades a clear "no address" error
for a ten-minute hang.

Resolution happens once in `parse_args`, in the parent shell — `attach` forks a
job per VM, so lazy memoization would not survive.

Why internal works at all, from
`gcloud compute networks get-effective-firewalls default --project=rbe-tpu-oss`:

```
org-firewall  priority 1-31  ALLOW INGRESS   (#24 = 10.0.0.0/8, 100.64.0.0/10, ...)
org-firewall  priority 32    DENY  INGRESS   0.0.0.0/0
org-firewall  priority 33    DENY  INGRESS   ::/0
org-firewall  priority 34    ALLOW EGRESS    0.0.0.0/0   tcp, udp, icmp
org-firewall  priority 37    DENY  EGRESS    0.0.0.0/0
org-firewall  priority 38    DENY  EGRESS    ::/0
```

RFC1918 is allowed at 24, ahead of the blanket deny at 32. TPU VMs sit on
`10.164.0.0/20`. [`relay_ssh.sh`](../ci/tools/relay_ssh.sh) needs no change —
the corp `ProxyCommand` in `/etc/ssh/ssh_config` only matches GCP external
ranges.

**Ingress is shut, egress is open.** The allow at 34 covers all of tcp, udp and
icmp to `0.0.0.0/0` and sits ahead of the denies at 37 and 38, so a TPU VM can
open a connection to anything — a blob store, a cache, an artifact bucket —
with no policy change. Nothing outside the allowed ingress ranges can open one
to it. That asymmetry decides what is cheap to build: anything where the VM
pulls needs no approval, anything where an outside worker pushes needs an
exception on a policy we do not own.

> [!NOTE]
> Known gap: a GCE VM in a *different* VPC resolves to `internal` and then fails
> to connect. `auto` cannot tell "on GCE" from "on GCE in the right VPC". Escape
> hatch is `--ip-mode external`.

## Failure handling

**Preemption vs. local fault.** These are kept separate on purpose. Collapsing
them once reported 64 shards as preempted against a VM that was `READY` the
whole time — the fault was local and the report blamed Google Cloud. So
`check_preemption` distinguishes "the VM is gone" (`PREEMPTED` / `TERMINATED`)
from "I could not ask" (gcloud failed), and says which.

**Reboot detection.** A rebooted VM comes back with `/tmp` emptied, so the base
cache is gone. SSH still works, the node still reports `READY`, and the test
runs without an interpreter and dies with a bare exit 1. The `test -d` probe at
step 3 catches this and quarantines the session.

**Quarantine.** A bad session gets a `<session>.env.quarantine` marker and drops
out of the pool. The run continues on the remaining chips.

**Every path writes JUnit XML.** A relay-level failure that produced no remote
XML still gets a synthesized single-case report, so infrastructure faults show
up as named failures instead of an empty result.

## Invariants

Things that will quietly break if changed:

1. The three payload archives must stay disjoint (GNU tar 1.34, above).
2. `--ip-mode auto` must check `corp-ssh-helper` before the metadata server.
3. `--local_test_jobs` must never exceed the VM count.
4. Base cache keys must be content-derived, or concurrent runs corrupt each
   other.
5. CI must use `attach`/`detach`, never `up`/`down`.
6. `ALLOWED_PROJECT` is `rbe-tpu-oss`, enforced in five scripts. Note the
   ambient gcloud default on a corp workstation is often something else.

## Sharing the relay directory between users

`/tmp/torch_tpu_relay` on each VM is shared by everyone who uses the fleet. It
used to be created by a plain `mkdir -p`, so it ended up owned by whoever
attached first with no write permission for anyone else. The second person's
`mkdir` of the claim directory failed `EACCES`, the script read *any* `mkdir`
failure as having lost the race, and reported `The fleet is busy: 28 of 28
VM(s) held`. `--wait` then polled forever, because waiting cannot change a
permission bit. That cost somebody twenty minutes against a completely idle
fleet.

Three things changed:

| Change | Where |
| :--- | :--- |
| Tell a lost race apart from a permission wall, and print the mode and owner found | `claim_script` in [`spot_tpu_fleet.sh`](../scripts/spot_tpu_fleet.sh) |
| Stop `--wait` retrying a `CLAIM_ERROR`, and name the command that fixes it | `die_if_claims_are_broken` |
| Create every shared directory `0777`, including the claim and queue-ticket directories themselves | all five `mkdir` sites |

The mode is `0777` with **no sticky bit**, on purpose. Sticky lets only the
owner delete an entry, and three things here delete entries another user
created: reaping a claim past its TTL, `--force-claim`, and the base store
reaper in `stage_relay_base.sh`. Sticky would turn a stale four-hour claim into
a permanent one. The tradeoff is that anyone can delete anyone else's cache or
sandbox, which on an ephemeral `/tmp` shared by a few trusted people is the
intended behaviour.

Creating the directory correctly only helps the first time. A VM whose tree
already belongs to one user is repaired with:

```
scripts/spot_tpu_fleet.sh prepare --zone europe-west4-b
```

which installs `/etc/tmpfiles.d/torch-tpu-relay.conf` and runs
`systemd-tmpfiles --create`. Because systemd re-applies it at boot, the tree
comes back root-owned and world-writable every time `/tmp` is wiped, and no
user ever owns it. The rule deliberately does **not** list `claim`: that
directory *is* the mutex, and pre-creating it would make every VM read as
permanently held.

## Known rough edges

Flagged, not fixed:

- **`DEFAULT_ZONES` does not describe the fleet.** It is
  `(europe-west4-b us-central1-a)`
  ([`spot_tpu_fleet.sh:99`](../scripts/spot_tpu_fleet.sh#L99)) — one empty zone
  and one real one, missing three of the four zones the 28 VMs actually live in.
  So a plain `status` reports 8 of 28. `--zone` now works, so there is a flag
  that fixes it, but the default is still wrong. Left alone until the full
  reservation lands and the zone list changes anyway.
- **ControlMaster socket path** `/tmp/tpu_cm_${ip}_22_${ssh_user}` has no local
  UID. Two users on one shared host collide on a `0600` socket. More likely now
  that GCE drivers are supported.
- **Queue host selection** picks the first candidate by name, so callers with
  different `--zone` lists pick different queue hosts and miss each other's
  tickets. Costs ordering, not safety.
- **Thundering herd** when several waiters see `ahead==0` at once. Jitter would
  help.
- **Payload cache publish/read race** in `remote_tpu_executor.sh` — `cp -al`
  against `mv -T`/`rm -rf`. A `.complete` marker, like `stage_relay_base.sh`
  already uses, would close it.
- **`claim_owner`** is interpolated unquoted into remote heredocs; breaks on
  shell metacharacters in usernames.

## Numbers

Measured on PR 3730, 4 zones, 28 chips. The run is **two separate bazel
invocations**, not one invocation with a build phase inside it:

| | | |
| :--- | ---: | :--- |
| Targets | 57 | |
| Build invocation, elapsed | 19.8s | `relay_reports/timing_check/bazel_build.log` |
| Test invocation, elapsed | 424.8s | `bazel_presubmit.log`, critical path 414.9s |
| Reported total | 426s | 56/57 pass |
| Total wall clock | 8m15s | |
| Baseline GitHub runner | ~1 hour | |
| Single small target, cold pool | 63s | end to end |

Both invocations report **zero remote executions and zero remote cache fetches**
— every hit came from the local action cache or the local disk cache. On a warm
driver the RBE workers contribute nothing to these numbers. They earn their keep
on a cold driver or in CI, where the disk cache is empty.

**Inference, not measurement:** the 57 targets report 7,032.8s of test time in
aggregate. Over 28 chips that is a 251s floor if packing were perfect, against
426s actual — so roughly 40% of the run is per-action overhead (lease, ssh,
payload, sandbox setup and teardown) plus the tail where the long shards are
still going and most chips have nothing left to do. The 414.9s critical path
says some of that tail is unavoidable rather than waste.

Set `TORCH_TPU_RELAY_TIMING=1` for per-phase marks on both the host and VM
sides.
