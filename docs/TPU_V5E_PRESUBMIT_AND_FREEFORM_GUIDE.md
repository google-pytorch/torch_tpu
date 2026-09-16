# Run the TPU v5e Presubmit in 10 Minutes Instead of an Hour

The single-chip v5e presubmit takes 54–83 minutes on the shared GitHub `ct5lp`
runner. The standing fleet of 28 Cloud TPU v5e chips in `rbe-tpu-oss` can run
the same tests in about 10 minutes.

This guide explains how to run presubmits against your PRs or execute arbitrary
test targets across the fleet using the GCS mailbox runner.

--------------------------------------------------------------------------------

## Prerequisites

Before running tests, ensure you have:

1.  **GCP Project Access**: Membership in `torchtpu-dev@google.com` or
    `xla-torchtpu-team@google.com` with read/write access to
    `gs://torch-tpu-mailbox-eval`.
2.  **gcloud Credentials**: Run `gcloud auth application-default login` so your
    local environment has active Application Default Credentials (ADC).
3.  **GitHub CLI (for PR presubmit)**: `gh` logged in (`gh auth status`).

--------------------------------------------------------------------------------

## 1. Running Presubmits on a Pull Request

Run `scripts/rbe_tpu/rbe_tpu_presubmit_pr.sh` pointing to your PR:

```bash
scripts/rbe_tpu/rbe_tpu_presubmit_pr.sh --pr <PR_NUMBER> --mode replacement --add-label
```

Leave it running. It will execute the full presubmit suite across the 28 chips
and publish results directly to GitHub.

### Which Mode and Label Do I Want?

<!-- mdformat off(this file is mirrored to GitHub; keep a plain GFM table) -->

| Mode | Label Applied | Status Context | When to Use |
| :--- | :--- | :--- | :--- |
| `--mode replacement` | `ci:replace-tpu-v5` | `Presubmit on linux-x86-ct5lp-224-8tpu` | You want the fast run to satisfy the gating check. |
| `--mode shadow` | `ci:relay-tpu-v5` | `TPU v5e relay (shadow)` | Advisory run. Runs alongside normal CI without affecting merge gating. |

<!-- mdformat on -->

Both modes run the identical test suite on the identical hardware.

> **The label does not cancel the ct5lp job.** `presubmit.yml` never reads
> `ci:replace-tpu-v5`; the ct5lp matrix entry is not gated on it. What the label
> does is trigger the `hardware_handoff` job in `test_rbe_opt_in.yml`, whose
> first step holds the required check open with a pending status so branch
> protection cannot go green before the fleet run reports. The real 8-TPU job
> still runs to completion on the shared runner. What you save is waiting for
> it, not the runner time itself.

In `replacement` mode the script refuses to run if it cannot apply the label.
Publishing a green verdict under the gating context while the normal workflow is
unaware of the fleet run would leave two rows with the same name on your PR, so
the script exits instead of posting. Add the label by hand and re-run, or use
`--mode shadow`.

--------------------------------------------------------------------------------

## 2. Running Arbitrary Bazel Tests (No PR Needed)

You do not need a pull request to run tests on the fleet. Use
`scripts/rbe_tpu/rbe_tpu_bazel_test.sh`:

```bash
# Run a single target
scripts/rbe_tpu/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test

# Run with custom bazel flags after `--`
scripts/rbe_tpu/rbe_tpu_bazel_test.sh //tests/pallas:pallas_test -- --runs_per_test=20

# Run an entire package
scripts/rbe_tpu/rbe_tpu_bazel_test.sh //tests/pallas/...

# Run with a tag filter
scripts/rbe_tpu/rbe_tpu_bazel_test.sh --scope //tests/... --filter presubmit-v5
```

Patterns go through `bazel query tests(...)`, so only test rules are selected.
Any flags after `--` pass straight to `bazel test`.

--------------------------------------------------------------------------------

## 3. Running from Cloudtop, GCE VMs, or GitHub Actions

Unlike direct SSH relays, the mailbox runner requires no direct SSH ingress into
the TPU VMs:

-   **No `corp-ssh-helper` needed**: Work and artifacts transfer entirely
    through `gs://torch-tpu-mailbox-eval` over HTTPS.
-   **Any network**: Drivers can run from a Cloudtop, a GCE VM in any
    project/VPC, or a CI runner without needing VPC peering or firewall
    exceptions.
-   **Concurrent runs**: Leases use atomic GCS object generation preconditions
    (`bindings/<tpu>.json`). Multiple developers can run tests simultaneously;
    idle chips are claimed on-demand without lock collisions.

--------------------------------------------------------------------------------

## 4. Known Behaviors and Caveats

1.  **Single-chip only**: Each leased VM has 1 chip (`v5litepod-1`). Multi-chip
    targets (e.g. tagged `requires-tpu-v5lite:8`) are automatically skipped.
2.  **Known test flake**:
    `test_kernel_compiled_mode_donation_invalidates_deferred_op` exhibits a
    ~1-in-3 flake on v5e tracked in [b/561202917](http://b/561202917).
3.  **Memory overcommit**: TPU error tests (`tpu_errors_test`) require
    `vm.overcommit_memory=1` on the VM so the kernel lets the TPU runtime catch
    allocation bounds rather than aborting early on the CPU.
