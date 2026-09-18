# Native Remote Build Execution on Cloud TPU Architecture

This document explains the architecture, networking, and deployment of native
Remote Build Execution (RBE) for `google-pytorch/torch_tpu` on physical Cloud
TPU hardware.

--------------------------------------------------------------------------------

## 1. Problem & Context

Hardware tests for `torch_tpu` require physical TPU chips (`/dev/vfio/0`). Until
now, CI relied on two constrained models:

1.  **A Single Shared Runner (`linux-x86-ct5lp-224-8tpu`):** A single physical
    machine runs presubmit jobs sequentially. When multiple developers push PRs,
    tests queue up behind one another, stalling pull requests.
2.  **GCP Managed RBE (Foundry):** Google Cloud RBE only provisions standard CPU
    worker pools. It does not provide Cloud TPU instances as RBE execution
    backends.
3.  **The GCS Mailbox Workaround:** To parallelize across the project's 28
    standing Cloud TPU v5e VMs, we previously built a custom GCS mailbox
    protocol. Bazel wrapped tests in a Python harness that bundled inputs,
    zipped runfiles, and uploaded task descriptors to a Cloud Storage bucket.
    TPU daemons continuously polled GCS, acquired locks, ran tests, and wrote
    results back. While functional, this required ~3,800 lines of custom wire
    protocol, heartbeat management, and polling logic.

--------------------------------------------------------------------------------

## 2. Solution: Native REAPI v2 via Buildbarn

Instead of building custom task queues over object storage, we run an
open-source Remote Execution API (REAPI v2) service using Buildbarn. Bazel
speaks REAPI v2 natively.

```mermaid
flowchart TD
    subgraph Client ["Client (Cloudtop / GitHub Actions Runner)"]
        Bazel["Bazel Client<br/>bazel test --config=rbe_native_tpu<br/>--strategy=TestRunner=remote"]
        Tunnel["Cloud IAP TCP Tunnel<br/>localhost:8980 -> rbe-tunnel-gateway:8980"]
        Bazel --> Tunnel
    end

    subgraph Gateway ["VPC Gateway (us-east5-a)"]
        GW["rbe-tunnel-gateway (10.202.0.13:8980)<br/>Debian 12 GCE VM with socat proxy"]
        Tunnel --> GW
    end

    subgraph ControlPlane ["GKE Control Plane & CPU Workers (rbe-tpu-oss-east5-cluster)"]
        FE["bb-frontend (NodePort 30980)<br/>REAPI v2 Client Surface & CAS/AC Router"]
        SCHED["bb-scheduler (NodePort 30983)<br/>Matches platform tags & maintains build queue"]
        STOR["bb-storage (ClusterIP 8981)<br/>CAS (32 GB disk) & Action Cache (1 GB disk)"]
        CPUWorkers["224-CPU Worker Pool (7 Nodes x 32 CPUs)<br/>Executes remote C++/Python compilations"]

        GW --> FE
        FE --> SCHED
        FE --> STOR
        SCHED --> CPUWorkers
    end

    subgraph Fleet ["Standing Cloud TPU v5e Fleet (28 Nodes Across 4 Zones)"]
        subgraph Node ["Cloud TPU VM (spot-tpu-v5e-*)"]
            Worker["bb_worker daemon<br/>(concurrency: 1, maxCache: 20 GB)"]
            Runner["bb_runner daemon<br/>(unix:///tmp/torch_tpu_rbe/runner.sock)"]
            Hardware["Physical TPU Chip<br/>(/dev/vfio/0)"]

            Worker -->|"Outbound gRPC long-poll<br/>10.202.0.3:30983"| SCHED
            Worker -->|"Fetch inputs / upload results<br/>10.202.0.3:30980"| FE
            Worker -->|"UNIX Domain Socket"| Runner
            Runner --> Hardware
        end
    end
```

--------------------------------------------------------------------------------

## 3. How It Works

### Step 1: Worker Registration (Inversion of Control)

Each TPU VM runs `bb_worker` as a systemd service. On startup, `bb_worker` dials
outbound to the Buildbarn scheduler at `10.202.0.3:30983` over internal VPC
networking.

-   The worker advertises platform keys:

    ```json
    "platform": {
      "properties": [
        { "name": "Accelerator", "value": "tpu-v5e" },
        { "name": "OSFamily", "value": "Linux" }
      ]
    }
    ```
-   Because workers initiate outbound connections to the scheduler, TPU VMs
    require zero listening ports, zero public IPs, and no incoming firewall
    rules.

### Step 2: Bazel Test Invocation

When a developer or CI runs:

```bash
bazel test --config=rbe_native_tpu //tests/pallas:pallas_test
```

Bazel evaluates `.bazelrc`:

-   `--spawn_strategy=local`: C++ compilations, code generation, and test
    binaries build locally on the client (or pull from local cache).
-   `--strategy=TestRunner=remote`: Only the test execution step dispatches
    remotely over REAPI v2.
-   `--remote_default_exec_properties=Accelerator=tpu-v5e`: Tells the scheduler
    to dispatch this action only to workers with a TPU v5e attached.
-   `--noremote_upload_local_results`: Prevents Bazel from uploading thousands
    of intermediate compiler objects to the CAS over the WAN.

### Step 3: Action Execution on Hardware

1.  The scheduler matches the test action to an idle TPU worker and hands off
    the action digest.
2.  `bb_worker` pulls input runfiles (such as `libtpu.so`, python virtual
    environments, and test scripts) from `bb-storage` into
    `/tmp/torch_tpu_rbe/build/`.
3.  `bb_worker` delegates execution to `bb_runner` via a local UNIX socket.
4.  `bb_runner` runs the test command directly on the host OS as root or runner
    user. The process opens `/dev/vfio/0` directly with hardware performance.
5.  `bb_worker` uploads test outputs (such as `test.xml`, `test.log`) to CAS and
    reports exit code `0` to the scheduler.

### Step 4: Remote Action Caching

Buildbarn stores the test result digest in its Action Cache (AC). If the inputs
and test target do not change, subsequent invocations hit the cache:

-   Fresh execution: ~18 seconds (staging dependencies + running tests on chip).
-   Cached execution: **0.348 seconds** (`(cached) PASSED`).

--------------------------------------------------------------------------------

## 4. Network Layout & Multi-Zone Topology

The control plane lives in `us-east5`. The 28 standing Cloud TPU v5e nodes sit
across four regions in the same GCP VPC network (`default`):

| Region / Zone    | TPU Nodes         | Round-Trip Latency to Scheduler |
| :--------------- | :---------------- | :------------------------------ |
| `us-east5-a`     | GKE Control Plane | Local (<1 ms)                   |
:                  : (`10.202.0.3`)    :                                 :
| `us-south1-a`    | 8 nodes           | ~32 ms                          |
| `us-west4-a`     | 4 nodes           | ~55 ms                          |
| `us-west1-c`     | 8 nodes           | ~68 ms                          |
| `europe-west4-b` | 8 nodes           | ~95 ms                          |

### Runfile Caching

`libtpu.so` is ~700 MB, and the Python virtual environment runfiles total ~2.6
GB. To avoid downloading 2.6 GB over cross-region links on every test:

-   Each TPU VM configures a 20 GB local worker cache
    (`/tmp/torch_tpu_rbe/cache`).
-   Files downloaded from the CAS remain on local disk across actions.
-   Successive tests on the same VM hard-link cached inputs instantly, keeping
    network traffic minimal.

--------------------------------------------------------------------------------

## 5. Repository Configuration

### `.bazelrc`

```ini
# Native Remote Build Execution on Cloud TPU v5e hardware via Buildbarn (REAPI v2)
common:rbe_native_tpu --remote_executor=grpc://127.0.0.1:8980
common:rbe_native_tpu --remote_cache=grpc://127.0.0.1:8980
common:rbe_native_tpu --remote_timeout=600s
common:rbe_native_tpu --noremote_upload_local_results
common:rbe_native_tpu --spawn_strategy=local
common:rbe_native_tpu --strategy=TestRunner=remote
common:rbe_native_tpu --remote_default_exec_properties=Accelerator=tpu-v5e
common:rbe_native_tpu --remote_default_exec_properties=OSFamily=Linux

# CI config matching .github/workflows/test_rbe_opt_in.yml
common:ci_tpu_v5_full_rbe --config=rbe_native_tpu
common:ci_tpu_v5_full_rbe --build_tests_only
common:ci_tpu_v5_full_rbe --color=yes
common:ci_tpu_v5_full_rbe --curses=no
common:ci_tpu_v5_full_rbe --build_tag_filters=-nobuild,requires-tpu-v5lite,requires-tpu-v5lite:8,requires-tpu-v5p,requires-tpu-v5p:4,requires-tpu-v6e,requires-tpu-v7x,requires-tpu-v7x:4
test:ci_tpu_v5_full_rbe --test_output=errors
```

### GitHub Actions Presubmit (`.github/workflows/test_rbe_opt_in.yml`)

CI jobs start a background SSH tunnel to forward port `8980` to the GKE node
(`10.202.0.3:30980`) before launching Bazel, and terminate the tunnel on job
completion.

### CLI Presubmit Helper (`rbe_tpu_presubmit_pr.sh`)

Developers can test any PR branch over native RBE with:

```bash
rbe_tpu/deploy/rbe_tpu_presubmit_pr.sh --rbe --pr <PR_NUMBER> --mode replacement
```

This runs the test suite across the 28-node pool, generates a markdown summary,
and posts the gating commit check `Presubmit on linux-x86-ct5lp-224-8tpu` back
to GitHub.

--------------------------------------------------------------------------------

## 6. Worker Daemon Management (systemd)

Each TPU node runs two systemd services:

1.  **`bb_runner.service`**: Listens on UNIX domain socket
    `/tmp/torch_tpu_rbe/runner.sock`. Executes actions delivered by `bb_worker`.
2.  **`bb_worker.service`**: Maintains the outbound gRPC session to
    `10.202.0.3:30983`, fetches action inputs, passes execution requests to
    `bb_runner`, and uploads results.

```bash
# Check worker status on any TPU node:
sudo systemctl status bb_worker.service
sudo systemctl status bb_runner.service

# View live worker logs:
journalctl -u bb_worker.service -f
```
