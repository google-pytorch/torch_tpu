# Native Remote Build Execution on Cloud TPU Architecture

This document explains the architecture, networking, and deployment of native
Remote Build Execution (RBE) for `google-pytorch/torch_tpu` on physical Cloud
TPU hardware.

--------------------------------------------------------------------------------

## 1. Problem & Context

Hardware tests for `torch_tpu` require physical Cloud TPU chips (`/dev/vfio/0`).
Testing against real accelerators presents two scaling constraints:

1.  **A Single Shared Runner (`linux-x86-ct5lp-224-8tpu`):** Standard presubmit
    runs all tests sequentially on one dedicated machine. When multiple
    developers push changes, tests queue behind one another, delaying pull
    request verification.
2.  **Managed RBE Lacks TPU Workers:** Managed Remote Build Execution offerings
    (such as Google Cloud Foundry) provision standard CPU worker pools. They do
    not provide Cloud TPU instances as execution backends.

To run hardware tests in parallel across a pool of standing Cloud TPU v5e
virtual machines, `torch_tpu` runs a native Remote Build Execution cluster.

--------------------------------------------------------------------------------

## 2. REAPI v2 Architecture with Buildbarn

The cluster implements the open-source Remote Execution API (REAPI v2) using
Buildbarn. Bazel communicates directly with Buildbarn over gRPC.

```mermaid
flowchart TD
    subgraph CIRunner ["CI Runner (ml-velocity-actions-production)"]
        BazelCI["Bazel Client<br/>bazel test --config=ci_rbe_presubmit"]
        PSCEndpoint["PSC Endpoint (default network)<br/>10.202.0.67:8980"]
        BazelCI --> PSCEndpoint
    end

    subgraph DevClient ["Developer Workstation (Cloudtop)"]
        BazelDev["Bazel Client<br/>bazel test --config=rbe_native_tpu"]
        SSHTunnel["SSH Port Forward<br/>localhost:8980 -> port 22"]
        BazelDev --> SSHTunnel
    end

    subgraph ProducerVPC ["RBE Producer Network (rbe-tpu-oss / us-east5)"]
        PSCSA["PSC Service Attachment<br/>rbe-tpu-service-attachment"]
        ILB["Internal TCP Load Balancer<br/>rbe-internal-ilb:8980"]
        GW["rbe-tunnel-gateway (10.202.0.13:8980)<br/>HAProxy TCP Forwarder"]

        PSCEndpoint -->|"Google Backbone"| PSCSA
        PSCSA --> ILB
        ILB --> GW
        SSHTunnel -->|"Corp SSH (port 22)"| GW
    end

    subgraph ControlPlane ["GKE Control Plane & CPU Workers (rbe-tpu-oss-east5-cluster)"]
        FE["bb-frontend (NodePort 30980)<br/>REAPI v2 Surface & CAS/AC Router"]
        SCHED["bb-scheduler (NodePort 30983)<br/>Queue & Platform Matcher"]
        STOR["bb-storage (ClusterIP 8981)<br/>CAS (32 GB) & Action Cache (1 GB)"]
        CPUWorkers["224-CPU Worker Pool (7 Nodes x 32 Slots)<br/>ml-build Image | Standard, 16 GB, 20 GB Queues"]

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

### Step 1: Worker Registration & Multi-Queue Memory Pools

Workers connect to the scheduler via outbound gRPC sessions, eliminating the
need for inbound firewall rules or public IP addresses. The cluster maintains
two distinct pools:

#### TPU Worker Pool (Hardware Execution)

Each standing TPU VM runs `bb_worker` as a systemd service. On startup,
`bb_worker` dials outbound to the Buildbarn scheduler at `10.202.0.3:30983` over
internal VPC networking and advertises:

```json
"platform": {
  "properties": [
    { "name": "Accelerator", "value": "tpu-v5e" },
    { "name": "OSFamily", "value": "Linux" }
  ]
}
```

Because workers initiate outbound connections, TPU VMs require zero listening
ports, zero public IPs, and no incoming firewall rules.

#### CPU Worker Pool & Multi-Queue Memory (`worker-cpu.jsonnet`)

The CPU worker pool runs inside GKE across 7 nodes with 32 slots each (224
concurrent slots). CPU worker pods run the official `ml-build` runner image:

```
us-central1-docker.pkg.dev/ml-velocity-actions-production/ci-runners/ml-build
```

This replaces legacy generic runner images and ensures compilers, system
libraries, Python runtimes, and LLVM/sysroot dependencies match the CI
environment.

In `worker-cpu.jsonnet`, the pool partitions into three memory queues to handle
different action types without memory exhaustion:

-   **Standard Runner (`OSFamily=Linux`):** Handles standard C++ compilation,
    Python packaging, and genrules.
-   **High-Memory 16 GB Runner (`OSFamily=Linux`, `mem=16g`):** Matches
    `cpp_link.mem=16g` set in `//build_files:build_defs.bzl` (for example, in
    `//csrc/distributed:process_group_tpu_test`). C++ link actions with heavy
    symbol tables or AddressSanitizer (ASan) instrumentation land on this queue.
-   **High-Memory 20 GB Runner (`OSFamily=Linux`, `mem=20g`):** Handles large
    monolithic link steps and heavy binary generation.

### Step 2: Bazel Invocation & Dual-Pool Platform Routing

When a developer or CI runs:

```bash
bazel test --config=rbe_native_tpu //tests/pallas:pallas_test
```

Bazel resolves the execution platform configured in `//bazel/platforms:BUILD.bazel`:

```python
platform(
    name = "buildbarn_native_tpu_platform",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "OSFamily": "Linux",
        "test.Accelerator": "tpu-v5e",
        "test.OSFamily": "Linux",
    },
    visibility = ["//visibility:public"],
)
```

This configuration splits execution across both pools using Bazel execution
groups:

-   **Default Execution Group (`OSFamily=Linux`):** Applies to all compilation,
    linking, and genrules. The scheduler routes these actions to the 224-slot CPU
    worker pool.
-   **Test Execution Group (`test.Accelerator=tpu-v5e`, `test.OSFamily=Linux`):**
    Applies exclusively to Bazel's built-in `test` execution group. The scheduler
    routes test runner actions to the 28 standing Cloud TPU v5e workers.

Two design requirements govern this platform definition:

-   **No `parents` inheritance:** Buildbarn keys worker queues on exact platform
    property matching. Inheriting properties like `cache-silo-key` from
    `@ml_build_config_platform//:platform` causes queue mismatches where actions
    find no matching workers. The platform defines its constraints directly.
-   **Explicit `test.OSFamily`:** Buildbarn treats execution group properties as
    complete sets rather than merging with defaults. Repeating `test.OSFamily=Linux`
    ensures the test action matches the TPU worker's advertised properties
    (`Accelerator=tpu-v5e`, `OSFamily=Linux`).

Additional flags applied from `.bazelrc`:

-   `--jobs=200`: Sized for the 224-slot CPU pool to maximize concurrency. Without
    this flag, Bazel sizes jobs to the local client core count.
-   `--noremote_upload_local_results`: Prevents untrusted local builds from writing
    to the shared cache. Trusted CI runs re-enable uploads in `ci_rbe_presubmit`.
-   `--remote_timeout=3600s`: Extends per-RPC deadlines so concurrent actions
    uploading inputs do not time out under load.

### Step 3: Action Execution on Hardware

1.  The scheduler matches the action to an idle worker matching the requested
    platform properties (CPU pool for compilation, TPU pool for tests).
2.  For test actions, `bb_worker` on the TPU node pulls input runfiles (such as
    `libtpu.so`, Python virtual environments, and test binaries) from `bb-storage`
    into `/tmp/torch_tpu_rbe/build/`.
3.  `bb_worker` delegates execution to `bb_runner` via a local UNIX domain socket
    (`/tmp/torch_tpu_rbe/runner.sock`).
4.  `bb_runner` runs the test command directly on the host OS as root or runner
    user. The process opens `/dev/vfio/0` directly on the physical chip.
5.  `bb_worker` uploads test outputs (such as `test.xml`, `test.log`) to CAS and
    reports exit code `0` to the scheduler.

### Step 4: Remote Action Caching

Buildbarn stores action digests in its Action Cache (AC) backed by `bb-storage`.
When target sources and input hashes do not change, subsequent builds skip
execution entirely:

-   Fresh execution: Runs across the worker pools (~55m cold presubmit).
-   Cached execution: Instant cache hit in **0.348 seconds** (`(cached) PASSED`).

--------------------------------------------------------------------------------

## 4. Network Layout & Multi-Zone Topology

### Tunnel Gateway (`rbe-tunnel-gateway`)

The `rbe-tunnel-gateway` instance (`10.202.0.13`) is a dedicated Debian 12 GCE
virtual machine in zone `us-east5-a`. HAProxy listens on port `8980` and
forwards incoming TCP connections to the GKE NodePort `30980`
(`10.202.0.3:30980`) serving the Buildbarn frontend (`bb-frontend`). It runs in
TCP mode with a one hour idle timeout on each side, so a Bazel gRPC stream stays
up for the length of a build. `option tcplog` only writes its line when a
session closes, so an in-flight build produces no access log entries at all.

A `socat` unit listens on `8443` and forwards to the same backend. It exists
only so the load balancer health check has a port it can reach: the hierarchical
firewall policy permits the health check ranges on 80, 443, 8080, 8443, 9230 and
10402, and a probe to 8980 hits the deny-all rule instead.

### CI Runner Connectivity (Private Service Connect)

GitHub Actions runners in `ml-velocity-actions-production` reach the Buildbarn
frontend through Private Service Connect (PSC).

1.  A consumer forwarding rule in `ml-velocity-actions-production` publishes the
    reserved internal address `10.202.0.67:8980` on the runner's VPC (`default`,
    subnet `10.202.0.0/20` in `us-east5`). No DNS zone is registered
    (`no_automate_dns_zone = true`), so `.bazelrc` dials the address directly.
2.  Traffic routes across Google's private datacenter backbone into the producer
    service attachment (`rbe-tpu-service-attachment`) in `us-east5`. PSC
    rewrites the source to the attachment's NAT subnet (`10.0.100.0/24`), which
    is what the producer firewall rules actually see.
3.  The service attachment hands off to an internal TCP load balancer
    (`rbe-internal-ilb:8980`), which forwards packets to `rbe-tunnel-gateway`.
4.  HAProxy on the gateway forwards the connection to the Buildbarn frontend
    (`bb-frontend`) on GKE NodePort `30980`.

Both sides number their subnets out of `10.202.0.0/20`, so the producer ILB
(`10.202.0.15`), the gateway (`10.202.0.13`) and the consumer endpoint
(`10.202.0.67`) all look like neighbours. They are not. They sit in separate
VPCs in separate projects and only the PSC attachment joins them.

Because packets stay on Google's private backbone, PSC traffic never touches the
public internet and bypasses edge ingress firewall restrictions.

### Developer Connectivity (Cloudtop / Corporate Workstations)

Developers connect to the Buildbarn frontend through `rbe-tunnel-gateway` on
port `8980` using a background SSH port-forwarding tunnel:

```bash
gcloud compute ssh rbe-tunnel-gateway \
  --zone=us-east5-a \
  --project=rbe-tpu-oss \
  -- -N -L 8980:127.0.0.1:8980 &
```

This route connects over Google corporate network CIDRs, which are explicitly
permitted by the project's hierarchical firewall policy (`163641247602` rules 30
and 31) on port 22. Once the local forward is listening, Bazel points to
`grpc://127.0.0.1:8980` as configured in `.bazelrc`.

### GCE Virtual Machine Connectivity (In-VPC)

Virtual machines running inside the `rbe-tpu-oss` VPC network bypass tunnels
entirely. They communicate directly with the Buildbarn frontend over internal IP
`10.202.0.13:8980`:

```bash
bazel test --config=rbe_native_tpu \
  --remote_executor=grpc://10.202.0.13:8980 \
  --remote_cache=grpc://10.202.0.13:8980 \
  //tests/pallas:pallas_test
```

### Security & Ingress Policy (Hierarchical Firewall)

Project `rbe-tpu-oss` enforces an organization-level hierarchical firewall
policy (`163641247602`):

-   **Rules 30 & 31**: Permit ingress exclusively from Google corporate network
    CIDRs on ports 22, 443, and 3389.
-   **Rule 28**: Permits ingress from designated cluster egress IPs on port 22.
-   **Rule 32**: Denies all other external ingress traffic (`0.0.0.0/0`).

Hierarchical firewall rules evaluate before project-level VPC rules. This drops
traffic from unwhitelisted external ranges, including Cloud IAP proxy CIDR
`35.235.240.0/20` and public GitHub runner IPs, at the edge.

Private Service Connect bypasses these edge ingress rules entirely because
consumer traffic enters the VPC through a dedicated service attachment on
Google's private backbone. For developer workstations, traffic connects over
authorized corporate CIDRs via SSH forwarding.

#### Current Security Boundaries

Access control relies on the network transport layer:
-   Private Service Connect uses connection policy `ACCEPT_MANUAL` restricted
    exclusively to consumer project `ml-velocity-actions-production`.
-   The endpoint is an internal RFC 1918 address (`10.202.0.67`) and is not
    internet-routable.
-   Developer access requires authenticated corporate SSH access through Google's
    hierarchical firewall.

#### Authentication Roadmap

Future hardening will add application-layer authentication on the Buildbarn
frontend (`bb-frontend`). CI runners will supply JWT / bearer tokens via
Bazel's `--remote_header` flag, enabling per-caller identity verification and
revocation at the gRPC layer.

### Fleet Distribution & Latency

The GKE control plane lives in `us-east5`. The 28 standing Cloud TPU v5e nodes
sit across four regions in the GCP VPC network (`default`):

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

## 5. Repository Configuration & Bazel Usage

### 5.1 `.bazelrc`

```ini
# Native Remote Build Execution on Cloud TPU v5e hardware via Buildbarn (REAPI v2)
common:rbe_native_tpu --remote_executor=grpc://127.0.0.1:8980
common:rbe_native_tpu --remote_cache=grpc://127.0.0.1:8980
common:rbe_native_tpu --noremote_upload_local_results
common:rbe_native_tpu --extra_execution_platforms="//bazel/platforms:buildbarn_native_tpu_platform"
common:rbe_native_tpu --jobs=200
common:rbe_native_tpu --remote_timeout=3600s
common:rbe_native_tpu --grpc_keepalive_time=15s
common:rbe_native_tpu --grpc_keepalive_timeout=15s
common:rbe_native_tpu --experimental_remote_execution_keepalive=true

# CI config matching .github/workflows/presubmit.yml
common:ci_rbe_presubmit --config=rbe_native_tpu
common:ci_rbe_presubmit --remote_executor=grpc://10.202.0.67:8980
common:ci_rbe_presubmit --remote_cache=grpc://10.202.0.67:8980
common:ci_rbe_presubmit --build_tests_only
common:ci_rbe_presubmit --color=yes
common:ci_rbe_presubmit --curses=no
common:ci_rbe_presubmit --keep_going
common:ci_rbe_presubmit --remote_download_minimal
common:ci_rbe_presubmit --remote_upload_local_results=true
common:ci_rbe_presubmit --build_tag_filters=-nobuild,requires-tpu-v5lite,-requires-tpu-v5lite:8
common:ci_rbe_presubmit --test_tag_filters=presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild,-requires-tpu-v5p,-requires-tpu-v6e,-requires-tpu-v7x,-requires-tpu-v5lite:8,-requires-net:external
test:ci_rbe_presubmit --test_output=errors
```

#### Flag and Filter Explanations

-   **gRPC Keepalives (`--grpc_keepalive_time=15s`, `--grpc_keepalive_timeout=15s`, `--experimental_remote_execution_keepalive=true`):**
    Maintains active pings on gRPC connections every 15 seconds. This prevents
    intermediary stateful firewalls, HAProxy sessions, and NAT gateways from
    dropping idle TCP sockets during extended compilation steps or test
    executions.
-   **Distributed 8-Chip Exclusion (`-requires-tpu-v5lite:8`):**
    The standing TPU pool consists of single-chip v5e (v5lite-1) nodes. Tests
    requiring an 8-chip slice topology cannot execute on single-chip worker nodes
    and run instead on dedicated multi-chip runners (such as
    `linux-x86-ct5lp-224-8tpu`).
-   **External Network Exclusion (`-requires-net:external`):**
    Excludes tests (such as `remote_tier3_compilation_cache_test`) that connect
    to external network services or Google Cloud Storage. Buildbarn RBE workers
    operate hermetically with no outbound internet access and no cloud storage
    write credentials.

### 5.2 Running Tests Locally via Bazel

With the tunnel running on `localhost:8980`, execute tests directly with Bazel:

```bash
# Run an individual test target on a remote Cloud TPU:
bazel test --config=rbe_native_tpu //tests/pallas:pallas_test

# Run an entire test suite matching CI filters. Use rbe_native_tpu, not
# ci_rbe_presubmit: the CI config points at the PSC endpoint, which is only
# routable from inside the CI VPC.
bazel test --config=rbe_native_tpu \
  --test_tag_filters=presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild \
  //tests:ops_test_tpu-v5lite
```

To run multiple tests in parallel across the standing TPU fleet, increase test
jobs:

```bash
bazel test --config=rbe_native_tpu --jobs=28 //tests/...
```

### 5.3 Presubmit CI Workflow Details (`.github/workflows/presubmit.yml`)

The native RBE workflow runs as a first-class job named `rbe_presubmit`
(`Presubmit on RBE`) directly inside `.github/workflows/presubmit.yml`.

#### Runner Environment

The job executes on a standard CPU runner:
-   **Runner:** `linux-x86-n4-16`
-   **Container:** `us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build@sha256:480f397c57207671315a27173ccec790ea9c66ef030af96cbe6ac9b4a6613996`
-   **IPC:** `--ipc=host`

The runner executes `bazel test --config=ci_rbe_presubmit` directly against
`10.202.0.67:8980` over Private Service Connect using `bazel-diff`. Because PSC
provides a direct internal route across VPCs, the runner requires no background
tunnels or proxy daemons.

#### Label Event Guard

GitHub Actions fires `pull_request` triggers on every label event. To prevent
unrelated label edits from triggering builds or cancelling running jobs,
`rbe_presubmit` enforces a label event guard:

```yaml
if: >-
  (
    github.event_name != 'pull_request'
    || github.event.action != 'labeled'
    || contains(fromJSON('["ci:replace-tpu-v5", "ci:relay-tpu-v5", "run-rbe"]'), github.event.label.name)
  ) && (
    github.event_name == 'workflow_dispatch'
    || contains(github.event.pull_request.labels.*.name, 'ci:replace-tpu-v5')
    || contains(github.event.pull_request.labels.*.name, 'ci:relay-tpu-v5')
    || contains(github.event.pull_request.labels.*.name, 'run-rbe')
  )
```

When a pull request triggers a `labeled` event, this guard verifies that
`github.event.label.name` is one of the opt-in labels (`ci:replace-tpu-v5`,
`ci:relay-tpu-v5`, or `run-rbe`). If a user adds an unrelated label (such as
`automerge` or `documentation`) to an already labeled PR, the workflow ignores
the event and leaves in-progress jobs undisturbed.

When an opt-in label is present, the standard presubmit job (`run_tests`) skips
its single-machine `ci_tpu_v5_presubmit` run, delegating all TPU v5 testing to
`rbe_presubmit`.

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

--------------------------------------------------------------------------------

## 7. Performance & Verified Benchmarks

Real end-to-end presubmit runs verify build acceleration across both pools:

| Metric | Cold Presubmit Run | Warm Presubmit Run (Cached) | Notes |
| :--- | :--- | :--- | :--- |
| **Total Duration** | ~55 minutes | **23 minutes 19 seconds** | **58% overall speedup** |
| **Action Cache Hits** | 0 actions | **>20,000 actions** | Instant cache hits from Action Cache |
| **CPU Actions Executed** | Full C++ compilation | 8 actions | CPU phase completes in ~4 minutes |
| **TPU Test Targets** | 72 test targets | 72 test targets | Dispatched in parallel across 28 nodes |
| **Test Pass Rate** | 71 / 72 passed (98.6%) | 71 / 72 passed (98.6%) | Expected passing rate |

### Cold vs. Warm Execution Breakdown

1.  **Cold Run (~55m):**
    -   Compiles all C++, LLVM/OpenXLA dependencies, and PyTorch extensions across
        the 224 CPU worker slots.
    -   Dispatches 72 test targets in parallel across the 28 standing Cloud TPU v5e
        nodes.

2.  **Warm Run (23m 19s):**
    -   More than 20,000 compilation and toolchain actions hit the Action Cache
        instantly.
    -   Bazel executes only 8 non-cached actions on the CPU pool, finishing the CPU
        phase in about 4 minutes.
    -   All 72 test targets run across the 28 TPU workers in parallel, finishing in
        ~19 minutes.
    -   71 of 72 tests pass cleanly. The filtered 8-chip distributed tests and
        external-network tests are excluded by `-requires-tpu-v5lite:8` and
        `-requires-net:external`.
